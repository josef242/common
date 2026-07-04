#!/usr/bin/env python
"""
test_swa_cache.py — parity tests for the SWA ROLLING WINDOW KV cache.

Ground truth = the TRAINING forward (windowed flex/SDPA attention over the full
sequence), which the cached generation path must reproduce position-by-position:
logits_cached(p) == logits_train(p) for every p, including far past the window.

Pure CPU (tiny dims): both rigs are production — no GPU required or touched.
    python test_swa_cache.py
"""
import sys

import torch

sys.path.insert(0, ".")
from model_v2 import Transformer, ModelArgs

PASS = [0]
FAIL = [0]
TOL = 3e-4  # fp32 CPU; flex-vs-SDPA kernel-order differences dominate


def check(name, cond, extra=""):
    ok = bool(cond)
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  {extra}" if extra and not ok else ""))
    (PASS if ok else FAIL)[0] += 1
    return ok


def make_model(**over):
    kw = dict(dim=64, n_layers=4, n_heads=4, n_kv_heads=2, vocab_size=64,
              max_seq_len=256, dropout=0.0, qk_norm_mode="before_rope",
              use_keel=True, use_activation_checkpointing=False,
              tie_word_embeddings=False, bos_token_id=5)
    kw.update(over)
    torch.manual_seed(11)
    return Transformer(ModelArgs(**kw)).float().eval()


def train_forward_logits(model, tokens):
    """Ground truth: the training/eval forward (windowed semantics for SWA)."""
    with torch.no_grad():
        logits, _ = model(tokens)
    return logits[0]  # [S, V]


def cached_logits(model, tokens, chunks):
    """Drive generate-style cached forwards over `chunks` (list of lengths
    summing to S); return per-position last-layer logits [S, V]."""
    S = tokens.shape[1]
    model.setup_caches(max_batch_size=1, max_seq_len=S, force=True)
    outs = []
    pos = 0
    with torch.no_grad():
        for n in chunks:
            logits, _ = model(tokens[:, pos:pos + n], start_pos=pos)
            outs.append(logits[0])
            pos += n
    model.clear_caches()
    return torch.cat(outs, dim=0)  # [S, V]


def max_diff(a, b):
    return (a - b).abs().max().item()


# ============================== tests ==============================

def t_global_unchanged():
    print("non-SWA model: cached path == training forward (original semantics intact)")
    m = make_model()
    tokens = torch.randint(6, 64, (1, 96))
    ref = train_forward_logits(m, tokens)
    # prefill-then-decode
    got = cached_logits(m, tokens, [64] + [1] * 32)
    check("prefill+decode parity", max_diff(ref, got) < TOL, f"{max_diff(ref, got):.2e}")
    # capacity bookkeeping
    m.setup_caches(max_batch_size=1, max_seq_len=96, force=True)
    check("capacity reports logical msl", m.cache_capacity() == (1, 96),
          str(m.cache_capacity()))
    ck = m.layers[0].attention.cache_k
    check("global cache is full-length", ck.shape[1] == 96, str(ck.shape))
    m.clear_caches()


def t_windowed_decode_parity():
    print("ALL-LOCAL swa model, seq >> W: per-position parity through the wrap")
    W = 16
    m = make_model(swa_enabled=True, swa_window=W, swa_global_interleave=10**6)
    S = 80  # 5x the window
    tokens = torch.randint(6, 64, (1, S))
    ref = train_forward_logits(m, tokens)
    got = cached_logits(m, tokens, [1] * S)  # pure decode from position 0
    d_pre = max_diff(ref[:W], got[:W])
    d_post = max_diff(ref[W:], got[W:])
    check(f"parity BEFORE wrap (pos < {W})", d_pre < TOL, f"{d_pre:.2e}")
    check(f"parity AFTER wrap  (pos >= {W}) — the rolling window", d_post < TOL,
          f"{d_post:.2e}")
    # cache shape: local layers hold only W slots
    m.setup_caches(max_batch_size=1, max_seq_len=S, force=True)
    ck = m.layers[0].attention.cache_k
    check("local cache holds W slots, not S", ck.shape[1] == W, str(ck.shape))
    check("capacity still reports logical msl", m.cache_capacity() == (1, S))
    m.clear_caches()


def t_hybrid_parity():
    print("HYBRID model (interleave 4): local+global layers, long sequence")
    W = 16
    m = make_model(swa_enabled=True, swa_window=W, swa_global_interleave=4)
    S = 72
    tokens = torch.randint(6, 64, (1, S))
    ref = train_forward_logits(m, tokens)
    got = cached_logits(m, tokens, [40] + [1] * (S - 40))  # prefill > W, then decode
    d = max_diff(ref, got)
    check("prefill(>W)+decode parity across hybrid stack", d < TOL, f"{d:.2e}")
    m.setup_caches(max_batch_size=1, max_seq_len=S, force=True)
    shapes = [blk.attention.cache_k.shape[1] for blk in m.layers]
    check("per-layer cache lengths follow the local/global census",
          shapes == [W, W, W, S], str(shapes))
    m.clear_caches()


def t_chunk_continuation():
    print("cross-turn suffix chunks (start_pos>0, S>1) — the chat pattern")
    W = 16
    m = make_model(swa_enabled=True, swa_window=W, swa_global_interleave=4)
    S = 90
    tokens = torch.randint(6, 64, (1, S))
    ref = train_forward_logits(m, tokens)
    # prefill 30 (chunk > W), suffix chunk 25, suffix chunk 20, then decodes
    got = cached_logits(m, tokens, [30, 25, 20] + [1] * 15)
    d = max_diff(ref, got)
    check("multi-chunk continuation parity", d < TOL, f"{d:.2e}")
    # chunk LONGER than the window in continuation position
    got2 = cached_logits(m, tokens, [20, 50] + [1] * 20)
    d2 = max_diff(ref, got2)
    check("continuation chunk longer than W", d2 < TOL, f"{d2:.2e}")


def t_swa_with_docmask_gen():
    print("swa + doc-mask model: generation path (mask off at inference) still parity")
    # doc flags change TRAINING semantics; generate path never builds doc masks.
    # Ground truth here: a training forward on a token stream with NO BOS, where
    # doc-mask==causal (t_no_bos_parity precedent) -> windowed-only semantics.
    W = 16
    m = make_model(swa_enabled=True, swa_window=W, swa_global_interleave=4,
                   doc_attn_mask=True, doc_pos_reset=True)
    S = 60
    tokens = torch.randint(6, 64, (1, S))  # ids 6.. : no BOS(5) anywhere
    ref = train_forward_logits(m, tokens)
    got = cached_logits(m, tokens, [24] + [1] * (S - 24))
    d = max_diff(ref, got)
    check("no-BOS stream: cached gen == training forward", d < TOL, f"{d:.2e}")


def t_cache_growth_preserve():
    print("cache growth preserve: ring buffers survive a max_seq_len extension")
    W = 16
    m = make_model(swa_enabled=True, swa_window=W, swa_global_interleave=4)
    tokens = torch.randint(6, 64, (1, 100))
    ref = train_forward_logits(m, tokens[:, :100])
    # run 40 positions at msl=60, GROW to 100 (preserve path), continue
    m.setup_caches(max_batch_size=1, max_seq_len=60, force=True)
    outs = []
    with torch.no_grad():
        logits, _ = m(tokens[:, :40], start_pos=0)
        outs.append(logits[0])
        m.setup_caches(max_batch_size=1, max_seq_len=100)  # growth, no force
        for p in range(40, 100):
            logits, _ = m(tokens[:, p:p + 1], start_pos=p)
            outs.append(logits[0])
    got = torch.cat(outs, dim=0)
    d = max_diff(ref, got)
    check("parity across a preserve-growth boundary", d < TOL, f"{d:.2e}")
    m.clear_caches()
    check("clear resets logical capacity", m.cache_capacity() == (None, None))


def t_rewind_contract():
    print("rewind contract: post-wrap rewind is UNSOUND at model level (the sharp edge,")
    print("pinned) and min_rolling_cache_len exposes the guard threshold for callers")
    W = 16
    m = make_model(swa_enabled=True, swa_window=W, swa_global_interleave=10**6)
    S = 80
    tokens = torch.randint(6, 64, (1, S))
    ref = train_forward_logits(m, tokens)
    m.setup_caches(max_batch_size=1, max_seq_len=S, force=True)
    with torch.no_grad():
        m(tokens[:, :S], start_pos=0)                       # materialize; rings wrap 5x
        rew, _ = m(tokens[:, 40:41], start_pos=40)          # REWIND decode behind HWM
    d_rewind = max_diff(ref[40], rew[0, 0])
    check("post-wrap rewind DIVERGES at model level (callers MUST guard — if this "
          "ever passes, the guard can be removed)", d_rewind > 1e-3, f"{d_rewind:.2e}")
    check("min_rolling_cache_len reports the ring size", m.min_rolling_cache_len() == W,
          str(m.min_rolling_cache_len()))
    m.clear_caches()
    # hybrid + non-SWA reporting
    mh = make_model(swa_enabled=True, swa_window=W, swa_global_interleave=4)
    mh.setup_caches(max_batch_size=1, max_seq_len=64, force=True)
    check("hybrid reports min ring (local W), not global len",
          mh.min_rolling_cache_len() == W)
    mh.clear_caches()
    mp = make_model()
    mp.setup_caches(max_batch_size=1, max_seq_len=64, force=True)
    check("non-SWA model reports None (rewind stays allowed)",
          mp.min_rolling_cache_len() is None)
    mp.clear_caches()
    # UNWRAPPED rewind stays sound (ledger <= ring): the guard's fast path
    m2 = make_model(swa_enabled=True, swa_window=W, swa_global_interleave=10**6)
    short = tokens[:, :12]  # 12 < W: ring never wraps
    ref2 = train_forward_logits(m2, short)
    m2.setup_caches(max_batch_size=1, max_seq_len=12, force=True)
    with torch.no_grad():
        m2(short, start_pos=0)
        rew2, _ = m2(short[:, 8:9], start_pos=8)            # rewind, no wrap
    d2 = max_diff(ref2[8], rew2[0, 0])
    check("UNWRAPPED rewind is sound (guard's ledger<=ring fast path)", d2 < TOL,
          f"{d2:.2e}")
    m2.clear_caches()


def main():
    print(f"\n=== SWA rolling-cache parity tests (CPU, torch {torch.__version__}) ===\n")
    for t in (t_global_unchanged, t_windowed_decode_parity, t_hybrid_parity,
              t_chunk_continuation, t_swa_with_docmask_gen, t_cache_growth_preserve,
              t_rewind_contract):
        t()
        print()
    print(f"=== {PASS[0]} passed, {FAIL[0]} failed ===")
    sys.exit(1 if FAIL[0] else 0)


if __name__ == "__main__":
    main()
