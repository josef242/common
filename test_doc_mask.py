#!/usr/bin/env python
"""
test_doc_mask.py — behavior tests for the doc_attn_mask feature (branch doc-mask).

Packed windows separate documents with BOS; doc_attn_mask confines attention to
(causal AND same-document) via FlexAttention; doc_pos_reset restarts RoPE
positions at each BOS. Drives the REAL Attention/Transformer modules against
reference implementations and property tests. Needs a CUDA device (FlexAttention);
run on a rig:

    python test_doc_mask.py
"""
import math
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, ".")
from model_v2 import (
    ModelArgs, Attention, Transformer,
    doc_ids_from_tokens, doc_position_ids,
    apply_rotary_emb, precompute_freqs_cis,
    flex_attention, create_block_mask,
)

PASS = [0]
FAIL = [0]
DEV = "cuda"
BOS = 5  # test-vocab BOS id (feature is id-configurable; production uses 32000)


def check(name, cond, extra=""):
    ok = bool(cond)
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  {extra}" if extra and not ok else ""))
    (PASS if ok else FAIL)[0] += 1
    return ok


def make_block_mask(tokens):
    doc = doc_ids_from_tokens(tokens, BOS)

    def mask_mod(b, h, q_idx, kv_idx):
        return (q_idx >= kv_idx) & (doc[b, q_idx] == doc[b, kv_idx])

    B, S = tokens.shape
    return create_block_mask(mask_mod, B=B, H=None, Q_LEN=S, KV_LEN=S, device=tokens.device)


def ref_doc_attention(q, k, v, doc, n_rep):
    """Reference: explicit dense mask, fp32 softmax. q,k,v: [B, H, S, D]."""
    if n_rep > 1:
        k = k.repeat_interleave(n_rep, dim=1)
        v = v.repeat_interleave(n_rep, dim=1)
    B, H, S, D = q.shape
    scores = (q @ k.transpose(-2, -1)) / math.sqrt(D)
    causal = torch.tril(torch.ones(S, S, dtype=torch.bool, device=q.device))
    same_doc = doc[:, :, None] == doc[:, None, :]           # [B, S, S]
    allowed = (causal[None] & same_doc)[:, None]             # [B, 1, S, S]
    scores = scores.masked_fill(~allowed, float("-inf"))
    return (F.softmax(scores.float(), dim=-1).to(q.dtype) @ v)


def rand_packed_tokens(B, S, vocab=64, mean_doc=48, seed=0):
    """Random windows with geometric-ish document lengths (BOS separators)."""
    g = torch.Generator().manual_seed(seed)
    t = torch.randint(6, vocab, (B, S), generator=g)  # ids 6.. leave BOS=5 unused
    for b in range(B):
        pos = int(torch.randint(1, mean_doc, (1,), generator=g))
        while pos < S:
            t[b, pos] = BOS
            pos += max(2, int(torch.randint(mean_doc // 2, mean_doc * 2, (1,), generator=g)))
    return t.to(DEV)


# ============================== tests ==============================

def t_doc_ids():
    print("doc ids + positions: BOS starts a new doc; fragment keeps window-relative positions")
    t = torch.tensor([[7, 8, BOS, 9, 10, BOS, BOS, 11]])
    ids = doc_ids_from_tokens(t, BOS)
    check("doc ids exact", ids.tolist() == [[0, 0, 1, 1, 1, 2, 3, 3]], str(ids.tolist()))
    pos = doc_position_ids(t, BOS)
    check("positions exact", pos.tolist() == [[0, 1, 0, 1, 2, 0, 0, 1]], str(pos.tolist()))
    t2 = torch.tensor([[7, 8, 9, 10]])  # no BOS anywhere
    check("no-BOS window: ids all 0", doc_ids_from_tokens(t2, BOS).unique().tolist() == [0])
    check("no-BOS window: positions = arange", doc_position_ids(t2, BOS).tolist() == [[0, 1, 2, 3]])


def t_flex_vs_reference():
    print("flex doc-masked output == dense reference (fp32, GQA)")
    B, S, Hq, Hkv, D = 2, 256, 4, 2, 32
    torch.manual_seed(1)
    tokens = rand_packed_tokens(B, S)
    doc = doc_ids_from_tokens(tokens, BOS)
    q = torch.randn(B, Hq, S, D, device=DEV)
    k = torch.randn(B, Hkv, S, D, device=DEV)
    v = torch.randn(B, Hkv, S, D, device=DEV)
    bm = make_block_mask(tokens)
    out = flex_attention(q, k, v, block_mask=bm, enable_gqa=True)
    ref = ref_doc_attention(q, k, v, doc, n_rep=Hq // Hkv)
    diff = (out - ref).abs().max().item()
    check("max |flex - ref| < 1e-4", diff < 1e-4, f"diff={diff:.3e}")


def t_no_bos_parity():
    print("window without BOS: doc-masked flex == plain SDPA causal")
    B, S, Hq, Hkv, D = 2, 256, 4, 2, 32
    torch.manual_seed(2)
    tokens = torch.randint(6, 64, (B, S), device=DEV)  # no BOS
    q = torch.randn(B, Hq, S, D, device=DEV)
    k = torch.randn(B, Hkv, S, D, device=DEV)
    v = torch.randn(B, Hkv, S, D, device=DEV)
    bm = make_block_mask(tokens)
    out = flex_attention(q, k, v, block_mask=bm, enable_gqa=True)
    sdpa = F.scaled_dot_product_attention(q, k, v, is_causal=True, enable_gqa=True)
    diff = (out - sdpa).abs().max().item()
    check("max |flex - sdpa| < 1e-4", diff < 1e-4, f"diff={diff:.3e}")


def t_attention_module():
    print("Attention module: block_mask arg wires through (KEEL-style args, before_rope qk-norm)")
    args = ModelArgs(dim=64, n_layers=2, n_heads=4, n_kv_heads=2, vocab_size=64,
                     max_seq_len=256, dropout=0.0, qk_norm_mode="before_rope",
                     use_activation_checkpointing=False, tie_word_embeddings=False)
    torch.manual_seed(3)
    attn = Attention(args).to(DEV).float()
    S = 256
    fc, fs = precompute_freqs_cis(64 // 4, S)
    fc, fs = fc.to(DEV), fs.to(DEV)
    x = torch.randn(2, S, 64, device=DEV)
    tokens = rand_packed_tokens(2, S, seed=3)
    bm = make_block_mask(tokens)
    out_causal = attn(x, fc, fs)                 # block_mask=None -> SDPA path
    out_masked = attn(x, fc, fs, bm)             # flex path
    check("masked differs from causal when BOS present",
          (out_causal - out_masked).abs().max().item() > 1e-6)
    tokens_nb = torch.randint(6, 64, (2, S), device=DEV)
    bm_nb = make_block_mask(tokens_nb)
    out_masked_nb = attn(x, fc, fs, bm_nb)
    diff = (out_causal - out_masked_nb).abs().max().item()
    check("no-BOS mask reproduces causal through the module", diff < 1e-4, f"diff={diff:.3e}")


def t_pos_reset_rope():
    print("doc_pos_reset: gathered per-sample freqs == per-document fresh rope")
    D_head, S = 32, 64
    fc, fs = precompute_freqs_cis(D_head, S)
    fc, fs = fc.to(DEV), fs.to(DEV)
    t = torch.tensor([[7, 8, BOS, 9, 10, BOS, 11, 12]], device=DEV)
    pos = doc_position_ids(t, BOS)
    xq = torch.randn(1, 8, 2, D_head, device=DEV)
    xk = torch.randn(1, 8, 2, D_head, device=DEV)
    q3, k3 = apply_rotary_emb(xq, xk, fc[pos], fs[pos])       # 3D gathered freqs
    # reference: rotate each token individually at its within-doc position
    q_ref = torch.empty_like(q3)
    k_ref = torch.empty_like(k3)
    for i in range(8):
        p = int(pos[0, i])
        qi, ki = apply_rotary_emb(xq[:, i:i+1], xk[:, i:i+1], fc[p:p+1], fs[p:p+1])
        q_ref[:, i:i+1] = qi
        k_ref[:, i:i+1] = ki
    diff = max((q3 - q_ref).abs().max().item(), (k3 - k_ref).abs().max().item())
    check("max |gathered - per-token| < 1e-5", diff < 1e-5, f"diff={diff:.3e}")


def _tiny_model(**over):
    kw = dict(dim=64, n_layers=4, n_heads=4, n_kv_heads=2, vocab_size=64,
              max_seq_len=256, dropout=0.0, qk_norm_mode="before_rope",
              use_keel=True, use_activation_checkpointing=False,
              tie_word_embeddings=False, bos_token_id=BOS)
    kw.update(over)
    torch.manual_seed(4)
    return Transformer(ModelArgs(**kw)).to(DEV).float().eval()


def t_independence():
    print("independence (the property that IS the feature): perturbing doc 0 must not move doc 1")
    S = 256
    m_mask = _tiny_model(doc_attn_mask=True, doc_pos_reset=True)
    m_free = _tiny_model()  # same seed -> identical weights
    tokens = torch.randint(6, 64, (1, S), device=DEV)
    tokens[0, 100] = BOS  # doc 0 = [0,100), doc 1 = [100, 256)
    tokens_pert = tokens.clone()
    tokens_pert[0, :100] = torch.randint(6, 64, (100,), device=DEV)  # rewrite doc 0
    with torch.no_grad():
        la, _ = m_mask(tokens)
        lb, _ = m_mask(tokens_pert)
        fa, _ = m_free(tokens)
        fb, _ = m_free(tokens_pert)
    d_masked = (la[0, 100:] - lb[0, 100:]).abs().max().item()
    d_free = (fa[0, 100:] - fb[0, 100:]).abs().max().item()
    check("masked: doc-1 logits identical under doc-0 rewrite", d_masked == 0.0,
          f"diff={d_masked:.3e}")
    check("unmasked: doc-1 logits DO move (sanity that the test can fail)", d_free > 1e-4,
          f"diff={d_free:.3e}")


def t_model_train_smoke():
    print("training smoke: flags on + KEEL + activation checkpointing, fwd/bwd, finite loss")
    # bf16: the CCE backward kernel requires bf16/fp16 embeddings (production dtype)
    m = _tiny_model(doc_attn_mask=True, doc_pos_reset=True,
                    use_activation_checkpointing=True).to(torch.bfloat16).train()
    tokens = rand_packed_tokens(2, 256, seed=5)
    targets = torch.roll(tokens, -1, dims=1)
    _, loss = m(tokens, targets=targets)
    check("loss finite", loss is not None and torch.isfinite(loss).item(), f"loss={loss}")
    loss.backward()
    g = [p.grad for p in m.parameters() if p.grad is not None]
    check("grads exist", len(g) > 0)
    check("grads finite", all(torch.isfinite(t).all().item() for t in g))


def t_swa():
    print("SWA: window independence, hybrid dispatch, W>=S degeneracy, doc composition")
    S = 256
    # single ALL-LOCAL layer (interleave huge -> no global layers): a token beyond the
    # window must have EXACTLY zero influence (multi-layer would re-propagate through
    # depth, so n_layers=1 isolates the window semantics)
    m1 = _tiny_model(n_layers=1, swa_enabled=True, swa_window=64,
                     swa_global_interleave=10**6)
    tokens = torch.randint(6, 64, (1, S), device=DEV)
    tokens_pert = tokens.clone()
    tokens_pert[0, 10] = (tokens[0, 10] + 1) % 58 + 6  # perturb ONE early token
    with torch.no_grad():
        a, _ = m1(tokens)
        b, _ = m1(tokens_pert)
    d_near = (a[0, 11:74] - b[0, 11:74]).abs().max().item()   # inside window of pos 10
    d_far = (a[0, 100:] - b[0, 100:]).abs().max().item()      # beyond window reach
    check("inside-window positions DO move", d_near > 1e-6, f"{d_near:.2e}")
    check("beyond-window positions are BIT-identical (1 local layer)", d_far == 0.0,
          f"{d_far:.2e}")
    # hybrid: layer 4k-1 global -> perturbation reaches far positions again
    mh = _tiny_model(n_layers=4, swa_enabled=True, swa_window=64, swa_global_interleave=4)
    with torch.no_grad():
        ha, _ = mh(tokens)
        hb, _ = mh(tokens_pert)
    check("hybrid: global layer restores long-range reach",
          (ha[0, 100:] - hb[0, 100:]).abs().max().item() > 1e-6)
    check("hybrid layer kinds: 3 local + 1 global",
          [blk.swa_local for blk in mh.layers] == [True, True, True, False],
          str([blk.swa_local for blk in mh.layers]))
    # W >= S degenerates to full causal: must match the plain model exactly
    mw = _tiny_model(n_layers=2, swa_enabled=True, swa_window=S, swa_global_interleave=4)
    mp = _tiny_model(n_layers=2)  # same seed -> identical weights
    with torch.no_grad():
        wa, _ = mw(tokens)
        pa, _ = mp(tokens)
    dW = (wa - pa).abs().max().item()
    check("W>=S == plain causal through the model", dW < 1e-4, f"{dW:.3e}")
    # composition: swa + doc mask — cross-doc leak blocked even INSIDE the window
    mc = _tiny_model(n_layers=1, swa_enabled=True, swa_window=S,
                     swa_global_interleave=10**6, doc_attn_mask=True)
    t2 = torch.randint(6, 64, (1, S), device=DEV)
    t2[0, 50] = BOS
    t2p = t2.clone()
    t2p[0, :50] = torch.randint(6, 64, (50,), device=DEV)
    with torch.no_grad():
        ca, _ = mc(t2)
        cb, _ = mc(t2p)
    d_doc = (ca[0, 50:] - cb[0, 50:]).abs().max().item()
    check("window + doc: doc-0 rewrite cannot leak into doc-1 (in-window)", d_doc == 0.0,
          f"{d_doc:.2e}")


def t_meta_init_freqs():
    print("meta-init regression: to_empty clobbers RoPE buffers; init_weights must refill them")
    args = ModelArgs(dim=64, n_layers=2, n_heads=4, n_kv_heads=2, vocab_size=64,
                     max_seq_len=128, dropout=0.0, qk_norm_mode="before_rope",
                     use_keel=True, use_activation_checkpointing=False,
                     tie_word_embeddings=False)
    with torch.device("meta"):
        m = Transformer(args)
    m = m.to_empty(device=DEV)          # clobbers the value-carrying buffers
    torch.manual_seed(7)
    m.init_weights()                     # must recompute the tables
    ref_cos, ref_sin = precompute_freqs_cis(64 // 4, 128, args.rope_theta)
    ok_c = torch.allclose(m.freqs_cos.cpu().float(), ref_cos, atol=1e-5)
    ok_s = torch.allclose(m.freqs_sin.cpu().float(), ref_sin, atol=1e-5)
    check("freqs_cos correct after meta->to_empty->init_weights", ok_c,
          f"absmax={m.freqs_cos.float().abs().max().item():.4f}")
    check("freqs_sin correct after meta->to_empty->init_weights", ok_s)


def t_compile_smoke():
    print("torch.compile smoke: compiled Attention with BlockMask input == eager")
    args = ModelArgs(dim=64, n_layers=2, n_heads=4, n_kv_heads=2, vocab_size=64,
                     max_seq_len=256, dropout=0.0, qk_norm_mode="before_rope",
                     use_activation_checkpointing=False, tie_word_embeddings=False)
    torch.manual_seed(6)
    attn = Attention(args).to(DEV).float()
    attn_c = torch.compile(attn)
    S = 256
    fc, fs = precompute_freqs_cis(64 // 4, S)
    fc, fs = fc.to(DEV), fs.to(DEV)
    x = torch.randn(2, S, 64, device=DEV)
    bm = make_block_mask(rand_packed_tokens(2, S, seed=6))
    eager = attn(x, fc, fs, bm)
    comp = attn_c(x, fc, fs, bm)
    diff = (eager - comp).abs().max().item()
    check("max |eager - compiled| < 2e-4", diff < 2e-4, f"diff={diff:.3e}")
    # second batch with a DIFFERENT BlockMask must not recompile into wrong results
    bm2 = make_block_mask(rand_packed_tokens(2, S, seed=7))
    diff2 = (attn(x, fc, fs, bm2) - attn_c(x, fc, fs, bm2)).abs().max().item()
    check("fresh BlockMask on same compiled module still correct", diff2 < 2e-4,
          f"diff={diff2:.3e}")


def main():
    if not torch.cuda.is_available():
        print("CUDA required (FlexAttention) — run on a rig.")
        sys.exit(2)
    print(f"\n=== doc_attn_mask tests (torch {torch.__version__}, {torch.cuda.get_device_name(0)}) ===\n")
    for t in (t_doc_ids, t_flex_vs_reference, t_no_bos_parity, t_attention_module,
              t_pos_reset_rope, t_independence, t_model_train_smoke, t_swa,
              t_meta_init_freqs, t_compile_smoke):
        t()
        print()
    print(f"=== {PASS[0]} passed, {FAIL[0]} failed ===")
    sys.exit(1 if FAIL[0] else 0)


if __name__ == "__main__":
    main()
