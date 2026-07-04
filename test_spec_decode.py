#!/usr/bin/env python
"""
test_spec_decode.py — correctness tests for MTP self-speculative decoding.

THE property: speculative decoding is an ACCELERATION, not an approximation —
greedy spec output must equal greedy vanilla output token-for-token, regardless
of acceptance rate (an untrained model rejects most drafts: the rejection path
gets hammered). Runs on CPU; both rigs are production.

    python test_spec_decode.py
"""
import sys

import torch

sys.path.insert(0, ".")
from model_v2 import Transformer, ModelArgs
import neo_common as nc

PASS = [0]
FAIL = [0]


def check(name, cond, extra=""):
    ok = bool(cond)
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  {extra}" if extra and not ok else ""))
    (PASS if ok else FAIL)[0] += 1
    return ok


class StubTok:
    eos_id = None
    def encode(self, text, bos=True, eos=False): raise RuntimeError("use prompt_ids")
    # composition-consistent: decode(a+b) == decode(a)+decode(b), so the
    # token-by-token streaming path and full-list decode agree (required for
    # stop-sequence tests to see the same text the stream accumulates)
    def decode(self, ids): return "".join(f"<{i}>" for i in ids)


def make_model(vocab=16, **over):
    kw = dict(dim=64, n_layers=4, n_heads=4, n_kv_heads=2, vocab_size=vocab,
              max_seq_len=256, dropout=0.0, qk_norm_mode="before_rope",
              use_keel=True, use_activation_checkpointing=False,
              tie_word_embeddings=False, bos_token_id=5, mtp_enabled=True,
              swa_enabled=True, swa_window=16, swa_global_interleave=4,
              doc_attn_mask=True, doc_pos_reset=True)
    kw.update(over)
    torch.manual_seed(23)
    return Transformer(ModelArgs(**kw)).float().eval()


def vanilla_greedy(model, prompt_ids, n_new):
    total = len(prompt_ids) + n_new
    model.setup_caches(max_batch_size=1, max_seq_len=total, force=True)
    out = []
    with torch.no_grad():
        toks = torch.tensor([prompt_ids], dtype=torch.long)
        logits = model.generate_forward(toks, 0)
        out.append(int(logits[0, -1].argmax()))
        pos = len(prompt_ids)
        while len(out) < n_new:
            logits = model.generate_forward(
                torch.tensor([[out[-1]]], dtype=torch.long), pos)
            out.append(int(logits[0, -1].argmax()))
            pos += 1
    model.clear_caches()
    return out


# ============================== tests ==============================

def t_greedy_equality():
    print("THE property: greedy spec == greedy vanilla, token-for-token (full stack)")
    m = make_model()
    prompt = [5, 7, 9, 11, 6, 8, 10, 12, 7, 9, 6, 13]
    n_new = 48
    ref = vanilla_greedy(m, prompt, n_new)
    m.setup_caches(max_batch_size=1, max_seq_len=len(prompt) + n_new, force=True)
    res = nc.spec_generate(m, StubTok(), None, n_new, len(prompt) + n_new,
                           temperature=0.0, prompt_ids=prompt)
    m.clear_caches()
    check("token-for-token equality", res["token_ids"] == ref,
          f"spec={res['token_ids'][:10]}... ref={ref[:10]}...")
    bits = res["accept_bits"]
    print(f"       (rounds={res['rounds']}, acceptance={res['acceptance_rate']:.2f}, "
          f"both-branches-exercised={'yes' if 0 in bits and 1 in bits else 'NO — rerun with different arch'})")
    check("ran a sensible number of rounds", 0 < res["rounds"] <= n_new)
    check("emitted exactly n_new", res["tokens_generated"] == n_new)


def t_greedy_equality_no_swa():
    print("greedy equality on a plain-MTP model (full-length caches)")
    m = make_model(swa_enabled=False, doc_attn_mask=False, doc_pos_reset=False)
    prompt = [5, 6, 7, 8, 9, 10]
    n_new = 40
    ref = vanilla_greedy(m, prompt, n_new)
    m.setup_caches(max_batch_size=1, max_seq_len=len(prompt) + n_new, force=True)
    res = nc.spec_generate(m, StubTok(), None, n_new, len(prompt) + n_new,
                           temperature=0.0, prompt_ids=prompt)
    m.clear_caches()
    check("token-for-token equality", res["token_ids"] == ref)


def t_accept_branch_coverage():
    print("accept-branch coverage: tiny vocab forces argmax collisions (greedy) and")
    print("high temperature forces near-uniform p~q (sampled) — equality must hold THROUGH accepts")
    m = make_model(vocab=4, bos_token_id=1)
    prompt = [1, 2, 3, 2, 0, 3]
    n_new = 60
    ref = vanilla_greedy(m, prompt, n_new)
    m.setup_caches(max_batch_size=1, max_seq_len=len(prompt) + n_new, force=True)
    res = nc.spec_generate(m, StubTok(), None, n_new, len(prompt) + n_new,
                           temperature=0.0, prompt_ids=prompt)
    m.clear_caches()
    bits = res["accept_bits"]
    check("greedy equality with accepts in the mix", res["token_ids"] == ref)
    check("BOTH branches exercised (greedy)", 0 in bits and 1 in bits,
          f"acceptance={res['acceptance_rate']:.2f} over {res['rounds']} rounds")
    # sampled, near-uniform: acceptance should be substantial
    m2 = make_model()
    prompt2 = [5, 7, 9, 11]
    total2 = len(prompt2) + 40
    m2.setup_caches(max_batch_size=1, max_seq_len=total2, force=True)
    r2 = nc.spec_generate(m2, StubTok(), None, 40, total2,
                          temperature=3.0, top_p=1.0, seed=7, prompt_ids=prompt2)
    m2.clear_caches()
    check("sampled accepts occur at high temperature", 1 in r2["accept_bits"],
          f"acceptance={r2['acceptance_rate']:.2f}")


def t_sampled_determinism():
    print("sampled mode: seeded determinism + mechanics")
    m = make_model()
    prompt = [5, 7, 9, 11]
    n_new = 30
    total = len(prompt) + n_new
    m.setup_caches(max_batch_size=1, max_seq_len=total, force=True)
    a = nc.spec_generate(m, StubTok(), None, n_new, total,
                         temperature=0.8, top_p=0.95, seed=123, prompt_ids=prompt)
    m.setup_caches(max_batch_size=1, max_seq_len=total, force=True)
    b = nc.spec_generate(m, StubTok(), None, n_new, total,
                         temperature=0.8, top_p=0.95, seed=123, prompt_ids=prompt)
    m.setup_caches(max_batch_size=1, max_seq_len=total, force=True)
    c = nc.spec_generate(m, StubTok(), None, n_new, total,
                         temperature=0.8, top_p=0.95, seed=999, prompt_ids=prompt)
    m.clear_caches()
    check("same seed -> identical trajectory", a["token_ids"] == b["token_ids"])
    check("different seed -> different trajectory (overwhelmingly likely)",
          a["token_ids"] != c["token_ids"])
    check("acceptance rate in [0,1]", 0.0 <= a["acceptance_rate"] <= 1.0)
    check("emitted n_new", a["tokens_generated"] == n_new)


def t_eos_ledger_truth():
    print("EOS handling: stop promptly on ANY arrival path; ledger == materialized cache")
    class EosTok(StubTok):
        eos_id = 3
    m = make_model()  # vocab 16; eos id 3 will appear naturally at high temp
    prompt = [5, 7, 9, 11]
    n_new = 40
    total = len(prompt) + n_new
    eos_stops = 0
    for seed in range(30):
        m.setup_caches(max_batch_size=1, max_seq_len=total, force=True)
        res = nc.spec_generate(m, EosTok(), None, n_new, total,
                               temperature=2.5, top_p=1.0, seed=seed,
                               prompt_ids=prompt, stop_on_eos=True)
        ledger = m.get_cache_ledger()
        m.clear_caches()
        if res["stop_reason"] != "eos":
            continue
        eos_stops += 1
        ids = res["token_ids"]
        if ids[-1] != 3:
            check(f"seed {seed}: eos-stop ends with eos", False, str(ids[-8:]))
            return
        # THE invariant the buried-EOS bug violated: the ledger must equal the
        # PHYSICALLY MATERIALIZED tokens — prompt + emitted, at most one
        # never-forwarded trailing token short, and NEVER positions beyond it.
        tail = ledger[len(prompt):]
        ok_tail = (tail == ids) or (tail == ids[:-1])
        ok_bound = len(ledger) <= len(prompt) + len(ids)
        ok_rounds = res["rounds"] <= len(ids) + 1  # no post-EOS speculation
        if not (ok_tail and ok_bound and ok_rounds and ledger[:len(prompt)] == prompt):
            check(f"seed {seed}: ledger truth", False,
                  f"ledger_tail={tail[-6:]}, ids={ids[-6:]}, rounds={res['rounds']}")
            return
    check("ledger-truth invariant held for every eos stop", True)
    check("eos stops actually occurred across the seed scan", eos_stops >= 5,
          f"eos_stops={eos_stops}/30")


def t_mtp_cache_lifecycle():
    print("MTP block cache: allocated by setup_caches, cleared by clear_caches")
    m = make_model()
    m.setup_caches(max_batch_size=1, max_seq_len=64, force=True)
    ck = m.mtp.block.attention.cache_k
    check("allocated full-length (global; swa_local forced False)",
          ck is not None and ck.shape[1] == 64, str(None if ck is None else ck.shape))
    m.clear_caches()
    check("cleared", m.mtp.block.attention.cache_k is None)


def t_stream_engine_parity():
    print("stream_generate_kv: spec engine is a DROP-IN — greedy text bit-identical to")
    print("classic, stop-sequences honored mid-round, spec stats surfaced")

    class EncTok(StubTok):
        def __init__(self, ids): self._ids = list(ids)
        def encode(self, text, bos=True, eos=False): return list(self._ids)

    prompt_ids = [5, 7, 9, 11, 6, 8]
    tok = EncTok(prompt_ids)
    m = make_model()
    n_new = 40
    ctx = len(prompt_ids) + n_new

    ref_text, ref_info = nc.stream_generate_kv(
        m, tok, "p", n_new, ctx, temperature=0.0, top_p=1.0,
        display=False, print_prompt=False, return_stop_info=True, spec=False)
    spec_text, spec_info = nc.stream_generate_kv(
        m, tok, "p", n_new, ctx, temperature=0.0, top_p=1.0,
        display=False, print_prompt=False, return_stop_info=True, spec=None)
    check("greedy text bit-identical (auto-spec vs classic)", spec_text == ref_text,
          f"spec[:60]={spec_text[:60]!r} ref[:60]={ref_text[:60]!r}")
    check("spec stats surfaced in stop_info", "spec" in spec_info,
          str(spec_info))
    check("classic path has no spec stats", "spec" not in ref_info)
    check("token counts match", spec_info["tokens_generated"] == ref_info["tokens_generated"])

    # stop-sequence parity: stop on the decoded text of a frequent token
    ref2, ri2 = nc.stream_generate_kv(
        m, tok, "p", n_new, ctx, temperature=0.0, top_p=1.0, display=False,
        print_prompt=False, return_stop_info=True, spec=False,
        stop_sequences=[ref_text[len(ref_text)//2: len(ref_text)//2 + 4]])
    spec2, si2 = nc.stream_generate_kv(
        m, tok, "p", n_new, ctx, temperature=0.0, top_p=1.0, display=False,
        print_prompt=False, return_stop_info=True, spec=None,
        stop_sequences=[ref_text[len(ref_text)//2: len(ref_text)//2 + 4]])
    check("stop-sequence truncation identical", spec2 == ref2,
          f"spec={spec2[-40:]!r} ref={ref2[-40:]!r}")
    check("both report stop_sequence", ri2["reason"] == si2["reason"] == "stop_sequence",
          f"{ri2['reason']} vs {si2['reason']}")

    # plain model: auto-spec silently falls back to classic
    mp = make_model(mtp_enabled=False)
    p3, i3 = nc.stream_generate_kv(
        mp, tok, "p", 20, len(prompt_ids) + 20, temperature=0.0, top_p=1.0,
        display=False, print_prompt=False, return_stop_info=True, spec=None)
    check("no-mtp model: auto falls back to classic (no spec stats)", "spec" not in i3)


def t_no_mtp_raises():
    print("plain checkpoint: spec_generate refuses loudly")
    m = make_model(mtp_enabled=False)
    m.setup_caches(max_batch_size=1, max_seq_len=32, force=True)
    try:
        nc.spec_generate(m, StubTok(), None, 8, 32, prompt_ids=[5, 6, 7])
        check("raises ValueError", False)
    except ValueError:
        check("raises ValueError", True)
    m.clear_caches()


def main():
    print(f"\n=== MTP speculative decoding tests (CPU, torch {torch.__version__}) ===\n")
    for t in (t_greedy_equality, t_greedy_equality_no_swa, t_accept_branch_coverage,
              t_sampled_determinism,
              t_eos_ledger_truth, t_stream_engine_parity, t_mtp_cache_lifecycle, t_no_mtp_raises):
        t()
        print()
    print(f"=== {PASS[0]} passed, {FAIL[0]} failed ===")
    sys.exit(1 if FAIL[0] else 0)


if __name__ == "__main__":
    main()
