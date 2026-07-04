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


class StubTokEnc(StubTok):
    """StubTok whose encode() returns a settable id list — for driving
    stream_generate_kv (which encodes prompt_text) with controlled tokens.
    set_ids() lets a multi-turn harness re-encode the growing stream each turn."""
    def __init__(self, ids):
        self._ids = list(ids)
    def set_ids(self, ids):
        self._ids = list(ids)
    def encode(self, text, bos=True, eos=False):
        return list(self._ids)


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


def vanilla_sample(model, prompt_ids, n_new, temperature, top_p, generator):
    """Direct (warped) sampling from the trunk — the GROUND-TRUTH distribution
    the spec engine must reproduce. Every token is drawn directly from p (the
    warped trunk distribution); no draft/verify. Uses nc._warp_probs so the
    warp is byte-identical to what spec_generate applies, isolating the ONLY
    difference to draft/verify-vs-direct."""
    total = len(prompt_ids) + n_new
    model.setup_caches(max_batch_size=1, max_seq_len=total, force=True)
    out = []
    with torch.no_grad():
        toks = torch.tensor([prompt_ids], dtype=torch.long)
        logits = model.generate_forward(toks, 0)
        p = nc._warp_probs(logits[0, -1].float(), temperature, top_p)
        out.append(int(torch.multinomial(p, 1, generator=generator)))
        pos = len(prompt_ids)
        while len(out) < n_new:
            logits = model.generate_forward(
                torch.tensor([[out[-1]]], dtype=torch.long), pos)
            p = nc._warp_probs(logits[0, -1].float(), temperature, top_p)
            out.append(int(torch.multinomial(p, 1, generator=generator)))
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


def t_spec_verify_primitive():
    print("accept/reject PRIMITIVE (_spec_verify_step, the SINGLE source of truth for")
    print("both engines): emitted-token law == p1 for ANY draft law q — tested with q")
    print("chosen ADVERSARIALLY far from p, where the residual/reject path carries the")
    print("mass. (Untrained p~=q can't stress this; that is why the integration")
    print("differential below has little to see — the POWER lives HERE.)")
    V = 6
    M = 300_000

    def tvd(a, b):
        return 0.5 * float((a - b).abs().sum())

    def emitted_law(p1, q, seed):
        """Empirical distribution of the token _spec_verify_step emits, over M
        draws of (draft ~ q, then verify). Must equal p1 if the primitive is
        exact — for ANY q."""
        g = torch.Generator()
        g.manual_seed(seed)
        c = torch.zeros(V)
        for _ in range(M):
            draft = int(torch.multinomial(q, 1, generator=g))
            acc, emit = nc._spec_verify_step(p1, q, draft, 1.0, generator=g)
            c[draft if acc else emit] += 1
        return c / M

    cases = [
        ("p peaked / q anti-peaked", [.60, .20, .10, .05, .03, .02], [.02, .03, .05, .10, .20, .60]),
        ("p uniform  / q peaked",    [1/6]*6,                        [.70, .10, .08, .06, .04, .02]),
        ("near-disjoint support",    [.45, .45, .05, .03, .01, .01], [.01, .01, .03, .05, .45, .45]),
    ]
    noise = (V / M) ** 0.5     # ~0.004: MC floor a CORRECT sampler reaches
    worst = 0.0
    for name, pl, ql in cases:
        p1 = torch.tensor(pl); p1 = p1 / p1.sum()
        q = torch.tensor(ql); q = q / q.sum()
        d = emitted_law(p1, q, 4242)
        e = tvd(d, p1)
        worst = max(worst, e)
        e_acc = float(torch.minimum(p1, q).sum())   # expected accept rate = sum min(p,q)
        print(f"       {name:<26} TVD(emit,p1)={e:.4f}  E[accept]={e_acc:.2f}  "
              f"TVD(q,p1)={tvd(q, p1):.2f}")
        check(f"[{name}] emitted law == p1 (divergent q, {100*(1-e_acc):.0f}% reject)",
              e < 0.01, f"TVD={e:.4f} (MC floor ~{noise:.4f})")
        # SELF-WITNESSING POWER: the naive/broken 'always accept' emits ~q, whose
        # TVD from p1 is huge here — so clearing the <0.01 bound is something only
        # a CORRECT accept/reject+residual sampler can do (a mutated one lands near
        # TVD(q,p1), far above the bound). No transcribed mutation needed.
        check(f"[{name}] bound is meaningful (broken sampler ~q would fail it)",
              tvd(q, p1) > 0.05)
    check("primitive exact across all adversarial (p,q)", worst < 0.01, f"worst={worst:.4f}")


def t_sampled_distribution_equivalence():
    print("SAMPLED mode INTEGRATION: the real engine feeds p (trunk) and q (MTP) into")
    print("the verified primitive and plumbs the result — end-to-end output law must")
    print("match direct sampling. (Formula POWER is proven by t_spec_verify_primitive;")
    print("this + greedy bit-identity confirm the PLUMBING.) Differential, self-calibrated.")
    V = 6
    m = make_model(vocab=V)
    prompt = [5, 3, 1, 4, 2]
    T, P = 1.0, 1.0            # top_p=1.0 isolates accept/reject from nucleus edges
    N = 2500                   # smoke-scale: the primitive test carries the power

    def joint(sampler, seed_base):
        """Empirical joint distribution of (tok0, tok1) over N independent draws.
        tok0 is direct-sampled in BOTH engines; tok1 is the FIRST speculative
        decision (accept a q-draft w.p. min(1,p/q), else residual) — so any
        p/q-misalignment or residual bug shows up here and nowhere in greedy."""
        C = torch.zeros(V, V)
        for s in range(N):
            ids = sampler(seed_base + s)
            C[ids[0], ids[1]] += 1
        return C / C.sum()

    def draw_direct(seed):
        g = torch.Generator()
        g.manual_seed(seed)
        return vanilla_sample(m, prompt, 2, T, P, g)

    def draw_spec(seed):
        return nc.spec_generate(m, StubTok(), None, 2, len(prompt) + 2,
                                temperature=T, top_p=P, seed=seed,
                                prompt_ids=prompt)["token_ids"]

    # Disjoint seed ranges so no run is shared between the three empirical joints.
    D_a = joint(draw_direct, 1)            # direct sampling, batch A
    D_b = joint(draw_direct, 1 + N)        # direct sampling, batch B -> NULL noise
    S   = joint(draw_spec, 1 + 2 * N)      # spec engine

    def tvd(a, b):
        return 0.5 * float((a - b).abs().sum())

    null = tvd(D_a, D_b)                    # direct-vs-direct: pure sampling noise
    test = tvd(D_a, S)                      # spec-vs-direct
    m_null = tvd(D_a.sum(0), D_b.sum(0))    # tok1 marginal (V cells: high power)
    m_test = tvd(D_a.sum(0), S.sum(0))
    print(f"       joint  TVD  spec-vs-direct={test:.4f}   null direct-vs-direct={null:.4f}")
    print(f"       tok1   TVD  spec-vs-direct={m_test:.4f}   null direct-vs-direct={m_null:.4f}")
    # If spec's distribution == direct's, `test` is two independent N-estimates
    # of the SAME law, so E[test]≈E[null]; 2*null+0.01 covers fluctuation while a
    # genuine bias (which grows with the error, not the noise) blows past it.
    check("joint (tok0,tok1) distribution matches within null noise",
          test <= 2 * null + 0.01, f"test={test:.4f} null={null:.4f}")
    check("tok1 marginal distribution matches within null noise",
          m_test <= 2 * m_null + 0.01, f"test={m_test:.4f} null={m_null:.4f}")
    # guard against a vacuous pass: confirm accept AND reject both actually fired
    r = nc.spec_generate(m, StubTok(), None, 24, len(prompt) + 24, temperature=T,
                         top_p=P, seed=42, prompt_ids=prompt)
    bits = r["accept_bits"]
    check("accept/reject path exercised (both branches present)",
          0 in bits and 1 in bits, f"acceptance={r['acceptance_rate']:.2f}")

    # cross-implementation: the INLINED stream engine and standalone spec_generate
    # share the accept/reject math but are separate code — their (tok0,tok1)
    # joints must agree too (differential between the two spec implementations).
    import re as _re
    def draw_stream_spec(seed):
        torch.manual_seed(seed)   # stream engine uses global RNG
        txt = nc.stream_generate_kv(m, StubTokEnc(prompt), "p", 2, len(prompt) + 2,
                                    temperature=T, top_p=P, display=False,
                                    print_prompt=False, spec=None)
        return [int(x) for x in _re.findall(r'<(\d+)>', txt)][:2]
    Nc = 2500
    Csg = torch.zeros(V, V)
    Cst = torch.zeros(V, V)
    for s in range(Nc):
        a = draw_spec(7_000_000 + s)
        b = draw_stream_spec(9_000_000 + s)
        Csg[a[0], a[1]] += 1
        Cst[b[0], b[1]] += 1
    Csg /= Csg.sum(); Cst /= Cst.sum()
    xtest = tvd(Csg, Cst)
    # both are N=2500 estimates of the (claimed) same law -> expect ~sqrt(2) larger
    # per-cell noise than the N=5000 null; bound generously.
    print(f"       joint  TVD  standalone-vs-stream spec={xtest:.4f}")
    check("standalone spec_generate == inlined stream engine (distribution)",
          xtest <= 3 * null + 0.02, f"xtest={xtest:.4f} null={null:.4f}")


def t_spec_reuse_equivalence():
    print("cross-turn KV reuse: multi-turn spec decode WITH reuse == a no-reuse")
    print("spec decode over the concatenated stream (greedy -> bit-exact). Drives")
    print("the spec engine's materialized_len (+2/+1) and _mtp_cache_len lockstep")
    print("bookkeeping, INCLUDING a spec->classic->spec turn sequence (the")
    print("stale-_mtp_cache_len path the integration review flagged).")
    import re as _re

    def parse_ids(txt):
        return [int(x) for x in _re.findall(r'<(\d+)>', txt)]

    def run_turns(m, base, per_turn, n_turns, ctx, reuse, spec_per_turn):
        m.clear_caches()
        tok = StubTokEnc(base)
        state = nc.CacheState(reusable=True) if reuse else None
        stream = list(base)
        for t in range(n_turns):
            tok.set_ids(stream)          # each turn re-encodes the FULL stream so far
            txt = nc.stream_generate_kv(
                m, tok, "p", per_turn, ctx, temperature=0.0, top_p=1.0,
                display=False, print_prompt=False, cache_state=state,
                spec=spec_per_turn[t])
            stream += parse_ids(txt)
        m.clear_caches()
        return stream

    for label, kw in [("non-SWA (isolates spec bookkeeping)",
                       dict(swa_enabled=False, doc_attn_mask=False, doc_pos_reset=False)),
                      ("full-stack SWA", dict())]:
        m = make_model(**kw)
        base = [5, 7, 9, 11, 6, 8, 10, 12]
        pt, nt, ctx = 5, 3, 96
        ref = run_turns(m, base, pt, nt, ctx, reuse=False, spec_per_turn=[None] * nt)
        reu = run_turns(m, base, pt, nt, ctx, reuse=True, spec_per_turn=[None] * nt)
        mix = run_turns(m, base, pt, nt, ctx, reuse=True, spec_per_turn=[None, False, None])
        check(f"[{label}] spec reuse stream == no-reuse (bit-exact)", reu == ref,
              f"reu_tail={reu[-6:]} ref_tail={ref[-6:]}")
        check(f"[{label}] spec->classic->spec reuse == no-reuse", mix == ref,
              f"mix_tail={mix[-6:]} ref_tail={ref[-6:]}")


def t_spec_reuse_rewind_swa():
    print("SWA rewind guard vs the spec-REJECT phantom write (regression). A terminal")
    print("reject forwards a 2-token chunk whose rejected draft is physically written")
    print("one ring slot PAST the ledger, so at cache length == window the ring has")
    print("wrapped (slot 0 evicted) while the ledger looks unwrapped. A later REWIND")
    print("reuse turn must NOT read that phantom — the guard must degrade to full")
    print("re-prefill. We hit the exact boundary and require bit-exact vs no-reuse.")
    import re as _re

    def parse_ids(txt):
        return [int(x) for x in _re.findall(r'<(\d+)>', txt)]

    # small window so the local rings wrap almost immediately; greedy on an
    # untrained model rejects every draft, so the turn always ENDS on a reject.
    m = make_model(swa_window=4, doc_attn_mask=False, doc_pos_reset=False)
    prompt = [7, 9]                    # short: prompt_len + gen lands on Lc
    ctx = 128
    m.setup_caches(max_batch_size=1, max_seq_len=ctx, force=True)
    Lc = m.min_rolling_cache_len()     # ring capacity is only known post-alloc
    m.clear_caches()
    if Lc is None:
        check("model has a rolling SWA ring to exercise", False, "min_rolling_cache_len is None")
        return
    # ledger after a greedy (all-reject) turn = prompt_len + gen1 - 1; the phantom
    # bites at ledger == Lc, i.e. gen1 == Lc - prompt_len + 1. Sweep around it.
    target = Lc - len(prompt) + 1
    rewind = [prompt[0], 13]           # shares 1 token -> start_pos=1 (a rewind); id < vocab

    def turn2(reuse, prime_gen1):
        m.clear_caches()
        tok = StubTokEnc(prompt)
        state = nc.CacheState(reusable=True) if reuse else None
        if reuse:
            # turn 1 primes + persists the cache/ledger at length prompt+gen1-1
            nc.stream_generate_kv(m, tok, "p", prime_gen1, ctx, temperature=0.0,
                                  top_p=1.0, display=False, print_prompt=False,
                                  cache_state=state, spec=None)
        tok.set_ids(rewind)
        txt = nc.stream_generate_kv(m, tok, "p", 6, ctx, temperature=0.0, top_p=1.0,
                                    display=False, print_prompt=False,
                                    cache_state=state, spec=None)
        m.clear_caches()
        return parse_ids(txt)

    fails = 0
    for gen1 in range(max(1, target - 2), target + 3):
        ref = turn2(reuse=False, prime_gen1=gen1)          # fresh cache each time
        reu = turn2(reuse=True, prime_gen1=gen1)           # rewind reuse
        ok = (reu == ref)
        if not ok:
            fails += 1
            check(f"gen1={gen1} (ledger~{len(prompt)+gen1-1}, Lc={Lc}): rewind-reuse == no-reuse",
                  ok, f"reu={reu} ref={ref}")
    check(f"rewind-reuse bit-exact across the wrap boundary (Lc={Lc}, target gen1={target})",
          fails == 0)


def t_spec_stop_reason_honest():
    print("stop_reason honesty (regression): spec_generate must NOT report")
    print("stop_reason=='eos' unless the LAST emitted token is eos. The bug: an")
    print("accepted draft fills the token cap, the pair's second token nxt is dropped,")
    print("but the nxt==eos check still set 'eos' (final token = draft != eos).")
    class EosTok(StubTok):
        eos_id = 3
    m = make_model()
    prompt = [5, 7, 9, 11]
    violations = 0
    checked = 0
    for M in range(2, 13):
        for seed in range(60):
            total = len(prompt) + M
            m.setup_caches(max_batch_size=1, max_seq_len=total, force=True)
            r = nc.spec_generate(m, EosTok(), None, M, total, temperature=2.5,
                                 top_p=1.0, seed=seed, prompt_ids=prompt, stop_on_eos=True)
            m.clear_caches()
            checked += 1
            if r["stop_reason"] == "eos" and (not r["token_ids"] or r["token_ids"][-1] != 3):
                violations += 1
    check(f"stop_reason=='eos' implies final token is eos ({checked} runs scanned)",
          violations == 0, f"{violations} runs reported eos with a non-eos final token")


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
              t_sampled_determinism, t_spec_verify_primitive,
              t_sampled_distribution_equivalence, t_spec_reuse_equivalence,
              t_spec_reuse_rewind_swa, t_spec_stop_reason_honest,
              t_eos_ledger_truth, t_stream_engine_parity, t_mtp_cache_lifecycle, t_no_mtp_raises):
        t()
        print()
    print(f"=== {PASS[0]} passed, {FAIL[0]} failed ===")
    sys.exit(1 if FAIL[0] else 0)


if __name__ == "__main__":
    main()
