#!/usr/bin/env python
"""
test_body_lr_controller_auto.py — behavior tests for the self-anchored LR-track ('auto') mode of
BodyLRController (Math Q11). Drives the REAL controller through synthetic frozen-body plants
(pdr = K*m) and checks the properties Math signed off on. Pure-Python (no torch) — runs anywhere:

    python test_body_lr_controller_auto.py

Plant model (frozen body): the natural per-step pdr is K(t)*m, with K = base*lr(t)*drift(t).
- base*lr(t) is the "K proportional to lr" frozen-body physics (so LR-track => m~1).
- drift(t) is the slow K-drift the closed loop must cancel (or, if it falls, expose as no-authority).
"""
import math, sys

sys.path.insert(0, ".")
from body_lr_controller import BodyLRController, _STATE_VERSION

CAD = 100
TOK_PER_STEP = 131072


def cosine_lr(step, max_lr=3.5e-4, min_lr=3.5e-5, warmup=0, total=8000):
    if step < warmup:
        return max_lr * step / max(1, warmup)
    p = min(1.0, max(0.0, (step - warmup) / (total - warmup)))
    return min_lr + 0.5 * (max_lr - min_lr) * (1.0 + math.cos(math.pi * p))


def make_auto(**over):
    cfg = dict(enabled=True, warmup_step=0, m_floor=0.20, m_max=1.0,
               k_ema_alpha=0.30, pdr_ema_alpha=0.30, rate_down=0.10, rate_up=0.10,
               reference=dict(mode="auto", anchor_step=500, anchor_samples=5))
    # shallow-merge reference overrides
    if "reference" in over:
        cfg["reference"].update(over.pop("reference"))
    cfg.update(over)
    return BodyLRController(cfg)


def run(ctrl, K_fn, lr_fn, f_fn, last_step, start=0):
    """Drive the controller at cadence. pdr_measured = K(step) * m_inflight (the held m)."""
    traj = []
    step = start
    while step <= last_step:
        m_inflight = ctrl.current_multiplier()
        pdr = K_fn(step) * m_inflight
        ctrl.observe(step, step * TOK_PER_STEP / 1e6, pdr,
                     scheduled_lr=lr_fn(step), f_now=f_fn(step))
        d = ctrl._last
        traj.append(dict(step=step, K=K_fn(step), lr=lr_fn(step), m=ctrl.m,
                         pdr=pdr, r=d.get("r"), m_ff_raw=d.get("m_ff_raw")))
        step += CAD
    return traj


RESULTS = []
def check(name, cond, detail=""):
    RESULTS.append((name, bool(cond), detail))
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}" + (f"  — {detail}" if detail else ""))


# ---------------------------------------------------------------------------
def test_lrtrack_m_about_one():
    """K proportional to lr (no drift) => LR-track reference == natural pdr => commanded m ~ 1.
    Uses a realistic-slope cosine (total=200k like dn3); the K_ema lag — and so m's dip below 1 — is a
    function of per-sample LR change, which is ~0.1%/sample in a real run (vs ~3%/sample if compressed)."""
    base = 10.0
    lr = lambda s: cosine_lr(s, total=200000)
    ctrl = make_auto()
    K = lambda s: base * lr(s)                  # K ∝ lr exactly
    traj = run(ctrl, K, lr, lambda s: 1.0, 6000)
    check("auto latched", ctrl.anchor_set, f"K_anchor={ctrl.K_anchor:.3e} lr_anchor={ctrl.lr_anchor:.3e}")
    active = [p for p in traj if p["r"] is not None][3:]   # drop the warm-in samples
    ms = [p["m"] for p in active]
    check("m≈1 under lr-track (K∝lr)", all(abs(m - 1.0) < 0.02 for m in ms),
          f"m range [{min(ms):.3f}, {max(ms):.3f}] (want ~1.0)")
    # pdr should track the reference tightly
    rel = max(abs(p["pdr"] / p["r"] - 1.0) for p in active)
    check("pdr tracks r (lr-track)", rel < 0.02, f"max rel err {rel:.3%}")


def test_kdrift_compensation():
    """Positive K-drift (K rises above the ∝lr line): the loop must CUT m so pdr stays on r."""
    base = 10.0
    lr = lambda s: cosine_lr(s, total=200000)
    def drift(s):  # +60% over the run, starting after the anchor latches (~step 900)
        return 1.0 + 0.60 * max(0.0, (s - 1000)) / 5000.0
    ctrl = make_auto()
    K = lambda s: base * lr(s) * drift(s)
    traj = run(ctrl, K, lr, lambda s: 1.0, 6000)
    active = [p for p in traj if p["r"] is not None and p["step"] >= 5500]
    rel = max(abs(p["pdr"] / p["r"] - 1.0) for p in active)
    mfin = active[-1]["m"]
    # at +60% K, m should land near 1/1.6 ≈ 0.625 to hold pdr on r
    check("K-drift cut: m<1 to compensate", mfin < 0.80, f"m_final={mfin:.3f} (want ~0.63)")
    check("K-drift: pdr still tracks r", rel < 0.03, f"max rel err {rel:.3%}")
    check("K-drift: no false alarms", (not ctrl.alarm) and (not ctrl.upper_alarm),
          f"alarm={ctrl.alarm} upper={ctrl.upper_alarm}")


def test_upper_rail_no_authority():
    """K falls BELOW the ∝lr line: controller wants m>1, capped at 1 => upper alarm, NOT lower."""
    base = 10.0
    def drop(s):   # K drifts down to 0.5x after the anchor — body cooler than reference
        return 1.0 - 0.5 * min(1.0, max(0.0, (s - 1000)) / 1500.0)
    ctrl = make_auto()
    K = lambda s: base * cosine_lr(s) * drop(s)
    traj = run(ctrl, K, cosine_lr, lambda s: 1.0, 6000)
    check("upper-rail alarm fires", ctrl.upper_alarm, f"m_ff_raw={ctrl._m_ff_raw:.2f} (wants >1)")
    check("lower-rail alarm does NOT fire", not ctrl.alarm, "")
    check("m pegged at m_max", abs(ctrl.m - ctrl.m_max) < 1e-6, f"m={ctrl.m:.3f}")


def test_lower_rail_insufficient_cooling():
    """K drifts up hard (8x): even m_floor can't cool to target => lower alarm, m at floor."""
    base = 10.0
    def surge(s):
        return 1.0 + 7.0 * min(1.0, max(0.0, (s - 1000)) / 1500.0)
    ctrl = make_auto()
    K = lambda s: base * cosine_lr(s) * surge(s)
    traj = run(ctrl, K, cosine_lr, lambda s: 1.0, 6000)
    check("lower-rail alarm fires", ctrl.alarm, f"m={ctrl.m:.3f} at floor {ctrl.m_floor}")
    check("upper-rail alarm does NOT fire", not ctrl.upper_alarm, "")
    check("m driven to floor", abs(ctrl.m - ctrl.m_floor) < 1e-6, f"m={ctrl.m:.3f}")


def test_f_gating():
    """Past anchor_step but f<1 => must NOT anchor (still growing). Anchors only after f reaches 1."""
    base = 10.0
    ctrl = make_auto(reference=dict(anchor_step=500))
    f = lambda s: 0.0 if s < 2000 else 1.0    # freeze completes at step 2000, well past anchor_step
    K = lambda s: base * cosine_lr(s)
    # run only up to step 1500 (>= anchor_step=500 but f still 0)
    run(ctrl, K, cosine_lr, f, 1500)
    check("no anchor while f<1 (even past anchor_step)", not ctrl.anchor_set,
          f"collected={len(ctrl._anchor_buf)}")
    # continue past the freeze
    run(ctrl, K, cosine_lr, f, 4000, start=1600)
    check("anchors after f reaches 1", ctrl.anchor_set, f"K_anchor={ctrl.K_anchor}")


def test_geometric_mean_anchor():
    """K_anchor must be the GEOMETRIC mean of the collected post-freeze pdr samples (m=1)."""
    ctrl = make_auto(reference=dict(anchor_step=300, anchor_samples=4))
    # feed 4 known pdr values directly at/after anchor_step with f=1, lr=const
    vals = [2.0e-3, 2.4e-3, 1.8e-3, 2.2e-3]
    step = 300
    for v in vals:
        ctrl.observe(step, step * TOK_PER_STEP / 1e6, v, scheduled_lr=1.5e-4, f_now=1.0)
        step += CAD
    geo = math.exp(sum(math.log(v) for v in vals) / len(vals))
    check("K_anchor == geometric mean", ctrl.anchor_set and abs(ctrl.K_anchor - geo) < 1e-12,
          f"K_anchor={ctrl.K_anchor:.6e} geomean={geo:.6e}")


def test_sanity_bands():
    """Anchor sanity bands must trip on realistic capture bugs at the DEFAULT config (anchor_samples=8),
    and the pre-freeze baseline must NOT include the anchor samples themselves (Math Q11 item 8 — if the
    baseline absorbs the anchor values the warn/fatal bands desensitize toward ratio 1.0)."""
    def build(anchor_ratio, samples=8, anchor_step=1000):
        ctrl = make_auto(reference=dict(anchor_step=anchor_step, anchor_samples=samples))
        s = 0
        while s < anchor_step:                   # pre-anchor baseline ~2e-3 (strictly before anchor_step)
            ctrl.observe(s, s * TOK_PER_STEP / 1e6, 2.0e-3, scheduled_lr=1.5e-4, f_now=1.0)
            s += CAD
        for _ in range(samples):                 # anchor samples at anchor_ratio * baseline
            ctrl.observe(s, s * TOK_PER_STEP / 1e6, 2.0e-3 * anchor_ratio, scheduled_lr=1.5e-4, f_now=1.0)
            s += CAD
        return ctrl
    c1 = build(1.0)
    check("clean anchor: no warn/fatal", c1.anchor_set and c1.anchor_warn is None and c1.anchor_fatal is None, "")
    c2 = build(1.6)   # outside warn [0.6,1.4], inside fatal [0.35,2.0]
    check("warn band trips at 1.6x (samples=8)", c2.anchor_warn is not None and c2.anchor_fatal is None,
          (c2.anchor_warn or "")[:60])
    c3 = build(4.0)   # outside fatal [0.35,2.0] — the contamination-defeat case
    check("fatal band trips at 4x (samples=8, uncontaminated baseline)", c3.anchor_fatal is not None,
          (c3.anchor_fatal or "")[:60])


def test_state_roundtrip_idempotent():
    """state_dict/load_state_dict v2 preserves the anchor; a restored controller does NOT re-capture."""
    base = 10.0
    a = make_auto()
    K = lambda s: base * cosine_lr(s)
    run(a, K, cosine_lr, lambda s: 1.0, 3000)
    assert a.anchor_set
    sd = a.state_dict()
    check("state_dict version is 2", sd["version"] == _STATE_VERSION, f"v={sd['version']}")
    b = make_auto()
    b.load_state_dict(sd)
    check("anchor restored", b.anchor_set and abs(b.K_anchor - a.K_anchor) < 1e-15
          and abs(b.lr_anchor - a.lr_anchor) < 1e-18, f"K {a.K_anchor:.3e}->{b.K_anchor:.3e}")
    # continue BOTH and confirm identical m (restored controller does not re-anchor / diverge)
    ta = run(a, K, cosine_lr, lambda s: 1.0, 4000, start=3100)
    tb = run(b, K, cosine_lr, lambda s: 1.0, 4000, start=3100)
    same = all(abs(x["m"] - y["m"]) < 1e-12 for x, y in zip(ta, tb))
    check("resumed trajectory bit-identical", same, "")
    check("no spurious re-capture (buf stable)", b.K_anchor == a.K_anchor, "")


def test_v1_migration():
    """A v1 (knots-only) checkpoint loads into the v2 code: knots state restored, auto fields default."""
    v1 = {"version": 1, "m": 0.73, "logK": math.log(2.5e-3), "pdr_ema": 2.5e-3,
          "alarm_run": 0, "alarm": False, "alarm_ever": True, "inspect": False,
          "dropped": 0, "pid": {"I": 0.0, "prev_pv": None}}
    c = BodyLRController(dict(enabled=True, warmup_step=0,
                             reference=dict(knots=[[1, 3e-3], [1000, 1e-3]])))
    c.load_state_dict(v1)
    check("v1 m restored", abs(c.m - 0.73) < 1e-12, f"m={c.m}")
    check("v1->v2 auto fields default", (c.K_anchor is None) and (not c.anchor_set)
          and (c._anchor_buf == []), "")


def test_knots_mode_still_works():
    """Regression: knots mode constructs, tracks its reference, and is unaffected by the new fields."""
    c = BodyLRController(dict(enabled=True, warmup_step=0, m_floor=0.3,
                             reference=dict(knots=[[0, 3.0e-3], [800, 1.5e-3]])))
    base_ratio = 1.0
    # plant K constant; controller should pull m down to track the declining knots reference
    K = lambda s: 3.0e-3
    t = run(c, K, lambda s: 1.5e-4, lambda s: None, 80000)  # tok_m drives the knot interp
    last = t[-1]
    check("knots mode runs + tracks", last["r"] is not None and abs(last["pdr"] / last["r"] - 1) < 0.1,
          f"pdr={last['pdr']:.3e} r={last['r']:.3e} m={last['m']:.3f}")
    check("knots mode: m in [floor,1]", all(c.m_floor - 1e-9 <= p["m"] <= 1.0 + 1e-9 for p in t), "")


def test_baseline_none_warns():
    """If NO genuine pre-freeze samples are seen (collection starts immediately at anchor_step with f=1),
    the relative sanity band is skipped — and the controller must say so via anchor_warn."""
    ctrl = make_auto(reference=dict(anchor_step=100, anchor_samples=4))
    s = 100
    for _ in range(4):                          # collect immediately at anchor_step, f=1, no pre-freeze
        ctrl.observe(s, s * TOK_PER_STEP / 1e6, 2.0e-3, scheduled_lr=1.5e-4, f_now=1.0)
        s += CAD
    check("latched with no pre-freeze baseline", ctrl.anchor_set and ctrl._pre_freeze_pdr_ema is None, "")
    check("anchor_warn flags skipped relative band",
          ctrl.anchor_warn is not None and "relative" in (ctrl.anchor_warn or "").lower(),
          (ctrl.anchor_warn or "")[:60])


def test_phase_labels():
    """Phase must reflect reality: 'pre-anchor' (step<anchor_step), 'awaiting-freeze' (step>=anchor_step
    but f<1, NOT collecting), 'anchoring' (collecting)."""
    c = make_auto(reference=dict(anchor_step=1000, anchor_samples=4))
    c.observe(200, 200 * TOK_PER_STEP / 1e6, 2.0e-3, scheduled_lr=1.5e-4, f_now=1.0)   # step<anchor_step
    check("phase=pre-anchor before anchor_step", c._last.get('phase') == 'pre-anchor', c._last.get('phase'))
    c2 = make_auto(reference=dict(anchor_step=100, anchor_samples=4))
    c2.observe(200, 200 * TOK_PER_STEP / 1e6, 2.0e-3, scheduled_lr=1.5e-4, f_now=0.5)  # past, f<1
    check("phase=awaiting-freeze past anchor_step but f<1",
          c2._last.get('phase') == 'awaiting-freeze' and not c2.anchor_set and len(c2._anchor_buf) == 0,
          c2._last.get('phase'))
    c2.observe(300, 300 * TOK_PER_STEP / 1e6, 2.0e-3, scheduled_lr=1.5e-4, f_now=1.0)  # now frozen
    check("phase=anchoring once collecting", c2._last.get('phase') == 'anchoring' and len(c2._anchor_buf) == 1,
          c2._last.get('phase'))


def main():
    for fn in [test_lrtrack_m_about_one, test_kdrift_compensation, test_upper_rail_no_authority,
               test_lower_rail_insufficient_cooling, test_f_gating, test_geometric_mean_anchor,
               test_sanity_bands, test_baseline_none_warns, test_phase_labels,
               test_state_roundtrip_idempotent, test_v1_migration,
               test_knots_mode_still_works]:
        print(f"\n=== {fn.__name__} ===")
        fn()
    n_fail = sum(1 for _, ok, _ in RESULTS if not ok)
    print(f"\n{'='*60}\n{len(RESULTS)-n_fail}/{len(RESULTS)} checks passed.")
    sys.exit(1 if n_fail else 0)


if __name__ == "__main__":
    main()
