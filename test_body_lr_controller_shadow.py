#!/usr/bin/env python
"""
test_body_lr_controller_shadow.py — behavior tests for the shadow-norm modes
(auto_shadow_growth / auto_shadow_partial) of BodyLRController (Math Q12 + shadow-norm spec).
Drives the REAL controller with synthetic radial inputs and checks the properties Math signed off
on. Pure-Python (no torch) — runs anywhere:

    python test_body_lr_controller_shadow.py

radial input per observe: {param_name: (R, ΔR_free_accum, γ)} accumulated by the trainer since the
last observe; eta_accum = Σ scheduled_lr over the window (for the f=0 WD shrink of S).
"""
import math, sys

sys.path.insert(0, ".")
from body_lr_controller import BodyLRController, _STATE_VERSION, _median

PASS = [0]
FAIL = [0]


def check(name, cond, extra=""):
    ok = bool(cond)
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  {extra}" if extra and not ok else ""))
    (PASS if ok else FAIL)[0] += 1
    return ok


def make_shadow(mode="auto_shadow_growth", **over):
    cfg = dict(enabled=True, m_min_full=0.20, m_max=1.0, rho=0.20,
               lambda_max=0.02, lambda_min=0.002, k_ema_alpha=0.30,
               rate_down=0.10, rate_up=0.10, reference=dict(mode=mode))
    if "reference" in over:
        cfg["reference"].update(over.pop("reference"))
    cfg.update(over)
    return BodyLRController(cfg)


# ============================== tests ==============================

def t_idle():
    print("idle (f=0): m=1, S tracks R, WD at the f=0 ceiling")
    c = make_shadow()
    m = c.observe(0, 0.0, 3e-3, scheduled_lr=3.5e-4, f_now=0.0,
                  radial={"w1": (200.0, 0.0, 0.1)}, eta_accum=0.03)
    check("m==1 at f=0", abs(m - 1.0) < 1e-12, f"m={m}")
    check("S tracks R at idle", abs(c.S["w1"] - 200.0) < 1e-9)
    check("shadow not active until f>0", c.shadow_active is False)
    check("WD = lambda_max at f=0 (rho*gamma=0.02)", abs(c.current_wd() - 0.02) < 1e-9,
          f"lam={c.current_wd()}")


def t_ramp_monotone():
    print("ramp: m decreases monotonically as S outgrows R, never below the f-aware floor")
    c = make_shadow()
    c.observe(0, 0.0, 3e-3, scheduled_lr=3.5e-4, f_now=0.0,
              radial={"w1": (200.0, 0.0, 0.1)}, eta_accum=0.03)  # init at idle
    f = 0.5
    floor = 1.0 - f * (1.0 - 0.20)  # = 0.60
    ms = []
    for k in range(40):
        # R held constant, S accumulates a steady free-growth increment -> R/S falls
        m = c.observe(50 * (k + 1), 0.0, 3e-3, scheduled_lr=3.5e-4, f_now=f,
                      radial={"w1": (200.0, 2.0, 0.1)}, eta_accum=0.03)
        ms.append(m)
    strictly_down = all(b <= a + 1e-12 for a, b in zip(ms, ms[1:]))
    check("m monotonically non-increasing during ramp", strictly_down, f"{ms[:5]}...")
    check("m actually moved below 1", ms[-1] < 0.999, f"m_end={ms[-1]:.4f}")
    check("m never below f-aware floor", min(ms) >= floor - 1e-9, f"min={min(ms):.4f} floor={floor}")


def t_floor_binds():
    print("f-aware floor binds under extreme growth")
    c = make_shadow()
    c.observe(0, 0.0, 3e-3, scheduled_lr=3.5e-4, f_now=0.0,
              radial={"w1": (200.0, 0.0, 0.1)}, eta_accum=0.03)
    f = 0.5
    floor = 1.0 - f * (1.0 - 0.20)
    m = 1.0
    for k in range(60):
        m = c.observe(50 * (k + 1), 0.0, 3e-3, scheduled_lr=3.5e-4, f_now=f,
                      radial={"w1": (200.0, 80.0, 0.1)}, eta_accum=0.03)  # huge growth
    check("m clamps at the f-aware floor", abs(m - floor) < 1e-6, f"m={m:.4f} floor={floor}")


def t_wd_law():
    print("radial-budget WD: lambda_body=clamp(rho(1-f)gamma), lambda_S at f=0")
    c = make_shadow()
    # seed gamma_ema=0.1
    c.observe(0, 0.0, 3e-3, scheduled_lr=3.5e-4, f_now=0.0,
              radial={"w1": (200.0, 0.0, 0.1)}, eta_accum=0.03)
    check("gamma_ema seeded", abs(c._gamma_ema - 0.1) < 1e-9, f"g={c._gamma_ema}")
    # f=0.5 -> rho*0.5*0.1 = 0.01
    c.observe(50, 0.0, 3e-3, scheduled_lr=3.5e-4, f_now=0.5,
              radial={"w1": (200.0, 2.0, 0.1)}, eta_accum=0.03)
    check("lambda_body at f=0.5 == 0.01", abs(c.current_wd() - 0.01) < 1e-6, f"lam={c.current_wd()}")
    check("lambda_S (f=0) >= lambda_body (f>0)", c._lambda(0.0) >= c.current_wd() - 1e-12)
    # f=0.9 -> rho*0.1*0.1 = 0.002 = lam_min
    c.observe(100, 0.0, 3e-3, scheduled_lr=3.5e-4, f_now=0.9,
              radial={"w1": (200.0, 2.0, 0.1)}, eta_accum=0.03)
    check("lambda_body floors at lam_min at f=0.9", abs(c.current_wd() - 0.002) < 1e-6,
          f"lam={c.current_wd()}")
    check("lambda_S still ceilings at lam_max at f=0", abs(c._lambda(0.0) - 0.02) < 1e-9)


def t_handoff_growth():
    print("growth mode: f=1 handoff latches once + m continuous; LR-track tail after")
    c = make_shadow("auto_shadow_growth")
    K0 = 0.05  # fixed plant gain: pdr = K0 * m  -> K_ema -> K0
    c.observe(0, 0.0, K0 * 1.0, scheduled_lr=3.5e-4, f_now=0.0,
              radial={"w1": (200.0, 0.0, 0.1)}, eta_accum=0.03)
    # ramp a bit so m < 1 and K_ema settles
    m_pre = 1.0
    for k in range(30):
        m_pre = c.observe(50 * (k + 1), 0.0, K0 * c.current_multiplier(), scheduled_lr=3.5e-4,
                          f_now=0.6, radial={"w1": (200.0, 2.0, 0.1)}, eta_accum=0.03)
    check("not frozen during ramp", c.frozen is False)
    # cross f=1 -> latch
    m_latch = c.observe(2000, 0.0, K0 * c.current_multiplier(), scheduled_lr=3.5e-4, f_now=1.0,
                        radial={"w1": (200.0, 2.0, 0.1)}, eta_accum=0.03)
    check("frozen latched at f>=1", c.frozen is True)
    check("r_freeze captured", c.r_freeze is not None and c.r_freeze > 0)
    check("latch was idempotent flag", c._just_latched is True)
    # next frozen step with SAME lr -> LR-track tail r=r_freeze -> m unchanged (continuity)
    m_tail = c.observe(2050, 0.0, K0 * c.current_multiplier(), scheduled_lr=3.5e-4, f_now=1.0,
                       radial={"w1": (200.0, 2.0, 0.1)}, eta_accum=0.03)
    check("frozen stays latched (no re-capture)", c.frozen is True)
    check("m continuous across the seam (<1% at constant lr)", abs(m_tail - m_latch) < 0.01 * m_latch,
          f"latch={m_latch:.4f} tail={m_tail:.4f}")
    # tail tracks lr: halve lr -> r halves -> m should fall (LR-track)
    m_lower = c.observe(2100, 0.0, K0 * c.current_multiplier(), scheduled_lr=1.75e-4, f_now=1.0,
                        radial={"w1": (200.0, 2.0, 0.1)}, eta_accum=0.03)
    check("frozen tail rides the LR down", m_lower < m_tail + 1e-9, f"tail={m_tail:.4f} lower={m_lower:.4f}")


def t_partial_no_handoff():
    print("partial mode: f plateaus < 1, never hands off, stays in ramp law")
    c = make_shadow("auto_shadow_partial")
    c.observe(0, 0.0, 3e-3, scheduled_lr=3.5e-4, f_now=0.0,
              radial={"w1": (200.0, 0.0, 0.1)}, eta_accum=0.03)
    last = 1.0
    for k in range(40):
        last = c.observe(50 * (k + 1), 0.0, 3e-3, scheduled_lr=3.5e-4, f_now=0.75,
                         radial={"w1": (200.0, 2.0, 0.1)}, eta_accum=0.03)
    check("partial never freezes", c.frozen is False)
    check("partial r_freeze never captured", c.r_freeze is None)
    floor = 1.0 - 0.75 * (1.0 - 0.20)  # = 0.40
    check("partial m above its f=0.75 floor", last >= floor - 1e-9, f"m={last:.4f} floor={floor}")
    check("partial phase label", c._last.get("phase") == "partial", c._last.get("phase"))


def t_multi_matrix_median():
    print("multi-matrix: m = exp(median_i log(R_i/S_i))")
    c = make_shadow()
    c.observe(0, 0.0, 3e-3, scheduled_lr=3.5e-4, f_now=0.0,
              radial={"a": (100.0, 0.0, 0.1), "b": (200.0, 0.0, 0.1), "c": (300.0, 0.0, 0.1)},
              eta_accum=0.0)  # idle
    c.observe(50, 0.0, 3e-3, scheduled_lr=3.5e-4, f_now=0.5,
              radial={"a": (100.0, 0.0, 0.1), "b": (200.0, 0.0, 0.1), "c": (300.0, 0.0, 0.1)},
              eta_accum=0.0)  # ENGAGE (S=R, m=1)
    # one step with distinct growth so the three ratios differ but no guardrail binds
    c.observe(100, 0.0, 3e-3, scheduled_lr=3.5e-4, f_now=0.5,
              radial={"a": (100.0, 0.5, 0.1), "b": (200.0, 2.0, 0.1), "c": (300.0, 1.0, 0.1)},
              eta_accum=0.0)
    expected = math.exp(_median([math.log(100.0 / c.S["a"]),
                                 math.log(200.0 / c.S["b"]),
                                 math.log(300.0 / c.S["c"])]))
    check("m equals log-median of per-matrix R/S", abs(c.m - expected) < 1e-9,
          f"m={c.m:.6f} exp={expected:.6f}")


def t_state_roundtrip():
    print("v3 state round-trips (S name-keyed, frozen, gamma_ema, m)")
    c = make_shadow()
    c.observe(0, 0.0, 0.05, scheduled_lr=3.5e-4, f_now=0.0,
              radial={"w1": (200.0, 0.0, 0.1)}, eta_accum=0.03)
    for k in range(20):
        c.observe(50 * (k + 1), 0.0, 0.05 * c.current_multiplier(), scheduled_lr=3.5e-4,
                  f_now=0.6, radial={"w1": (200.0, 2.0, 0.1)}, eta_accum=0.03)
    sd = c.state_dict()
    check("state version is v3", sd["version"] == _STATE_VERSION == 3)
    c2 = make_shadow()
    c2.load_state_dict(sd)
    check("S restored by name", c2.S.get("w1") is not None and abs(c2.S["w1"] - c.S["w1"]) < 1e-9)
    check("m restored", abs(c2.m - c.m) < 1e-12)
    check("gamma_ema restored", abs(c2._gamma_ema - c._gamma_ema) < 1e-12)
    check("shadow_active restored", c2.shadow_active == c.shadow_active)
    # continuing from the restored controller reproduces the same next m (integral survived)
    m_a = c.observe(2000, 0.0, 0.05 * c.current_multiplier(), scheduled_lr=3.5e-4, f_now=0.6,
                    radial={"w1": (200.0, 2.0, 0.1)}, eta_accum=0.03)
    m_b = c2.observe(2000, 0.0, 0.05 * c2.current_multiplier(), scheduled_lr=3.5e-4, f_now=0.6,
                     radial={"w1": (200.0, 2.0, 0.1)}, eta_accum=0.03)
    check("resumed controller reproduces next m", abs(m_a - m_b) < 1e-12, f"{m_a} vs {m_b}")


def t_v2_migration():
    print("v2 -> v3 migration loads clean (shadow fields default)")
    c = make_shadow()
    v2 = {"version": 2, "m": 0.8, "logK": math.log(0.05), "pdr_ema": 0.04,
          "alarm_run": 0, "alarm": False, "alarm_ever": False, "inspect": False,
          "dropped": 0, "pid": {}, "ref_mode": "auto", "K_anchor": 2e-3, "lr_anchor": 3e-4,
          "anchor_set": True, "anchor_buf": [], "pre_freeze_pdr_ema": 2e-3,
          "upper_run": 0, "upper_alarm": False, "upper_alarm_ever": False, "m_ff_raw": 0.9,
          "lr_fingerprint": None}
    try:
        c.load_state_dict(v2)
        ok = True
    except Exception as e:  # noqa
        ok = False
        print("    ", e)
    check("v2 state loads without error", ok)
    check("S defaults empty after v2 load", c.S == {})
    check("shadow_active defaults False after v2 load", c.shadow_active is False)
    check("m restored from v2", abs(c.m - 0.8) < 1e-12)


def t_stale_hold():
    print("missing radial sample holds m (stale), does not reset S")
    c = make_shadow()
    c.observe(0, 0.0, 3e-3, scheduled_lr=3.5e-4, f_now=0.0,
              radial={"w1": (200.0, 0.0, 0.1)}, eta_accum=0.03)
    c.observe(50, 0.0, 3e-3, scheduled_lr=3.5e-4, f_now=0.5,
              radial={"w1": (200.0, 5.0, 0.1)}, eta_accum=0.03)
    m_before, S_before = c.m, c.S["w1"]
    m_hold = c.observe(100, 0.0, 3e-3, scheduled_lr=3.5e-4, f_now=0.5, radial=None, eta_accum=0.03)
    check("m held on missing radial", abs(m_hold - m_before) < 1e-12)
    check("S unchanged on missing radial", abs(c.S["w1"] - S_before) < 1e-12)
    check("stale flag set", c._last.get("stale") is True)


def t_engage_seam():
    print("engage seam: first f>0 observe HOLDS m=1 (S=R), integral starts the NEXT window")
    c = make_shadow()
    c.observe(0, 0.0, 3e-3, scheduled_lr=3.5e-4, f_now=0.0,
              radial={"w1": (200.0, 0.0, 0.1)}, eta_accum=0.03)  # idle
    # first f>0 observe carries a FULL window of free growth (dR=50) but m must stay 1 (no double-count)
    m0 = c.observe(50, 0.0, 3e-3, scheduled_lr=3.5e-4, f_now=0.5,
                   radial={"w1": (200.0, 50.0, 0.1)}, eta_accum=0.03)
    check("first f>0 observe holds m=1 (engage)", abs(m0 - 1.0) < 1e-9, f"m0={m0}")
    check("S seeded = R (not R + window growth)", abs(c.S["w1"] - 200.0) < 1e-6, f"S={c.S['w1']}")
    check("phase == engage", c._last.get("phase") == "engage", c._last.get("phase"))
    # second f>0 observe begins the integral -> m cuts below 1
    m1 = c.observe(100, 0.0, 3e-3, scheduled_lr=3.5e-4, f_now=0.5,
                   radial={"w1": (200.0, 5.0, 0.1)}, eta_accum=0.03)
    check("second f>0 observe cuts m below 1", m1 < 1.0, f"m1={m1}")


def t_dR_clip():
    print("ΔR_free outlier clip: a single spike can't move S by more than ~50%/window")
    c = make_shadow()
    c.observe(0, 0.0, 3e-3, scheduled_lr=3.5e-4, f_now=0.0,
              radial={"w1": (200.0, 0.0, 0.1)}, eta_accum=0.0)            # idle
    c.observe(50, 0.0, 3e-3, scheduled_lr=3.5e-4, f_now=0.5,
              radial={"w1": (200.0, 0.0, 0.1)}, eta_accum=0.0)            # engage (S=200)
    S0 = c.S["w1"]
    c.observe(100, 0.0, 3e-3, scheduled_lr=3.5e-4, f_now=0.5,
              radial={"w1": (200.0, 1.0e6, 0.1)}, eta_accum=0.0)          # massive spike, eta=0 (no WD)
    grew = c.S["w1"] - S0
    check("spike clipped to <= 0.5*S (not 1e6)", grew <= 0.5 * S0 + 1e-6, f"grew={grew} cap={0.5*S0}")


def t_glide_gain():
    print("glide_gain: g>1 steepens the cut (deeper than pure R/S); Θ drift tracked")
    def run_to_ramp(g):
        c = make_shadow(glide_gain=g)
        c.observe(0, 0.0, 3e-3, scheduled_lr=3.5e-4, f_now=0.0,
                  radial={"w1": (200.0, 0.0, 0.1)}, eta_accum=0.0)   # idle
        c.observe(50, 0.0, 3e-3, scheduled_lr=3.5e-4, f_now=0.5,
                  radial={"w1": (200.0, 0.0, 0.1)}, eta_accum=0.0)   # engage
        last = 1.0
        for k in range(20):
            last = c.observe(100 + 50 * k, 0.0, 3e-3, scheduled_lr=3.5e-4, f_now=0.5,
                             radial={"w1": (200.0, 2.0, 0.1)}, eta_accum=0.0)
        return c, last
    c1, m1 = run_to_ramp(1.0)
    c2, m2 = run_to_ramp(2.0)
    check("g=1 default still cuts below 1", m1 < 1.0, f"m1={m1:.4f}")
    check("g=2 cuts deeper than g=1", m2 < m1 - 1e-6, f"m(g=1)={m1:.4f} m(g=2)={m2:.4f}")
    check("Θ drift accumulated (telemetry live)", c1.theta_actual > 0 and c1.theta_ref > 0)
    sd = c1.state_dict(); c3 = make_shadow(); c3.load_state_dict(sd)
    check("Θ restored", abs(c3.theta_actual - c1.theta_actual) < 1e-9 and
          abs(c3.theta_ref - c1.theta_ref) < 1e-9)


def t_freeze_handoff_off():
    print("growth + freeze_handoff:false: f=1 does NOT latch; R/S law continues (no kink), m keeps falling")
    c = make_shadow("auto_shadow_growth", freeze_handoff=False)
    check("freeze_handoff flag read", c.freeze_handoff is False)
    c.observe(0, 0.0, 3e-3, scheduled_lr=3.5e-4, f_now=0.0,
              radial={"w1": (200.0, 0.0, 0.1)}, eta_accum=0.03)
    # ramp at f=0.6 so m settles below 1
    m_ramp = 1.0
    for k in range(20):
        m_ramp = c.observe(50 * (k + 1), 0.0, 3e-3, scheduled_lr=3.5e-4, f_now=0.6,
                           radial={"w1": (200.0, 2.0, 0.1)}, eta_accum=0.03)
    check("not frozen during ramp", c.frozen is False)
    # cross f=1: with handoff OFF the controller must NOT latch — it keeps commanding m=median(R/S)
    ms = []
    for k in range(30):
        ms.append(c.observe(2000 + 50 * k, 0.0, 3e-3, scheduled_lr=3.5e-4, f_now=1.0,
                            radial={"w1": (200.0, 2.0, 0.1)}, eta_accum=0.03))
    check("freeze_handoff:false never latches at f=1", c.frozen is False)
    check("r_freeze never captured (no handoff)", c.r_freeze is None)
    check("phase == 'open' at f=1 (no-handoff)", c._last.get("phase") == "open", c._last.get("phase"))
    nonincr = all(b <= a + 1e-12 for a, b in zip(ms, ms[1:]))
    check("m keeps falling into the frozen phase (no kink)", nonincr, f"{[round(x,3) for x in ms[:4]]}...")
    check("m continued below the ramp level (R/S kept cutting)", ms[-1] < m_ramp + 1e-9,
          f"ramp={m_ramp:.4f} end={ms[-1]:.4f}")
    check("m stays above the f=1 floor (m_min_full=0.20)", min(ms) >= 0.20 - 1e-9, f"min={min(ms):.4f}")
    # regression: default (handoff ON) DOES still latch
    c2 = make_shadow("auto_shadow_growth")
    check("default freeze_handoff is True", c2.freeze_handoff is True)


def t_freeze_handoff_resume_guard():
    print("freeze_handoff: checkpointed for the trainer mismatch-guard; config wins on load; anchor-once belt")
    c = make_shadow("auto_shadow_growth", freeze_handoff=False)
    sd = c.state_dict()
    check("freeze_handoff serialized in state_dict", sd.get("freeze_handoff") is False)
    # load into a controller built with the OPPOSITE config -> config WINS (authoritative), ckpt stashed
    c2 = make_shadow("auto_shadow_growth", freeze_handoff=True)
    c2.load_state_dict(sd)
    check("config freeze_handoff wins on load (not overwritten by ckpt)", c2.freeze_handoff is True)
    check("checkpointed value stashed for the trainer mismatch-guard", c2._ckpt_freeze_handoff is False)
    # pre-feature checkpoint (no key) -> stash None (guard treats as 'unknown, warn')
    sd_old = dict(sd); sd_old.pop("freeze_handoff")
    c3 = make_shadow("auto_shadow_growth", freeze_handoff=True); c3.load_state_dict(sd_old)
    check("pre-feature ckpt -> _ckpt_freeze_handoff None", c3._ckpt_freeze_handoff is None)
    # anchor-once belt: even if a resume restored frozen=False with r_freeze already set, never re-capture
    c4 = make_shadow("auto_shadow_growth")  # handoff ON
    c4.observe(0, 0.0, 0.05, scheduled_lr=3.5e-4, f_now=0.0, radial={"w1": (200.0, 0.0, 0.1)}, eta_accum=0.03)
    for k in range(8):
        c4.observe(50 * (k + 1), 0.0, 0.05 * c4.current_multiplier(), scheduled_lr=3.5e-4, f_now=0.6,
                   radial={"w1": (200.0, 2.0, 0.1)}, eta_accum=0.03)
    c4.observe(2000, 0.0, 0.05 * c4.current_multiplier(), scheduled_lr=3.5e-4, f_now=1.0,
               radial={"w1": (200.0, 2.0, 0.1)}, eta_accum=0.03)
    r0 = c4.r_freeze
    check("anchored r_freeze at f=1", r0 is not None and c4.frozen is True)
    c4.frozen = False  # simulate the pathological restore (frozen lost, r_freeze kept)
    c4.observe(2050, 0.0, 0.05 * c4.current_multiplier(), scheduled_lr=3.5e-4, f_now=1.0,
               radial={"w1": (200.0, 2.0, 0.1)}, eta_accum=0.03)
    check("anchor-once belt: r_freeze NOT re-captured at the resume step", c4.r_freeze == r0,
          f"r0={r0} now={c4.r_freeze}")


def main():
    print(f"\n=== shadow-norm controller tests (state v{_STATE_VERSION}) ===\n")
    for t in (t_idle, t_engage_seam, t_ramp_monotone, t_floor_binds, t_wd_law, t_handoff_growth,
              t_partial_no_handoff, t_freeze_handoff_off, t_freeze_handoff_resume_guard,
              t_multi_matrix_median, t_dR_clip, t_glide_gain,
              t_state_roundtrip, t_v2_migration, t_stale_hold):
        t()
        print()
    print(f"=== {PASS[0]} passed, {FAIL[0]} failed ===")
    sys.exit(1 if FAIL[0] else 0)


if __name__ == "__main__":
    main()
