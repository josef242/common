"""
body_lr_controller.py — FFN-only pdr feedback controller (kv3).

Restores the relative-LR anneal that tangent projection removed, by controlling the FFN
body angular step size (pdr = ||dW||/||W|| ~ angular LR) toward a reference trajectory,
while leaving the QK-norm-self-regularized attention path at m=1.0.

Design: docs/KV3_CONTROLLER_DESIGN.md (Code <-> Math, Briefs #7 + Q8). Validated offline in
tools/pdr_controller_sim.py (FF-only, smooth dn2_merge, no freeze/alarm across 15 K-scenarios).

Plant (verified):  pdr = K(t) * m,   K = group_lr * ||update|| / ||W||   (measurable: K = pdr/m).
Controller:        feedforward inversion  m = r(t) / K_ema   + optional log-space PI trim (off
                   by default). K smoothed by a log-space EMA. m in [m_floor, 1], asymmetric
                   rate limit, warmup gate, anti-windup. FFN params only.

This module is OPTIMIZER-AGNOSTIC: it computes a scalar multiplier for the FFN body group given
the latest measured FFN pdr. The training loop is responsible for (a) feeding pdr at the control
cadence, (b) writing the returned multiplier into lr_scale_overrides[id(p)] for FFN body params
every step, and (c) checkpointing via state_dict()/load_state_dict().
"""
from __future__ import annotations
import math
from typing import Optional, Sequence, Tuple, Dict, Any

_STATE_VERSION = 3  # v3: shadow-norm modes (auto_shadow_growth/partial) — name-keyed S + radial-budget WD.
                    # v1 (knots-only) checkpoints migrate cleanly (auto fields default; knots restore complete).


def _interp(knots: Sequence[Tuple[float, float]], x: float) -> float:
    """Linear interpolation over (x, y) knots; flat extrapolation past the ends."""
    if not knots:
        return float("nan")
    if x <= knots[0][0]:
        return knots[0][1]
    if x >= knots[-1][0]:
        return knots[-1][1]
    for i in range(1, len(knots)):
        x0, y0 = knots[i - 1]
        x1, y1 = knots[i]
        if x <= x1:
            return y0 + (y1 - y0) * (x - x0) / (x1 - x0)
    return knots[-1][1]


def _smoothstep(t: float, t0: float, t1: float) -> float:
    """C1-continuous 0->1 ramp on [t0, t1] (3u^2 - 2u^3), clamped outside."""
    if t1 <= t0:
        return 1.0 if t >= t1 else 0.0
    u = max(0.0, min(1.0, (t - t0) / (t1 - t0)))
    return u * u * (3.0 - 2.0 * u)


def _median(xs) -> float:
    """Plain median of a list (no numpy). Even length -> mean of the two middle."""
    s = sorted(xs)
    n = len(s)
    if n == 0:
        return float("nan")
    if n % 2 == 1:
        return s[n // 2]
    return 0.5 * (s[n // 2 - 1] + s[n // 2])


class PIDController:
    """Standalone log-space PID trim kernel (extracted from adaptive_wd.py w_rms_target).

    Returns a multiplicative trim exp(kp*e + I + kd*d) on a log-space error `e`. With
    kp=ki=kd=0 it is an exact no-op (returns 1.0) — the kv3 run-1 default (feedforward only).
    Anti-windup is handled by the caller (it knows the output-saturation state); call
    `freeze_integral()` to roll the integral back when the output is railed.
    """

    def __init__(self, kp: float = 0.0, ki: float = 0.0, kd: float = 0.0,
                 integral_clamp: float = 0.5):
        self.kp, self.ki, self.kd = kp, ki, kd
        self.iclamp = integral_clamp
        self.I = 0.0
        self._prev_pv: Optional[float] = None
        self._I_before: float = 0.0  # last pre-commit integral, for freeze/anti-windup
        # pending commit state — initialized here so accept()/freeze_integral() can never read
        # an undefined attribute even if (mistakenly) called before the first trim().
        self._pending_I: float = 0.0
        self._pending_pv: Optional[float] = None

    @property
    def active(self) -> bool:
        return not (self.kp == 0.0 and self.ki == 0.0 and self.kd == 0.0)

    def trim(self, error: float, pv: Optional[float] = None) -> float:
        """Compute trim from log-error `error`. `pv` (process variable, e.g. pdr_ema) is used
        for D-on-PV to avoid setpoint-change kick. Does not finalize the integral commit
        until the caller resolves anti-windup via accept()/freeze_integral()."""
        self._I_before = self.I
        new_I = max(-self.iclamp, min(self.iclamp, self.I + self.ki * error))
        if pv is None or self._prev_pv is None:
            d = 0.0
        else:
            d = pv - self._prev_pv
        self._pending_I = new_I
        self._pending_pv = pv
        return math.exp(self.kp * error + new_I + self.kd * d)

    def accept(self):
        """Commit the pending integral/PV after a non-railed step."""
        self.I = self._pending_I
        self._prev_pv = self._pending_pv

    def freeze_integral(self):
        """Anti-windup: keep the prior integral (do not wind further), but advance PV."""
        self.I = self._I_before
        self._prev_pv = self._pending_pv

    def state_dict(self) -> Dict[str, Any]:
        return {"I": self.I, "prev_pv": self._prev_pv}

    def load_state_dict(self, sd: Dict[str, Any]):
        self.I = sd.get("I", 0.0)
        self._prev_pv = sd.get("prev_pv", None)


class BodyLRController:
    """FFN-only pdr feedback controller. Config-gated; inert when disabled.

    Call pattern from the training loop:
      - every step:        m = ctrl.current_multiplier()      # held value to write to side-dict
      - at control cadence: ctrl.observe(step, tok_m, pdr_ffn) # updates the held m from fresh pdr
      - logging:           ctrl.diagnostics()                  # dict for a [ffn-ctrl] log line
      - checkpoint:        ctrl.state_dict() / load_state_dict()
    `enabled` False => current_multiplier() always 1.0 and observe() is a no-op.
    """

    def __init__(self, cfg: Optional[dict]):
        cfg = cfg or {}
        self.enabled: bool = bool(cfg.get("enabled", False))
        self.warmup_step: int = int(cfg.get("warmup_step", 1500))
        # NOTE: there is NO `cadence` field. observe() is driven by the trainer at its val_step cadence,
        # so val_step IS the control cadence. The EMA alphas + rate limits below are per-observe-sample
        # (i.e. per val_step) and are tuned for val_step ~100. (A stray `cadence:` key in a config is ignored.)
        # DEBUG open-loop override: if set, current_multiplier() returns this FIXED value with the feedback
        # loop BYPASSED — for actuator probes (pin a big cut, watch whether pdr responds). None = normal.
        _fm = cfg.get("force_m", None)
        self.force_m: Optional[float] = float(_fm) if _fm is not None else None
        self.m_floor: float = float(cfg.get("m_floor", 0.30))
        self.m_max: float = float(cfg.get("m_max", 1.0))
        self.k_alpha: float = float(cfg.get("k_ema_alpha", 0.15))
        self.pdr_alpha: float = float(cfg.get("pdr_ema_alpha", 0.15))
        self.rate_down: float = float(cfg.get("rate_down", 0.05))
        self.rate_up: float = float(cfg.get("rate_up", 0.02))
        # --- reference pdr curve(s) ---
        # The reference is a single `knots` curve (token-in-M -> target ffn-median pdr). OPTIONALLY,
        # a run that needs an early-regime target distinct from the mature glide (e.g. a from-scratch
        # run riding a high early-plasticity pdr before settling onto a measured glide) can supply
        # `blend_from`: a second knot curve that is smoothstep-crossfaded INTO `knots` over
        # [start_tok_m, end_tok_m]. With no `blend_from`, reference() is simply interp(knots) — the
        # common case. Fields are named by ROLE; the provenance of any given curve (which experiment
        # it was measured from) belongs in the config's comments, not the schema.
        ref = cfg.get("reference", {}) or {}
        # reference mode: 'knots' (hand-fit curve, default/legacy) or 'auto' (self-anchored LR-track:
        # capture the body's own frozen pdr at the freeze point, then ride r(t)=K_anchor*lr(t)/lr_anchor;
        # the loop's only job is to cancel the slow K-drift). docs/MATH_AGENT_Q11. 'auto' needs NO knots.
        self.ref_mode = str(ref.get("mode", "knots"))
        self.knots = [tuple(k) for k in ref.get("knots", [])]
        # --- self-anchoring (auto) config ---
        self.anchor_step = int(ref.get("anchor_step", 0))        # freeze point: f=1 AND body WD-taper done
        self.anchor_samples = int(ref.get("anchor_samples", 8))  # geometric-mean window (5-10 post-freeze)
        _wb = ref.get("anchor_warn_band", [0.6, 1.4])            # sanity bands vs trailing pre-freeze pdr EMA
        self.anchor_warn_lo, self.anchor_warn_hi = float(_wb[0]), float(_wb[1])
        _fb = ref.get("anchor_fatal_band", [0.35, 2.0])
        self.anchor_fatal_lo, self.anchor_fatal_hi = float(_fb[0]), float(_fb[1])
        _ab = ref.get("anchor_abs_warn", [3.0e-4, 1.0e-2])       # absolute plausible pdr range (loose,
        # widened so the scale-FREE relative-to-pre-freeze band is the real bug-catcher and this one
        # doesn't nuisance-warn on a hotter frozen body / different model scale)
        self.anchor_abs_lo, self.anchor_abs_hi = float(_ab[0]), float(_ab[1])
        self.upper_margin = float(cfg.get("upper_alarm_margin", 0.05))  # upper-rail trips if m_ff_raw > 1+margin
        _bf = ref.get("blend_from") or None
        if _bf:
            self.blend_knots: Optional[list] = [tuple(k) for k in _bf.get("knots", [])]
            self.blend_t0: Optional[float] = float(_bf["start_tok_m"])
            self.blend_t1: Optional[float] = float(_bf["end_tok_m"])
        else:
            self.blend_knots = None
            self.blend_t0 = None
            self.blend_t1 = None
        # Validate knot lists: non-empty (when enabled) + strictly ascending in x (else _interp
        # silently mis-extrapolates / returns NaN) + strictly positive y (r feeds m_ff=r/K_ema and
        # e=log(r/pdr_ema); a zero/negative knot would make log(r/…) raise or go -inf). Cheap guard
        # against a future edited config.
        # 'auto' mode discovers its own reference (no knots); only validate knots in 'knots' mode.
        _lists = [("knots", self.knots)] if self.ref_mode == "knots" else []
        if self.blend_knots is not None:
            _lists.append(("blend_from.knots", self.blend_knots))
        for _name, _kn in _lists:
            if not _kn:
                if _name == "knots" and not self.enabled:
                    continue        # disabled controller never calls reference(); no curve needed
                raise ValueError(f"ffn_pdr_controller.reference.{_name} is empty")
            _xs = [k[0] for k in _kn]
            if any(b <= a for a, b in zip(_xs, _xs[1:])):
                raise ValueError(f"ffn_pdr_controller.reference.{_name} x must be strictly ascending: {_xs}")
            if any(k[1] <= 0 for k in _kn):
                raise ValueError(f"ffn_pdr_controller.reference.{_name} y (pdr) must be > 0: "
                                 f"{[k[1] for k in _kn]}")
        # blend window must be a real interval (else the smoothstep degenerates to a hard step).
        if self.blend_knots is not None and self.blend_t0 >= self.blend_t1:
            raise ValueError("ffn_pdr_controller.reference.blend_from.start_tok_m must be < end_tok_m "
                             f"(got {self.blend_t0} >= {self.blend_t1})")
        # PI trim (off in run 1)
        self.pid = PIDController(kp=float(cfg.get("kp", 0.0)), ki=float(cfg.get("ki", 0.0)),
                                 kd=float(cfg.get("kd", 0.0)),
                                 integral_clamp=float(cfg.get("integral_clamp", 0.5)))
        # guardrails
        self.authority_low_m: float = float(cfg.get("authority_low_m", 0.5))
        self.alarm_pdr_ratio: float = float(cfg.get("alarm_pdr_ratio", 1.1))
        self.alarm_consecutive: int = int(cfg.get("alarm_consecutive", 3))
        # --- shadow-norm modes (auto_shadow_growth / auto_shadow_partial; Math Q12) ---
        # m = log-median(R/S); body WD = radial-budget law λ=clamp(λmin,λmax,ρ(1−f)γ_EMA).
        self.m_min_full: float = float(cfg.get("m_min_full", 0.20))   # f-aware floor base: m_min(f)=1−f(1−m_min_full)
        self.rho: float = float(cfg.get("rho", 0.20))                 # WD radial-budget fraction
        self.lam_max: float = float(cfg.get("lambda_max", 0.02))
        self.lam_min: float = float(cfg.get("lambda_min", 0.002))
        # glide gain: m = exp(g * median[log(R/S)]). g=1.0 = pure R/S (track the shadow counterfactual).
        # g>1 STEEPENS the cut -> cooler reference -> pulls the controlled pdr toward a free-growth run
        # (the shadow norm S, built from the cooled trajectory, runs a hair warm; g compensates). Math's
        # anticipated "learned-glide" knob, preferred band [1.0, 1.3]; a tuning dial, default inert.
        self.glide_gain: float = float(cfg.get("glide_gain", 1.0))
        # freeze_handoff: at f=1, hand off to the Q11 LR-track tail (True, default) OR keep the shadow
        # R/S law running into the frozen phase (False) — a continuous anneal with NO law-switch/kink.
        # False makes auto_shadow_growth behave like the partial law at f=1 (S keeps carrying history,
        # m keeps falling); the body is still fully frozen (bounded norm), it just isn't handed off.
        self.freeze_handoff: bool = bool(cfg.get("freeze_handoff", True))
        self._ckpt_freeze_handoff: Optional[bool] = None  # set by load_state_dict; trainer mismatch-guards it
        # acts_on_attn: broadcast the SAME FFN-computed m to the attention matrices (wq/wk/wv/wo) too.
        # The controller is unchanged (one m, FFN-only median/S); the trainer just applies the held m to
        # attn ids as well. Relies on attn~ffn free-growth (the picket-hyg lock-step) so one m serves both.
        # Pure pdr/LR control — attn keeps its flat base WD (NOT the radial-budget λ, which is FFN-only).
        self.acts_on_attn: bool = bool(cfg.get("acts_on_attn", False))

        # live state
        self.m: float = 1.0
        self._logK: Optional[float] = None
        self._pdr_ema: Optional[float] = None
        self._last: Dict[str, Any] = {}     # last observe() snapshot, for diagnostics
        self._alarm_run: int = 0
        self._dropped: int = 0              # consecutive missing/invalid pdr samples held
        self.alarm: bool = False            # lower rail: out of cooling authority (floor pinned + pdr>1.1 r)
        self.alarm_ever: bool = False       # historical record: lower-rail alarm fired at least once
        self.inspect: bool = False          # m below authority floor before the merge region
        # upper rail (Math Q11): m pinned at m_max while the UNCLAMPED demand wants more (m_ff_raw>1) =
        # reference unreachable / no upward authority (body cooler than target). Informational, NOT hot-body.
        self._upper_run: int = 0
        self.upper_alarm: bool = False
        self.upper_alarm_ever: bool = False
        # last unclamped demand — telemetry + the upper rail's watch quantity (closed loop: r/K_ema;
        # shadow modes: the R/S-law m, or r/K_ema in the frozen LR-track tail)
        self._m_ff_raw: Optional[float] = None
        # self-anchoring (auto) live state
        self.K_anchor: Optional[float] = None       # latched frozen-body plant gain (= pdr_anchor/m_anchor)
        self.lr_anchor: Optional[float] = None       # LR at the anchor step (for r=K_anchor*lr/lr_anchor)
        self.anchor_set: bool = False
        self._anchor_buf: list = []                  # post-freeze pdr samples being collected
        self._pre_freeze_pdr_ema: Optional[float] = None  # trailing EMA for the anchor sanity band
        self.anchor_warn: Optional[str] = None       # set once at latch; trainer logs (HEALTH WARNING)
        self.anchor_fatal: Optional[str] = None      # set once at latch; trainer escalates to fatal_error
        self._just_latched: bool = False             # transient: True only on the latch step (trainer logs)
        self.lr_fingerprint: Optional[tuple] = None  # LR-schedule id captured by the trainer; resume guard
        # shadow-norm (auto_shadow_growth / auto_shadow_partial) live state
        self.S: Dict[str, float] = {}                # per-FFN-matrix shadow norm, keyed by PARAM NAME (resume-stable)
        self.shadow_active: bool = False             # latched once f>0 (S initialized = R)
        self.frozen: bool = False                    # latched once f>=1 (growth-mode handoff to LR-track)
        self.r_freeze: Optional[float] = None        # target pdr captured at f=1 (post-guardrail K_ema*m)
        self.lr_freeze: Optional[float] = None       # LR at the f=1 crossing (for r(t)=r_freeze*lr/lr_freeze)
        self._gamma_ema: Optional[float] = None      # EMA(median_i γ_i) — drives the radial-budget WD law
        self.lam_body: Optional[float] = None        # current radial-budget body WD λ (trainer reads via current_wd)
        self.theta_actual: float = 0.0               # Σ pdr_ffn (cumulative realized angle)  — Math's
        self.theta_ref: float = 0.0                  # Σ r       (cumulative reference angle)  — drift monitor

    # ---- reference (knots mode; 'auto' mode computes r=K_anchor*lr/lr_anchor inline in observe) ----
    def reference(self, tok_m: float) -> float:
        if self.blend_knots is None:
            return _interp(self.knots, tok_m)
        a = _smoothstep(tok_m, self.blend_t0, self.blend_t1)
        return (1.0 - a) * _interp(self.blend_knots, tok_m) + a * _interp(self.knots, tok_m)

    # ---- self-anchoring: latch the frozen-body plant gain from the post-freeze pdr samples ----
    def _latch_anchor(self, scheduled_lr: float) -> None:
        """Latch K_anchor = geometric-mean(post-freeze pdr) / m_anchor (m=1 during collection) and
        lr_anchor = scheduled_lr. Warm the K_ema at K_anchor so the first active step is transient-free.
        Run the (loose) sanity bands vs the trailing pre-freeze pdr EMA; set anchor_warn/anchor_fatal
        for the trainer to surface (fatal_error lives in the trainer; this module stays torch-free)."""
        buf = self._anchor_buf
        pdr_anchor = math.exp(sum(math.log(x) for x in buf) / len(buf))  # geometric mean (pdr noise is multiplicative)
        m_anchor = self.m                                                # 1.0 throughout collection
        self.K_anchor = pdr_anchor / max(m_anchor, 1e-6)
        # lr_anchor = LR at the LAST collection step. K_anchor (geomean over the window) reflects ~the
        # window-midpoint LR, so K_anchor/lr_anchor carries a sub-percent bias over an
        # anchor_samples*cadence window on a slow cosine — well below the upper-rail margin, and a
        # constant offset the live K_ema absorbs within a few active samples. Left simple deliberately.
        self.lr_anchor = float(scheduled_lr)
        self.anchor_set = True
        self._just_latched = True
        # warm-start the smoothers at the anchor so the first active observe doesn't lurch
        self._logK = math.log(self.K_anchor)
        self._pdr_ema = pdr_anchor
        # sanity bands (loose — catch capture bugs, not enforce theory): relative to the trailing
        # pre-freeze pdr EMA, then an absolute plausibility check.
        base = self._pre_freeze_pdr_ema
        if base is not None and base > 0:
            ratio = pdr_anchor / base
            if ratio < self.anchor_fatal_lo or ratio > self.anchor_fatal_hi:
                self.anchor_fatal = (f"pdr_anchor={pdr_anchor:.3e} is {ratio:.2f}x the trailing pre-freeze pdr "
                                     f"({base:.3e}) — outside fatal band [{self.anchor_fatal_lo},{self.anchor_fatal_hi}]; "
                                     f"likely a capture bug (wrong anchor_step / unsettled plant).")
            elif ratio < self.anchor_warn_lo or ratio > self.anchor_warn_hi:
                self.anchor_warn = (f"pdr_anchor={pdr_anchor:.3e} is {ratio:.2f}x the trailing pre-freeze pdr "
                                    f"({base:.3e}) — outside warn band [{self.anchor_warn_lo},{self.anchor_warn_hi}].")
        else:
            # No genuine pre-freeze samples (e.g. anchor_step ~ warmup_step, or f hit 1.0 immediately):
            # the scale-free RELATIVE band could not run — only the loose absolute band guarded the
            # capture. Make that explicit so the operator isn't lulled into thinking it was fully checked.
            self.anchor_warn = ("no pre-freeze pdr samples were seen before the anchor window, so the "
                                "relative sanity band was SKIPPED (only the absolute band ran). Consider a "
                                "larger gap between warmup_step and anchor_step.")
        if pdr_anchor < self.anchor_abs_lo or pdr_anchor > self.anchor_abs_hi:
            _amsg = (f"pdr_anchor={pdr_anchor:.3e} outside absolute plausible range "
                     f"[{self.anchor_abs_lo:.1e},{self.anchor_abs_hi:.1e}].")
            self.anchor_warn = (self.anchor_warn + " " + _amsg) if self.anchor_warn else _amsg

    # ---- actuator value (held; written every step by the loop) ----
    def current_multiplier(self) -> float:
        if self.force_m is not None:
            return self.force_m            # DEBUG open-loop override: m pinned, feedback bypassed
        return 1.0 if not self.enabled else self.m

    def current_wd(self) -> Optional[float]:
        """Radial-budget body WD λ_body for the shadow modes (None when not set / not a shadow mode).
        The trainer writes this into wd_overrides for the FFN body matrices each cadence."""
        return self.lam_body

    def _update_gamma_ema(self, radial: Optional[dict]) -> None:
        """EMA of the median per-matrix free radial-growth rate γ = max(0, −⟨U,W⟩/‖W‖²)."""
        if not radial:
            return
        gammas = [max(0.0, g) for (_R, _dR, g) in radial.values()]
        if not gammas:
            return
        med = _median(gammas)
        self._gamma_ema = med if self._gamma_ema is None else \
            (1 - self.k_alpha) * self._gamma_ema + self.k_alpha * med

    def _lambda(self, f: float) -> float:
        """Radial-budget WD: clamp(λ_min, λ_max, ρ(1−f)γ_EMA). Before any γ sample, defaults to
        λ_max (the high-early-WD regularization prior — used when the budget can support it)."""
        if self._gamma_ema is None:
            return self.lam_max
        raw = self.rho * (1.0 - f) * self._gamma_ema
        return max(self.lam_min, min(self.lam_max, raw))

    # ---- control update (at cadence, when fresh pdr is available) ----
    def observe(self, step: int, tok_m: float, pdr_ffn: float,
                scheduled_lr: Optional[float] = None, f_now: Optional[float] = None,
                radial: Optional[dict] = None, eta_accum: Optional[float] = None) -> float:
        """Update the held multiplier from a fresh FFN-median pdr sample.

        `scheduled_lr` (current cosine body LR) and `f_now` (current tangent_project_strength) are
        REQUIRED for ref_mode 'auto' (the LR-track reference rides scheduled_lr; the anchor latches
        only once f_now>=1 — the true freeze point). Both are ignored in 'knots' mode.

        SHADOW modes ('auto_shadow_growth' / 'auto_shadow_partial') additionally take `radial`
        ({param_name: (R, ΔR_free_accum, γ)} accumulated by the trainer since the last observe) and
        `eta_accum` (Σ scheduled_lr over the window, for the f=0 WD shrink of S). They ignore the
        warmup gate (engagement is f-gated) and tolerate a bad pdr_ffn (m comes from R/S, not pdr).
        """
        self._just_latched = False
        if not self.enabled:
            return 1.0
        if self.ref_mode in ("auto_shadow_growth", "auto_shadow_partial"):
            return self._observe_shadow(step, pdr_ffn, scheduled_lr, f_now, radial, eta_accum)
        if step < self.warmup_step:
            self.m = 1.0                      # warmup gate: hold unity, loop frozen
            self._last = dict(step=step, tok_m=tok_m, pdr=pdr_ffn, r=None,
                              K_ema=None, m=self.m, gated=True)
            return self.m
        # HOLD on a missing/invalid sample. CRUCIALLY reject non-finite pdr: a NaN/inf passes
        # `<= 0.0` (nan<=0 is False, inf<=0 is False), would poison the log-EMAs PERMANENTLY
        # (NaN is sticky through exp/log; inf -> log(r/inf)=log(0)=ValueError), and that poisoned
        # state is checkpointed — one loss-spike sample could kill the controller for the whole
        # run with no recovery. Reject and hold the last good m instead.
        if pdr_ffn is None or not math.isfinite(pdr_ffn) or pdr_ffn <= 0.0:
            self._dropped += 1
            # Reached only AFTER the warmup gate, so clear any leftover gated flag (else
            # log_line would mislabel an engaged-but-dropped controller as "warmup-gated").
            self._last = dict(self._last, gated=False, stale=True, dropped=self._dropped)
            return self.m
        self._dropped = 0

        # ---- ref_mode 'auto': self-anchored LR-track (Math Q11) ----
        if self.ref_mode == "auto":
            if not self.anchor_set:
                # Pre-anchor: hold m=1 (the still-growing/transitioning body self-anneals).
                self.m = 1.0
                # Collect anchor samples only at/past anchor_step AND once the body is truly FROZEN
                # (f_now>=1). f_now=None trusts anchor_step alone — only the pure-Python tests pass None;
                # Settings validation forces tangent_project on for auto, so the trainer always passes a
                # real f. Needs the live LR to latch lr_anchor.
                frozen = (f_now is None) or (f_now >= 1.0 - 1e-6)
                ready = (step >= self.anchor_step) and frozen and (scheduled_lr is not None and scheduled_lr > 0)
                if ready:
                    self._anchor_buf.append(pdr_ffn)
                    if len(self._anchor_buf) >= self.anchor_samples:
                        self._latch_anchor(scheduled_lr)
                else:
                    # Build the trailing pre-freeze pdr EMA (the latch-time sanity-band baseline) ONLY
                    # from genuinely pre-freeze samples — NEVER from the anchor-collection samples, or the
                    # baseline absorbs the very values it must validate against and the warn/fatal bands
                    # desensitize toward ratio 1.0 (Math Q11 item 8; would silently miss a 2-4x capture bug).
                    self._pre_freeze_pdr_ema = pdr_ffn if self._pre_freeze_pdr_ema is None else math.exp(
                        (1 - self.pdr_alpha) * math.log(self._pre_freeze_pdr_ema)
                        + self.pdr_alpha * math.log(pdr_ffn))
                # phase reflects what is ACTUALLY happening: 'anchoring' only while collecting; past
                # anchor_step but f<1 (body not yet frozen) is 'awaiting-freeze', not stalled collection.
                _phase = "anchoring" if ready else ("awaiting-freeze" if step >= self.anchor_step else "pre-anchor")
                self._last = dict(step=step, tok_m=tok_m, pdr=pdr_ffn, r=None, K_ema=None, m=self.m,
                                  gated=False, phase=_phase, collected=len(self._anchor_buf))
                return self.m
            # Anchored: LR-track reference rides the live LR. Without it we cannot form r — hold + stale.
            if scheduled_lr is None or scheduled_lr <= 0 or self.lr_anchor is None:
                self._dropped += 1
                self._last = dict(self._last, gated=False, stale=True, dropped=self._dropped)
                return self.m
            r = self.K_anchor * (scheduled_lr / self.lr_anchor)
        else:
            # ---- ref_mode 'knots' (legacy/hand-fit; math unchanged) ----
            r = self.reference(tok_m)

        # ---- common closed loop (identical math for both modes once r is set) ----
        # log-space EMAs. K_inst is intentionally RAW (pdr/m) — it is itself smoothed into the
        # _logK EMA; _pdr_ema is a SEPARATE PV smoother used only by the PI trim's error term, so
        # the two are not redundant (the feedforward uses K_ema, the trim uses _pdr_ema).
        self._pdr_ema = pdr_ffn if self._pdr_ema is None else math.exp(
            (1 - self.pdr_alpha) * math.log(self._pdr_ema) + self.pdr_alpha * math.log(pdr_ffn))
        K_inst = pdr_ffn / max(self.m, 1e-6)
        self._logK = math.log(K_inst) if self._logK is None else \
            (1 - self.k_alpha) * self._logK + self.k_alpha * math.log(K_inst)
        K_ema = max(math.exp(self._logK), 1e-12)   # floor: never divide by zero/denormal below
        # feedforward inversion + optional PI trim (log-error)
        e = math.log(r / self._pdr_ema)
        m_ff = r / K_ema
        self._m_ff_raw = m_ff                       # unclamped demand (telemetry + upper-rail alarm)
        trim = self.pid.trim(e, pv=self._pdr_ema) if self.pid.active else 1.0
        m_cmd = m_ff * trim
        # asymmetric rate limit (cool fast, reheat slow)
        m_cmd = max(self.m * (1 - self.rate_down), min(self.m * (1 + self.rate_up), m_cmd))
        # output clamp
        m_clamped = max(self.m_floor, min(self.m_max, m_cmd))
        # anti-windup: freeze integral if pinned at a rail in the error's direction
        if self.pid.active:
            if (m_clamped >= self.m_max and e > 0) or (m_clamped <= self.m_floor and e < 0):
                self.pid.freeze_integral()
            else:
                self.pid.accept()
        self.m = m_clamped
        # ---- guardrails: BOTH rails, regime-aware (Math Q11). Current-state (not latched) so each
        # clears when resolved; `*_ever` keeps the historical record. All checkpointed. ----
        # `inspect`: m below the authority floor while still inside the early blend region (knots only).
        _blend_until = self.blend_t1 if self.blend_t1 is not None else float("-inf")
        self.inspect = (self.m < self.authority_low_m and tok_m < _blend_until)
        # LOWER rail — the REAL hot-body / insufficient-COOLING-authority alarm: pinned at the floor yet
        # pdr still > 1.1 r (cutting as hard as allowed and still can't cool to target).
        if self.m <= self.m_floor + 1e-9 and pdr_ffn > self.alarm_pdr_ratio * r:
            self._alarm_run += 1
        else:
            self._alarm_run = 0
        self.alarm = self._alarm_run >= self.alarm_consecutive
        self.alarm_ever = self.alarm_ever or self.alarm
        # UPPER rail — reference unreachable / no UPWARD authority: pinned at m_max yet the unclamped
        # demand wants more (m_ff_raw > 1+margin). The body is BELOW target (cooler), NOT hot — amplifying
        # is deliberately forbidden, so this is informational (anchor/ref too high, base LR too low,
        # or the body cooling faster than asked). Becomes worrying only if the run also underfits.
        if self.m >= self.m_max - 1e-9 and m_ff > 1.0 + self.upper_margin:
            self._upper_run += 1
        else:
            self._upper_run = 0
        self.upper_alarm = self._upper_run >= self.alarm_consecutive
        self.upper_alarm_ever = self.upper_alarm_ever or self.upper_alarm
        self._last = dict(step=step, tok_m=tok_m, pdr=pdr_ffn, pdr_ema=self._pdr_ema, r=r,
                          K_ema=K_ema, m=self.m, gated=False, m_ff_raw=m_ff)
        return self.m

    # ---- shadow-norm control (auto_shadow_growth / auto_shadow_partial; Math Q12) ----
    def _observe_shadow(self, step, pdr_ffn, scheduled_lr, f_now, radial, eta_accum) -> float:
        """m = log-median(R_i/S_i) replacing the LR/anneal knots; body WD = radial-budget law.
        S is the f=0, m=1, no-projection counterfactual norm (its WD shrink uses λ_S at f=0). At
        f=1, growth mode hands off to the Q11 LR-track tail; partial mode stays in the shadow law.
        See docs/SHADOW_NORM_PDR_CONTROLLER_SPEC.md."""
        partial = (self.ref_mode == "auto_shadow_partial")
        f = 0.0 if f_now is None else float(f_now)

        # γ_EMA drives BOTH WD values (live from step 0 so high-early-WD comes via the λ_max ceiling).
        self._update_gamma_ema(radial)
        lam_S = self._lambda(0.0)              # shadow WD: the f=0 counterfactual value (Math)
        self.lam_body = self._lambda(f)        # actual optimizer WD: radial-budget at the live f

        # ---- idle (f==0): hold m=1; shadow tracks R (no divergence yet) ----
        # Only track S=R BEFORE the controller has ever engaged (not self.shadow_active). Once the
        # integral is live, an f-dip back to 0 must NOT wipe it (defensive — shadow modes also
        # require a monotone-up f schedule via Settings; this is belt-and-suspenders).
        if f <= 0.0:
            self.m = 1.0
            if radial and not self.shadow_active:
                for name, (R, _dR, _g) in radial.items():
                    self.S[name] = R
            self._last = dict(step=step, f=f, m=1.0, phase="idle", r=None, pdr=pdr_ffn,
                              lam=self.lam_body, lam_S=lam_S, gamma_ema=self._gamma_ema,
                              shadow_n=len(self.S), gated=False)
            return self.m

        # ---- f>0: need a radial sample to act; else hold m + λ stale ----
        if not radial:
            self._dropped += 1
            self._last = dict(self._last, stale=True, dropped=self._dropped, phase="ramp", gated=False)
            return self.m
        self._dropped = 0

        # init shadow norms at the first f>0 step (eager, ALL matrices) and HOLD m=1 this window —
        # begin the integral the NEXT window (spec §2.3: "snapshot S=R; accumulate THEREAFTER").
        # Accumulating this window's ΔR_free on top of an already-end-of-window R would make S>R
        # instantly => a spurious ~one-window LR cut at f-onset. The trainer resets the ΔR_free
        # window after every observe, so the next window starts the integral cleanly.
        if not self.shadow_active:
            for name, (R, _dR, _g) in radial.items():
                self.S[name] = R
            self.shadow_active = True
            m_min_f = 1.0 - f * (1.0 - self.m_min_full)
            self.m = max(m_min_f, min(self.m_max, 1.0))
            self._last = dict(step=step, f=f, m=self.m, phase="engage", r=None, pdr=pdr_ffn,
                              lam=self.lam_body, lam_S=lam_S, gamma_ema=self._gamma_ema,
                              m_min_f=m_min_f, shadow_n=len(self.S), gated=False)
            return self.m

        # accumulate S = S + ΔR_free − η·λ_S·S  (WD shrink at the f=0 / m=1 counterfactual rate)
        eta_acc = 0.0 if eta_accum is None else float(eta_accum)
        ratios_log = []
        for name, (R, dR_free, _g) in radial.items():
            S = self.S.get(name, R)            # late-appearing matrix: seed S=R
            # robust outlier guard (spec §5): a single anomalous step (e.g. a loss spike driving
            # _dot wild) must not PERMANENTLY corrupt the checkpointed S. Bound the window's net
            # free-growth to ±50% of S (≫ the ~3%/window normal, so only true spikes ever clip).
            _cap = 0.5 * S
            dR_free = _cap if dR_free > _cap else (-_cap if dR_free < -_cap else dR_free)
            S = S + dR_free - eta_acc * lam_S * S
            if S < 1e-12:
                S = 1e-12                      # guard (growth should dominate; never divide by ~0)
            self.S[name] = S
            if R > 0:
                ratios_log.append(math.log(R / S))
        if not ratios_log:
            self._last = dict(self._last, stale=True, phase="ramp", gated=False)
            return self.m
        # m = exp(g · median[log(R/S)]) = geomean(R/S)^g. g=1.0 (default) = pure shadow ratio; g>1
        # STEEPENS the cut (cooler reference) — the glide-gain tuning knob.
        m_raw = math.exp(self.glide_gain * _median(ratios_log))

        # K_ema from THIS pdr sample and the m that produced it (current self.m, pre-update):
        # telemetry + the frozen-tail inversion + the lower-rail alarm reference.
        if pdr_ffn is not None and math.isfinite(pdr_ffn) and pdr_ffn > 0:
            K_inst = pdr_ffn / max(self.m, 1e-6)
            self._logK = math.log(K_inst) if self._logK is None else \
                (1 - self.k_alpha) * self._logK + self.k_alpha * math.log(K_inst)
        K_ema = math.exp(self._logK) if self._logK is not None else None

        # ---- phase + target m ----
        # freeze_handoff gates the f=1 latch: with it OFF the body is still fully frozen (f=1 clamp),
        # but we keep commanding m = log-median(R/S) instead of handing off to the LR-track tail ->
        # a continuous anneal with NO law-switch/kink (S keeps growing while R is frozen -> m keeps
        # falling). partial mode never latches regardless (freeze_handoff is a no-op there).
        frozen_now = (not partial) and self.freeze_handoff and (f >= 1.0 - 1e-6)
        if frozen_now and self.frozen:
            # LR-track tail: r(t) = r_freeze * lr/lr_freeze ; m = r / K_ema
            if K_ema is None or scheduled_lr is None or scheduled_lr <= 0 or self.lr_freeze is None:
                self._last = dict(self._last, stale=True, phase="frozen", gated=False)
                return self.m
            r = self.r_freeze * (float(scheduled_lr) / self.lr_freeze)
            m_target = r / K_ema
            phase = "frozen"
        else:
            r = (K_ema * m_raw) if K_ema is not None else None   # logged target pdr (telemetry)
            m_target = m_raw
            if partial:
                phase = "partial"
            elif frozen_now:
                phase = "frozen-latch"
            elif f >= 1.0 - 1e-6:
                phase = "open"          # growth + freeze_handoff:false at f=1 -> R/S continues (no kink)
            else:
                phase = "ramp"
        # the unclamped demand of THIS window (R/S law, or r/K_ema in the LR-track tail) — the
        # quantity the upper rail watches; checkpointed telemetry, same meaning as the closed loop's.
        self._m_ff_raw = m_target

        # guardrails: asymmetric slew -> f-aware floor -> hard clamp
        m_cmd = max(self.m * (1 - self.rate_down), min(self.m * (1 + self.rate_up), m_target))
        m_min_f = 1.0 - f * (1.0 - self.m_min_full)
        m_cmd = max(m_min_f, min(self.m_max, m_cmd))
        self.m = m_cmd

        # capture the freeze anchor ONCE, AFTER guardrails (post-guardrail commanded m; Math). The
        # `r_freeze is None` belt makes "anchor at most once, EVER" structural: even if a resume
        # restored frozen=False with an anchor already set, we never silently re-capture at a later step.
        if (frozen_now and not self.frozen and self.r_freeze is None
                and K_ema is not None and scheduled_lr and scheduled_lr > 0):
            self.r_freeze = K_ema * self.m
            self.lr_freeze = float(scheduled_lr)
            self.frozen = True
            self._just_latched = True

        # lower-rail alarm: pinned at the f-aware floor yet pdr still > 1.1 r (out of cooling authority)
        r_eff = r if r is not None else ((K_ema * self.m) if K_ema is not None else None)
        if (r_eff is not None and pdr_ffn is not None and math.isfinite(pdr_ffn)
                and self.m <= m_min_f + 1e-9 and pdr_ffn > self.alarm_pdr_ratio * r_eff):
            self._alarm_run += 1
        else:
            self._alarm_run = 0
        self.alarm = self._alarm_run >= self.alarm_consecutive
        self.alarm_ever = self.alarm_ever or self.alarm

        # UPPER rail (Math Q11, shadow analogue): pinned at m_max while the unclamped demand wants
        # more (m_target > 1+margin). Post-engagement the free-growth counterfactual S must outgrow
        # the clamped R, so a persistent R/S >= 1+margin means S is drifting/corrupt (a WIPED S is
        # fataled at resume; a drifting one is not) — or, in the LR-track tail, the reference is
        # unreachable. Either would otherwise hold m at m_max SILENTLY. Informational, NOT hot-body
        # (amplification stays forbidden); mirrors the closed-loop rail above.
        if self.m >= self.m_max - 1e-9 and m_target > 1.0 + self.upper_margin:
            self._upper_run += 1
        else:
            self._upper_run = 0
        self.upper_alarm = self._upper_run >= self.alarm_consecutive
        self.upper_alarm_ever = self.upper_alarm_ever or self.upper_alarm

        # cumulative-angle drift monitor (Math §6): Θ_actual=Σpdr vs Θ_ref=Σr. A persistently growing
        # gap = the controller drifting off its OWN reference (shadow estimate / guardrails too weak).
        # Telemetry only — does NOT touch the control path. (Tracking the *reference*; the gap to a
        # free-growth twin like picket-hyg is a reference-LEVEL question, separate from this.)
        if r_eff is not None and pdr_ffn is not None and math.isfinite(pdr_ffn) and pdr_ffn > 0:
            self.theta_actual += pdr_ffn
            self.theta_ref += r_eff

        self._last = dict(step=step, f=f, m=self.m, phase=phase, m_ff_raw=self._m_ff_raw, r=r_eff,
                          K_ema=K_ema, pdr=pdr_ffn, lam=self.lam_body, lam_S=lam_S,
                          gamma_ema=self._gamma_ema, m_min_f=m_min_f, shadow_n=len(self.S),
                          theta_drift=(self.theta_actual - self.theta_ref), gated=False)
        return self.m

    # ---- logging ----
    def diagnostics(self) -> Dict[str, Any]:
        d = dict(self._last)
        d.update(enabled=self.enabled, alarm=self.alarm, upper_alarm=self.upper_alarm,
                 inspect=self.inspect, anchor_set=self.anchor_set,
                 K_anchor=self.K_anchor, lr_anchor=self.lr_anchor)
        return d

    def log_line(self) -> str:
        """One-line status, parallel to the [body-pdr] line."""
        if not self.enabled:
            return ""
        d = self._last
        # shadow-norm modes: own status line (m from R/S, plus γ_EMA + λ_body telemetry).
        if self.ref_mode in ("auto_shadow_growth", "auto_shadow_partial"):
            if not d or d.get('m') is None:
                return f"  [ffn-ctrl] SHADOW m={self.current_multiplier():.3f} (pre-observe)"
            def _t(label, k, fmt):
                v = d.get(k)
                return (f" {label}=" + format(v, fmt)) if v is not None else ""
            _flags = ("" if not d.get('stale') else f" STALE×{d.get('dropped', 0)}") + \
                     ("" if not self.alarm else " ALARM:base-LR-too-high") + \
                     ("" if not self.upper_alarm else " upper:no-upward-authority")
            return (f"  [ffn-ctrl] SHADOW[{d.get('phase', '?')}] m={d.get('m', float('nan')):.3f}"
                    f"{_t('m_raw', 'm_ff_raw', '.3f')}"
                    f"{_t('pdr', 'pdr', '.3e')}{_t('r', 'r', '.3e')}{_t('K', 'K_ema', '.3e')}"
                    f"{_t('g', 'gamma_ema', '.3e')}{_t('lam', 'lam', '.4f')}"
                    f"{_t('drift', 'theta_drift', '+.2e')}"
                    f" Sn={d.get('shadow_n', 0)}{_flags}")
        # auto-mode pre-anchor / awaiting-freeze / anchoring phase: no reference yet, m held at 1.
        if d.get('phase') in ('pre-anchor', 'awaiting-freeze', 'anchoring') and not d.get('stale'):
            return (f"  [ffn-ctrl] AUTO {d['phase']}: m=1.000 holding "
                    f"(pdr_ffn={d.get('pdr', float('nan')):.3e}, anchor "
                    f"{d.get('collected', 0)}/{self.anchor_samples})")
        if not d or 'pdr' not in d or d.get('gated') or d.get('r') is None:
            # No successful CLOSED-LOOP observe to report yet. Distinguish: a fresh warmup start; a
            # resume that restored a non-unity m but hasn't re-observed; or dropped samples before the
            # first good one. (Don't mislabel an engaged controller "warmup-gated".)
            if d.get('gated'):
                tag = "warmup-gated"
            elif d.get('stale'):
                tag = f"no valid pdr yet, dropped×{d.get('dropped', 0)}"
            elif self.m >= 1.0 - 1e-9:
                tag = "warmup-gated"
            else:
                tag = "resumed, pre-observe"
            return f"  [ffn-ctrl] m={self.current_multiplier():.3f} ({tag})"
        flags = ("" if not self.alarm else " ALARM:base-LR-too-high") + \
                ("" if not self.upper_alarm else " upper:no-upward-authority") + \
                ("" if not self.inspect else " inspect:low-m-early") + \
                ("" if not d.get("stale") else f" STALE:pdr-dropped×{d.get('dropped', 0)}")
        _raw = d.get('m_ff_raw')
        _rawtag = f" m_raw={_raw:.3f}" if _raw is not None else ""
        return (f"  [ffn-ctrl] pdr_ffn={d['pdr']:.3e} r={d['r']:.3e} "
                f"K_ema={d['K_ema']:.3e} m_ffn={d['m']:.3f}{_rawtag}{flags}")

    # ---- checkpoint (mirror the AWD versioned, rank-0 .pt convention) ----
    def state_dict(self) -> Dict[str, Any]:
        return {"version": _STATE_VERSION, "m": self.m, "logK": self._logK,
                "pdr_ema": self._pdr_ema, "alarm_run": self._alarm_run,
                "alarm": self.alarm, "alarm_ever": self.alarm_ever, "inspect": self.inspect,
                "dropped": self._dropped, "pid": self.pid.state_dict(),
                # v2 (self-anchored 'auto' mode + symmetric upper rail):
                "ref_mode": self.ref_mode,
                "K_anchor": self.K_anchor, "lr_anchor": self.lr_anchor,
                "anchor_set": self.anchor_set, "anchor_buf": list(self._anchor_buf),
                "pre_freeze_pdr_ema": self._pre_freeze_pdr_ema,
                "upper_run": self._upper_run, "upper_alarm": self.upper_alarm,
                "upper_alarm_ever": self.upper_alarm_ever, "m_ff_raw": self._m_ff_raw,
                "lr_fingerprint": self.lr_fingerprint,
                # v3 (shadow-norm modes): S keyed by PARAM NAME (resume-stable, NOT id()).
                "S": dict(self.S), "shadow_active": self.shadow_active, "frozen": self.frozen,
                "r_freeze": self.r_freeze, "lr_freeze": self.lr_freeze,
                "gamma_ema": self._gamma_ema, "lam_body": self.lam_body,
                "theta_actual": self.theta_actual, "theta_ref": self.theta_ref,
                # freeze_handoff is CONFIG-authoritative (kept from cfg on load, like ref_mode), but it
                # gates the post-f=1 LAW (LR-track tail vs continuous R/S) — checkpointed purely so the
                # trainer can FATAL if it's flipped across a resume (silent law-switch / r_freeze rebase).
                "freeze_handoff": self.freeze_handoff}

    def load_state_dict(self, sd: Dict[str, Any]):
        v = sd.get("version", 0)
        if v in (1, 2):
            # v1 (knots-only) / v2 (auto) -> v3: clean superset. The restored mode's state comes back
            # fully; the newer-mode fields below simply default (an older run never used them). Benign.
            print(f"[body-lr-ctrl] migrating state v{v} -> v{_STATE_VERSION} "
                  f"(restore complete; newer-mode fields default).")
        elif v != _STATE_VERSION:
            # Unknown version: restore best-effort but warn loudly (silently-wrong field semantics risk).
            print(f"[body-lr-ctrl] WARNING: state version {v} != {_STATE_VERSION}; "
                  f"restoring best-effort (m only fully guaranteed).")
        self.m = sd.get("m", 1.0)
        self._logK = sd.get("logK", None)
        self._pdr_ema = sd.get("pdr_ema", None)
        self._alarm_run = sd.get("alarm_run", 0)
        self.alarm = sd.get("alarm", False)
        self.alarm_ever = sd.get("alarm_ever", self.alarm)
        self.inspect = sd.get("inspect", False)
        self._dropped = sd.get("dropped", 0)
        if "pid" in sd:
            self.pid.load_state_dict(sd["pid"])
        # v2 additions (default for v1 checkpoints). ref_mode is CONFIG-authoritative — keep the
        # constructed value (you never switch modes mid-run); the checkpointed one is informational.
        self.K_anchor = sd.get("K_anchor", None)
        self.lr_anchor = sd.get("lr_anchor", None)
        self.anchor_set = bool(sd.get("anchor_set", False))
        self._anchor_buf = list(sd.get("anchor_buf", []))
        self._pre_freeze_pdr_ema = sd.get("pre_freeze_pdr_ema", None)
        self._upper_run = sd.get("upper_run", 0)
        self.upper_alarm = sd.get("upper_alarm", False)
        self.upper_alarm_ever = sd.get("upper_alarm_ever", self.upper_alarm)
        self._m_ff_raw = sd.get("m_ff_raw", None)
        self.lr_fingerprint = sd.get("lr_fingerprint", None)
        # v3 additions (default for v1/v2 checkpoints). S is keyed by PARAM NAME (resume-stable) — the
        # trainer re-resolves name->live-param each run, so the integral survives across process restarts.
        self.S = dict(sd.get("S", {}))
        self.shadow_active = bool(sd.get("shadow_active", False))
        self.frozen = bool(sd.get("frozen", False))
        self.r_freeze = sd.get("r_freeze", None)
        self.lr_freeze = sd.get("lr_freeze", None)
        self._gamma_ema = sd.get("gamma_ema", None)
        self.lam_body = sd.get("lam_body", None)
        self.theta_actual = float(sd.get("theta_actual", 0.0))
        self.theta_ref = float(sd.get("theta_ref", 0.0))
        # Stash the checkpointed freeze_handoff WITHOUT overwriting self.freeze_handoff (config wins).
        # None for pre-feature (v1/v2 or older v3) checkpoints — the trainer's mismatch guard treats
        # None as "unknown, can't check". The trainer compares this against the live config value.
        self._ckpt_freeze_handoff = sd.get("freeze_handoff", None)
