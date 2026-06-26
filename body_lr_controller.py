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

_STATE_VERSION = 1


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
        self.cadence: int = int(cfg.get("cadence", 100))
        self.m_floor: float = float(cfg.get("m_floor", 0.30))
        self.m_max: float = float(cfg.get("m_max", 1.0))
        self.k_alpha: float = float(cfg.get("k_ema_alpha", 0.15))
        self.pdr_alpha: float = float(cfg.get("pdr_ema_alpha", 0.15))
        self.rate_down: float = float(cfg.get("rate_down", 0.05))
        self.rate_up: float = float(cfg.get("rate_up", 0.02))
        # reference (smooth dn2_merge)
        ref = cfg.get("reference", {}) or {}
        self.merge_t0: float = float(ref.get("merge_start_tok_m", 197.0))
        self.merge_t1: float = float(ref.get("merge_end_tok_m", 575.0))
        self.kv2_early_knots = [tuple(k) for k in ref.get(
            "kv2_early_knots", [[197, 3.42e-3], [400, 3.10e-3], [600, 2.90e-3]])]
        # DN2's ACTUAL recorded FFN-median pdr, sampled out to the full run horizon (~26B tok).
        # Past the last knot _interp flat-extrapolates at DN2's late plateau (~1.26e-3) — DN2's
        # own data flattened there, so this is faithful. (Earlier the curve stopped at 1000M and
        # flat-lined at 2.20e-3 for ~97% of the run — a design-fidelity bug the 850M sim missed.)
        self.dn2_ffn_knots = [tuple(k) for k in ref.get(
            "dn2_ffn_knots", [[197, 1.66e-3], [393, 2.94e-3], [590, 2.58e-3], [787, 2.28e-3],
                              [1000, 2.20e-3], [2000, 1.85e-3], [4000, 1.64e-3], [8000, 1.38e-3],
                              [12000, 1.30e-3], [16000, 1.28e-3], [20000, 1.23e-3],
                              [26000, 1.26e-3]])]
        # Validate knot lists: non-empty and strictly ascending in x (else _interp silently
        # mis-extrapolates / returns NaN). Cheap guard against a future edited config.
        for _name, _kn in (("kv2_early_knots", self.kv2_early_knots),
                           ("dn2_ffn_knots", self.dn2_ffn_knots)):
            if not _kn:
                raise ValueError(f"ffn_pdr_controller.reference.{_name} is empty")
            _xs = [k[0] for k in _kn]
            if any(b <= a for a, b in zip(_xs, _xs[1:])):
                raise ValueError(f"ffn_pdr_controller.reference.{_name} x must be strictly ascending: {_xs}")
            # y (pdr target) must be strictly positive: r feeds m_ff=r/K_ema and e=log(r/pdr_ema);
            # a zero/negative knot would make log(r/…) raise or go -inf.
            if any(k[1] <= 0 for k in _kn):
                raise ValueError(f"ffn_pdr_controller.reference.{_name} y (pdr) must be > 0: "
                                 f"{[k[1] for k in _kn]}")
        # PI trim (off in run 1)
        self.pid = PIDController(kp=float(cfg.get("kp", 0.0)), ki=float(cfg.get("ki", 0.0)),
                                 kd=float(cfg.get("kd", 0.0)),
                                 integral_clamp=float(cfg.get("integral_clamp", 0.5)))
        # guardrails
        self.authority_low_m: float = float(cfg.get("authority_low_m", 0.5))
        self.alarm_pdr_ratio: float = float(cfg.get("alarm_pdr_ratio", 1.1))
        self.alarm_consecutive: int = int(cfg.get("alarm_consecutive", 3))

        # live state
        self.m: float = 1.0
        self._logK: Optional[float] = None
        self._pdr_ema: Optional[float] = None
        self._last: Dict[str, Any] = {}     # last observe() snapshot, for diagnostics
        self._alarm_run: int = 0
        self._dropped: int = 0              # consecutive missing/invalid pdr samples held
        self.alarm: bool = False            # CURRENTLY out of authority (floor pinned + pdr>1.1 r)
        self.alarm_ever: bool = False       # historical record: alarm fired at least once
        self.inspect: bool = False          # m below authority floor before the merge region

    # ---- reference ----
    def reference(self, tok_m: float) -> float:
        a = _smoothstep(tok_m, self.merge_t0, self.merge_t1)
        return (1.0 - a) * _interp(self.kv2_early_knots, tok_m) \
            + a * _interp(self.dn2_ffn_knots, tok_m)

    # ---- actuator value (held; written every step by the loop) ----
    def current_multiplier(self) -> float:
        return 1.0 if not self.enabled else self.m

    # ---- control update (at cadence, when fresh pdr is available) ----
    def observe(self, step: int, tok_m: float, pdr_ffn: float) -> float:
        if not self.enabled:
            return 1.0
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
        r = self.reference(tok_m)
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
        # guardrails. `alarm` reflects the CURRENT state (out of authority right now), not a
        # latch — so it clears when the condition clears, avoiding permanent log spam / fatigue.
        # `alarm_ever` keeps the historical record. Both are checkpointed.
        self.inspect = (self.m < self.authority_low_m and tok_m < self.merge_t1)
        if self.m <= self.m_floor + 1e-9 and pdr_ffn > self.alarm_pdr_ratio * r:
            self._alarm_run += 1
        else:
            self._alarm_run = 0
        self.alarm = self._alarm_run >= self.alarm_consecutive
        self.alarm_ever = self.alarm_ever or self.alarm
        self._last = dict(step=step, tok_m=tok_m, pdr=pdr_ffn, pdr_ema=self._pdr_ema, r=r,
                          K_ema=K_ema, m=self.m, gated=False)
        return self.m

    # ---- logging ----
    def diagnostics(self) -> Dict[str, Any]:
        d = dict(self._last)
        d.update(enabled=self.enabled, alarm=self.alarm, inspect=self.inspect)
        return d

    def log_line(self) -> str:
        """One-line status, parallel to the [body-pdr] line."""
        if not self.enabled:
            return ""
        d = self._last
        if not d or 'pdr' not in d or d.get('gated'):
            # No successful observe to report yet. Distinguish: a fresh warmup start; a resume
            # that restored a non-unity m but hasn't re-observed; or dropped samples before the
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
                ("" if not self.inspect else " inspect:low-m-early") + \
                ("" if not d.get("stale") else f" STALE:pdr-dropped×{d.get('dropped', 0)}")
        return (f"  [ffn-ctrl] pdr_ffn={d['pdr']:.3e} r={d['r']:.3e} "
                f"K_ema={d['K_ema']:.3e} m_ffn={d['m']:.3f}{flags}")

    # ---- checkpoint (mirror the AWD versioned, rank-0 .pt convention) ----
    def state_dict(self) -> Dict[str, Any]:
        return {"version": _STATE_VERSION, "m": self.m, "logK": self._logK,
                "pdr_ema": self._pdr_ema, "alarm_run": self._alarm_run,
                "alarm": self.alarm, "alarm_ever": self.alarm_ever, "inspect": self.inspect,
                "dropped": self._dropped, "pid": self.pid.state_dict()}

    def load_state_dict(self, sd: Dict[str, Any]):
        v = sd.get("version", 0)
        if v != _STATE_VERSION:
            # No migration path exists yet (only v1). Restore best-effort but warn loudly: a
            # future schema change could otherwise reload with silently-wrong field semantics.
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
