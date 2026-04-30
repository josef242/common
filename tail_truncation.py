# tail_truncation.py
# Progressive Tail Truncation — training-only technique that randomly
# truncates the network at a point in the tail, forcing head/mid layers
# to build stronger representations via shorter gradient paths.
# -----------------------------------------------------------------------------
import random
import math
from typing import List, Tuple, Union


class ProgressiveTailTruncation:
    """
    Randomly truncates the tail of the network during training.

    On each step, with probability `truncation_prob`, picks a cut point in the
    truncation zone (layers above `safe_fraction * n_layers`) and returns the
    number of layers to actually execute.  Layers below the safe zone always
    run.  The final norm + output head are always applied regardless.

    Uses a step-seeded Python stdlib RNG so all FSDP ranks make identical
    decisions — no communication needed.
    """

    def __init__(self, n_layers: int, config: dict):
        self.n_layers = n_layers
        self.enabled = config.get('enabled', False)
        self.depth_power = float(config.get('depth_power', 2.0))
        self.loss_weight = config.get('loss_weight', 1.0)
        self.bypass_compile = config.get('bypass_compile', True)

        # Parse scheduled or static values
        self._safe_schedule = self._parse_schedule(
            config.get('safe_fraction', 0.75))
        self._prob_schedule = self._parse_schedule(
            config.get('truncation_prob', 0.25))

    # ------------------------------------------------------------------
    @staticmethod
    def _parse_schedule(value: Union[float, int, list]) -> List[Tuple[int, float]]:
        """Parse a static value or list of [step, value] waypoints.

        Values are fractions in [0.0, 1.0].
        """
        if isinstance(value, (int, float)):
            return [(0, float(value))]
        return [(int(s), float(v)) for s, v in value]

    @staticmethod
    def _interpolate(schedule: List[Tuple[int, float]], step: int) -> float:
        """Linearly interpolate between waypoints."""
        if len(schedule) == 1:
            return schedule[0][1]
        if step <= schedule[0][0]:
            return schedule[0][1]
        if step >= schedule[-1][0]:
            return schedule[-1][1]
        for i in range(len(schedule) - 1):
            s0, v0 = schedule[i]
            s1, v1 = schedule[i + 1]
            if s0 <= step <= s1:
                frac = (step - s0) / (s1 - s0)
                return v0 + frac * (v1 - v0)
        return schedule[-1][1]

    # ------------------------------------------------------------------
    def get_truncation_point(self, step: int) -> int:
        """
        Return the number of layers to execute this step.

        Returns ``n_layers`` for a full forward pass (no truncation), or a
        value in ``[safe_layer, n_layers - 1]`` for a truncated pass.

        CRITICAL: Uses step-seeded Python stdlib RNG (not torch) so that
        all FSDP ranks make the identical truncation decision every step.
        """
        if not self.enabled:
            return self.n_layers

        safe_fraction = self._interpolate(self._safe_schedule, step)
        truncation_prob = self._interpolate(self._prob_schedule, step)

        safe_layer = int(self.n_layers * safe_fraction)
        zone_size = self.n_layers - safe_layer

        if zone_size <= 0 or truncation_prob <= 0:
            return self.n_layers

        # Deterministic RNG seeded by step — all ranks get same result
        rng = random.Random(step + 42)

        # First decision: truncate at all?
        if rng.random() > truncation_prob:
            return self.n_layers

        # Second decision: where in the zone?
        # depth_power controls bias: 1.0 = uniform, 2.0 = shallow-biased (linear),
        # 3.0 = strongly shallow-biased.  u^(1/power) concentrates near 1.0
        # (shallow cuts, near end of network).
        u = rng.random()
        position = u ** (1.0 / self.depth_power) if self.depth_power != 1.0 else u

        trunc_layer = safe_layer + int(position * zone_size)
        trunc_layer = min(trunc_layer, self.n_layers - 1)

        return trunc_layer

    def get_loss_weight(self, active_layers: int) -> float:
        """Return loss multiplier for this pass."""
        if active_layers == self.n_layers:
            return 1.0
        return self.loss_weight

    # ------------------------------------------------------------------
    @staticmethod
    def fmt_schedule(sched: List[Tuple[int, float]]) -> str:
        """Format a schedule for logging."""
        if len(sched) == 1:
            return f"{sched[0][1]}"
        return " -> ".join(f"{v} @step {s}" for s, v in sched)
