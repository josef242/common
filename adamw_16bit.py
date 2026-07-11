# adamw_16bit.py
"""
16-bit AdamW optimizer variants for FSDP2 training.

Stores optimizer states (exp_avg, exp_avg_sq) in half-precision to reduce VRAM
by ~50% vs FP32 while preserving much more fidelity than 8-bit quantization.

Designed for KEEL architecture where thin gradient signals across many layers
are corrupted by 8-bit quantization noise.

Uses stochastic rounding when copying FP32 intermediates back to BF16 state
tensors.  Deterministic round-to-nearest silently drops EMA updates smaller
than BF16's 0.78% precision step, causing the second moment to stall on slow
gradient growth.  A stale denominator makes the effective LR too large,
triggering a positive-feedback loop that ends in gradient explosion.
Stochastic rounding preserves small updates in expectation, eliminating the
stalling failure mode.

Includes:
- AdamW16bit: Direct subclass of torchao's _AdamBase (FSDP2/DTensor compatible)
- AdamC16bit: Wrapper adding AdamC corrected weight decay
"""
import torch
from torch import Tensor
from typing import Optional
from torch.distributed._tensor import DTensor

from torchao.optim.adam import _AdamBase


VALID_STATE_DTYPES = {"mixed", "fp16", "bf16"}


# ---------------------------------------------------------------------------
# Stochastic rounding helpers
# ---------------------------------------------------------------------------

def _fp32_to_bf16_sr(x_f32: Tensor) -> Tensor:
    """Stochastic rounding from FP32 to BF16.

    BF16 truncates the lower 16 mantissa bits of FP32.  Deterministic
    round-to-nearest drops EMA updates smaller than ~0.78% of the current
    value, causing the second moment to stall.

    Stochastic rounding: add uniform noise in [0, 2^16) to the lower 16 bits
    before truncating.  Carry overflow rounds the value up with probability
    equal to its fractional position between adjacent BF16 values.  Works
    correctly for both positive and negative IEEE 754 floats (sign-magnitude
    ordering means adding to lower bits always increases magnitude).
    """
    bits = x_f32.view(torch.int32)
    noise = torch.randint_like(bits, 0, 1 << 16)
    bits = bits + noise
    bits = bits & -65536  # 0xFFFF0000: clear lower 16 bits
    return bits.view(torch.float32).to(torch.bfloat16)


def _single_param_adam_16bit(
    p: Tensor,
    grad: Tensor,
    step: Tensor,
    exp_avg: Tensor,
    exp_avg_sq: Tensor,
    max_exp_avg_sq: Optional[Tensor],
    lr: Tensor,
    beta1: float,
    beta2: float,
    weight_decay: float,
    eps: float,
    IS_ADAMW: bool,
    BF16_STOCHASTIC_ROUND: bool,
):
    """Adam update with stochastic rounding for BF16 optimizer states.

    Mirrors torchao's single_param_adam but applies stochastic rounding
    when copying FP32 intermediates back to BF16 state tensors.  This
    prevents the second moment from stalling on slow gradient growth,
    which causes denominator staleness and eventual gradient explosion
    in deep models.

    FP16 states use deterministic copy (10-bit mantissa has ~0.1% precision,
    10x finer than BF16 — sufficient to track EMA updates).
    """
    # Compute in FP32 for accurate calculations
    p_f32 = p.float()
    grad_f32 = grad.float()

    if IS_ADAMW:
        p_f32 = p_f32 - lr * weight_decay * p_f32
    else:
        grad_f32 = grad_f32 + weight_decay * p_f32

    bias_correction1 = 1 - beta1 ** step
    bias_correction2 = 1 - beta2 ** step

    # Keep high precision copy for param update
    exp_avg_f32 = exp_avg.float().lerp(grad_f32, 1 - beta1)
    exp_avg_sq_f32 = exp_avg_sq.float().lerp(grad_f32.square(), 1 - beta2)

    # Copy states back — stochastic rounding for BF16, deterministic for FP16
    if exp_avg.dtype == torch.bfloat16:
        exp_avg.copy_(_fp32_to_bf16_sr(exp_avg_f32))
    else:
        exp_avg.copy_(exp_avg_f32)

    if exp_avg_sq.dtype == torch.bfloat16:
        exp_avg_sq.copy_(_fp32_to_bf16_sr(exp_avg_sq_f32))
    else:
        exp_avg_sq.copy_(exp_avg_sq_f32)

    if max_exp_avg_sq is not None:
        max_exp_avg_sq_f32 = torch.maximum(max_exp_avg_sq.float(), exp_avg_sq_f32)
        # SR for bf16, like its sibling states above — deterministic rounding
        # of a slow-moving value in bf16 is the EMA-stall mode the module
        # docstring warns about (audit 2026-07-11; unreachable until amsgrad
        # is ever enabled, fixed for completeness).
        if max_exp_avg_sq.dtype == torch.bfloat16:
            max_exp_avg_sq.copy_(_fp32_to_bf16_sr(max_exp_avg_sq_f32))
        else:
            max_exp_avg_sq.copy_(max_exp_avg_sq_f32)
        denom = (max_exp_avg_sq_f32.sqrt() / bias_correction2.sqrt()) + eps
    else:
        denom = (exp_avg_sq_f32.sqrt() / bias_correction2.sqrt()) + eps

    p_f32 = p_f32 - lr * (exp_avg_f32 / bias_correction1) / denom

    if BF16_STOCHASTIC_ROUND:
        p.copy_(_fp32_to_bf16_sr(p_f32))
    else:
        p.copy_(p_f32)


class AdamW16bit(_AdamBase):
    """
    AdamW with 16-bit optimizer states for FSDP2.

    All computation is done in FP32 (same as torchao's quantized optimizers).
    Only the stored state tensors are in half-precision.

    Default "mixed" mode uses FP16 for exp_avg (10-bit mantissa for precise
    gradient direction tracking) and BF16 for exp_avg_sq (FP32-matching
    exponent range to avoid underflow on squared gradients).

    Args:
        params: Parameters to optimize
        lr: Learning rate (default: 1e-3)
        betas: Adam beta coefficients (default: (0.9, 0.999))
        eps: Numerical stability epsilon (default: 1e-8)
        weight_decay: Weight decay coefficient (default: 1e-2)
        amsgrad: Use AMSGrad variant (default: False)
        state_dtype: How to store optimizer states. Options:
            - "mixed" (default): FP16 for exp_avg, BF16 for exp_avg_sq
            - "fp16": Both states in FP16 (caution: exp_avg_sq may underflow)
            - "bf16": Both states in BF16 (safe range, less precision)
        bf16_stochastic_round: Use stochastic rounding for BF16 weight updates
    """

    def __init__(
        self,
        params,
        lr=1e-3,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=1e-2,
        amsgrad=False,
        *,
        state_dtype="mixed",
        bf16_stochastic_round=False,
    ) -> None:
        if state_dtype not in VALID_STATE_DTYPES:
            raise ValueError(
                f"Invalid state_dtype='{state_dtype}'. "
                f"Valid options: {sorted(VALID_STATE_DTYPES)}"
            )
        self.state_dtype = state_dtype

        super().__init__(
            params,
            lr,
            betas,
            eps,
            weight_decay,
            amsgrad,
            block_size=1,  # unused — _new_buffer is overridden
            bf16_stochastic_round=bf16_stochastic_round,
            is_adamw=True,
        )

    def _get_state_dtype(self, signed: bool) -> torch.dtype:
        """Return the storage dtype for a state tensor.

        Args:
            signed: True for exp_avg (1st moment), False for exp_avg_sq (2nd moment)
        """
        if self.state_dtype == "mixed":
            return torch.float16 if signed else torch.bfloat16
        elif self.state_dtype == "fp16":
            return torch.float16
        else:  # "bf16"
            return torch.bfloat16

    def _new_buffer(self, p: Tensor, signed: bool):
        """Create a half-precision state buffer, wrapped in DTensor if needed.

        Replaces _AdamBase._new_buffer entirely. Instead of creating a
        quantized tensor subclass, we create a plain half-precision tensor.
        """
        dtype = self._get_state_dtype(signed)

        # Extract local tensor from DTensor (FSDP2 shards params as DTensors)
        local_p = p.to_local() if isinstance(p, DTensor) else p

        # Create the half-precision state tensor
        out = torch.zeros(local_p.shape, dtype=dtype, device=local_p.device)

        # Re-wrap in DTensor if the param is a DTensor (for FSDP2 compatibility)
        # NOTE: local tensor may have different shapes across ranks when the 1st
        # dim is not divisible by WORLD_SIZE. We must supply global shape/stride
        # for DTensor metadata (not the local shard's layout).
        if isinstance(p, DTensor):
            out = DTensor.from_local(
                local_tensor=out,
                device_mesh=p.device_mesh,
                placements=p.placements,
                run_check=False,
                shape=p.shape,
                stride=p.stride(),
            )

        # Handle CPU offload: DTensor.from_local() may move to device_mesh device,
        # but with CPU offload p.device is cpu. Move back if needed.
        out = out.to(p.device)
        return out

    @staticmethod
    def _subclass_zeros(p: Tensor, signed: bool, block_size: int):
        # Not used — _new_buffer is overridden. Must exist to satisfy ABC.
        raise NotImplementedError("AdamW16bit uses _new_buffer override, not _subclass_zeros")

    def load_state_dict(self, state_dict):
        """Restore state, then UNDO the stock loader's dtype promotion.

        torch.optim.Optimizer.load_state_dict casts every non-'step' float
        state tensor to the PARAM dtype — for fp32 params that silently turns
        the 16-bit exp_avg/exp_avg_sq into fp32 after any resume: optimizer
        state VRAM doubles and the bf16 stochastic-rounding path goes dead,
        with no error or log (audit 2026-07-11). Re-cast to the configured
        storage dtypes; a plain .to() is exact here because the checkpoint
        values were produced in these dtypes in the first place."""
        super().load_state_dict(state_dict)
        for group in self.param_groups:
            for p in group["params"]:
                st = self.state.get(p)
                if not st:
                    continue
                for key, signed in (("exp_avg", True), ("exp_avg_sq", False),
                                    ("max_exp_avg_sq", False)):
                    t = st.get(key)
                    if t is not None and t.is_floating_point():
                        want = self._get_state_dtype(signed)
                        if t.dtype != want:
                            st[key] = t.to(want)

    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step.

        Overrides _AdamBase.step() to use stochastic rounding when
        copying FP32 intermediates back to BF16 state tensors.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue

                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError("Sparse gradient is not supported")

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state["step"] = torch.tensor(0.0)
                    state["exp_avg"] = self._new_buffer(p, True)
                    state["exp_avg_sq"] = self._new_buffer(p, False)
                    if group["amsgrad"]:
                        state["max_exp_avg_sq"] = self._new_buffer(p, False)

                state["step"] += 1

                if not isinstance(group["lr"], Tensor):
                    # torchao's tensor-lr contract exists to avoid torch.compile
                    # recompiles — but this step is deliberately NOT compiled (see
                    # below), so a float lr is mathematically fine. It also arrives
                    # legitimately: DCP resume (_init_optim_state) rewrites group lr
                    # as float 0.0 and runs a state-priming step() BEFORE loading —
                    # the old raise here crashed every adamw_16bit/adamc_16bit
                    # resume. Normalize instead of raising (audit 2026-07-11).
                    group["lr"] = torch.tensor(float(group["lr"]))

                # Call directly (not compiled) — torch.compile + DTensor
                # random ops (randint_like for stochastic rounding) conflict
                # during fake tensor tracing.  Optimizer step is <5% of total
                # step time, so the perf impact is negligible.
                _single_param_adam_16bit(
                    p.detach(),
                    grad,
                    state["step"],
                    state["exp_avg"],
                    state["exp_avg_sq"],
                    state.get("max_exp_avg_sq", None),
                    group["lr"],
                    group["betas"][0],
                    group["betas"][1],
                    group["weight_decay"],
                    group["eps"],
                    self.is_adamw,
                    self.bf16_stochastic_round and p.dtype is torch.bfloat16,
                )

        return loss


class AdamC16bit:
    """
    16-bit AdamC using AdamW16bit as the base optimizer.

    Wraps AdamW16bit and applies AdamC's corrected weight decay
    for normalized layers after the base optimizer step.

    FSDP2/DTensor compatible.

    Args:
        params: iterable of parameters to optimize or dicts defining parameter groups
        lr: learning rate (default: 1e-3)
        betas: coefficients for computing running averages (default: (0.9, 0.999))
        eps: numerical stability epsilon (default: 1e-8)
        weight_decay: weight decay coefficient (default: 0.01)
        state_dtype: How to store optimizer states ("mixed", "fp16", "bf16")
    """

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=0.01, state_dtype="mixed"):
        if not 0.0 <= lr:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= eps:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 0: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 1: {betas[1]}")
        if not 0.0 <= weight_decay:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")

        self.max_lr = lr
        self._weight_decay = weight_decay

        # Process params into param_groups format if needed
        if isinstance(params, dict):
            param_groups = [params]
        else:
            param_groups = list(params)
            if len(param_groups) > 0 and not isinstance(param_groups[0], dict):
                param_groups = [{'params': param_groups}]

        # Store our extra metadata (is_normalized, our weight_decay) per group
        self._group_metadata = []
        base_params = []
        for group in param_groups:
            self._group_metadata.append({
                'is_normalized': group.get('is_normalized', False),
                'weight_decay': group.get('weight_decay', weight_decay),
            })
            # For base optimizer, use weight_decay=0 (we handle it ourselves)
            base_group = {
                'params': list(group['params']),
                'lr': group.get('lr', lr),
                'betas': group.get('betas', betas),
                'eps': group.get('eps', eps),
                'weight_decay': 0,
            }
            base_params.append(base_group)

        # Create the underlying 16-bit optimizer
        self._base_optimizer = AdamW16bit(base_params, state_dtype=state_dtype)

        # Side-dict for per-param WD overrides (keyed by id(param))
        self.wd_overrides = {}

    @property
    def param_groups(self):
        """Return the base optimizer's param_groups (which have tensor LRs)."""
        return self._base_optimizer.param_groups

    @param_groups.setter
    def param_groups(self, value):
        """Set the base optimizer's param_groups."""
        self._base_optimizer.param_groups = value

    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step with AdamC weight decay correction."""
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        # Let the base optimizer do the Adam update (no weight decay)
        self._base_optimizer.step()

        # Now apply weight decay with AdamC correction for normalized layers
        for group, meta in zip(self._base_optimizer.param_groups, self._group_metadata):
            group_wd = meta['weight_decay']
            lr_val = group['lr'].item() if isinstance(group['lr'], torch.Tensor) else group['lr']

            for p in group['params']:
                if p.grad is None:
                    continue

                wd = self.wd_overrides.get(id(p), group_wd)
                if wd != 0:
                    if meta.get('is_normalized', False):
                        # AdamC: weight decay scales with lr²/max_lr
                        effective_decay = wd * (lr_val**2 / self.max_lr)
                        p.mul_(1 - effective_decay)
                    else:
                        # Standard AdamW weight decay
                        p.mul_(1 - lr_val * wd)

        return loss

    def zero_grad(self, set_to_none=True):
        """Clear gradients."""
        self._base_optimizer.zero_grad(set_to_none=set_to_none)

    # Optimizer-shaped state delegation: this wrapper is not a
    # torch.optim.Optimizer subclass, so without these any external tool
    # calling optimizer.state_dict() got AttributeError (the trainer's
    # getattr(_base_optimizer) unwrap still works and stays canonical).
    # AdamW16bit.load_state_dict's 16-bit dtype re-cast rides along free.
    def state_dict(self):
        return self._base_optimizer.state_dict()

    def load_state_dict(self, state_dict):
        return self._base_optimizer.load_state_dict(state_dict)

    @property
    def state(self):
        """Return the base optimizer's state."""
        return self._base_optimizer.state

    @state.setter
    def state(self, value):
        """Set the base optimizer's state."""
        self._base_optimizer.state = value

    @property
    def defaults(self):
        """Return the base optimizer's defaults."""
        return self._base_optimizer.defaults
