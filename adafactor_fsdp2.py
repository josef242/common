# adafactor_fsdp2.py
"""
FSDP2-compatible Adafactor optimizer with factorized second moments.

Key features:
- Memory efficient: stores row + col factors instead of full second moment matrix
- FSDP2 compatible: properly all-reduces col_var across sharded ranks
- Decoupled weight decay (AdamW-style)
- Supports both auto beta2 scheduling and fixed beta2
- Automatic stochastic rounding for BF16/FP16 weights (essential for small batches)

References:
- Adafactor paper: https://arxiv.org/abs/1804.04235
- Small batch training: https://arxiv.org/abs/2507.07101
- Colossal-AI distributed implementation (for FSDP2 fix inspiration)

Note on stochastic rounding (paper Section A.3):
    When weights are stored in BF16 (~2.4 decimal digits), deterministic rounding
    biases small updates toward zero. This is particularly harmful with small batches
    where updates are smaller. Stochastic rounding removes this bias by rounding
    up/down with probability proportional to the fractional part.
"""

import math
import torch
from torch.optim import Optimizer
from torch.distributed.tensor import DTensor
import torch.distributed as dist

__all__ = ["AdafactorFSDP2"]


class AdafactorFSDP2(Optimizer):
    """
    Adafactor optimizer with FSDP2 support.

    Memory usage per 2D parameter [d_out, d_in]:
    - Adam: 2 * d_out * d_in (exp_avg + exp_avg_sq)
    - Adafactor: d_out + d_in (row_var + col_var)

    For a 4096x4096 matrix: 32M -> 8K floats (~4000x reduction!)

    Args:
        params: Iterable of parameters or param groups
        lr: Learning rate (required, no internal LR scheduling)
        beta2: Second moment decay. If None, uses auto-scheduling: 1 - step^(-0.8).
               For small batch training, use a fixed value like 0.999 (recommended).
               Auto-schedule was designed for large batches.
        eps: (eps1, eps2) tuple for numerical stability
        weight_decay: Decoupled weight decay coefficient
        clip_threshold: Gradient clipping threshold (1.0 = no effect for most updates)
    """

    def __init__(
        self,
        params,
        lr: float = 1e-3,
        beta2: float = None,  # None = auto-schedule, or fixed value like 0.999
        eps: tuple = (1e-30, 1e-3),
        weight_decay: float = 0.0,
        clip_threshold: float = 1.0,
    ):
        if lr <= 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay: {weight_decay}")

        defaults = dict(
            lr=lr,
            beta2=beta2,
            eps=eps,
            weight_decay=weight_decay,
            clip_threshold=clip_threshold,
        )
        super().__init__(params, defaults)

    def _get_beta2(self, group, step):
        """Get beta2 for current step - either fixed or auto-scheduled."""
        if group["beta2"] is not None:
            return group["beta2"]
        # Auto-schedule: 1 - step^(-0.8) from original Adafactor
        return 1.0 - math.pow(step, -0.8)

    def _rms(self, tensor):
        """Compute RMS of tensor, handling DTensor with all-reduce."""
        sum_sq = tensor.pow(2).sum()
        numel = tensor.numel()

        # If DTensor (FSDP2), all-reduce to get global RMS
        if isinstance(tensor, DTensor):
            pg = tensor.device_mesh.get_group()
            world_size = tensor.device_mesh.size()
            # sum_sq is also a DTensor after reduction, use local tensor for all-reduce
            if isinstance(sum_sq, DTensor):
                dist.all_reduce(sum_sq._local_tensor, group=pg)
                sum_sq = sum_sq._local_tensor
            else:
                dist.all_reduce(sum_sq, group=pg)
            numel = numel * world_size

        return (sum_sq / numel).sqrt()

    def _factored_dims(self, shape):
        """
        Determine if parameter should use factored second moment.
        Returns (factored, row_dim, col_dim) or (False, None, None).

        Factored only for 2D+ tensors. For 2D, row=-2, col=-1.
        """
        if len(shape) < 2:
            return False, None, None
        return True, -2, -1

    @torch.no_grad()
    def step(self, closure=None):
        """Perform a single optimization step."""
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            eps1, eps2 = group["eps"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError("AdafactorFSDP2 does not support sparse gradients")

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state["step"] = 0

                    factored, _, _ = self._factored_dims(grad.shape)
                    state["factored"] = factored

                    if factored:
                        # Row variance: reduce along last dim -> shape [..., d_out, 1]
                        # Col variance: reduce along second-to-last dim -> shape [..., 1, d_in]
                        # Use mean operations on grad to get correct DTensor placement
                        state["row_var"] = grad.mean(dim=-1, keepdim=True).mul_(0)  # Zero init, keeps DTensor
                        state["col_var"] = grad.mean(dim=-2, keepdim=True).mul_(0)  # Zero init, keeps DTensor
                    else:
                        # Non-factored: store full second moment (like Adam)
                        state["exp_avg_sq"] = torch.zeros_like(grad)

                state["step"] += 1
                step = state["step"]
                beta2 = self._get_beta2(group, step)

                # Compute squared gradient for second moment update
                grad_sq = grad.pow(2).add_(eps1)

                if state["factored"]:
                    row_var = state["row_var"]
                    col_var = state["col_var"]

                    # Update row variance: mean along last dim (columns)
                    # row_var shape: [..., d_out, 1]
                    row_mean = grad_sq.mean(dim=-1, keepdim=True)
                    row_var.mul_(beta2).add_(row_mean, alpha=1.0 - beta2)

                    # Update col variance: mean along second-to-last dim (rows)
                    # col_var shape: [..., 1, d_in]
                    col_mean = grad_sq.mean(dim=-2, keepdim=True)
                    col_var.mul_(beta2).add_(col_mean, alpha=1.0 - beta2)

                    # =============================================================
                    # FSDP2 FIX: col_var needs all-reduce across sharded ranks
                    # =============================================================
                    # FSDP2 shards on dim-0 (rows), so each rank only sees partial
                    # rows when computing col_mean. We need the global average.
                    if isinstance(col_var, DTensor):
                        pg = col_var.device_mesh.get_group()
                        world_size = col_var.device_mesh.size()
                        # All-reduce the local tensor inside the DTensor
                        dist.all_reduce(col_var._local_tensor, group=pg)
                        col_var._local_tensor.div_(world_size)

                    # Reconstruct variance estimate: outer product of factors
                    # row_var: [..., d_out, 1], col_var: [..., 1, d_in]
                    # Result: [..., d_out, d_in]
                    row_var_mean = row_var.mean(dim=-2, keepdim=True)  # Scalar-ish
                    var_estimate = (row_var / row_var_mean.clamp(min=eps1)) * col_var

                    # Bias correction for fixed beta2 (not needed for auto-schedule)
                    # Without this, early steps underestimate variance -> updates too large
                    # We need to SCALE UP the raw variance to get the true estimate
                    # var_corrected = var_raw / (1 - beta2^step)
                    if group["beta2"] is not None:
                        bias_correction2 = 1.0 - beta2 ** step
                        var_estimate = var_estimate / bias_correction2

                    # Compute update: grad / sqrt(var)
                    update = grad / var_estimate.sqrt().clamp(min=eps2)
                else:
                    # Non-factored path (1D tensors like biases)
                    exp_avg_sq = state["exp_avg_sq"]
                    exp_avg_sq.mul_(beta2).add_(grad_sq, alpha=1.0 - beta2)

                    # Bias correction for fixed beta2
                    if group["beta2"] is not None:
                        bias_correction2 = 1.0 - beta2 ** step
                        exp_avg_sq_corrected = exp_avg_sq / bias_correction2
                        update = grad / exp_avg_sq_corrected.sqrt().clamp(min=eps2)
                    else:
                        update = grad / exp_avg_sq.sqrt().clamp(min=eps2)

                # Gradient clipping by RMS
                update_rms = self._rms(update)
                update.div_(max(1.0, update_rms / group["clip_threshold"]))

                # Apply update with stochastic rounding for low-precision weights
                # BF16 has only ~2.4 decimal digits of precision, so small updates
                # get rounded away with deterministic rounding, biasing training.
                # Stochastic rounding removes this bias (see Adafactor paper A.3).
                if p.dtype in (torch.bfloat16, torch.float16):
                    # Compute update in FP32 to avoid precision loss
                    p_fp32 = p.float()

                    # Decoupled weight decay (AdamW-style)
                    if group["weight_decay"] != 0:
                        p_fp32.mul_(1.0 - group["lr"] * group["weight_decay"])

                    # Apply the update
                    p_fp32.add_(update.float(), alpha=-group["lr"])

                    # Stochastic rounding: add uniform noise scaled to dtype precision
                    # before casting back. This makes rounding probabilistic based on
                    # the fractional part, removing systematic bias.
                    # BF16/FP16 have ~2^-7 relative precision, so noise in [-0.5, 0.5) * ulp
                    noise = torch.empty_like(p_fp32).uniform_(-0.5, 0.5)
                    noise.mul_((2**-7) * p_fp32.abs().clamp_(min=1e-30))
                    p.copy_((p_fp32 + noise).to(p.dtype))
                else:
                    # FP32 path - no stochastic rounding needed
                    if group["weight_decay"] != 0:
                        p.mul_(1.0 - group["lr"] * group["weight_decay"])
                    p.add_(update, alpha=-group["lr"])

        return loss
