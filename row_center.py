"""Output-head row-centering (gauge subtraction).

Removes the CE-invisible common-mode "gauge" from the LM readout head by
subtracting the vocab-row mean mu from every row of the head weight:

    mu = (1/V) * sum_i W_i           # W in R^[V, D]
    W <- W - 1 mu^T

This shifts every vocab logit for a token by the SAME scalar (h . mu):

    z_i' = h^T (W_i - mu) = z_i - h^T mu

so softmax probabilities, CE loss, sampling, and top-k/top-p ordering are
mathematically unchanged (allclose up to fp roundoff / tie-breaks, NOT bit
identity). This is a GAUGE CHOICE, not a regularizer and not a head-norm
brake -- it preserves the next-token distribution. It is NOT centered z-loss.

The probe finding that motivates it: the raw logZ of KEEL heads is ~77-81%
a common-mode offset (u1 . ones ~ 0.93 on every checkpoint), while the real
centered margin (logZ_c ~ 99-110) is healthy. Centering strips the inert 80%
and leaves the real structure W_c untouched.

OPERATIONAL CORRECTNESS (the only ways to get this wrong):
  * GLOBAL mean across vocab shards, not per-shard -- per-shard means subtract
    DIFFERENT offsets from different vocab regions and DO change probabilities.
  * Project the Adam first moment with ITS OWN row-mean (different tensor,
    different units) -- pure CE has zero common-mode gradient, but Adam's
    elementwise preconditioning manufactures a nonzero-row-mean update, so the
    gauge regrows unless momentum is stripped too.
  * Do NOT center the second moment (it's a positive variance accumulator, not
    a signed gauge). The residual common-mode it injects via m/sqrt(v) is why
    the per-step projection of W remains necessary as the backstop.
  * Compute in fp32; write back with stochastic rounding for bf16 buffers, or
    the gauge persists in low-precision optimizer state.

Assumes the head is UNTIED and has NO bias (the dn2 clean case). Tied
embeddings make direct head modification non-function-preserving (it moves the
input embeddings too); an output bias has its own gauge (the bias mean) that
must be handled separately. Callers must guarantee these assumptions.
"""

import torch
import torch.distributed as dist

try:
    from torch.distributed.tensor import DTensor
except Exception:  # older torch layout
    try:
        from torch.distributed._tensor import DTensor
    except Exception:
        DTensor = ()  # isinstance(x, ()) is always False -> treat all as plain


def _fp32_to_bf16_sr(x_f32):
    """Stochastic rounding fp32 -> bf16 (self-contained copy of the kernel in
    adamw_16bit.py — duplicated deliberately so row-centering has NO dependency
    on the 16-bit optimizer module, which pulls in torchao at import time even
    for fp32 runs). Add uniform noise to the low 16 mantissa bits before
    truncating: carry-overflow rounds up with probability equal to the value's
    fractional position between adjacent bf16 values. Unbiased."""
    bits = x_f32.view(torch.int32)
    noise = torch.randint_like(bits, 0, 1 << 16)
    bits = bits + noise
    bits = bits & -65536  # 0xFFFF0000: clear lower 16 bits
    return bits.view(torch.float32).to(torch.bfloat16)


def _global_row_mean(weight, vocab_dim=0):
    """Global row-mean of a [V, D] tensor (vector in R^D), correct under FSDP2
    vocab-sharding. Computed in fp32. Returns (mu_fp32 [D], V_global).

    For a DTensor sharded on the vocab axis, each rank holds only V_local rows;
    the global mean needs sum-of-rows and global V all-reduced over the head's
    mesh. We reduce the local row-SUM (and local row-COUNT) rather than local
    means so unequal shard sizes are handled exactly.
    """
    is_dt = isinstance(weight, DTensor)
    local = weight._local_tensor if is_dt else weight
    if vocab_dim != 0:
        local = local.transpose(0, vocab_dim)
    local_f32 = local.float()
    # sum over the local vocab rows -> [D]; count = local row count
    row_sum = local_f32.sum(dim=0)
    row_cnt = torch.tensor(
        float(local_f32.shape[0]), device=local_f32.device, dtype=torch.float32
    )
    if is_dt and dist.is_available() and dist.is_initialized():
        # Reduce over the DTensor's device mesh (the dp mesh the head lives on).
        # Use the mesh's process group so we don't accidentally reduce over a
        # wider/narrower world than the head is sharded across.
        pg = weight.device_mesh.get_group()
        dist.all_reduce(row_sum, op=dist.ReduceOp.SUM, group=pg)
        dist.all_reduce(row_cnt, op=dist.ReduceOp.SUM, group=pg)
    mu = row_sum / row_cnt.clamp_min(1.0)
    return mu, int(row_cnt.item())


def _subtract_row_mean_(tensor, mu, vocab_dim=0):
    """In-place subtract mu [D] from every vocab row of `tensor` ([V, D]),
    matching the buffer's dtype with stochastic rounding for bf16. Operates on
    the LOCAL shard of a DTensor (mu is global, so each shard subtracts the
    same offset -> a uniform shift, the whole point). no_grad caller."""
    is_dt = isinstance(tensor, DTensor)
    local = tensor._local_tensor if is_dt else tensor
    view = local if vocab_dim == 0 else local.transpose(0, vocab_dim)
    centered = view.float() - mu.to(view.device, torch.float32).unsqueeze(0)
    if view.dtype is torch.bfloat16:
        view.copy_(_fp32_to_bf16_sr(centered))
    else:
        view.copy_(centered.to(view.dtype))


def row_norm_of_mean(weight, vocab_dim=0):
    """||mu(W)|| -- the magnitude of the current gauge. Telemetry diagnostic;
    pre-projection this is the per-step gauge regrowth rate."""
    mu, _ = _global_row_mean(weight, vocab_dim)
    return mu.norm().item()


@torch.no_grad()
def row_center_head_(weight, exp_avg=None, vocab_dim=0):
    """Project the gauge out of the LM head in place, and (if given) out of the
    Adam first moment using ITS OWN row-mean.

    Returns a telemetry dict:
      mu_w_pre   : ||mu(W)|| before projecting W (gauge to remove this step)
      mu_w_post  : ||mu(W)|| after  (should be ~0 by construction)
      m_bar      : ||m_bar(exp_avg)|| before projecting the first moment
      proj_fro   : ||1 mu^T||_F = sqrt(V) * ||mu|| (Frobenius norm of the shift)
      proj_ratio : proj_fro / ||W||_F  (relative size of the gauge in the head)
    """
    mu_w, V = _global_row_mean(weight, vocab_dim)
    mu_w_pre = mu_w.norm().item()

    # ||W||_F (global, fp32) for the projection-ratio telemetry, BEFORE we
    # mutate W. Reuse the local-sq -> all_reduce -> sqrt idiom.
    is_dt = isinstance(weight, DTensor)
    w_local = (weight._local_tensor if is_dt else weight).float()
    w_sq = (w_local * w_local).sum()
    if is_dt and dist.is_available() and dist.is_initialized():
        dist.all_reduce(w_sq, op=dist.ReduceOp.SUM, group=weight.device_mesh.get_group())
    w_fro = w_sq.clamp_min(0).sqrt().item()

    _subtract_row_mean_(weight, mu_w, vocab_dim)

    # Post-projection gauge (recompute -> should be ~0). Cheap; confirms the
    # projection actually took on this buffer/layout.
    mu_w_after, _ = _global_row_mean(weight, vocab_dim)
    mu_w_post = mu_w_after.norm().item()

    m_bar = None
    if exp_avg is not None:
        mu_m, _ = _global_row_mean(exp_avg, vocab_dim)  # OWN row-mean
        m_bar = mu_m.norm().item()
        _subtract_row_mean_(exp_avg, mu_m, vocab_dim)

    proj_fro = (V ** 0.5) * mu_w_pre
    return {
        "mu_w_pre": mu_w_pre,
        "mu_w_post": mu_w_post,
        "m_bar": m_bar,
        "proj_fro": proj_fro,
        "proj_ratio": (proj_fro / w_fro) if w_fro > 0 else 0.0,
    }
