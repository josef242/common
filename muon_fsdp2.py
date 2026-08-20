# ruff: noqa
# type: ignore
# fmt: off

# credits to https://gist.github.com/main-horse/7314170780e36f7443d1926418d75823
# MuonSphere implementation based on "Controlled LLM Training on Spectral Sphere" (arXiv:2601.08393)

import math
from typing import Protocol
import torch
from torch.distributed.tensor import DTensor, Shard
from torch.distributed import gather, scatter, broadcast, all_reduce, get_rank
from collections import deque

__version__ = "0.5.0"  # Configurable 16-bit Adam states

VALID_ADAM_STATE_DTYPES = {"fp32", "mixed", "fp16", "bf16"}

__all__ = ["Muon"]


# =============================================================================
# MuonSphere: Spectral Sphere Optimization Helpers
# =============================================================================

_SPHERE_GENERATORS: dict[str, torch.Generator] = {}


def _sphere_generator(device) -> torch.Generator:
    """Per-device generator for power-iteration init, isolated from the global
    RNG stream (seed fixed; not checkpointed — power iteration's convergence
    does not depend on the particular random init)."""
    key = str(device)
    if key not in _SPHERE_GENERATORS:
        g = torch.Generator(device=device)
        g.manual_seed(0x5EED)
        _SPHERE_GENERATORS[key] = g
    return _SPHERE_GENERATORS[key]


def power_iteration(W: torch.Tensor, num_iters: int = 10) -> tuple[float, torch.Tensor, torch.Tensor]:
    """
    Compute spectral norm (largest singular value) and top singular vectors via power iteration.

    Args:
        W: Weight matrix [d_out, d_in]
        num_iters: Number of power iteration steps (10 for init, 3-5 with caching)

    Returns:
        (sigma, u, v) where:
        - sigma: Spectral norm ||W||_2 (largest singular value)
        - u: Left singular vector [d_out]
        - v: Right singular vector [d_in]
    """
    d_out, d_in = W.shape

    # Initialize v randomly and normalize. Dedicated generator, NOT the global
    # RNG: this runs on the per-param dest_rank only, so drawing from the
    # default stream permanently desynchronizes per-rank RNG states (and makes
    # muonsphere runs irreproducible) — audit 2026-07-11.
    v = torch.randn(d_in, device=W.device, dtype=W.dtype,
                    generator=_sphere_generator(W.device))
    v = v / v.norm()

    # Power iteration: alternately compute u = Wv, v = W'u
    for _ in range(num_iters):
        u = W @ v
        u = u / (u.norm() + 1e-12)
        v = W.T @ u
        v = v / (v.norm() + 1e-12)

    # Final u computation for accuracy
    u = W @ v
    sigma = u.norm().item()
    u = u / (sigma + 1e-12)

    return sigma, u, v


def compute_spectral_radius(d_out: int, d_in: int, radius_scale: float = 2.0) -> float:
    """
    Compute target spectral radius R for μP scaling.

    R = c × √(d_out/d_in) where c ≈ 2.0 is optimal per SSO paper.

    This ensures activations stay at Θ(1) scale regardless of layer width.
    """
    return radius_scale * math.sqrt(d_out / d_in)



def _nsloop_eager(X: torch.Tensor, steps: int, *, a=3.4445, b=-4.7750, c=2.0315):
    for _ in range(steps):
        A = X @ X.mT
        B = b * A + c * A @ A
        X = a * X + B @ X
    return X

# Compiled variant — set MUON_NS_COMPILE=0 to disable (avoids slow first-call autotuning)
import os as _os
if _os.environ.get("MUON_NS_COMPILE", "1") != "0":
    nsloop_torch = torch.compile(_nsloop_eager, fullgraph=True)
else:
    nsloop_torch = _nsloop_eager

def zeropower_via_newtonschulz5(G, steps: int):
    """
    Newton-Schulz iteration to compute the zeroth power / orthogonalization of G. We opt to use a
    quintic iteration whose coefficients are selected to maximize the slope at zero. For the purpose
    of minimizing steps, it turns out to be empirically effective to keep increasing the slope at
    zero even beyond the point where the iteration no longer converges all the way to one everywhere
    on the interval. This iteration therefore does not produce UV^T but rather something like US'V^T
    where S' is diagonal with S_{ii}' ~ Uniform(0.5, 1.5), which turns out not to hurt model
    performance at all relative to UV^T, where USV^T = G is the SVD.
    """
    assert G.ndim >= 2 # batched Muon implementation by @scottjmaddox, and put into practice in the record by @YouJiacheng
    a, b, c = (3.4445, -4.7750,  2.0315)
    X = G.bfloat16()
    if G.size(-2) > G.size(-1):
        X = X.mT

    # Ensure spectral norm is at most 1
    X = X / (X.norm(dim=(-2, -1), keepdim=True) + 1e-7)
    # Perform the NS iterations
    X = nsloop_torch(X, steps, a=a, b=b, c=c)
    
    if G.size(-2) > G.size(-1):
        X = X.mT
    return X

def apply_momentum(grad, momentum, beta, nesterov):
    momentum.lerp_(grad, 1 - beta)
    # NEVER return the momentum buffer object itself: callers (Fsdp1dWork) assign the
    # result to param.grad and later scatter/rescale/project INTO that storage in place —
    # returning `momentum` would alias the buffer and silently destroy accumulation
    # (each step's "momentum" becomes last step's scaled NS output). grad is scratch
    # either way, so materialize the nesterov=False update into it.
    update = grad.lerp_(momentum, beta) if nesterov else grad.copy_(momentum)
    if update.ndim == 4: # for the case of conv filters
        update = update.view(len(update), -1)
    return update

def apply_scaling(grad, rms_scale=False ):
    if rms_scale:
        # https://github.com/MoonshotAI/Moonlight/blob/5afcb6911077e7f182d05865fe90d9f39abcbcbd/examples/toy_train.py#L146
        grad *= 0.2 * math.sqrt(max(grad.shape[1], grad.shape[0]))
        return grad
    else:
        # https://github.com/KellerJordan/Muon/blob/f90a42b28e00b8d9d2d05865fe90d9f39abcbcbd/muon.py#L40
        grad *= max(1, grad.size(-2) / grad.size(-1))**0.5
        return grad

def adam_update(grad, buf1, buf2, step, betas, eps):
    buf1.lerp_(grad, 1 - betas[0])
    buf2.lerp_(grad.square(), 1 - betas[1])
    buf1c = buf1 / (1 - betas[0]**step)
    buf2c = buf2 / (1 - betas[1]**step)
    return buf1c / (buf2c.sqrt() + eps)


# ---------------------------------------------------------------------------
# 16-bit Adam state helpers (imported lazily to avoid hard dep on adamw_16bit)
# ---------------------------------------------------------------------------
_adam16bit_imports = None

def _ensure_adam16bit_imports():
    global _adam16bit_imports
    if _adam16bit_imports is None:
        from adamw_16bit import _fp32_to_bf16_sr, _single_param_adam_16bit
        _adam16bit_imports = (_fp32_to_bf16_sr, _single_param_adam_16bit)
    return _adam16bit_imports


# ---------------------------------------------------------------------------
# Head applied-update gauge projection (dn4 head-hygiene; lazy import, no hard
# dep on row_center). Removes the CE-invisible common-mode gauge from the LM
# head's APPLIED Adam update U=m/sqrt(v): U <- U - 1*mean_vocab(U)^T, using the
# GLOBAL vocab-row mean (fp32 + stochastic-rounding write-back for bf16). NOT a
# weight projection and NOT an exp_avg projection. See docs/DN4_HEAD_HYGIENE_SPEC.
# ---------------------------------------------------------------------------
_row_center_imports = None

def _ensure_row_center_imports():
    global _row_center_imports
    if _row_center_imports is None:
        from row_center import _global_row_mean, _subtract_row_mean_
        _row_center_imports = (_global_row_mean, _subtract_row_mean_)
    return _row_center_imports


def _project_head_update_gauge_(update, verify=False):
    """In-place: U <- U - 1*Ubar^T, Ubar = global mean over vocab rows (dim 0).
    Returns (||Ubar|| before, ||Ubar|| after-or-None). 'after' (verify=True)
    recomputes the post-write gauge -> ~0 confirms the SR write-back landed; a
    nonzero 'after' flags a biased bf16 residual. Cheap [D] reductions on the
    single head param."""
    _global_row_mean, _subtract_row_mean_ = _ensure_row_center_imports()
    mu, _ = _global_row_mean(update, vocab_dim=0)
    ubar_pre = mu.norm().item()
    _subtract_row_mean_(update, mu, vocab_dim=0)
    ubar_post = None
    if verify:
        mu_after, _ = _global_row_mean(update, vocab_dim=0)
        ubar_post = mu_after.norm().item()
    return ubar_pre, ubar_post


def _get_adam_state_dtype(state_dtype: str, signed: bool) -> torch.dtype:
    """Return storage dtype for an Adam state tensor.

    Args:
        state_dtype: "fp32", "mixed", "fp16", or "bf16"
        signed: True for exp_avg (1st moment), False for exp_avg_sq (2nd moment)
    """
    if state_dtype == "fp32":
        return torch.float32
    elif state_dtype == "mixed":
        return torch.float16 if signed else torch.bfloat16
    elif state_dtype == "fp16":
        return torch.float16
    else:  # "bf16"
        return torch.bfloat16


def _new_adam_buffer(p: torch.Tensor, signed: bool, state_dtype: str) -> torch.Tensor:
    """Create a (possibly half-precision) Adam state buffer, DTensor-aware."""
    dtype = _get_adam_state_dtype(state_dtype, signed)

    local_p = p.to_local() if isinstance(p, DTensor) else p
    out = torch.zeros(local_p.shape, dtype=dtype, device=local_p.device)

    if isinstance(p, DTensor):
        out = DTensor.from_local(
            local_tensor=out,
            device_mesh=p.device_mesh,
            placements=p.placements,
            run_check=False,
            shape=p.shape,
            stride=p.stride(),
        )
        out = out.to(p.device)

    return out


def apply_normuon(update, second_momentum, beta2):
    """
    NorMuon neuron-wise normalization - applied after Newton-Schulz orthogonalization.
    From https://arxiv.org/abs/2510.05491

    This normalizes each neuron's update by its running second moment, which helps
    stabilize training and can improve convergence.
    """
    vnorm = update.norm(dim=(-2, -1), keepdim=True)
    v_mean = torch.mean(update * update, dim=-1, keepdim=True)
    second_momentum.lerp_(v_mean, 1 - beta2)
    step_size = 1 / second_momentum.sqrt().add_(1e-10)
    update.mul_(step_size)
    vnorm_new = update.norm(dim=(-2, -1), keepdim=True)
    update.mul_(vnorm / (vnorm_new.add_(1e-10)))  # Keep update norm same as pre-normalization
    return update




class Work(Protocol):

    def __init__(self, param, state, group, index: int, wd_overrides: dict, lr_scale_overrides: dict):
        ...

    def start(self):
        ...

    def finish(self):
        ...
    
    
def apply_cautious_weight_decay(param, momentum_buffer, lr, weight_decay):
    """
    Cautious Weight Decay (CWD) - only decay weights where momentum and weight have same sign.

    Reference: "Cautious Weight Decay" (Chen et al., arXiv 2510.12402), Algorithm 1.

    Standard weight decay: param.mul_(1 - lr * wd)
    Cautious weight decay: only decay where (momentum * param) >= 0

    Args:
        param: Parameter tensor
        momentum_buffer: Raw momentum buffer (NOT bias-corrected)
        lr: Current learning rate
        weight_decay: Weight decay coefficient
    """
    # Mask: 1 where momentum and param have same sign, 0 otherwise
    mask = (momentum_buffer * param.data >= 0).float()
    # Apply weight decay only to masked elements: param -= mask * param * lr * wd
    param.data.add_(mask * param.data, alpha=-lr * weight_decay)


class Fsdp1dWork:
    """
    muon handle for fsdp2 1d mesh.
    """

    def __init__(self, param, state, group, index: int, wd_overrides: dict, lr_scale_overrides: dict,
                 allow_uneven: bool = False):
        self.param = param
        self.state = state
        self.group = group
        self.wd_overrides = wd_overrides
        self.lr_scale_overrides = lr_scale_overrides

        self.index = index

        self._intermediate_state = None

        # Fail fast on uneven Shard(0) locals: start() sizes every gather buffer as
        # zeros_like(the DEST rank's own local), so a dim-0 not divisible by world_size
        # (torch.chunk semantics -> unequal shards) size-mismatches the NCCL gather and
        # HANGS silently at the first optimizer step (verified empirically). Turn that
        # into an immediate, explainable launch-time error.
        # allow_uneven=True is passed ONLY by _Fsdp1dBatchedPipeline, whose padded
        # transport handles uneven locals; this work's own start()/finish() path
        # still cannot (and the pipeline never calls them).
        if isinstance(param, DTensor) and param.device_mesh.ndim == 1:
            _ws = param.device_mesh.size()
            if param.shape[0] % _ws != 0 and not allow_uneven:
                raise ValueError(
                    f"Muon FSDP2 gather requires dim-0 divisible by world_size: param shape "
                    f"{tuple(param.shape)} over {_ws} ranks gives uneven Shard(0) locals, which "
                    f"deadlocks the gather in Fsdp1dWork.start (silent NCCL hang). Adjust the "
                    f"model dims or the GPU count, or enable muon_comm_batch (padded transport).")
            # The gather/cat(dim=0)/chunk/scatter round-trip below is ONLY the
            # identity for Shard(0). A Replicate or Shard(1) DTensor passes the
            # divisibility check above, then NS runs on a wrongly-stacked matrix
            # and each rank receives a DIFFERENT slice of it — silent cross-rank
            # parameter desync (audit 2026-07-11). Refuse anything else.
            if tuple(param.placements) != (Shard(0),):
                raise ValueError(
                    f"Muon FSDP2 Fsdp1dWork requires Shard(0) placement; got "
                    f"{tuple(param.placements)} for param shape {tuple(param.shape)}.")
    
    def start(self):

        self.param.grad = apply_momentum(self.param.grad, self.state["momentum_buffer"] , self.group["momentum"], self.group["nesterov"])

        grad = self.param.grad
        assert isinstance(grad, DTensor), "only supports DTensor parameters"
        assert grad.device_mesh.ndim == 1, "only supports 1D mesh"

        world_size = grad.device_mesh.size()
        pg = grad.device_mesh.get_group()
        # GROUP-relative rank throughout: dest_rank = index % world_size is a
        # group-relative id, and every collective below addresses it via
        # group_dst/group_src. The old code compared it against the GLOBAL
        # mesh rank and used global-rank src= kwargs — correct only while the
        # 1D mesh spans the full world starting at rank 0; any sub-world mesh
        # (HSDP shard dim, sub-mesh experiments) would gather to one physical
        # rank and scatter from another (audit 2026-07-11).
        rank = get_rank(pg)

        dest_rank = self.index % world_size

        # Gather gradient to dest_rank (existing behavior)
        if rank == dest_rank:
            gather_lists = [torch.zeros_like(input=grad.to_local()) for _ in range(world_size)]
            gather_handle = gather(grad.to_local(), gather_lists, group_dst=dest_rank, group=pg, async_op=True)

        else:
            gather_lists = None
            gather_handle = gather(grad.to_local(), None, group_dst=dest_rank, group=pg, async_op=True)

        self._intermediate_state = [dest_rank, gather_handle, gather_lists]

        # MuonSphere: Also gather weights for spectral norm computation
        if self.group.get("use_muonsphere", False):
            if rank == dest_rank:
                w_gather_lists = [torch.zeros_like(input=self.param.to_local()) for _ in range(world_size)]
                w_gather_handle = gather(self.param.to_local(), w_gather_lists, group_dst=dest_rank, group=pg, async_op=True)
            else:
                w_gather_lists = None
                w_gather_handle = gather(self.param.to_local(), None, group_dst=dest_rank, group=pg, async_op=True)

            self._intermediate_state.extend([w_gather_handle, w_gather_lists])

    def finish(self):

        assert self._intermediate_state is not None, "gather work must be called first"

        grad = self.param.grad
        world_size = grad.device_mesh.size()
        pg = grad.device_mesh.get_group()
        rank = get_rank(pg)  # group-relative, matching dest_rank (see start())

        dest_rank, gather_handle, gather_lists = self._intermediate_state[:3]
        gather_handle.wait()

        # =========================================================================
        # MuonSphere: Spectral retraction BEFORE Newton-Schulz
        # =========================================================================
        use_muonsphere = self.group.get("use_muonsphere", False)
        R = 1.0  # Default scaling factor (no μP scaling when MuonSphere disabled)

        if use_muonsphere:
            w_gather_handle, w_gather_lists = self._intermediate_state[3:5]
            w_gather_handle.wait()

            # Compute spectral norm and retraction scale on dest_rank
            if rank == dest_rank:
                W_full = torch.cat(w_gather_lists, dim=0)
                d_out, d_in = W_full.shape

                # Compute target spectral radius R = c × √(d_out/d_in)
                radius_scale = self.group.get("radius_scale", 2.0)
                R = compute_spectral_radius(d_out, d_in, radius_scale)

                # Compute current spectral norm via power iteration
                power_iters = self.group.get("power_iters", 10)
                sigma, _, _ = power_iteration(W_full.float(), power_iters)

                # Scale factor to retract to spectral sphere
                scale_factor = R / (sigma + 1e-12)

                # Prepare tensors for broadcast
                scale_tensor = torch.tensor([scale_factor], device=W_full.device, dtype=torch.float32)
                R_tensor = torch.tensor([R], device=W_full.device, dtype=torch.float32)
            else:
                # Non-dest ranks create placeholder tensors for broadcast
                scale_tensor = torch.tensor([0.0], device=self.param.device, dtype=torch.float32)
                R_tensor = torch.tensor([0.0], device=self.param.device, dtype=torch.float32)

            # Broadcast scale factor and R to all ranks (group-relative src)
            broadcast(scale_tensor, group_src=dest_rank, group=pg)
            broadcast(R_tensor, group_src=dest_rank, group=pg)

            scale_factor = scale_tensor.item()
            R = R_tensor.item()

            # Each rank retracts their local weight shard: W ← W × (R/σ)
            self.param.to_local().mul_(scale_factor)

        # =========================================================================
        # Newton-Schulz orthogonalization (existing behavior)
        # =========================================================================
        if rank == dest_rank:
            g_full_block = torch.cat(gather_lists, dim=0)
            g_full_block.copy_(zeropower_via_newtonschulz5(g_full_block, self.group["ns_steps"]))
            g_full_block = g_full_block.type_as(grad)
            chunks = list(g_full_block.chunk(chunks=world_size, dim=0))
            scatter(grad.to_local(), scatter_list=chunks, group_src=dest_rank, group=pg, async_op=False)
        else:
            scatter(grad.to_local(), None, group_src=dest_rank, group=pg, async_op=False)

        self._apply_local_update(R, use_muonsphere)

    def _apply_local_update(self, R=1.0, use_muonsphere=False):
        """Everything AFTER the NS'd update has landed in grad's local storage:
        rms scaling -> NorMuon -> tangent projection -> WD -> apply. Extracted
        verbatim from finish() so the batched transport path (muon_comm_batch)
        can reuse the exact same math on the exact same storage — any change
        here changes BOTH paths, keeping them bit-identical by construction."""
        grad = self.param.grad
        pg = grad.device_mesh.get_group()

        update = apply_scaling(grad, self.group["rms_scale"])

        # Apply NorMuon neuron-wise normalization if enabled
        if self.group.get("use_normuon", False):
            update = apply_normuon(update, self.state["second_momentum_buffer"], self.group["beta2"])

        # =========================================================================
        # TANGENT PROJECTION (opt-in) — strip the radial component of the FINAL update
        # =========================================================================
        # Newton-Schulz's singular-value flattening turns the (radial-null) CE gradient into
        # an update with a small but 100%-consistent ANTI-radial component (cos(update,W)≈-0.013),
        # which descent flips to +radial -> body ‖W‖ grows -> WD-starvation. Project it out:
        #   U ← U − W·⟨U,W⟩/‖W‖²   (GLOBAL coefficient, all-reduced over the FSDP shards).
        # Done AFTER NS+scale+normuon (normuon's per-neuron rescale is not Frobenius-orthogonal,
        # so projecting earlier would let it reintroduce radial — Math Agent). Body matrices only;
        # gate via param group 'tangent_project'. Optional global norm-preserving rescale.
        if self.group.get("tangent_project", False):
            _u_loc = update.to_local() if hasattr(update, "to_local") else update
            _w_loc = self.param.to_local() if hasattr(self.param, "to_local") else self.param
            _uf = _u_loc.reshape(-1).float()
            _wf = _w_loc.reshape(-1).float()
            _stats = torch.stack([(_uf * _wf).sum(), (_wf * _wf).sum()])  # [<U,W>_local, ‖W‖²_local]
            all_reduce(_stats, group=pg)                                  # global sums over shards
            if self.group.get("tangent_project_sync_free", False):
                # Sync-free path (PERF_CAMPAIGN F2, 2026-08-13): the two .item()s
                # below each cost a full pipeline drain per body matrix (trace-
                # attributed ~12s/step fleet-wide). Here the projection coefficient
                # stays a 0-dim GPU tensor and radial_stats holds the RAW global
                # [⟨U,W⟩, ‖W‖²] vector; the train-loop consumer derives (‖W‖, γ)
                # after ONE batched .cpu() over all matrices, and drops wsq<=0
                # entries there (matching the _wsq>0 guard of the legacy path;
                # W≡0 also forces ⟨U,W⟩=0, so _c_t is 0 and the update is a no-op).
                _rs = getattr(self, "radial_stats", None)
                if _rs is not None:
                    _rs[id(self.param)] = _stats
                _strength = self.group.get("tangent_project_strength", 1.0)
                _c_t = (_stats[0] / _stats[1].clamp_min(1e-30)) * _strength
                # Audit hardening (2026-08-19): if _stats ever carries an inf
                # (upstream numeric event), inf*strength(0.0) -> NaN, and a NaN
                # coefficient poisons the weights PERSISTENTLY. Zero is the
                # correct degraded value: "no projection this step".
                _c_t = torch.nan_to_num(_c_t, nan=0.0, posinf=0.0, neginf=0.0)
                if self.group.get("tangent_project_preserve_norm", False):
                    _un0 = torch.stack([(_uf * _uf).sum()])
                    all_reduce(_un0, group=pg)
                    _u_loc.sub_(_w_loc.to(_u_loc.dtype).mul(_c_t))
                    _un1 = torch.stack([(_u_loc.reshape(-1).float() ** 2).sum()])
                    all_reduce(_un1, group=pg)
                    _u_loc.mul_(_un0[0].clamp_min(0).sqrt()
                                / _un1[0].clamp_min(0).sqrt().clamp_min(1e-30))
                else:
                    _u_loc.sub_(_w_loc.to(_u_loc.dtype).mul(_c_t))
                _dot = _wsq = 0.0  # skip the legacy block below — its work is done
            else:
                _dot, _wsq = _stats[0].item(), _stats[1].item()
            if _wsq > 0:
                # Shadow-norm body controller telemetry: ‖W‖ and the measured free radial-growth
                # rate γ = −⟨U,W⟩/‖W‖² (RAW, may be <0 = inward radial; the controller clamps/
                # smooths). Both GLOBAL (all-reduced above) and float (post-.item()), so keyed by
                # id(param) like wd_overrides/lr_scale_overrides. See docs/SHADOW_NORM_PDR_CONTROLLER_SPEC.md.
                _rs = getattr(self, "radial_stats", None)
                if _rs is not None:
                    _rs[id(self.param)] = (_wsq ** 0.5, -_dot / _wsq)
                # Partial-projection strength f in [0,1] (default 1.0): remove only fraction f of
                # the radial component, so ‖W‖ grows at (1-f) of its natural rate. f=1 = full
                # projection (flat ‖W‖, the original behavior); f=0 = no projection (free growth).
                # Updated per-step by the train loop when a schedule is configured. See
                # docs / configs tangent_project_strength.
                _strength = self.group.get("tangent_project_strength", 1.0)
                _c = (_dot / _wsq) * _strength
                if self.group.get("tangent_project_preserve_norm", False):
                    _n0 = _stats_norm = None
                    _un_loc = torch.stack([(_uf * _uf).sum()])
                    all_reduce(_un_loc, group=pg)
                    _norm_before = _un_loc[0].clamp_min(0).sqrt().item()
                # U ← U − c·W  (per-shard, with the global c)
                _u_loc.add_(_w_loc.to(_u_loc.dtype), alpha=-_c)
                if self.group.get("tangent_project_preserve_norm", False) and _norm_before > 0:
                    _un2 = torch.stack([(_u_loc.reshape(-1).float() ** 2).sum()])
                    all_reduce(_un2, group=pg)
                    _norm_after = _un2[0].clamp_min(0).sqrt().item()
                    if _norm_after > 0:
                        _u_loc.mul_(_norm_before / _norm_after)

        # =========================================================================
        # Weight Decay and Update Application
        # =========================================================================
        if use_muonsphere:
            # MuonSphere: NO weight decay (spectral retraction handles regularization)
            # μP-scaled update: W ← W - lr × R × Φ
            lr_scale = self.lr_scale_overrides.get(id(self.param), 1.0)
            self.param.add_(update.reshape(self.param.shape), alpha=-self.group["lr"] * R * lr_scale)
        else:
            # Standard path: apply weight decay then update.
            # effective_lr = lr * lr_scale; WD scales with it so setting
            # lr_scale_overrides[id(p)] = 0 freezes the param entirely
            # (no update AND no WD-driven decay). This invariant is relied
            # on by SCS / lr_mods / output_lr_batch_adjust.
            lr_scale = self.lr_scale_overrides.get(id(self.param), 1.0)
            effective_lr = self.group["lr"] * lr_scale
            wd = self.wd_overrides.get(id(self.param), self.group["weight_decay"])
            if wd != 0:
                if self.group.get("cautious_weight_decay", False):
                    # Cautious Weight Decay: use momentum buffer BEFORE Newton-Schulz
                    # Reference: Chen et al., arXiv 2510.12402
                    apply_cautious_weight_decay(
                        self.param,
                        self.state["momentum_buffer"],
                        effective_lr,
                        wd
                    )
                else:
                    # Standard weight decay
                    self.param.mul_(1 - effective_lr * wd)

            self.param.add_(update.reshape(self.param.shape), alpha=-effective_lr)


class _Fsdp1dBatchedPipeline:
    """Wave-batched transport for the Fsdp1dWork gather->NS->scatter round-trip
    (muon_comm_batch: true). The MATH per matrix is unchanged — same momentum
    call, NS on a byte-identical full-precision buffer, the update landing in
    the same grad-local storage, then the shared Fsdp1dWork._apply_local_update.
    What changes is transport only: instead of one NCCL gather + one scatter
    PER MATRIX (2 x ~600 latency-bound SendRecv rounds of ~3MB shards per step
    — measured 12s/step on rig-30's PCIe ring, 2026-08-04 trace), shards are
    coalesced into ONE gather and ONE scatter per destination rank per wave.

    Uneven Shard(0) locals (world sizes that do not divide dim-0, e.g. the
    (128, embd) KDA head matrix over 6 ranks) are handled by padding each
    matrix's per-rank segment to its max local row count; pad rows travel and
    are sliced off at both ends. This removes the divisibility fatal that
    restricted KDA models to world sizes 8/4/2 (ledger 2026-08-05).

    Not supported: use_muonsphere (per-matrix weight gathers + broadcasts are
    interleaved with transport in ways this pipeline does not reproduce —
    refused loudly below, fall back to muon_comm_batch: false).
    """

    def __init__(self, wave_mb: int = 256):
        self.wave_bytes = int(wave_mb) * 1024 * 1024

    @staticmethod
    def _chunk_sizes(dim0: int, world: int) -> list[int]:
        # torch.chunk semantics (what DTensor Shard(0) uses): ceil-sized chunks,
        # trailing ranks may be short or empty.
        c = -(-dim0 // world)
        return [max(0, min(c, dim0 - r * c)) for r in range(world)]

    def run(self, works: list["Fsdp1dWork"]):
        if not works:
            return
        if works[0].group.get("use_muonsphere", False):
            raise RuntimeError(
                "muon_comm_batch does not support use_muonsphere — set "
                "muon_comm_batch: false for MuonSphere runs.")

        grad0 = works[0].param.grad
        pg = grad0.device_mesh.get_group()
        world = grad0.device_mesh.size()
        rank = get_rank(pg)

        # Momentum for every matrix first (identical to Fsdp1dWork.start()).
        for w in works:
            w.param.grad = apply_momentum(
                w.param.grad, w.state["momentum_buffer"],
                w.group["momentum"], w.group["nesterov"])

        # Waves capped by total send-buffer bytes. REVIEW CORRECTION
        # (2026-08-16 critic pass): simultaneous residency is up to ~4x wave
        # bytes per rank (send + recv + glists-on-dest + scatter-out), not the
        # ~2x this comment originally claimed — at wave_mb=2048 that is ~6-7GB
        # transient during the optimizer phase. Empirically survived 16
        # optimizer steps at the T=4096/B=5 production shape (N1, peak-sampled
        # 22.2GB), but the margin is unmeasured; bracket optimizer.step() with
        # max_memory_allocated before shrinking headroom further.
        waves, wave, wave_bytes = [], [], 0
        for w in works:
            p = w.param
            max_rows = -(-p.shape[0] // world)
            seg_bytes = max_rows * p.shape[1] * p.grad.to_local().element_size()
            if wave and wave_bytes + seg_bytes > self.wave_bytes:
                waves.append(wave)
                wave, wave_bytes = [], 0
            wave.append(w)
            wave_bytes += seg_bytes
        if wave:
            waves.append(wave)

        # F10 (PERF_CAMPAIGN): wave pipelining. Legacy (flag 0/absent) runs
        # each wave to completion — including BLOCKING scatters — before the
        # next packs; the P2-2 trace zoom measured the result: 7.81s/step
        # optimizer phase with ZERO comm/NS overlap. The pipelined schedule
        # keeps the per-wave math and the per-rank collective ORDER identical
        # (pack k -> ns k-1 -> land k-2 is a pure function of the wave plan,
        # derived identically on every rank) while letting wave k+1's gathers
        # ride under wave k's NS and deferring scatter waits one slot.
        if works[0].group.get("muon_comm_pipeline", 0) and len(waves) > 1:
            ctxs: list = [None] * len(waves)
            for k, wv in enumerate(waves):
                ctxs[k] = self._pack(wv, pg, rank, world)
                if k >= 1:
                    self._ns(ctxs[k - 1], pg, rank, world)
                if k >= 2:
                    self._land(ctxs[k - 2], rank, world)
                    ctxs[k - 2] = None  # free buffers
            self._ns(ctxs[-1], pg, rank, world)
            if len(waves) >= 2:
                self._land(ctxs[-2], rank, world)
            self._land(ctxs[-1], rank, world)
        else:
            for wv in waves:
                self._flush(wv, pg, rank, world)

    # ── F10 pipelined stages ──────────────────────────────────────────────
    # _pack/_ns/_land are _flush's exact code split at its stage seams; the
    # ONLY behavioral deltas vs _flush are (a) scatters go async_op=True with
    # handles waited in _land, (b) stages of different waves interleave per
    # run()'s schedule. Per-matrix math is untouched and byte-identical.

    def _pack(self, wave, pg, rank, world):
        if not wave:
            return None
        buckets: dict[int, list] = {}
        for w in wave:
            p = w.param
            dim0, cols = p.shape[0], p.numel() // p.shape[0]
            sizes = self._chunk_sizes(dim0, world)
            max_rows = sizes[0] if sizes else 0
            loc = p.grad.to_local()
            assert loc.shape[0] == sizes[rank], (
                f"Shard(0) local rows {loc.shape[0]} != derived {sizes[rank]} "
                f"for {tuple(p.shape)} over {world} ranks — split-semantics drift")
            d = w.index % world
            buckets.setdefault(d, []).append((w, dim0, cols, sizes, max_rows))

        dests = sorted(buckets)
        send, recv, handles, glists = {}, {}, {}, {}
        for d in dests:
            numel = sum(mr * c for (_, _, c, _, mr) in buckets[d])
            g0 = buckets[d][0][0].param.grad
            buf = torch.empty(numel, dtype=g0.to_local().dtype, device=g0.to_local().device)
            off = 0
            for (w, dim0, cols, sizes, max_rows) in buckets[d]:
                loc = w.param.grad.to_local().reshape(sizes[rank], cols)
                assert loc.dtype == buf.dtype, (
                    f"mixed grad dtypes in one muon bucket ({loc.dtype} vs {buf.dtype}) "
                    f"— copy_ would silently cast and break unbatched parity")
                seg = buf[off:off + max_rows * cols].view(max_rows, cols)
                seg[:sizes[rank]].copy_(loc)
                if sizes[rank] < max_rows:
                    seg[sizes[rank]:].zero_()
                off += max_rows * cols
            send[d] = buf
            recv[d] = torch.empty_like(buf)
            if rank == d:
                glists[d] = [torch.empty_like(buf) for _ in range(world)]
                handles[d] = gather(buf, glists[d], group_dst=d, group=pg, async_op=True)
            else:
                glists[d] = None
                handles[d] = gather(buf, None, group_dst=d, group=pg, async_op=True)
        return {"wave": wave, "buckets": buckets, "dests": dests, "send": send,
                "recv": recv, "handles": handles, "glists": glists}

    def _ns(self, ctx, pg, rank, world):
        if ctx is None:
            return
        buckets, dests = ctx["buckets"], ctx["dests"]
        send, recv, handles, glists = ctx["send"], ctx["recv"], ctx["handles"], ctx["glists"]
        scat = {}
        for d in dests:
            handles[d].wait()
            if rank != d:
                continue
            lists = glists[d]
            out = [torch.empty_like(send[d]) for _ in range(world)]
            off = 0
            for (w, dim0, cols, sizes, max_rows) in buckets[d]:
                full = torch.empty(dim0, cols, dtype=send[d].dtype, device=send[d].device)
                ro = 0
                for r in range(world):
                    if sizes[r] == 0:
                        continue
                    seg = lists[r][off:off + max_rows * cols].view(max_rows, cols)
                    full[ro:ro + sizes[r]].copy_(seg[:sizes[r]])
                    ro += sizes[r]
                full.copy_(zeropower_via_newtonschulz5(full, w.group["ns_steps"]))
                full = full.type_as(w.param.grad)
                ro = 0
                for r in range(world):
                    seg = out[r][off:off + max_rows * cols].view(max_rows, cols)
                    seg[:sizes[r]].copy_(full[ro:ro + sizes[r]])
                    if sizes[r] < max_rows:
                        seg[sizes[r]:].zero_()
                    ro += sizes[r]
                off += max_rows * cols
            scat[d] = out
        # Review hardening (2026-08-16): hold scatter-source buffers in ctx
        # explicitly until _land, rather than relying on c10d recordStream
        # caching-allocator protection for function-local lifetimes.
        ctx["scat"] = scat
        ctx["scat_handles"] = {
            d: scatter(ctx["recv"][d], scatter_list=scat.get(d), group_src=d,
                       group=pg, async_op=True)
            for d in dests
        }

    def _land(self, ctx, rank, world):
        if ctx is None:
            return
        for d in ctx["dests"]:
            ctx["scat_handles"][d].wait()
        for d in ctx["dests"]:
            off = 0
            for (w, dim0, cols, sizes, max_rows) in ctx["buckets"][d]:
                seg = ctx["recv"][d][off:off + max_rows * cols].view(max_rows, cols)
                w.param.grad.to_local().reshape(sizes[rank], cols).copy_(seg[:sizes[rank]])
                off += max_rows * cols
        for w in ctx["wave"]:
            w._apply_local_update()

    def _flush(self, wave, pg, rank, world):
        if not wave:
            return
        # Bucket by destination rank; every rank derives the SAME buckets and
        # offsets (dest = index % world, wave order), so the collective
        # sequence below (gathers then scatters, ascending dest) is identical
        # across ranks — the ordering contract NCCL requires.
        buckets: dict[int, list] = {}
        for w in wave:
            p = w.param
            dim0, cols = p.shape[0], p.numel() // p.shape[0]
            sizes = self._chunk_sizes(dim0, world)
            max_rows = sizes[0] if sizes else 0
            loc = p.grad.to_local()
            # 2D view contract: Fsdp1dWork feeds NS a (dim0, cols) matrix.
            assert loc.shape[0] == sizes[rank], (
                f"Shard(0) local rows {loc.shape[0]} != derived {sizes[rank]} "
                f"for {tuple(p.shape)} over {world} ranks — split-semantics drift")
            d = w.index % world
            buckets.setdefault(d, []).append((w, dim0, cols, sizes, max_rows))

        dests = sorted(buckets)
        send, recv, handles, glists = {}, {}, {}, {}
        for d in dests:
            numel = sum(mr * c for (_, _, c, _, mr) in buckets[d])
            g0 = buckets[d][0][0].param.grad
            buf = torch.empty(numel, dtype=g0.to_local().dtype, device=g0.to_local().device)
            off = 0
            for (w, dim0, cols, sizes, max_rows) in buckets[d]:
                loc = w.param.grad.to_local().reshape(sizes[rank], cols)
                assert loc.dtype == buf.dtype, (
                    f"mixed grad dtypes in one muon bucket ({loc.dtype} vs {buf.dtype}) "
                    f"— copy_ would silently cast and break unbatched parity")
                seg = buf[off:off + max_rows * cols].view(max_rows, cols)
                seg[:sizes[rank]].copy_(loc)
                if sizes[rank] < max_rows:
                    seg[sizes[rank]:].zero_()
                off += max_rows * cols
            send[d] = buf
            recv[d] = torch.empty_like(buf)
            if rank == d:
                glists[d] = [torch.empty_like(buf) for _ in range(world)]
                handles[d] = gather(buf, glists[d], group_dst=d, group=pg, async_op=True)
            else:
                glists[d] = None
                handles[d] = gather(buf, None, group_dst=d, group=pg, async_op=True)

        # Own bucket: reconstruct each full matrix (byte-identical to the
        # unbatched cat of exact shards), NS it, lay the update back out into
        # per-source-rank padded segments.
        scat = {}
        for d in dests:
            handles[d].wait()
            if rank != d:
                continue
            lists = glists[d]
            out = [torch.empty_like(send[d]) for _ in range(world)]
            off = 0
            for (w, dim0, cols, sizes, max_rows) in buckets[d]:
                full = torch.empty(dim0, cols, dtype=send[d].dtype, device=send[d].device)
                ro = 0
                for r in range(world):
                    if sizes[r] == 0:
                        continue
                    seg = lists[r][off:off + max_rows * cols].view(max_rows, cols)
                    full[ro:ro + sizes[r]].copy_(seg[:sizes[r]])
                    ro += sizes[r]
                full.copy_(zeropower_via_newtonschulz5(full, w.group["ns_steps"]))
                full = full.type_as(w.param.grad)
                ro = 0
                for r in range(world):
                    seg = out[r][off:off + max_rows * cols].view(max_rows, cols)
                    seg[:sizes[r]].copy_(full[ro:ro + sizes[r]])
                    if sizes[r] < max_rows:
                        seg[sizes[r]:].zero_()
                    ro += sizes[r]
                off += max_rows * cols
            scat[d] = out

        for d in dests:
            scatter(recv[d], scatter_list=scat.get(d), group_src=d, group=pg, async_op=False)

        # Land updates in grad-local storage (the same destination the
        # unbatched scatter writes), then run the shared local math.
        for d in dests:
            off = 0
            for (w, dim0, cols, sizes, max_rows) in buckets[d]:
                seg = recv[d][off:off + max_rows * cols].view(max_rows, cols)
                w.param.grad.to_local().reshape(sizes[rank], cols).copy_(seg[:sizes[rank]])
                off += max_rows * cols
        for w in wave:
            w._apply_local_update()


class TpFsdp2dWork:
    """
    Muon work for TP + FSDP mesh
    """

    def __init__(self, param, state, group, index: int, wd_overrides: dict, lr_scale_overrides: dict):
        raise NotImplementedError("not implemented")

class EpFsdp2dWork:
    """
    Muon work for EP mesh
    """

    def __init__(self, param, state, group, index: int, wd_overrides: dict, lr_scale_overrides: dict):
        raise NotImplementedError("not implemented")

class TpEpFsdp3dWork:
    """
    Muon work for TP + EP mesh
    """

    def __init__(self, param, state, group, index: int, wd_overrides: dict, lr_scale_overrides: dict):
        raise NotImplementedError("not implemented")

class SingelDeviceWork:
    """
    muon handle for single device.
    """

    def __init__(self, param, state, group, index: int, wd_overrides: dict, lr_scale_overrides: dict):
        self.param = param
        self.state = state
        self.group = group
        self.wd_overrides = wd_overrides
        self.lr_scale_overrides = lr_scale_overrides

    def start(self):
        # Deliberately NOT implemented (audit 2026-07-11). The old body called an
        # undefined muon_update() — and even a working transcription would have
        # silently trained a DIFFERENT optimizer than Fsdp1dWork (no NorMuon, no
        # tangent projection, no MuonSphere, no radial telemetry): the exact
        # single-GPU-vs-8-GPU divergence failure mode this codebase has been
        # bitten by before. Fail loudly instead. A 1-rank FSDP mesh does NOT
        # land here (DTensor -> Fsdp1dWork, whose gather/scatter degenerate
        # cleanly) — so for single-device runs, wrap the model in a 1-rank
        # fully_shard instead of passing plain tensors. If a true plain-tensor
        # path is ever needed, implement it as a thin driver over the SAME
        # helpers Fsdp1dWork.finish uses, and add a parity test against a
        # 1-rank Fsdp1dWork before enabling it.
        raise NotImplementedError(
            "MuonFSDP2: plain-tensor (non-DTensor) Muon params are not supported — "
            "the single-device path was never finished and would diverge from the "
            "FSDP path. Use a 1-rank FSDP2 mesh (fully_shard) for single-device "
            "runs, or move this param to an Adam group."
        )

    def finish(self):
        pass
    
    
class Muon(torch.optim.Optimizer):
    """
    DTensor variant of Muon, original code https://github.com/KellerJordan/Muon/blob/f90a42b28e00b8d9d2d05865fe90d9f39abcbcbd/muon.py
    also support single device variant.
    
    Notable changes:
        - add rms_scale argument to the optimizer following the moonlight paper https://arxiv.org/abs/2502.16982
    
    example usage:
    
    ```python
    
    from muon_fsdp2 import Muon


    optimizer = Muon([
        dict(
            params=model.square_params(),
            lr=1e-3,
            use_muon=True
        ),
        dict(
            params=model.non_square_params(),
            lr=1e-3,
            use_muon=False
        )
    ])   
    ```
    
    
    param_groups args:
        lr: learning rate
        momentum: momentum
        weight_decay: weight decay
        use_muon: whether to use muon
        rms_scale: whether to scale the gradient by the RMS of the gradient . If true use the rms scale from the moonlight paper.
                https://github.com/MoonshotAI/Moonlight/blob/5afcb6911077e7f182d1d7faa3c2cd45acba4666/examples/toy_train.py#L146
                This variant adjust the update so that the RMS match the one of adam, allowing to only have one learning rate for all parameters.

    """
    def __init__(self, param_groups, adam_state_dtype="fp32"):
        if adam_state_dtype not in VALID_ADAM_STATE_DTYPES:
            raise ValueError(
                f"Invalid adam_state_dtype='{adam_state_dtype}'. "
                f"Valid options: {sorted(VALID_ADAM_STATE_DTYPES)}"
            )
        self.adam_state_dtype = adam_state_dtype
        self._use_16bit_adam = adam_state_dtype != "fp32"

        # Eagerly import 16-bit helpers if needed
        if self._use_16bit_adam:
            _ensure_adam16bit_imports()

        for group in param_groups:
            assert "use_muon" in group
            if group["use_muon"]:
                # defaults
                group["lr"] = group.get("lr", 0.02)
                group["momentum"] = group.get("momentum", 0.95)
                group["weight_decay"] = group.get("weight_decay", 0)
                group["rms_scale"] = group.get("rms_scale", True)
                group["nesterov"] = group.get("nesterov", True)
                group["ns_steps"] = group.get("ns_steps", 5)
                group["use_normuon"] = group.get("use_normuon", False)
                group["beta2"] = group.get("beta2", 0.95)
                group["cautious_weight_decay"] = group.get("cautious_weight_decay", False)
                # MuonSphere settings (spectral sphere optimization)
                group["use_muonsphere"] = group.get("use_muonsphere", False)
                group["radius_scale"] = group.get("radius_scale", 2.0)  # c parameter: R = c × √(d_out/d_in)
                group["power_iters"] = group.get("power_iters", 10)     # Power iteration steps for spectral norm
                # Tangent projection: strip the radial component of the final Muon update vs W
                group["tangent_project"] = group.get("tangent_project", False)
                group["tangent_project_preserve_norm"] = group.get("tangent_project_preserve_norm", False)
                # Partial-projection strength f (default 1.0 = full projection). The train loop
                # may overwrite this per-step from a schedule; the default keeps existing configs
                # bit-identical (f=1.0).
                group["tangent_project_strength"] = group.get("tangent_project_strength", 1.0)
                required_keys = {
                    "params", "lr", "momentum", "weight_decay", "use_muon", "rms_scale",
                    "nesterov", "ns_steps", "use_normuon", "beta2", "cautious_weight_decay",
                    "use_muonsphere", "radius_scale", "power_iters",  # MuonSphere keys
                    "tangent_project", "tangent_project_preserve_norm", "tangent_project_strength"
                }
                assert required_keys <= set(group.keys()), f"Muon group missing keys: {required_keys - set(group.keys())}"
            else:
                # defaults
                group["lr"] = group.get("lr", 3e-4)
                group["betas"] = group.get("betas", (0.9, 0.95))
                group["eps"] = group.get("eps", 1e-10)
                group["weight_decay"] = group.get("weight_decay", 0)
                group["cautious_weight_decay"] = group.get("cautious_weight_decay", False)
                required_keys = {"params", "lr", "betas", "eps", "weight_decay", "use_muon", "cautious_weight_decay"}
                assert required_keys <= set(group.keys()), f"Adam group missing keys: {required_keys - set(group.keys())}"
        super().__init__(param_groups, dict())

        # Side-dicts for per-param overrides (keyed by id(param))
        # External code (train_mara.py) assigns shared dicts to these after creation.
        self.wd_overrides = {}
        self.lr_scale_overrides = {}
        # Radial telemetry produced by the tangent-projection block (body matrices only),
        # consumed by the shadow-norm body controller: id(param) -> (‖W‖, γ) where
        # γ = −⟨U,W⟩/‖W‖² is the measured free radial-growth rate per unit LR. Transient
        # (overwritten each step, never checkpointed). EMPTY unless tangent_project is on.
        self.radial_stats = {}
        # Head applied-update gauge projection (dn4). EMPTY by default -> the hook
        # in the Adam path is a no-op membership test for every run that doesn't
        # opt in (train_mara fills this with {id(output.weight)} when enabled).
        self.head_gauge_ids = set()
        self._head_gauge_verify = False          # OFF by default. True (tests/debug) also recomputes
                                                 # the POST-write gauge -> an extra [D] all-reduce per
                                                 # head step, so left off on the production hot path.
        self._last_head_ubar_pre = None          # ||Ubar|| removed from the head update (last head step)
        self._last_head_ubar_post = None         # ||Ubar|| after the write-back (None unless verify)

    def load_state_dict(self, state_dict):
        """Restore state, then UNDO the stock loader's dtype promotion for the
        embedded 16-bit Adam states.

        torch.optim.Optimizer.load_state_dict casts every non-'step' float
        state tensor to the PARAM dtype — with fp32 params that silently turns
        16-bit exp_avg/exp_avg_sq (adam_state_dtype != 'fp32') into fp32 after
        any resume: state VRAM doubles and the stochastic-rounding path goes
        dead, no log (audit 2026-07-11; sibling of the AdamW16bit fix in
        adamw_16bit.py). The re-cast is exact: checkpoint values were produced
        in these dtypes. Muon-path buffers (momentum, second momentum) are
        param-dtype by construction, so the cast is a no-op for them and for
        adam_state_dtype='fp32' this whole override is a no-op."""
        super().load_state_dict(state_dict)
        for group in self.param_groups:
            if group["use_muon"]:
                continue
            for p in group["params"]:
                st = self.state.get(p)
                if not st:
                    continue
                if self._use_16bit_adam:
                    for key, signed in (("exp_avg", True), ("exp_avg_sq", False)):
                        t = st.get(key)
                        if t is not None and t.is_floating_point():
                            want = _get_adam_state_dtype(self.adam_state_dtype, signed)
                            if t.dtype != want:
                                st[key] = t.to(want)
                # Step-counter type differs by path (fp32: int, 16-bit: 0-dim
                # tensor). Coerce on load so an adam_state_dtype change across
                # a resume hands each path the type it expects.
                _step = st.get("step")
                if _step is not None:
                    if self._use_16bit_adam and not isinstance(_step, torch.Tensor):
                        st["step"] = torch.tensor(float(_step))
                    elif not self._use_16bit_adam and isinstance(_step, torch.Tensor):
                        st["step"] = int(_step.item())

    def _get_work_class(self, p: torch.Tensor) -> tuple[type[Work], int]:
        """
        dispatch the work class based on the mesh dimension.
        """
        if isinstance(p, DTensor):
            if p.device_mesh.ndim == 1:
                return Fsdp1dWork, 8
            elif p.device_mesh.ndim == 2:
                return TpFsdp2dWork, 8
            else:
                raise ValueError(f"Unsupported mesh dimension: {p.device_mesh.ndim}")
        else:
            return SingelDeviceWork, 1
        
    @torch.no_grad()
    def step(self, closure=None):

        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        dq: deque[Work] = deque()

        # Clear the radial telemetry each step so a matrix SKIPPED this step (lr_scale==0 freeze
        # short-circuit, or the _wsq==0 guard) genuinely yields NO entry — the shadow controller's
        # accumulator must not re-add a stale prior-step increment as phantom free-growth.
        self.radial_stats.clear()
        # Verify-off steps must not serve a stale post-write residual from an
        # earlier verify-on step (audit 2026-07-11).
        if not self._head_gauge_verify:
            self._last_head_ubar_post = None

        # In-file fence (audit 2026-07-11): the 16-bit Adam path implements
        # NEITHER head-gauge projection NOR cautious weight decay. train_mara
        # fatals these combos at boot, but that safety lives outside this file
        # — consolidate_optimizer (and any future caller) passes both settings
        # straight through. Refuse here so the divergence can't go silent.
        if self._use_16bit_adam:
            if self.head_gauge_ids:
                raise RuntimeError(
                    "MuonFSDP2: head_gauge_projection requires adam_state_dtype='fp32' "
                    "— the 16-bit Adam path has no gauge-projection hook.")
            for _g in self.param_groups:
                if not _g.get("use_muon") and _g.get("cautious_weight_decay", False):
                    raise RuntimeError(
                        "MuonFSDP2: cautious_weight_decay on Adam groups requires "
                        "adam_state_dtype='fp32' — not implemented in the 16-bit path.")

        for group in self.param_groups:

            if group["use_muon"]:
                _batch_works: list[Fsdp1dWork] = []
                for i, p in enumerate(group["params"]):
                    # Frozen via lr_scale=0 (SCS scaffold, lr_mods, etc.):
                    # skip the entire pipeline — no all_gather, no NS, no
                    # momentum/second-moment buffer update, no WD. Saves a
                    # significant chunk of optimizer cost during long
                    # partial-depth phases and prevents spurious decay of
                    # momentum buffers across them.
                    if self.lr_scale_overrides.get(id(p), 1.0) == 0.0:
                        continue
                    if p.grad is None:
                        p.grad = torch.zeros_like(p)  # Force synchronization
                    state = self.state[p]
                    if len(state) == 0:
                        state["momentum_buffer"] = torch.zeros_like(p)
                        if group.get("use_normuon", False):
                            state["second_momentum_buffer"] = torch.zeros_like(p[..., 0:1])
                    elif group.get("use_normuon", False) \
                            and "second_momentum_buffer" not in state:
                        # NorMuon enabled ACROSS a resume: momentum state exists
                        # from the checkpoint but the second-moment buffer does
                        # not — lazily init instead of KeyError'ing in finish()
                        # (audit 2026-07-11).
                        state["second_momentum_buffer"] = torch.zeros_like(p[..., 0:1])

                    class_work, prefetch_factor = self._get_work_class(p)

                    # muon_comm_batch: divert Fsdp1d params to the wave-batched
                    # transport (collected per group, run below). Flag off, or
                    # any non-Fsdp1d work class -> the historical per-param
                    # path, byte-identical to before the flag existed.
                    if group.get("muon_comm_batch", False) and class_work is Fsdp1dWork:
                        work = Fsdp1dWork(p, state, group, i, self.wd_overrides,
                                          self.lr_scale_overrides, allow_uneven=True)
                        work.radial_stats = self.radial_stats
                        _batch_works.append(work)
                        continue

                    work = class_work(p, state, group, i, self.wd_overrides, self.lr_scale_overrides)
                    work.radial_stats = self.radial_stats   # only Fsdp1dWork.finish reads it (via getattr)
                    work.start()
                    dq.append(work)

                    if len(dq) > prefetch_factor:
                        dq.popleft().finish()

                if _batch_works:
                    # Drain any per-param works first so the batched pipeline's
                    # collective sequence starts from the same point on every
                    # rank (mixed batched/unbatched groups are unusual but legal
                    # — e.g. a group with both Fsdp1d and single-device params).
                    while dq:
                        dq.popleft().finish()
                    _Fsdp1dBatchedPipeline(
                        wave_mb=group.get("muon_comm_batch_wave_mb", 256)
                    ).run(_batch_works)
            else:
                for p in group["params"]:
                    # Same freeze short-circuit for the Adam path. With
                    # effective_lr = lr * lr_scale = 0, both update and WD
                    # would be no-ops; skipping avoids the exp_avg /
                    # exp_avg_sq state updates as well.
                    if self.lr_scale_overrides.get(id(p), 1.0) == 0.0:
                        continue
                    if p.grad is None:
                        p.grad = torch.zeros_like(p)  # Force synchronization
                    state = self.state[p]

                    if self._use_16bit_adam:
                        # ---------------------------------------------------------
                        # 16-bit Adam path: half-precision states + stochastic rounding
                        # ---------------------------------------------------------
                        if len(state) == 0:
                            state["exp_avg"] = _new_adam_buffer(p, True, self.adam_state_dtype)
                            state["exp_avg_sq"] = _new_adam_buffer(p, False, self.adam_state_dtype)
                            state["step"] = torch.tensor(0.0)
                        state["step"] += 1

                        # Resolve per-param weight decay
                        wd = self.wd_overrides.get(id(p), group["weight_decay"])
                        lr_scale = self.lr_scale_overrides.get(id(p), 1.0)

                        # Build effective LR (scheduled_lr * lr_scale)
                        lr_val = group["lr"]
                        if not isinstance(lr_val, torch.Tensor):
                            lr_val = torch.tensor(lr_val, device=p.device)
                        effective_lr = lr_val * lr_scale

                        _, _single_param_adam_16bit = _adam16bit_imports
                        _single_param_adam_16bit(
                            p.detach(),
                            p.grad,
                            state["step"],
                            state["exp_avg"],
                            state["exp_avg_sq"],
                            None,  # no amsgrad
                            effective_lr,
                            group["betas"][0],
                            group["betas"][1],
                            wd,
                            group["eps"],
                            True,  # IS_ADAMW
                            p.dtype is torch.bfloat16,  # BF16_STOCHASTIC_ROUND
                        )
                    else:
                        # ---------------------------------------------------------
                        # FP32 Adam path (original)
                        # ---------------------------------------------------------
                        if len(state) == 0:
                            state["exp_avg"] = torch.zeros_like(p)
                            state["exp_avg_sq"] = torch.zeros_like(p)
                            state["step"] = 0
                        state["step"] += 1
                        update = adam_update(p.grad, state["exp_avg"], state["exp_avg_sq"],
                                             state["step"], group["betas"], group["eps"])

                        # dn4 head-hygiene: project the CE-invisible common-mode gauge out of
                        # the head's APPLIED update (vocab-row mean), BEFORE WD + the weight
                        # step. No-op unless this is the head param -- head_gauge_ids is empty
                        # for every run that doesn't opt in. Projecting U (not exp_avg, not the
                        # post-step weight) is the gauge-safe lever; see DN4_HEAD_HYGIENE_SPEC.
                        if id(p) in self.head_gauge_ids:
                            _pre, _post = _project_head_update_gauge_(
                                update, verify=self._head_gauge_verify)
                            self._last_head_ubar_pre = _pre
                            if _post is not None:
                                self._last_head_ubar_post = _post

                        # Weight Decay scaled by lr_scale so a zeroed-out
                        # lr_scale (SCS freeze, lr_mods, etc.) freezes the
                        # param entirely with no silent WD-driven decay.
                        lr_scale = self.lr_scale_overrides.get(id(p), 1.0)
                        effective_lr = group["lr"] * lr_scale
                        wd = self.wd_overrides.get(id(p), group["weight_decay"])
                        if wd != 0:
                            if group.get("cautious_weight_decay", False):
                                apply_cautious_weight_decay(
                                    p,
                                    state["exp_avg"],
                                    effective_lr,
                                    wd
                                )
                            else:
                                p.mul_(1 - effective_lr * wd)

                        p.add_(update, alpha=-effective_lr)

        for work in dq:
            work.finish()

        return loss
    


    
