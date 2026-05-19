# ruff: noqa
# type: ignore
# fmt: off

# credits to https://gist.github.com/main-horse/7314170780e36f7443d1926418d75823
# MuonSphere implementation based on "Controlled LLM Training on Spectral Sphere" (arXiv:2601.08393)

import math
from typing import Protocol
import torch
from torch.distributed.tensor import DTensor
from torch.distributed import gather, scatter, broadcast
from collections import deque

__version__ = "0.5.0"  # Configurable 16-bit Adam states

VALID_ADAM_STATE_DTYPES = {"fp32", "mixed", "fp16", "bf16"}

__all__ = ["Muon"]


# =============================================================================
# MuonSphere: Spectral Sphere Optimization Helpers
# =============================================================================

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

    # Initialize v randomly and normalize
    v = torch.randn(d_in, device=W.device, dtype=W.dtype)
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
    update = grad.lerp_(momentum, beta) if nesterov else momentum
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

    def __init__(self, param, state, group, index: int, wd_overrides: dict, lr_scale_overrides: dict):
        self.param = param
        self.state = state
        self.group = group
        self.wd_overrides = wd_overrides
        self.lr_scale_overrides = lr_scale_overrides

        self.index = index

        self._intermediate_state = None
    
    def start(self):

        self.param.grad = apply_momentum(self.param.grad, self.state["momentum_buffer"] , self.group["momentum"], self.group["nesterov"])

        grad = self.param.grad
        assert isinstance(grad, DTensor), "only supports DTensor parameters"
        assert grad.device_mesh.ndim == 1, "only supports 1D mesh"

        rank = grad.device_mesh.get_rank()
        world_size = grad.device_mesh.size()
        pg = grad.device_mesh.get_group()

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
        rank = grad.device_mesh.get_rank()
        world_size = grad.device_mesh.size()
        pg = grad.device_mesh.get_group()

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

            # Broadcast scale factor and R to all ranks
            broadcast(scale_tensor, src=dest_rank, group=pg)
            broadcast(R_tensor, src=dest_rank, group=pg)

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
            scatter(grad.to_local(), scatter_list=chunks, src=dest_rank, group=pg, async_op=False)
        else:
            scatter(grad.to_local(), None, src=dest_rank, group=pg, async_op=False)

        update = apply_scaling(grad, self.group["rms_scale"])

        # Apply NorMuon neuron-wise normalization if enabled
        if self.group.get("use_normuon", False):
            update = apply_normuon(update, self.state["second_momentum_buffer"], self.group["beta2"])

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
        # TODO: muon_update() is not defined — this code path (single-device Muon) is broken and unused
        update = muon_update(self.param.grad, self.state["momentum_buffer"], self.group["momentum"], self.group["nesterov"], self.group["ns_steps"], self.group["rms_scale"])

        # =============================================================
        # Weight Decay (standard or cautious) — scaled by lr_scale so a
        # zeroed-out lr_scale freezes the param entirely (no WD decay).
        # =============================================================
        lr_scale = self.lr_scale_overrides.get(id(self.param), 1.0)
        effective_lr = self.group["lr"] * lr_scale
        wd = self.wd_overrides.get(id(self.param), self.group["weight_decay"])
        if wd != 0:
            if self.group.get("cautious_weight_decay", False):
                # Cautious Weight Decay: use momentum buffer BEFORE Newton-Schulz
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
                required_keys = {
                    "params", "lr", "momentum", "weight_decay", "use_muon", "rms_scale",
                    "nesterov", "ns_steps", "use_normuon", "beta2", "cautious_weight_decay",
                    "use_muonsphere", "radius_scale", "power_iters"  # MuonSphere keys
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

        for group in self.param_groups:

            if group["use_muon"]:
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

                    class_work, prefetch_factor = self._get_work_class(p)

                    work = class_work(p, state, group, i, self.wd_overrides, self.lr_scale_overrides)
                    work.start()
                    dq.append(work)

                    if len(dq) > prefetch_factor:
                        dq.popleft().finish()
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
    


    
