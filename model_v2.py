# model_v2.py
"""
Dense Transformer with KV Caching - Training/Inference Path Isolation

Key Design Principles:
1. Training path is IDENTICAL to the original model_v1.py - zero overhead
2. KV caches are ONLY allocated when setup_caches() is explicitly called
3. Separate methods for training (forward) vs inference (generate_forward)
4. Activation checkpointing only applies to training path
5. Compatible with neo_common.py interface (setup_caches, clear_caches)

Usage:
    # Training - exactly like before, caches never allocated
    logits, loss = model(tokens, targets=targets)
    
    # Inference with KV caching
    model.setup_caches(max_batch_size=1, max_seq_len=2048)
    logits = model.generate_forward(tokens, start_pos=0)  # prefill
    logits = model.generate_forward(next_token, start_pos=seq_len)  # decode
    model.clear_caches()
"""

import os
import math
import inspect
from typing import List, Optional, Tuple
from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as cp
import torch._dynamo
import time
from contextlib import nullcontext, contextmanager

# MARA_OFFLOAD_DEBUG=1: log any pinned-buffer acquisition slower than 0.5s —
# discriminates offload-side host stalls from external ones (NAS shard
# fetches etc.) when hunting intermittent multi-second steps.
_OFFLOAD_DEBUG = os.environ.get("MARA_OFFLOAD_DEBUG") == "1"


class _ActOffloader:
    """Saved-tensor offload to pinned host RAM (rig-30 perf campaign 2026-08-05).

    One mechanism, two pools. Wrapped around an AC'd block, autograd only
    saves the checkpoint's boundary inputs -> those are what get offloaded
    (Pool A: frees ~B*T*embd per layer of VRAM for ~9ms of PCIe). Wrapped
    around a NON-checkpointed block, autograd saves every intermediate ->
    all of them ride to host and that block's backward recompute vanishes
    (Pool B: ~110ms/layer round trip). Which blocks get which is decided by
    Transformer.__init__ from ac_skip_layers/ac_input_offload/ac_offload.

    Mechanics: D2H copies run on a dedicated per-device stream that first
    waits on the compute stream; the source tensor is record_stream()'d so
    the caching allocator defers reuse until the copy lands. Backward
    demand-fetches: unpack synchronizes the pack event (a no-op that late),
    H2D's on the copy stream, and makes the compute stream wait on the copy
    event. Non-CUDA tensors and anything under min_bytes pass through.
    """

    def __init__(self, min_bytes: int):
        self.min_bytes = int(min_bytes)
        self._streams: dict = {}
        # Persistent pinned-buffer pool keyed by (shape, dtype). v1 allocated
        # fresh pinned buffers per pack and freed them per backward — the
        # resulting cudaHostAlloc/cudaFreeHost storms (FreeHost implicitly
        # device-syncs) produced periodic ~150s stalls every few steps
        # (Josef spotted the period, 2026-08-05). Buffers now live for the
        # run: steady state does ZERO host allocation. Reuse safety comes
        # from copy-stream ordering — a buffer returns to the pool at unpack
        # after its H2D is ISSUED on the copy stream, and any later D2H
        # reusing it is issued on that same stream, so the hardware serializes
        # them; no host-side wait needed.
        self._pool: dict = {}

    def _stream(self, dev):
        s = self._streams.get(dev)
        if s is None:
            s = torch.cuda.Stream(device=dev)
            self._streams[dev] = s
        return s

    def _get_buf(self, t):
        key = (tuple(t.shape), t.dtype)
        lst = self._pool.get(key)
        if lst:
            return key, lst.pop()
        return key, torch.empty_like(t, device="cpu", pin_memory=True)

    @torch.compiler.disable
    def _pack(self, t):
        if (not t.is_cuda) or t.numel() * t.element_size() < self.min_bytes:
            return t
        _t0 = time.perf_counter() if _OFFLOAD_DEBUG else 0.0
        key, cpu = self._get_buf(t)
        if _OFFLOAD_DEBUG:
            _dt = time.perf_counter() - _t0
            if _dt > 0.5:
                print(f"[act-offload][DEBUG] slow pinned get_buf: {_dt:.2f}s for "
                      f"{key[0]} pool_sizes={{k: len(v) for k, v in self._pool.items()}}",
                      flush=True)
        s = self._stream(t.device)
        s.wait_stream(torch.cuda.current_stream(t.device))
        with torch.cuda.stream(s):
            cpu.copy_(t, non_blocking=True)
            ev = torch.cuda.Event()
            ev.record(s)
        t.record_stream(s)
        return (key, cpu, t.device, ev)

    @torch.compiler.disable
    def _unpack(self, packed):
        if isinstance(packed, torch.Tensor):
            return packed
        key, cpu, dev, ev = packed
        ev.synchronize()  # D2H landed (fwd-era event; effectively free here)
        s = self._stream(dev)
        with torch.cuda.stream(s):
            gpu = cpu.to(dev, non_blocking=True)
            ev2 = torch.cuda.Event()
            ev2.record(s)
        cur = torch.cuda.current_stream(dev)
        cur.wait_event(ev2)
        gpu.record_stream(cur)
        self._pool.setdefault(key, []).append(cpu)  # stream-ordered reuse
        return gpu

    @contextmanager
    def hooks(self):
        with torch.autograd.graph.saved_tensors_hooks(self._pack, self._unpack):
            yield


def _in_backward_recompute() -> bool:
    """True inside an activation-checkpoint RECOMPUTE pass.

    Non-reentrant AC (use_reentrant=False, what this file uses) re-executes the
    checkpointed forward INSIDE loss.backward(), with grad enabled — so neither
    self.training nor torch.is_grad_enabled() can tell the passes apart. What
    can: the recompute runs within the autograd engine's backward graph task,
    where _current_graph_task_id() returns a valid id; every ordinary forward
    (stored-forward AC pass, plain training, eval) returns -1. Private API, but
    torch 2.9 exposes no public equivalent (verified empirically 2026-07-13;
    pinned by tests/test_moe.py). Used to run telemetry side effects exactly
    once per step — the forward MATH must stay identical in both passes."""
    return torch._C._current_graph_task_id() != -1

# ----------------------------------------------------------------------------
# Cross Entropy Helper (memory-efficient CCE)
# ----------------------------------------------------------------------------
try:
    if os.name != "nt":
        from cut_cross_entropy import linear_cross_entropy
        _use_lce = True
    else:
        raise ImportError("CCE disabled on Windows")
except Exception:
    _use_lce = False

    def linear_cross_entropy(hidden: torch.Tensor,
                             weight: torch.Tensor,
                             targets: torch.Tensor,
                             accum_e_fp32: bool = False,
                             accum_c_fp32: bool = False,
                             reduction: str = "mean",
                             **kw):
        logits = hidden @ weight.t()
        ignore_index = kw.get("ignore_index", -100)
        return F.cross_entropy(logits, targets, reduction=reduction, ignore_index=ignore_index)


# ── F9 op-level SAC (2026-08-17) ─────────────────────────────────────────────
# Selective activation checkpointing policy for the block checkpoint. The
# 'flex_save' policy MUST_SAVEs flex/scaled-dot-product attention outputs
# (the quadratic recompute, ~105MB/attn-layer at production shape) and
# PREFER_RECOMPUTEs everything else. MARA_SAC_DEBUG=1 prints each distinct op
# the policy sees ONCE (rank-local) — the reachability probe for whether the
# per-submodule-compile structure exposes the target ops to the dispatch mode.
_SAC_SEEN_OPS: set = set()


def _make_sac_context_fn(policy_name: str):
    from functools import partial
    from torch.utils.checkpoint import (
        create_selective_checkpoint_contexts, CheckpointPolicy)

    _debug = os.environ.get('MARA_SAC_DEBUG') == '1'

    def _policy_fn(ctx, op, *args, **kwargs):
        name = str(getattr(op, '__name__', op))
        if _debug and name not in _SAC_SEEN_OPS:
            _SAC_SEEN_OPS.add(name)
            print(f"[SAC-DEBUG] op: {name}", flush=True)
        if policy_name == 'flex_save':
            if ('flex_attention' in name
                    or 'scaled_dot_product' in name):
                return CheckpointPolicy.MUST_SAVE
            return CheckpointPolicy.PREFER_RECOMPUTE
        return CheckpointPolicy.PREFER_RECOMPUTE

    return partial(create_selective_checkpoint_contexts, _policy_fn)


# PERF_CAMPAIGN F1 (2026-08-13): precomputed CCE valids. The library's
# _build_flat_valids calls aten::nonzero on the GPU targets; its data-dependent
# output shape forces a full pipeline drain (~2.9s/call, 6 calls/step,
# trace-attributed). The valids depend ONLY on targets+ignore_index, so the
# train loop builds them on the CPU batch (cpu nonzero = no GPU sync) and
# ships them async. `_CCE_VALIDS_ABSENT` distinguishes "not provided" from a
# legitimate None (= all tokens valid).
_CCE_VALIDS_ABSENT = object()


def cce_cpu_valids(targets_cpu: torch.Tensor, ignore_index: int):
    """CPU twin of cut_cross_entropy.utils._build_flat_valids for shift=0.
    Returns int32 flat indices of valid targets, or None if all are valid."""
    t = targets_cpu.reshape(-1)
    v = (t != ignore_index).nonzero().to(torch.int32)
    return v.squeeze(1) if v.numel() != t.numel() else None


def _cce_loss_with_valids(hidden, weight, targets, valids, *,
                          accum_e_fp32=False, accum_c_fp32=False,
                          reduction="mean"):
    """Vendored tail of cut_cross_entropy.cce.cce_linear_cross_entropy
    (25.4.3) minus the _build_flat_valids call — byte-identical math, valids
    supplied by the caller. shift=0 / bias=None / softcap=None only (the only
    forms our hot path uses)."""
    from cut_cross_entropy.cce import CCEParams, linear_cross_entropy_apply
    from cut_cross_entropy.utils import _handle_eps
    assert hidden.size()[0:-1] == targets.size()
    e = hidden.contiguous()
    t = targets.contiguous()
    batch_shape = t.size()
    e = e.flatten(0, -2)
    t = t.flatten()
    if (t.data_ptr() % 16) != 0:
        t = torch.nn.functional.pad(t, (0, 1))[:-1]
    assert (t.data_ptr() % 16) == 0
    params = CCEParams(
        t, valids, None, reduction, _handle_eps("auto", e.dtype), 0,
        batch_shape, accum_e_fp32, accum_c_fp32,
        filter_e_grad=True, filter_c_grad=True, vocab_parallel_options=None,
    )
    return linear_cross_entropy_apply(e, weight, None, params)


@torch._dynamo.disable
def cce_loss(hidden, weight, targets, valids=_CCE_VALIDS_ABSENT, **kwargs):
    """Thin wrapper so CCE executes in eager mode.

    PROBE HOOK (env WD_CCE_IMPL): override the CCE implementation. The default fused
    'cce' (impl=1) Triton kernel REQUIRES bf16/fp16 weights — so a full fp32-param
    forward (the decisive sharded-path test) is impossible with it. Setting
    WD_CCE_IMPL=torch_compile selects the pure-PyTorch impl which accepts fp32,
    enabling the fp32-everywhere run. Unset => byte-identical to baseline. One env
    read per call is negligible at the head. (Josef's unblock for the fp32 test.)"""
    _impl = os.environ.get('WD_CCE_IMPL')
    if _impl and 'impl' not in kwargs:
        kwargs = dict(kwargs, impl=_impl)
    if valids is not _CCE_VALIDS_ABSENT and _use_lce and 'impl' not in kwargs:
        # F1 fast path: caller-supplied valids skip the library's GPU nonzero.
        # CONTRACT (review note 2026-08-16): the fast path DROPS ignore_index —
        # the caller's valids MUST have been built against the same ignore id
        # this call site would pass (today: params.pad_id on both ends). A
        # future call site with a different ignore_index must NOT pass valids.
        return _cce_loss_with_valids(
            hidden, weight, targets, valids,
            accum_e_fp32=kwargs.get('accum_e_fp32', False),
            accum_c_fp32=kwargs.get('accum_c_fp32', False),
            reduction=kwargs.get('reduction', 'mean'),
        )
    return linear_cross_entropy(hidden, weight, targets, **kwargs)


def _zloss_optionD(h_flat, weight, tgt_flat, pad_id, fp32_accum):
    """Differentiable z-loss statistics with NO [N,V] logits materialization.

    Returns (zloss, logz) = (mean(logZ**2), mean(logZ)) over non-pad tokens,
    where logZ is the per-token logsumexp of the (never-materialized) logits.

    The installed CCE (25.4.3) does NOT expose `return_lse`, so we reconstruct
    logZ from the identity  CE_per_token = logZ - logit_target  =>
        logZ = CE_none + logit_target
    using two pieces that each avoid the [N,V] logits tensor:
      - CE_none = linear_cross_entropy(reduction='none')  — CCE fused, differentiable.
      - logit_target = (h . W[target]) = (h * W[target_rows]).sum(-1). The gather
        W[targets] is [N, D] (same footprint as h), NOT [N, V].

    Precision (`fp32_accum`, set by the z_loss backend):
      - The reconstruction is a catastrophic cancellation: CE_none = logZ -
        logit_target with logZ, logit_target both ~O(8) and CE ~O(small). In
        bf16 the lost low-order bits make the z-loss GRADIENT ~0.99 cosine /
        ~12% norm-rel vs the fp32 truth (the forward logZ is fine). The
        cancellation happens INSIDE the CCE kernel, so a python-side .float()
        on the returned (already-fp32) CE_none cannot recover it.
      - backend='fp32_accum' passes CCE's accum_e_fp32/accum_c_fp32, forcing
        fp32 accumulation of the e/c gradient contractions INSIDE the CCE
        backward — where the cancelling target-class term is computed. Measured
        on rig (CCE 25.4.3): grad cosine 0.990 -> 0.999, norm-rel ~0.12 -> ~0.05,
        at ~+0.45 GB peak vs bf16 at dreadnought's head shape.
      - backend='bf16' (fp32_accum=False) accepts the ~0.99-cosine gradient for
        the lightest memory. Fine for a small annealed regularizer.

    Notes:
      - safe_targets: pad/ignore_index rows are clamped to row 0 for the gather
        (W[ignore_index] would mis-gather/crash for a non-vocab ignore_index like
        -100) and masked out of the zloss, so the bogus value never contributes.
      - VOCAB-PARALLEL CAVEAT: this manual W[targets] gather does NOT inherit
        CCE's vocab-parallel target remapping / rank-local handling. We are not
        vocab-parallel today; if VP is ever enabled, this gather must mirror
        CCE's target handling (rank-local vocab offset / valid mask).
      - Mask-MULTIPLY (not boolean indexing): static shapes, torch.compile-safe,
        and all-pad micro-batch -> exactly 0 (denom clamped) instead of NaN.
      - logZ is squared in fp32 (logZ ~O(10) -> O(100) loses bf16 precision).
    """
    out_dtype = weight.dtype
    if h_flat.dtype != out_dtype:
        h_flat = h_flat.to(out_dtype)

    kw = dict(reduction="none", ignore_index=pad_id)
    if fp32_accum:
        kw.update(accum_e_fp32=True, accum_c_fp32=True)
    ce_none = cce_loss(h_flat, weight, tgt_flat, **kw)        # [N], differentiable

    # safe_targets: rows we mask out of the z-loss are clamped into [0, vocab)
    # for the gather so a non-vocab ignore_index (e.g. -100) can't index out of
    # bounds. `valid` mirrors CE's ignored set (ignore_index == pad_id), so the
    # z-loss uses exactly the same token set as the CE it is consistent with;
    # the clamp additionally guards against any out-of-range index in those
    # already-excluded rows (their gathered value is discarded by the mask).
    vocab = weight.shape[0]
    valid = tgt_flat != pad_id
    safe_targets = tgt_flat.masked_fill(~valid, 0).clamp_(0, vocab - 1)
    w_rows = weight.index_select(0, safe_targets)            # [N, D] gather (NOT [N, V])
    logit_target = (h_flat * w_rows).sum(-1)                 # [N], differentiable

    lse = ce_none + logit_target                             # = logZ per token
    lse_f = lse.float()
    keep = valid.to(lse_f.dtype)
    denom = keep.sum().clamp_min(1.0)
    zloss = (lse_f * lse_f * keep).sum() / denom             # mean(logZ**2), differentiable
    logz = (lse_f * keep).sum() / denom                      # mean(logZ),    differentiable

    # Diagnostics (detached, logging only): rms = sqrt(mean logZ**2) = sqrt(zloss);
    # p95 over the valid tokens shows the tail of the partition function (the
    # outliers z-loss is meant to pull in), which the mean alone hides.
    with torch.no_grad():
        logz_rms = zloss.detach().clamp_min(0).sqrt()
        valid_lse = lse_f[valid]
        if valid_lse.numel() > 0:
            logz_p95 = torch.quantile(valid_lse, 0.95)
        else:
            logz_p95 = lse_f.new_zeros(())
    return zloss, logz, logz_rms, logz_p95


def _centered_zloss_deadband(h_flat, weight, tgt_flat, pad_id, tau, fp32_accum):
    """DEADBAND CENTERED z-loss (dn4 head-hygiene Lever 2; ships OFF).

    Penalizes the GAUGE-INVARIANT centered log-partition above a ceiling tau:

        logZ_c = logZ - h.mu      (mu = vocab-row mean of the head, a fn of W)
        loss   = mean_n( relu(logZ_c - tau)^2 )          [alpha applied by trainer]

    This is ZLOSS_CENTERED_PLAN Objective A (gradient flows through mu(W)) with a
    DEADBAND so it acts as a CEILING, not constant pressure: zero gradient while
    logZ_c <= tau, Math-approved (DN4_HEAD_HYGIENE_SPEC). Because the loss is a
    function of the gauge-invariant logZ_c, d(loss)/dW has EXACTLY zero common-mode
    component (Sum_v(softmax - 1/V) = 0) -- it cannot push the gauge, only the
    centered structure. Returns (loss, logZ_c_mean, h_mu_mean) over non-pad tokens
    (the latter two detached, for telemetry: h_mu flat + logZ_c draining = working).

    Reuses _zloss_optionD's no-[N,V] reconstruction (ce_none = logZ - logit_target). To
    dodge a catastrophic bf16 cancellation (logit_target ~ h.W_target and h.mu are BOTH
    gauge-dominated ~O(400)), it forms the CENTERED target logit h.(W_target - mu) DIRECTLY
    in fp32, so logZ_c = ce_none + h.(W_target - mu) never subtracts two large numbers
    (fp32_accum still governs ce_none's gradient precision). Note: mu = weight.float().mean(0)
    materializes a full [V,D] fp32 copy of the head + a dense grad through it -- bounded, but
    real when Lever 2 is enabled."""
    out_dtype = weight.dtype
    if h_flat.dtype != out_dtype:
        h_flat = h_flat.to(out_dtype)

    kw = dict(reduction="none", ignore_index=pad_id)
    if fp32_accum:
        kw.update(accum_e_fp32=True, accum_c_fp32=True)
    ce_none = cce_loss(h_flat, weight, tgt_flat, **kw)           # [N], = logZ - logit_target

    vocab = weight.shape[0]
    valid = tgt_flat != pad_id
    safe_targets = tgt_flat.masked_fill(~valid, 0).clamp_(0, vocab - 1)
    w_rows = weight.index_select(0, safe_targets)               # [N, D] gather (NOT [N, V])

    # logZ_c = logZ - h.mu = ce_none + h.(W_target - mu). Form the CENTERED target logit
    # h.(W_target - mu) DIRECTLY in fp32. The naive route (logit_target = h.W_target then
    # logZ_c = logZ - h.mu) subtracts two gauge-dominated ~O(400) quantities, and computing
    # logit_target's dot in BF16 left an O(0.3-1)/token error that survived the fp32
    # subtraction and corrupted the deadband threshold near tau (blind review 2026-06-28).
    # Centering W_target FIRST makes every term small (no large cancellation) and accumulates
    # in fp32. Gradient is unchanged: d(logZ_c)/dW_i = (p_i - 1/V).h  (Objective A).
    hf32 = h_flat.float()
    mu = weight.float().mean(dim=0)                            # [D], global vocab-row mean (the gauge)
    logit_target_c = (hf32 * (w_rows.float() - mu.unsqueeze(0))).sum(-1)   # [N] = h.(W_target - mu), fp32
    logZ_c = ce_none.float() + logit_target_c                 # [N], fp32, gauge-invariant
    h_mu = hf32 @ mu                                          # [N], telemetry only (gauge magnitude)

    excess = (logZ_c - float(tau)).clamp_min(0.0)             # relu(logZ_c - tau): the deadband
    keep = valid.to(logZ_c.dtype)
    denom = keep.sum().clamp_min(1.0)
    loss = (excess * excess * keep).sum() / denom            # mean( relu(logZ_c - tau)^2 )

    with torch.no_grad():
        logZ_c_mean = (logZ_c * keep).sum() / denom
        h_mu_mean = (h_mu * keep).sum() / denom
    return loss, logZ_c_mean, h_mu_mean


# ----------------------------------------------------------------------------
# Flash Attention (optional)
# ----------------------------------------------------------------------------
flash_attn_func = None  # Set to actual import if available

# ----------------------------------------------------------------------------
# FlexAttention (torch >= 2.5) — block-sparse attention for doc_attn_mask.
# Import-guarded so older torch still loads this module; the feature itself
# fatals at Settings validation when unavailable. NOTE: flex_attention is only
# fast inside a torch.compile'd region — the per-submodule compile of each
# Attention module provides that; uncompiled runs get eager flex (correct but
# slow), acceptable only for tests.
# ----------------------------------------------------------------------------
try:
    from torch.nn.attention.flex_attention import flex_attention, create_block_mask
except ImportError:
    flex_attention = None
    create_block_mask = None


# ----------------------------------------------------------------------------
# Gated DeltaNet / Kimi Delta Attention (FLA library, optional)
# ----------------------------------------------------------------------------
_GatedDeltaNet = None       # Lazy-loaded when gdn_enabled=True, gdn_impl='gdn'
_KimiDeltaAttention = None  # Lazy-loaded when gdn_enabled=True, gdn_impl='kda'


def _try_import_gdn(impl: str = 'gdn'):
    global _GatedDeltaNet, _KimiDeltaAttention
    if impl == 'kda':
        if _KimiDeltaAttention is not None:
            return
        try:
            from fla.layers import KimiDeltaAttention
            _KimiDeltaAttention = KimiDeltaAttention
        except ImportError:
            raise ImportError(
                "gdn_impl='kda' requires an FLA version that ships "
                "fla.layers.KimiDeltaAttention (Kimi Linear / K3 family). "
                "Install: pip install -U git+https://github.com/fla-org/flash-linear-attention"
            )
        return
    if _GatedDeltaNet is not None:
        return
    try:
        from fla.layers import GatedDeltaNet
        _GatedDeltaNet = GatedDeltaNet
    except ImportError:
        raise ImportError(
            "GDN hybrid attention requires the FLA library. "
            "Install: pip install -U git+https://github.com/fla-org/flash-linear-attention"
        )


def _new_fla_cache():
    """One FLA Cache per generation, shared by every delta-rule layer (each
    layer indexes it via its compacted _fla_cache_idx — see Transformer.__init__)."""
    from fla.models.utils import Cache
    return Cache()


# ----------------------------------------------------------------------------
# Model Components
# ----------------------------------------------------------------------------

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
        self._use_native = hasattr(F, 'rms_norm')

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        if self._use_native:
            return F.rms_norm(x.float(), self.weight.shape, self.weight.float(), self.eps).type_as(x)
        return self._norm(x.float()).type_as(x) * self.weight


class AuxHead(nn.Module):
    """
    Auxiliary next-token prediction head for intermediate-depth supervision.

    RMSNorm + Linear -> CE against the same shifted targets as the main LM head.
    Used to distribute readout-shaping pressure across the body so no single
    late block has to do all the abstract -> token-space translation.

    No weight sharing with the main LM head — each aux head learns its own readout.

    Forward is the loss path (not logits): with FSDP2 we want params to be
    unsharded on __call__ entry and resharded on exit, so the loss kernel has
    to live inside forward(). The whole module is wrapped with fully_shard().
    """
    def __init__(self, dim: int, vocab_size: int, norm_eps: float):
        super().__init__()
        self.norm = RMSNorm(dim, eps=norm_eps)
        self.linear = nn.Linear(dim, vocab_size, bias=False)

    def forward(self, h_tap, tgt_flat, pad_id, zloss_fp32_accum=None):
        """Compute CE loss at this tap point via the fused CCE kernel.

        Returns (loss, zloss, logz). z-loss is computed only when
        `zloss_fp32_accum` is not None (i.e. the trainer requested it):
            None  -> no z-loss; the CE call is byte-for-byte the original path,
                     loss tensor identical to baseline.
            False -> z-loss via option D, bf16 reconstruction (lightest memory).
            True  -> z-loss via option D, fp32 accumulation in the CCE backward
                     (near-exact gradient, ~+0.45 GB at the head shape).
        See _zloss_optionD for the reconstruction + precision rationale.
        """
        h_norm = self.norm(h_tap)
        h_flat = h_norm.reshape(-1, h_norm.size(-1))
        out_dtype = self.linear.weight.dtype
        if h_flat.dtype != out_dtype:
            h_flat = h_flat.to(out_dtype)
        accum_fp32 = out_dtype == torch.float32
        loss = cce_loss(
            h_flat,
            self.linear.weight,
            tgt_flat,
            accum_e_fp32=accum_fp32,
            accum_c_fp32=accum_fp32,
            reduction="mean",
            ignore_index=pad_id,
        )
        if zloss_fp32_accum is None:
            return loss, None, None
        # rms/p95 diagnostics are surfaced only for the main head (see the
        # Transformer.forward main branch); aux taps return mean only — rms is
        # derivable as sqrt(zloss) by the trainer if ever needed for a tap.
        zloss, logz, _rms, _p95 = _zloss_optionD(
            h_flat, self.linear.weight, tgt_flat, pad_id, zloss_fp32_accum
        )
        return loss, zloss, logz


@dataclass
class ModelArgs:
    dim: int = 4096
    n_layers: int = 32
    n_heads: int = 32
    n_kv_heads: Optional[int] = None
    vocab_size: int = -1
    inner_dim: Optional[int] = None
    norm_eps: float = 1e-5
    max_seq_len: int = 2048
    dropout: float = 0.0
    pad_id: int = 0
    use_activation_checkpointing: bool = True
    # ── Activation-memory levers (2026-08-05 rig-30 perf campaign). All are
    # RUNTIME perf knobs — resume-safe, deliberately absent from the resume
    # structural-guard triples. Default-off = byte-identical behavior. ──
    #   ac_skip_layers    — skip checkpointing on N evenly-striped blocks:
    #                       their recompute vanishes; intermediates cost VRAM
    #                       unless ac_offload ships them to host.
    #   ac_input_offload  — Pool A: offload the AC'd blocks' checkpoint
    #                       boundary inputs (~B*T*embd each) to pinned host
    #                       RAM. Frees VRAM (→ larger B) for ~9ms/layer PCIe.
    #   ac_offload        — Pool B: offload the AC-skipped blocks' full saved
    #                       intermediates (~110ms/layer PCIe round trip).
    #   ac_offload_min_mb — per-tensor floor; smaller saves stay on-device.
    ac_skip_layers: int = 0
    ac_input_offload: bool = False
    #   ac_input_offload_layers — partial Pool A (2026-08-07, born the night
    #     the RAM swap beeped): offload boundary inputs of only N evenly-
    #     striped blocks; the rest keep plain-AC behavior (input stays in
    #     VRAM). Trades pinned-host (N x B*T*embd per rank) against VRAM
    #     ((L-N) x same). 0 = all AC'd blocks (historical Pool A behavior).
    ac_input_offload_layers: int = 0
    ac_offload: bool = False
    ac_offload_min_mb: int = 8
    #   ac_sac_policy — F9 op-level SAC: None | 'flex_save' (save attention
    #     outputs inside the block checkpoint, recompute the rest).
    ac_sac_policy: str = None
    # QK-Norm Mode: None | "before_rope" | "after_rope_legacy"
    qk_norm_mode: Optional[str] = None
    # Tie input embeddings and output LM head weights
    tie_word_embeddings: bool = True
    # RoPE base frequency (higher = longer context support)
    rope_theta: float = 500000.0
    # Positional-encoding mode for the attention Q/K path (PE ablation 2026-07-30):
    #   "rope"     — standard rotary embedding (default; behavior unchanged)
    #   "nope"     — no positional encoding: identity tables (cos=1, sin=0) make
    #                apply_rotary_emb an exact no-op; position is inferable only
    #                through the causal mask
    #   "envelope" — deliberate reproduction of the pre-2026-07-02 meta-init
    #                corruption (cos=0, sin=cos-table): per-channel attention
    #                scores become (q·k)·cos(p·θ)·cos(m·θ), a separable
    #                positional envelope with no clean relative rotation
    rope_mode: str = "rope"
    # KEEL (Highway-style Post-LN) configuration
    # Paper: "Post-LayerNorm Is Back: Stable, Expressive, and Deep" (arXiv:2601.19895)
    use_keel: bool = False
    keel_alpha: Optional[float] = None  # If None, auto-set to n_layers * 2
    # MoE (Mixture of Experts) configuration
    moe_enabled: bool = False
    moe_num_experts: int = 8
    moe_top_k: int = 2
    moe_num_shared_experts: int = 1       # 0 = no shared expert
    moe_score_func: str = "sigmoid"       # "sigmoid" or "softmax"
    moe_score_before_experts: bool = True  # multiply scores before (True) or after (False) experts
    moe_route_norm: bool = False           # normalize top-k scores
    moe_route_scale: float = 1.0           # scale factor for router scores
    moe_load_balance_coeff: Optional[float] = 1e-3  # aux-loss-free balancing (None = disabled)
    moe_aux_balance_coeff: float = 0.0     # aux balance loss weight (0 = disabled)
    moe_bias_before_score: bool = False    # add expert_bias before score_func (True = old-style)
    moe_interleave_step: int = 1           # every Nth layer is MoE (1=all, 2=every other)
    moe_n_dense_layers: int = 0            # first N layers are always dense
    moe_n_tail_dense_layers: int = 0       # last N layers are always dense (synth layers)
    moe_capacity_factor: float = 0.0         # 0 = disabled, >0 = cap tokens/expert (e.g. 1.5)
    moe_inner_dim: Optional[int] = None    # expert FFN hidden dim (None = same as inner_dim)
    # Expert Parallel
    ep_degree: int = 1                     # EP degree (1 = no EP, all experts local)
    moe_shared_overlap: bool = False       # overlap shared_experts with EP on a side CUDA stream
    # Gated DeltaNet (GDN) hybrid attention configuration
    gdn_enabled: bool = False              # enable GDN hybrid attention
    gdn_impl: str = 'gdn'                  # 'gdn' = FLA GatedDeltaNet | 'kda' = FLA
                                           # KimiDeltaAttention (Kimi Linear / K3 family:
                                           # channel-wise decay instead of per-head scalar)
    gdn_interleave_step: int = 4           # every Nth layer is full-attention, rest are GDN
    gdn_n_tail_global: int = 1             # FORCE the last N layers global regardless of
                                           # interleave (K3 guarantees a global final layer —
                                           # the LM head must not decode from a delta layer's
                                           # ~100-token local summary). 0 = pure interleave.
                                           # >1 = experiment knob for multi-global tails
                                           # (KEEL denormalizing-tail interaction).
    n_gdn_heads: Optional[int] = None      # GDN head count (None = same as n_heads)
    gdn_head_dim: Optional[int] = None     # q/k head dim (None = impl default: 256 gdn / 128 kda)
    gdn_v_expand: Optional[float] = None   # value expansion ratio (None = impl default:
                                           # 2.0 gdn / 1.0 kda, matching each paper)
    gdn_short_conv_kernel: int = 4         # short convolution kernel size
    gdn_mode: str = 'chunk'                # FLA mode: 'chunk' (training) or 'fused_recurrent'
    # KDA-only knobs (K3's lower-bounded log-decay; ignored for gdn_impl='gdn'):
    kda_safe_gate: bool = True             # bounded decay parameterization (K3) vs
                                           # Kimi-Linear negative-softplus (unbounded)
    kda_lower_bound: float = -5.0          # K3 g_min: log-decay bounded to (g_min, 0)
    # Attention Residuals (AttnRes) — learned depth-wise attention over block representations
    # Paper: "Attention Residuals" — Kimi Team (2026)
    attn_res_enabled: bool = False         # enable Block AttnRes
    attn_res_block_size: int = 8           # layers per block (n_layers should be divisible by this)
    # Auxiliary prediction heads — RMSNorm + Linear at intermediate depths.
    # Distributes readout-shaping pressure across the body. List of 0-indexed
    # layer positions; head taps the output of layers[i] (i.e. the value that
    # becomes layers[i+1]'s input). Per-head loss weights are applied by the
    # trainer, not the model.
    aux_head_layers: List[int] = field(default_factory=list)
    # Document attention masking for packed windows (branch doc-mask).
    # The data stream packs documents separated by BOS; by default attention
    # flows across document boundaries. doc_attn_mask=True confines attention
    # to (causal AND same-document) via a FlexAttention BlockMask built per
    # micro-batch from token==bos_token_id. doc_pos_reset=True restarts RoPE
    # positions at each BOS (tokens before a window's first BOS keep
    # window-relative positions — same origin they'd get today). Training/eval
    # path only; the KV-cache generate path is single-document and unchanged.
    doc_attn_mask: bool = False
    doc_pos_reset: bool = False
    # -1 = "not set" sentinel. There is NO safe universal default: the ecosystem
    # has multiple document separators (SentencePiece BOS=1 in llama pretrain
    # shards, <|bos|>=32000 in extended-special chat data, <|endoftext|> in
    # tiktoken trees). Constructors that enable doc_attn_mask/doc_pos_reset must
    # pass the id explicitly — Transformer.__init__ enforces it — and the
    # trainer derives/cross-checks it against tokenizer.bos_id at boot
    # ([bos-check] banner). The old default here (32000) silently mismatched
    # the actual llama shard separator (1).
    bos_token_id: int = -1
    # Sliding-window attention (branch doc-mask, festival feature 2). Hybrid
    # local:global — a layer is GLOBAL when layer_id % swa_global_interleave ==
    # swa_global_interleave - 1 (mirrors the GDN interleave convention; 4 -> 3:1
    # local:global), else LOCAL with a causal window of swa_window tokens.
    # Local windows compose with doc_attn_mask (window AND same-doc AND causal).
    # Generation note: forward_with_cache uses the full causal cache, so sampling
    # is EXACT only while total sequence length <= swa_window; setup_caches warns.
    swa_enabled: bool = False
    swa_window: int = 512
    swa_global_interleave: int = 4
    # Multi-token prediction (festival feature 3): DeepSeek-V3-style sequential
    # module predicting t+2 through one extra block + the shared norm/head.
    # Loss weight is applied by the trainer (z-loss pattern), not the model.
    mtp_enabled: bool = False
    # Ignore MTP loss rows whose 2-token window crosses a document boundary
    # (targets[i] == BOS => the t+2 target is the NEXT doc's first content
    # token, conditioned on prev-doc state). Default False = original
    # DeepSeek-style behavior, byte-identical for existing runs.
    mtp_doc_boundary_mask: bool = False

def _block_attn_res_fn(partial_block, qk, eps, *blocks):
    """AttnRes core — all intermediates recomputed during backward via checkpoint.

    Factored so K tensor is never materialized:
        logit = (qk · V) / rms(V)
    Only ~5 passes over [N+1,B,S,D] in bf16 vs ~11 in the float32 version.
    """
    V = torch.stack(list(blocks) + [partial_block])           # [N+1, B, S, D]
    raw = torch.einsum('d, n b s d -> n b s', qk, V)         # [N+1, B, S]
    rms = V.pow(2).mean(-1).add(eps).sqrt()                   # [N+1, B, S]
    weights = (raw / rms).softmax(dim=0)                      # [N+1, B, S]
    return torch.einsum('n b s, n b s d -> b s d', weights, V)


def block_attn_res(blocks, partial_block, query, key_norm_weight, eps):
    """Block Attention Residuals with efficient activation checkpointing.

    preserve_rng_state=False is safe — no stochastic ops inside.
    qk precomputed outside checkpoint so gradients flow to params directly.
    """
    qk = query * key_norm_weight                              # [D]
    return cp.checkpoint(
        _block_attn_res_fn, partial_block, qk, eps, *blocks,
        use_reentrant=False, preserve_rng_state=False,
    )


def precompute_freqs_cis(dim: int, end: int, theta: float = 500000.0):
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2, device='cpu')[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device)
    freqs = torch.outer(t, freqs).float()
    freqs_cos = torch.cos(freqs)
    freqs_sin = torch.sin(freqs)
    return freqs_cos, freqs_sin


def compute_rope_tables(dim: int, end: int, theta: float, rope_mode: str = "rope"):
    """Positional tables keyed by rope_mode — single source of truth for
    Transformer.__init__, init_weights, and the trainer's [freqs-check] rail.

    "nope" is EXACTLY the identity through apply_rotary_emb (cos=1, sin=0; the
    fp32 round-trip of bf16 inputs is value-preserving, so scores reduce to raw
    q·k). "envelope" reproduces the pre-2026-07-02 meta-init corruption
    deterministically (cos = zero pages, sin = the recycled cos block)."""
    freqs_cos, freqs_sin = precompute_freqs_cis(dim, end, theta)
    if rope_mode == "rope":
        return freqs_cos, freqs_sin
    if rope_mode == "nope":
        return torch.ones_like(freqs_cos), torch.zeros_like(freqs_sin)
    if rope_mode == "envelope":
        return torch.zeros_like(freqs_cos), freqs_cos
    raise ValueError(
        f"unknown rope_mode {rope_mode!r} — expected 'rope' | 'nope' | 'envelope'")



""" Some debate about use of complex numbers for RoPE
def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor):
    ndim = x.ndim
    assert 0 <= 1 < ndim
    assert freqs_cis.shape == (x.shape[1], x.shape[-1])
    shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
    return freqs_cis.view(shape)

def apply_rotary_emb(
    xq: torch.Tensor, 
    xk: torch.Tensor, 
    freqs_cos: torch.Tensor, 
    freqs_sin: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    # Apply rotary position embeddings using complex number multiplication.
    xq_c = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_c = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    
    freqs_cis = torch.complex(freqs_cos.float(), freqs_sin.float())
    freqs_cis = reshape_for_broadcast(freqs_cis, xq_c)
    
    xq_out = torch.view_as_real(xq_c * freqs_cis).flatten(3)
    xk_out = torch.view_as_real(xk_c * freqs_cis).flatten(3)
    
    return xq_out.type_as(xq), xk_out.type_as(xk)
"""
def apply_rotary_emb(
    xq: torch.Tensor, 
    xk: torch.Tensor, 
    freqs_cos: torch.Tensor, 
    freqs_sin: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Apply rotary embeddings using real-valued operations (inductor-friendly)."""
    # xq, xk: [B, S, H, D]
    # freqs_cos, freqs_sin: [S, D//2] shared positions, or [B, S, D//2] when
    # doc_pos_reset gathers per-document positions (each window has its own
    # BOS layout, so positions differ per sample)

    # Split into even/odd (equivalent to real/imag in complex view)
    xq_r, xq_i = xq.float().reshape(*xq.shape[:-1], -1, 2).unbind(-1)
    xk_r, xk_i = xk.float().reshape(*xk.shape[:-1], -1, 2).unbind(-1)

    if freqs_cos.ndim == 3:
        cos = freqs_cos[:, :, None, :]  # [B, S, 1, D//2]
        sin = freqs_sin[:, :, None, :]
    else:
        cos = freqs_cos[None, :, None, :]  # [1, S, 1, D//2]
        sin = freqs_sin[None, :, None, :]
    
    # Complex multiplication: (a + bi)(c + di) = (ac - bd) + (ad + bc)i
    # Here c = cos, d = sin (unit vector rotation)
    xq_out_r = xq_r * cos - xq_i * sin
    xq_out_i = xq_r * sin + xq_i * cos
    xk_out_r = xk_r * cos - xk_i * sin
    xk_out_i = xk_r * sin + xk_i * cos
    
    # Interleave back: [B, S, H, D//2, 2] -> [B, S, H, D]
    xq_out = torch.stack([xq_out_r, xq_out_i], dim=-1).flatten(-2)
    xk_out = torch.stack([xk_out_r, xk_out_i], dim=-1).flatten(-2)
    
    return xq_out.type_as(xq), xk_out.type_as(xk)

def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """Repeat KV heads for GQA/MQA."""
    bs, slen, n_kv_heads, head_dim = x.shape
    if n_rep == 1:
        return x
    return (
        x[:, :, :, None, :]
        .expand(bs, slen, n_kv_heads, n_rep, head_dim)
        .reshape(bs, slen, n_kv_heads * n_rep, head_dim)
    )


def doc_ids_from_tokens(tokens: torch.Tensor, bos_id: int) -> torch.Tensor:
    """Per-token document ids for a packed window [B, S] -> [B, S] int32.

    Inclusive cumsum: a BOS token STARTS a new document (it belongs to the doc
    it opens, so later tokens of that doc can attend back to it). Tokens before
    a window's first BOS are the tail of a document cut by the window boundary
    and share doc id 0."""
    return (tokens == bos_id).cumsum(dim=-1, dtype=torch.int32)


def doc_cu_seqlens(tokens: torch.Tensor, bos_id: int) -> torch.Tensor:
    """Flattened-varlen boundaries for delta-rule (KDA/GDN) doc state resets.

    [B, S] tokens -> int32 cu_seqlens in FLA's flash-varlen convention: the
    batch is viewed as ONE flattened [1, B*S] sequence and cu_seqlens marks
    every segment start (plus the final total). Boundaries are placed at every
    ROW start (rows are independent packed windows — in normal [B, S] mode the
    recurrent state is per-row by construction, so row-start boundaries make
    the flattened form exactly equivalent) and at every BOS (the doc reset this
    exists for). A BOS sitting at a row start coincides with the row boundary
    and is naturally deduplicated by the boolean mask.
    See docs/KDA_VARLEN_DOC_RESET.md."""
    B, S = tokens.shape
    is_start = (tokens == bos_id).reshape(-1).clone()
    is_start[0::S] = True
    idx = is_start.nonzero(as_tuple=True)[0].to(torch.int32)
    total = torch.tensor([B * S], dtype=torch.int32, device=tokens.device)
    return torch.cat([idx, total])


def doc_position_ids(tokens: torch.Tensor, bos_id: int) -> torch.Tensor:
    """Position-within-document [B, S] -> [B, S] int64: 0 at each BOS,
    incrementing until the next BOS. Tokens before the first BOS keep
    window-relative positions — the same origin every window gets under
    shared (non-reset) positions, so the cut-document fragment trains
    exactly as it would today."""
    B, S = tokens.shape
    idx = torch.arange(S, device=tokens.device).expand(B, S)
    is_bos = tokens == bos_id
    # index of the most recent BOS at or before each position; -1 before any
    last_bos = torch.cummax(
        torch.where(is_bos, idx, torch.full_like(idx, -1)), dim=-1
    ).values
    start = torch.clamp(last_bos, min=0)
    return idx - start


class Attention(nn.Module):
    """
    Multi-head attention with GQA support and optional KV caching.
    
    Training: Uses forward() - no caching, identical to original model_v1.py
    Inference: Uses forward_with_cache() - KV caching for O(1) per-token generation
    """
    
    def __init__(self, args: ModelArgs, use_gate: bool = False):
        super().__init__()
        self.n_kv_heads = args.n_heads if args.n_kv_heads is None else args.n_kv_heads
        assert args.n_heads % self.n_kv_heads == 0
        model_parallel_size = 1
        self.n_local_heads = args.n_heads // model_parallel_size
        self.n_local_kv_heads = self.n_kv_heads // model_parallel_size
        self.n_rep = self.n_local_heads // self.n_local_kv_heads
        self.head_dim = args.dim // args.n_heads

        self.wq = nn.Linear(args.dim, args.n_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(args.dim, self.n_kv_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(args.dim, self.n_kv_heads * self.head_dim, bias=False)
        self.wo = nn.Linear(args.n_heads * self.head_dim, args.dim, bias=False)

        # Gated softmax attention: sigmoid gate on output (for GDN hybrid mode)
        self.use_gate = use_gate
        if use_gate:
            self.g_proj = nn.Linear(args.dim, args.n_heads * self.head_dim, bias=False)

        self.attn_dropout = nn.Dropout(args.dropout)
        self.resid_dropout = nn.Dropout(args.dropout)
        self.dropout = args.dropout

        # QK normalization mode
        self.qk_norm_mode = getattr(args, 'qk_norm_mode', None)
        self.norm_eps = args.norm_eps
        
        # Learnable RMSNorm for "before_rope" mode
        if self.qk_norm_mode == "before_rope":
            self.q_norm = RMSNorm(self.head_dim, eps=self.norm_eps)
            self.k_norm = RMSNorm(self.head_dim, eps=self.norm_eps)

        self.use_flashattn2 = (flash_attn_func is not None and torch.cuda.is_available())
        self.use_sdp = (not self.use_flashattn2 and hasattr(F, "scaled_dot_product_attention"))

        # Does this torch build support SDPA's enable_gqa flag?
        # (kept out of forward() so torch.compile doesn't see dynamic signature checks)
        self.sdp_enable_gqa = False
        if self.use_sdp:
            try:
                self.sdp_enable_gqa = "enable_gqa" in inspect.signature(
                    F.scaled_dot_product_attention
                ).parameters
            except (TypeError, ValueError):
                self.sdp_enable_gqa = False

        if not self.use_flashattn2 and not self.use_sdp:
            mask = torch.full((1, 1, args.max_seq_len, args.max_seq_len), float("-inf"))
            mask = torch.triu(mask, diagonal=1)
            self.register_buffer("mask", mask)

        # KV Cache placeholders - NOT allocated until setup_cache() is called
        # These are NOT nn.Parameters, just plain tensors when allocated
        self.cache_k: Optional[torch.Tensor] = None
        self.cache_v: Optional[torch.Tensor] = None
        # SWA rolling cache: set by Transformer.setup_caches — the window size W
        # for LOCAL layers (cache holds min(max_seq_len, W) slots, slot = pos % Lc),
        # None for global layers (full-length cache, original semantics).
        self.cache_window: Optional[int] = None

    def forward(self, x: torch.Tensor, freqs_cos: torch.Tensor, freqs_sin: torch.Tensor,
                block_mask=None):
        """
        TRAINING PATH - Identical to original model_v1.py
        No caching, no start_pos, no branching on cache existence.

        block_mask: optional FlexAttention BlockMask (doc_attn_mask feature).
        None -> the existing SDPA/flash causal paths, byte-identical to before.
        """
        bsz, seqlen, _ = x.shape
        
        # ➊ projections
        xq = self.wq(x).view(bsz, seqlen, self.n_local_heads, self.head_dim)
        xk = self.wk(x).view(bsz, seqlen, self.n_local_kv_heads, self.head_dim)
        xv = self.wv(x).view(bsz, seqlen, self.n_local_kv_heads, self.head_dim)

        # ➋ QK-norm BEFORE RoPE (learnable RMSNorm, recommended)
        if self.qk_norm_mode == "before_rope":
            xq = self.q_norm(xq)
            xk = self.k_norm(xk)

        # ➌ RoPE
        xq, xk = apply_rotary_emb(xq, xk, freqs_cos, freqs_sin)

        # ➍ QK-norm AFTER RoPE
        if self.qk_norm_mode == "after_rope_legacy":
            def _l2_norm(t: torch.Tensor) -> torch.Tensor:
                tf = t.float()
                inv = torch.rsqrt((tf * tf).sum(dim=-1, keepdim=True) + self.norm_eps)
                return (tf * inv).to(t.dtype)
            
            xq = _l2_norm(xq)
            xk = _l2_norm(xk)
            if self.qk_norm_mode == "after_rope_legacy":
                xq = xq * math.sqrt(self.head_dim)
                xk = xk * math.sqrt(self.head_dim)

        # ➎ attention computation (avoid materializing repeated KV heads when SDPA supports GQA)

        if block_mask is not None:
            # doc-masked block-sparse attention (FlexAttention). Causality AND
            # document confinement live in the BlockMask; cross-document blocks
            # are SKIPPED (block sparsity), not computed-then-masked. GQA is
            # native via enable_gqa. No dropout arg exists on this path —
            # Settings fatals if dropout>0 with doc_attn_mask enabled.
            q = xq.transpose(1, 2)   # [B, Hq, S, D]
            k = xk.transpose(1, 2)   # [B, Hkv, S, D]
            v = xv.transpose(1, 2)
            out = flex_attention(q, k, v, block_mask=block_mask,
                                 enable_gqa=(self.n_rep > 1))
            out = out.transpose(1, 2).contiguous().view(bsz, seqlen, -1)
        elif self.use_flashattn2:
            # flash-attn path expects matched head counts
            #xk_rep = repeat_kv(xk, self.n_rep)
            #xv_rep = repeat_kv(xv, self.n_rep)
            #out = flash_attn_func(
            #    xq.contiguous(), xk_rep.contiguous(), xv_rep.contiguous(),                
            #    dropout_p=self.dropout if self.training else 0.0,
            #    causal=True
            #)
            # Flash Attention 2 handles GQA natively - no need to repeat KV
            out = flash_attn_func(
                xq.contiguous(), 
                xk.contiguous(),  # [B, S, n_kv_heads, D]
                xv.contiguous(),
                dropout_p=self.dropout if self.training else 0.0,
                causal=True
            )           
            out = out.reshape(bsz, seqlen, -1)
        else:
            xq = xq.transpose(1, 2)  # [B, Hq, S, D]

            if self.use_sdp:
                # Keep K/V at Hkv; let SDPA do the head mapping when possible
                k = xk.transpose(1, 2)  # [B, Hkv, S, D]
                v = xv.transpose(1, 2)

                if self.n_rep > 1 and self.sdp_enable_gqa:
                    out = F.scaled_dot_product_attention(
                        xq, k, v,
                        attn_mask=None,
                        dropout_p=self.dropout if self.training else 0.0,
                        is_causal=True,
                        enable_gqa=True,
                    )
                else:
                    # Older SDPA without enable_gqa: materialize to Hq like before
                    if self.n_rep > 1:
                        k = repeat_kv(xk, self.n_rep).transpose(1, 2)
                        v = repeat_kv(xv, self.n_rep).transpose(1, 2)
                    out = F.scaled_dot_product_attention(
                        xq, k, v,
                        attn_mask=None,
                        dropout_p=self.dropout if self.training else 0.0,
                        is_causal=True
                    )
            else:
                # Manual attention path requires matched head counts
                k = repeat_kv(xk, self.n_rep).transpose(1, 2)
                v = repeat_kv(xv, self.n_rep).transpose(1, 2)

                scores = torch.matmul(xq, k.transpose(2, 3)) / math.sqrt(self.head_dim)
                scores = scores + self.mask[:, :, :seqlen, :seqlen]
                scores = F.softmax(scores.float(), dim=-1).type_as(xq)
                scores = self.attn_dropout(scores)
                out = torch.matmul(scores, v)

            out = out.transpose(1, 2).contiguous().view(bsz, seqlen, -1)

        # ➐ gated attention (GDN hybrid mode)
        if self.use_gate:
            out = out * torch.sigmoid(self.g_proj(x))

        # ➑ output projection
        out = self.wo(out)
        out = self.resid_dropout(out)
        return out

    def forward_with_cache(
        self, 
        x: torch.Tensor, 
        freqs_cos: torch.Tensor, 
        freqs_sin: torch.Tensor,
        start_pos: int
    ):
        """
        INFERENCE PATH - Uses KV caching for efficient generation.
        Must call setup_cache() before using this method.
        
        Args:
            x: Input tensor [B, S, D] - for prefill S=prompt_len, for decode S=1
            freqs_cos, freqs_sin: RoPE frequencies, sliced for positions [start_pos:start_pos+S]
            start_pos: Current position in the sequence (0 for prefill)
        """
        bsz, seqlen, _ = x.shape
        
        # Ensure freqs are on correct device
        if freqs_cos.device != x.device:
            freqs_cos = freqs_cos.to(x.device)
            freqs_sin = freqs_sin.to(x.device)
        
        # ➊ Projections
        xq = self.wq(x).view(bsz, seqlen, self.n_local_heads, self.head_dim)
        xk = self.wk(x).view(bsz, seqlen, self.n_local_kv_heads, self.head_dim)
        xv = self.wv(x).view(bsz, seqlen, self.n_local_kv_heads, self.head_dim)

        # ➋ QK-norm BEFORE RoPE
        if self.qk_norm_mode == "before_rope":
            xq = self.q_norm(xq)
            xk = self.k_norm(xk)

        # ➌ RoPE
        xq, xk = apply_rotary_emb(xq, xk, freqs_cos, freqs_sin)

        # ➍ QK-norm AFTER RoPE (apply to *new* Q/K once, then cache K)
        if self.qk_norm_mode == "after_rope_legacy":
            def _l2_norm(t: torch.Tensor) -> torch.Tensor:
                tf = t.float()
                inv = torch.rsqrt((tf * tf).sum(dim=-1, keepdim=True) + self.norm_eps)
                return (tf * inv).to(t.dtype)

            xq = _l2_norm(xq)
            xk = _l2_norm(xk)
            if self.qk_norm_mode == "after_rope_legacy":
                scale = math.sqrt(self.head_dim)
                xq = xq * scale
                xk = xk * scale

        # ➎ Update KV cache + select keys/values for this call
        assert self.cache_k is not None, "Must call setup_caches() before forward_with_cache()"
        if self.cache_window is None:
            # ---- GLOBAL layer: full-length cache (original path, byte-identical) ----
            self.cache_k[:bsz, start_pos:start_pos + seqlen] = xk
            self.cache_v[:bsz, start_pos:start_pos + seqlen] = xv

            # Retrieve all cached K/V up to current position
            keys = self.cache_k[:bsz, :start_pos + seqlen]
            values = self.cache_v[:bsz, :start_pos + seqlen]

            # For single-token decode (seqlen == 1) we attend to all past keys,
            # no mask. For prefill (seqlen > 1) we need a causal mask.
            #
            # IMPORTANT: query positions are [start_pos, start_pos+seqlen) but key
            # positions are [0, start_pos+seqlen) — a NON-SQUARE score matrix when
            # start_pos > 0 (cross-turn prefix reuse prefills only the suffix). SDPA's
            # is_causal=True applies a TOP-LEFT-aligned square mask, which is only
            # correct when start_pos == 0. For start_pos > 0 it would let suffix query
            # i attend to keys [0, i] instead of the correct [0, start_pos+i] — silent
            # wrong output. So: use is_causal=True ONLY for the start_pos == 0 prefill,
            # and build an explicit absolute-position-aligned mask otherwise.
            need_mask = (seqlen > 1)
            use_is_causal = need_mask and (start_pos == 0)
            attn_mask = None
            if need_mask and not use_is_causal:
                # Bottom-right / absolute-aligned causal mask: query row r (absolute
                # position start_pos+r) may attend to key cols <= start_pos+r.
                total_len = start_pos + seqlen
                attn_mask = torch.triu(
                    torch.full((seqlen, total_len), float("-inf"), device=x.device, dtype=xq.dtype),
                    diagonal=start_pos + 1,
                )
        else:
            # ---- LOCAL (SWA) layer: ROLLING WINDOW cache — slot(q) = q % Lc.
            # RoPE (and any qk-norm) is baked into cached keys, so attention is
            # over a SET: slot ORDER is irrelevant wherever no mask is needed.
            # Memory: Lc = min(max_seq_len, W) slots instead of max_seq_len —
            # and long-context decode attends ≤ W keys instead of the full
            # history (training-faithful semantics AND faster).
            if not self.use_sdp:
                raise RuntimeError("SWA rolling cache requires the SDPA path (torch>=2.0)")
            W = self.cache_window
            Lc = self.cache_k.shape[1]
            total = start_pos + seqlen
            need_mask = False
            use_is_causal = False
            attn_mask = None
            if seqlen == 1:
                # decode fast path: ring-write one slot, attend every valid slot,
                # NO mask ever — post-write the ring holds exactly positions
                # [max(0, total-Lc), total), all causal and all in-window (Lc==W).
                slot = start_pos % Lc
                self.cache_k[:bsz, slot] = xk[:, 0]
                self.cache_v[:bsz, slot] = xv[:, 0]
                n_valid = min(total, Lc)
                keys = self.cache_k[:bsz, :n_valid]
                values = self.cache_v[:bsz, :n_valid]
            else:
                # chunk path (prefill or cross-turn suffix): GATHER the previous
                # window in POSITIONAL order BEFORE writing (the chunk write may
                # evict slots the earliest chunk queries still need), then attend
                # [prev_window, chunk] under a banded causal∧window mask built on
                # absolute positions.
                n_prev = min(start_pos, Lc)
                if n_prev > 0:
                    ppos = torch.arange(start_pos - n_prev, start_pos, device=x.device)
                    keys = torch.cat([self.cache_k[:bsz, ppos % Lc], xk], dim=1)
                    values = torch.cat([self.cache_v[:bsz, ppos % Lc], xv], dim=1)
                else:
                    keys, values = xk, xv
                # ring-write the chunk tail (only the last min(S, Lc) positions can
                # survive; consecutive positions mod Lc are unique, so no in-write
                # slot collisions)
                tail = min(seqlen, Lc)
                wpos = torch.arange(total - tail, total, device=x.device)
                self.cache_k[:bsz, wpos % Lc] = xk[:, -tail:]
                self.cache_v[:bsz, wpos % Lc] = xv[:, -tail:]
                # banded mask: key positions are CONTIGUOUS [start_pos-n_prev, total);
                # query p attends key q iff q <= p AND q > p - W.
                kpos = torch.arange(start_pos - n_prev, total, device=x.device)
                qpos = torch.arange(start_pos, total, device=x.device)
                allowed = (kpos[None, :] <= qpos[:, None]) & (kpos[None, :] > qpos[:, None] - W)
                attn_mask = torch.zeros((seqlen, kpos.shape[0]), device=x.device, dtype=xq.dtype)
                attn_mask.masked_fill_(~allowed, float("-inf"))
                need_mask = True

        # (keys are already normalized/scaled in-cache for after_rope modes)

        # ➐ Attention (no dropout during inference)
        xq = xq.transpose(1, 2)       # [B, Hq, S_q, D]
        k = keys.transpose(1, 2)      # [B, Hkv, S_kv, D]
        v = values.transpose(1, 2)

        if self.use_sdp:
            if self.n_rep > 1 and self.sdp_enable_gqa:
                out = F.scaled_dot_product_attention(
                    xq, k, v,
                    attn_mask=attn_mask,
                    dropout_p=0.0,
                    is_causal=use_is_causal,
                    enable_gqa=True,
                )
            else:
                # Older SDPA without enable_gqa: materialize to Hq like before
                if self.n_rep > 1:
                    k = repeat_kv(keys, self.n_rep).transpose(1, 2)
                    v = repeat_kv(values, self.n_rep).transpose(1, 2)
                out = F.scaled_dot_product_attention(
                    xq, k, v,
                    attn_mask=attn_mask,
                    dropout_p=0.0,
                    is_causal=use_is_causal
                )
        else:
            # Manual attention path requires matched head counts
            k = repeat_kv(keys, self.n_rep).transpose(1, 2)
            v = repeat_kv(values, self.n_rep).transpose(1, 2)

            scores = torch.matmul(xq, k.transpose(2, 3)) / math.sqrt(self.head_dim)
            if need_mask:
                # Absolute-aligned causal mask covering both start_pos==0 and the
                # start_pos>0 (suffix-prefill) case. Reuse the mask built above
                # when present; build the start_pos==0 form otherwise so the two
                # attention paths can never diverge.
                if attn_mask is not None:
                    scores = scores + attn_mask
                else:
                    total_len = start_pos + seqlen
                    scores = scores + torch.triu(
                        torch.full((seqlen, total_len), float("-inf"), device=x.device, dtype=scores.dtype),
                        diagonal=start_pos + 1,
                    )
            scores = F.softmax(scores.float(), dim=-1).type_as(xq)
            out = torch.matmul(scores, v)

        out = out.transpose(1, 2).contiguous().view(bsz, seqlen, -1)
        
        # ➑ gated attention (GDN hybrid mode)
        if self.use_gate:
            out = out * torch.sigmoid(self.g_proj(x))

        # ➒ Output projection (no dropout during inference)
        out = self.wo(out)
        return out


class FeedForward(nn.Module):
    def __init__(self, dim: int, inner_dim: int, dropout: float):
        super().__init__()
        if inner_dim is None:
            inner_dim = 4 * dim
            inner_dim = int(2 * inner_dim / 3)
            inner_dim = 128 * ((inner_dim + 127) // 128)  # Round up to multiple of 128
        self.w1 = nn.Linear(dim, inner_dim, bias=False)
        self.w2 = nn.Linear(inner_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, inner_dim, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.dropout(self.w2(F.silu(self.w1(x)) * self.w3(x)))


def _compute_default_inner_dim(dim):
    """Compute default FFN hidden dim: 2/3 * 4 * dim, rounded up to multiple of 128."""
    inner_dim = 4 * dim
    inner_dim = int(2 * inner_dim / 3)
    return 128 * ((inner_dim + 127) // 128)


# =========================================================================
# Mixture of Experts (MoE) — adapted from TorchTitan
# =========================================================================

# Expert Parallel helpers (all-to-all dispatch/combine)
try:
    import torch.distributed as dist
    _has_dist = True
except ImportError:
    _has_dist = False


class _AllToAllSingleAutograd(torch.autograd.Function):
    """Differentiable all-to-all: backward reverses the split sizes."""

    @staticmethod
    def forward(ctx, x, output_splits, input_splits, group):
        ctx.output_splits = output_splits
        ctx.input_splits = input_splits
        ctx.group = group
        out = torch.empty(sum(output_splits), x.shape[1], dtype=x.dtype, device=x.device)
        dist.all_to_all_single(out, x, output_splits, input_splits, group=group)
        return out

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = torch.empty(
            sum(ctx.input_splits), grad_output.shape[1],
            dtype=grad_output.dtype, device=grad_output.device,
        )
        dist.all_to_all_single(
            grad_input, grad_output.contiguous(),
            ctx.input_splits, ctx.output_splits, group=ctx.group,
        )
        return grad_input, None, None, None


def _permute_for_ep(tokens, ep_degree, num_local_experts, counts):
    """Reorder from (rank, expert) layout to (expert, rank) layout after all-to-all.

    Args:
        tokens: received tokens from all-to-all
        ep_degree: number of EP ranks
        num_local_experts: experts per rank
        counts: pre-computed int list of per-(rank, expert) token counts
                (length = ep_degree * num_local_experts)

    Returns:
        (permuted_tokens, local_num_tpe_tensor, local_counts_list)
    """
    chunks = list(torch.split(tokens[:sum(counts)], counts))
    reordered = []
    local_counts = [0] * num_local_experts
    for e in range(num_local_experts):
        for r in range(ep_degree):
            reordered.append(chunks[r * num_local_experts + e])
            local_counts[e] += counts[r * num_local_experts + e]
    # MUST be int — FSDP mixed-precision casts float inputs to bf16, which
    # rounds large counts (e.g. 7346→7360) causing split_with_sizes mismatches.
    local_num_tpe = torch.tensor(local_counts, dtype=torch.int64, device=tokens.device)
    return torch.cat(reordered, dim=0), local_num_tpe, local_counts


def _unpermute_for_ep(tokens, ep_degree, num_local_experts, counts):
    """Reverse of _permute_for_ep: (expert, rank) -> (rank, expert).

    Args:
        counts: pre-computed int list in (expert, rank) order
                (length = num_local_experts * ep_degree)
    """
    chunks = list(torch.split(tokens[:sum(counts)], counts))
    reordered = []
    for r in range(ep_degree):
        for e in range(num_local_experts):
            reordered.append(chunks[e * ep_degree + r])
    return torch.cat(reordered, dim=0)

class GroupedExperts(nn.Module):
    """Expert weights stored as 3D tensors (num_experts, hidden_dim, dim).
    BMM forward for compiled training; for-loop fallback for eval.
    SM86 does not support torch._grouped_mm, so we use torch.bmm."""

    def __init__(self, dim: int, hidden_dim: int, num_experts: int):
        super().__init__()
        self.num_experts = num_experts
        self.w1 = nn.Parameter(torch.empty(num_experts, hidden_dim, dim))
        self.w2 = nn.Parameter(torch.empty(num_experts, dim, hidden_dim))
        self.w3 = nn.Parameter(torch.empty(num_experts, hidden_dim, dim))

    def forward(self, x: torch.Tensor, num_tokens_per_expert: torch.Tensor | None = None,
                *, _counts: list[int] | None = None) -> torch.Tensor:
        """Dual-mode forward:
        - BMM (training): x is (num_experts, capacity, dim), num_tokens_per_expert=None
        - For-loop (eval): x is (total_tokens, dim), num_tokens_per_expert is a tensor

        FSDP2 hooks fire in __call__ to unshard DTensor weights before this runs.
        The BMM path is compiled via OptimizedModule; the for-loop path runs eagerly
        through the original module (see MoE._eval_experts).
        """
        w1, w2, w3 = self.w1.to(x.dtype), self.w2.to(x.dtype), self.w3.to(x.dtype)

        if num_tokens_per_expert is not None:
            # For-loop path (eval — dynamic shapes, not compiled)
            num_tokens_list = _counts if _counts is not None else num_tokens_per_expert.int().tolist()
            total_assigned = sum(num_tokens_list)
            x_splits = torch.split(x[:total_assigned], num_tokens_list, dim=0)
            out_splits = []
            for i, x_expert in enumerate(x_splits):
                h = F.silu(x_expert @ w1[i].T) * (x_expert @ w3[i].T)
                out_splits.append(h @ w2[i].T)
            out = torch.cat(out_splits, dim=0)
            num_padding = x.shape[0] - total_assigned
            if num_padding > 0:
                out = torch.vstack((out, out.new_zeros((num_padding, out.shape[-1]))))
            return out

        # BMM path (training — static shapes, compiled)
        h = F.silu(torch.bmm(x, w1.transpose(1, 2))) * torch.bmm(x, w3.transpose(1, 2))
        return torch.bmm(h, w2.transpose(1, 2))


class TokenChoiceTopKRouter(nn.Module):
    """Token-choice top-K routing: each token selects its top-K experts."""

    def __init__(self, dim: int, num_experts: int, top_k: int,
                 score_func: str, route_norm: bool, route_scale: float,
                 aux_balance_coeff: float = 0.0, bias_before_score: bool = False):
        super().__init__()
        self.gate = nn.Linear(dim, num_experts, bias=False)
        self.num_experts = num_experts
        self.top_k = top_k
        self.score_func = score_func
        self.route_norm = route_norm
        self.route_scale = route_scale
        self.aux_balance_coeff = aux_balance_coeff
        self.bias_before_score = bias_before_score

    def forward(self, x: torch.Tensor, expert_bias: torch.Tensor = None):
        logits = self.gate(x)

        # Bias placement: before score_func shifts the sigmoid/softmax operating point
        if self.bias_before_score and expert_bias is not None:
            logits = logits + expert_bias

        if self.score_func == "sigmoid":
            scores = torch.sigmoid(logits.float())
        else:
            scores = F.softmax(logits.float(), dim=1)

        # Expert selection: pre-score bias already baked in, post-score adds here
        if self.bias_before_score or expert_bias is None:
            scores_for_choice = scores
        else:
            scores_for_choice = scores + expert_bias

        _, selected = torch.topk(scores_for_choice, k=self.top_k, dim=-1, sorted=False)
        top_scores = scores.gather(dim=1, index=selected)

        if self.route_norm:
            top_scores = top_scores / (top_scores.sum(dim=-1, keepdim=True) + 1e-20)
        top_scores = top_scores * self.route_scale

        num_tokens_per_expert = torch.histc(
            selected.view(-1).float(), bins=self.num_experts, min=0, max=self.num_experts,
        )

        # Aux balance loss: f_i * P_i encourages router to diversify
        aux_loss = None
        if self.aux_balance_coeff > 0 and self.training:
            N = x.shape[0]
            f_i = num_tokens_per_expert.detach() / (N * self.top_k)
            score_probs = scores / (scores.sum(dim=-1, keepdim=True) + 1e-20)
            P_i = score_probs.mean(dim=0)
            aux_loss = self.aux_balance_coeff * self.num_experts * (f_i * P_i).sum()

        return top_scores, selected, num_tokens_per_expert, aux_loss


def _scatter_to_padded(tokens: torch.Tensor, counts: list[int],
                       num_experts: int, capacity: int) -> torch.Tensor:
    """Scatter sorted flat tokens into (num_experts, capacity, dim) padded tensor.

    Fully differentiable — gradients flow through token slices back to input.
    Runs in uncompiled MoE.forward(); the Python loop is over num_experts (2-16).

    Args:
        tokens: (total_assigned, dim) — sorted by expert assignment
        counts: per-expert token counts (len = num_experts), all <= capacity
        num_experts: number of experts
        capacity: fixed capacity per expert (from capacity dropping)
    """
    dim = tokens.shape[-1]
    padded_list = []
    offset = 0
    for e in range(num_experts):
        n = counts[e]
        if n > 0:
            expert_tokens = tokens[offset:offset + n]
            if n < capacity:
                padded_list.append(torch.cat([
                    expert_tokens, expert_tokens.new_zeros(capacity - n, dim)
                ], dim=0))
            else:
                padded_list.append(expert_tokens)
        else:
            padded_list.append(tokens.new_zeros(capacity, dim))
        offset += n
    return torch.stack(padded_list, dim=0)


def _gather_from_padded(padded: torch.Tensor, counts: list[int]) -> torch.Tensor:
    """Gather real tokens from (num_experts, capacity, dim) padded tensor.

    Extracts the first counts[e] rows from each expert's padded slot.
    Fully differentiable — slicing + cat are standard autograd ops.
    """
    slices = []
    for e, n in enumerate(counts):
        if n > 0:
            slices.append(padded[e, :n])
    return torch.cat(slices, dim=0) if slices else padded.new_zeros(0, padded.shape[-1])


class MoE(nn.Module):
    """Mixture of Experts with token-choice routing and optional shared experts.

    Supports aux-loss-free load balancing via expert_bias buffer (updated externally).
    """

    def __init__(self, args: 'ModelArgs'):
        super().__init__()
        expert_hidden = args.moe_inner_dim or args.inner_dim or _compute_default_inner_dim(args.dim)

        # EP: each rank holds only its local experts
        self.num_experts = args.moe_num_experts
        self.ep_degree = getattr(args, 'ep_degree', 1)
        self.num_local_experts = self.num_experts // self.ep_degree

        self.experts = GroupedExperts(args.dim, expert_hidden, self.num_local_experts)
        self.router = TokenChoiceTopKRouter(
            args.dim, self.num_experts, args.moe_top_k,  # router still sees ALL experts
            args.moe_score_func, args.moe_route_norm, args.moe_route_scale,
            aux_balance_coeff=args.moe_aux_balance_coeff,
            bias_before_score=args.moe_bias_before_score,
        )
        self.shared_experts = (
            FeedForward(dim=args.dim, inner_dim=expert_hidden * args.moe_num_shared_experts, dropout=args.dropout)
            if args.moe_num_shared_experts > 0 else None
        )
        self.score_before_experts = args.moe_score_before_experts
        self.load_balance_coeff = args.moe_load_balance_coeff

        # Aux-loss-free load balancing buffers (always global num_experts)
        if self.load_balance_coeff is not None:
            self.register_buffer(
                "expert_bias", torch.zeros(self.num_experts, dtype=torch.float32), persistent=True,
            )
        else:
            self.expert_bias = None
        self.register_buffer(
            "tokens_per_expert", torch.zeros(self.num_experts, dtype=torch.float32), persistent=False,
        )

        # Aux balance loss stashed by forward() for Transformer to collect
        self._last_aux_loss = None
        # Capacity-based token dropping
        self.capacity_factor = args.moe_capacity_factor
        self._tokens_dropped_accum = 0  # accumulated across micro-batches, zeroed by balance hook

        # EP mesh — set externally via set_ep_mesh() before FSDP wrapping
        self._ep_mesh = None
        self._ep_group = None
        # Optional: overlap shared_experts with EP on a side CUDA stream
        self._shared_overlap = getattr(args, 'moe_shared_overlap', False)
        self._shared_stream: torch.cuda.Stream | None = None
        # BMM capacity for padded expert computation (computed on first training forward)
        self._bmm_capacity = None

    def set_ep_mesh(self, ep_mesh):
        """Attach EP mesh for all-to-all dispatch/combine. Call before FSDP wrapping."""
        self._ep_mesh = ep_mesh
        self._ep_group = ep_mesh.get_group()

    def _eval_experts(self, routed_input: torch.Tensor, num_tpe: torch.Tensor,
                      *, _counts: list[int] | None = None) -> torch.Tensor:
        """Call experts for eval — through original FSDP-wrapped module.

        Bypasses OptimizedModule to avoid compilation of the for-loop path,
        but still goes through __call__ to trigger FSDP2 unshard hooks.
        """
        orig = getattr(self.experts, '_orig_mod', self.experts)
        return orig(routed_input, num_tpe, _counts=_counts)

    def _experts_bmm(self, routed_input: torch.Tensor, counts: list[int]) -> torch.Tensor:
        """Scatter tokens → padded BMM → gather results (training path).

        Runs in uncompiled MoE.forward(). The scatter/gather use Python loops
        over num_experts (2-16 iterations). The BMM inside self.experts() is
        compiled with static shapes (num_experts, capacity, dim).
        """
        x_padded = _scatter_to_padded(routed_input, counts, len(counts), self._bmm_capacity)
        out_padded = self.experts(x_padded)
        return _gather_from_padded(out_padded, counts)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bs, slen, dim = x.shape
        x_flat = x.view(-1, dim)

        # Route tokens to experts (identical routing on all ranks — gate is replicated)
        top_scores, selected, num_tpe, aux_loss = self.router(x_flat, self.expert_bias)
        self._last_aux_loss = aux_loss
        # Balance-counter side effect runs EXACTLY ONCE per training forward
        # (audit 2026-07-11 #13): (a) training-gated — val forwards must not
        # feed the aux-loss-free bias signal; (b) recompute-gated — non-
        # reentrant activation checkpointing re-executes this forward inside
        # loss.backward(), which used to double every count. The forward MATH
        # is identical in both passes (required for correct grads); only the
        # side effect is skipped.
        if self.training and not _in_backward_recompute():
            with torch.no_grad():
                self.tokens_per_expert.add_(num_tpe)

        # ── Capacity-based token dropping (training only) ──
        if self.capacity_factor > 0 and self.training:
            N = x_flat.shape[0]
            capacity = max(1, math.ceil(
                self.capacity_factor * N * self.router.top_k / self.router.num_experts
            ))
            keep_mask = torch.ones_like(selected, dtype=torch.bool)
            flat_selected = selected.view(-1)
            flat_scores = top_scores.view(-1)
            for e in range(self.router.num_experts):
                expert_mask = (flat_selected == e)
                count = expert_mask.sum().item()
                if count > capacity:
                    expert_scores = flat_scores[expert_mask]
                    _, topk_idx = expert_scores.topk(capacity, sorted=False)
                    expert_positions = expert_mask.nonzero(as_tuple=True)[0]
                    drop = torch.ones(count, dtype=torch.bool, device=selected.device)
                    drop[topk_idx] = False
                    keep_mask.view(-1)[expert_positions[drop]] = False
            n_dropped = (~keep_mask).sum().item()
            if not _in_backward_recompute():  # once per step, like tokens_per_expert
                self._tokens_dropped_accum += n_dropped
            if n_dropped > 0:
                # A dropped slot simply loses its contribution (the residual
                # stream still carries the token) — matching every reference
                # that drops: Switch/GShard never rescale survivors, and
                # DeepSeek-V2's device-level dropping describes no compensation
                # (V3 drops nothing at all, which aux-loss-free balancing makes
                # attainable). The previous batch-GLOBAL rescale here
                # (sum_all/sum_keep applied to every kept slot) preserved the
                # batch's expected combine mass but was per-token biased:
                # tokens with NO dropped slots had their weights inflated by
                # OTHER tokens' drops — cross-token coupling that also broke
                # route_norm's per-token sum. Removed 2026-07-13 after
                # reference research (audit finding #7). Drop-rate telemetry:
                # _tokens_dropped_accum.
                top_scores = top_scores * keep_mask
                # Sentinel expert ID sorts dropped slots to the end
                selected = selected.clone()
                selected[~keep_mask] = self.router.num_experts
                # Recompute num_tpe for expert execution (post-drop)
                num_tpe = torch.histc(
                    selected.view(-1).float(),
                    bins=self.router.num_experts + 1,
                    min=0, max=self.router.num_experts + 1,
                )[:self.router.num_experts]
        # Reorder tokens by expert assignment
        token_indices_sorted = torch.argsort(selected.view(-1), stable=True)
        scores_sorted = top_scores.view(-1)[token_indices_sorted]

        routed_input = x_flat[token_indices_sorted // self.router.top_k]
        if self.score_before_experts:
            routed_input = (routed_input.float() * scores_sorted.unsqueeze(1)).to(x.dtype)

        # ── BMM capacity (computed once, cached for compile-stable shapes) ──
        if self._bmm_capacity is None and self.capacity_factor > 0 and self.training:
            N = x_flat.shape[0]
            per_rank_cap = max(1, math.ceil(
                self.capacity_factor * N * self.router.top_k / self.router.num_experts
            ))
            self._bmm_capacity = per_rank_cap * self.ep_degree
        use_bmm = self._bmm_capacity is not None and self.training

        # For EP with capacity dropping, truncate sentinel tokens before all-to-all
        # (all_to_all_single requires input size == sum of split sizes)
        n_total_slots = routed_input.shape[0]
        if self._ep_mesh is not None and self._tokens_dropped_accum > 0:
            total_assigned = num_tpe.sum().int().item()
            routed_input = routed_input[:total_assigned]

        if self._ep_mesh is not None and self._shared_overlap and self.shared_experts is not None:
            # ── Overlap shared experts with EP round-trip on a side CUDA stream ──
            if self._shared_stream is None:
                self._shared_stream = torch.cuda.Stream(device=x.device)
            self._shared_stream.wait_stream(torch.cuda.current_stream(x.device))
            with torch.cuda.stream(self._shared_stream):
                shared_out = self.shared_experts(x_flat)

            routed_input, local_num_tpe = self._ep_dispatch(routed_input, num_tpe)
            if use_bmm:
                routed_output = self._experts_bmm(routed_input, self._ep_local_counts)
            else:
                routed_output = self._eval_experts(routed_input, local_num_tpe, _counts=self._ep_local_counts)
            routed_output = self._ep_combine(routed_output)

            torch.cuda.current_stream(x.device).wait_stream(self._shared_stream)
        else:
            if self._ep_mesh is not None:
                # ── EP DISPATCH ──
                routed_input, local_num_tpe = self._ep_dispatch(routed_input, num_tpe)
                if use_bmm:
                    routed_output = self._experts_bmm(routed_input, self._ep_local_counts)
                else:
                    routed_output = self._eval_experts(routed_input, local_num_tpe, _counts=self._ep_local_counts)
                # ── EP COMBINE ──
                routed_output = self._ep_combine(routed_output)
            else:
                if use_bmm:
                    counts = num_tpe.int().tolist()
                    routed_output = self._experts_bmm(routed_input, counts)
                else:
                    routed_output = self._eval_experts(routed_input, num_tpe)

            shared_out = self.shared_experts(x_flat) if self.shared_experts is not None else None

        # Pad back to N*top_k after EP combine (sentinel slots get zeros)
        if routed_output.shape[0] < n_total_slots:
            routed_output = torch.cat([
                routed_output,
                routed_output.new_zeros(n_total_slots - routed_output.shape[0], dim),
            ])

        # Unsort back to original token positions (use x.dtype to guarantee bf16 output)
        out_unsorted = torch.zeros(
            bs * slen * self.router.top_k, dim,
            dtype=x.dtype, device=x.device,
        )
        out_unsorted[token_indices_sorted] = routed_output.to(x.dtype)
        out_unsorted = out_unsorted.reshape(-1, self.router.top_k, dim)

        if self.score_before_experts:
            out_experts = out_unsorted.sum(dim=1)
        else:
            out_experts = (
                torch.bmm(top_scores.unsqueeze(1).float(), out_unsorted.float())
                .to(x.dtype).squeeze(1)
            )

        if shared_out is not None:
            return (shared_out + out_experts).reshape(bs, slen, dim)
        return out_experts.reshape(bs, slen, dim)

    # ── Expert Parallel dispatch / combine ──

    def _ep_dispatch(self, routed_input, num_tpe):
        """Send tokens to the EP rank owning their assigned expert."""
        ep_degree = self.ep_degree
        num_local = self.num_local_experts

        with torch.no_grad():
            # Compute input_splits on GPU, async D2H (overlaps with count a2a)
            input_splits_gpu = num_tpe.reshape(ep_degree, num_local).sum(dim=1).int()
            input_splits_cpu = input_splits_gpu.to("cpu", non_blocking=True)

            # Count all-to-all (runs while input_splits transfers to CPU)
            num_tpe_received = torch.zeros_like(num_tpe)
            dist.all_to_all_single(num_tpe_received, num_tpe, group=self._ep_group)

            # Single GPU→CPU sync: get received counts as Python list
            rcv_counts = num_tpe_received.int().cpu().tolist()
            output_splits = [sum(rcv_counts[r * num_local:(r + 1) * num_local]) for r in range(ep_degree)]

            # Read input_splits (async D2H completed during count a2a)
            input_splits = input_splits_cpu.tolist()

        # Store for combine phase
        self._ep_input_splits = input_splits
        self._ep_output_splits = output_splits
        # Pre-compute unpermute counts (expert-major order) from rcv_counts (rank-major)
        self._ep_unpermute_counts = [rcv_counts[r * num_local + e]
                                     for e in range(num_local)
                                     for r in range(ep_degree)]

        # All-to-all tokens (autograd-aware)
        received = _AllToAllSingleAutograd.apply(
            routed_input, output_splits, input_splits, self._ep_group,
        )

        # Permute from (rank, expert) to (expert, rank) order — no GPU→CPU sync
        permuted, local_num_tpe, local_counts = _permute_for_ep(
            received, ep_degree, num_local, rcv_counts,
        )
        self._ep_local_counts = local_counts

        return permuted, local_num_tpe

    def _ep_combine(self, routed_output):
        """Send expert results back to the originating EP rank."""
        # Unpermute from (expert, rank) back to (rank, expert) order — no GPU→CPU sync
        unpermuted = _unpermute_for_ep(
            routed_output, self.ep_degree, self.num_local_experts,
            self._ep_unpermute_counts,
        )
        # All-to-all combine (reverse splits)
        result = _AllToAllSingleAutograd.apply(
            unpermuted, self._ep_input_splits, self._ep_output_splits, self._ep_group,
        )
        return result


class TransformerBlock(nn.Module):
    def __init__(self, layer_id: int, args: ModelArgs):
        super().__init__()
        self.n_heads = args.n_heads
        self.dim = args.dim
        self.head_dim = args.dim // args.n_heads
        self.layer_id = layer_id

        # GDN: conditionally replace Attention with GatedDeltaNet
        self.use_gdn = False
        if getattr(args, 'gdn_enabled', False):
            gdn_step = getattr(args, 'gdn_interleave_step', 4)
            _tail_global = getattr(args, 'gdn_n_tail_global', 1)
            self.use_gdn = (layer_id % gdn_step != gdn_step - 1) \
                and (layer_id < args.n_layers - _tail_global)

        # SWA hybrid: LOCAL (windowed) unless this is a global layer — same
        # interleave convention as GDN (every Nth layer, at step-1 offsets, is
        # the exception). False whenever swa is off.
        self.swa_local = False
        if getattr(args, 'swa_enabled', False):
            _swa_step = getattr(args, 'swa_global_interleave', 4)
            self.swa_local = (layer_id % _swa_step != _swa_step - 1)

        if self.use_gdn:
            _impl = getattr(args, 'gdn_impl', 'gdn')
            _try_import_gdn(_impl)
            n_gdn_heads = getattr(args, 'n_gdn_heads', None) or args.n_heads
            # Per-impl defaults track each paper: GDN 256/2.0, KDA (Kimi Linear
            # / K3) 128/1.0. Explicit config values always win.
            gdn_head_dim = getattr(args, 'gdn_head_dim', None) or (128 if _impl == 'kda' else 256)
            gdn_v_expand = getattr(args, 'gdn_v_expand', None) or (1.0 if _impl == 'kda' else 2.0)
            if _impl == 'kda':
                # layer_idx is provisional here — Transformer.__init__ rewrites
                # it to a COMPACTED index (see the fla-cache-idx pass) so a
                # shared FLA Cache stays densely indexed by delta-rule layers.
                self.gdn_attn = _KimiDeltaAttention(
                    hidden_size=args.dim,
                    num_heads=n_gdn_heads,
                    head_dim=gdn_head_dim,
                    expand_v=gdn_v_expand,
                    conv_size=getattr(args, 'gdn_short_conv_kernel', 4),
                    mode=getattr(args, 'gdn_mode', 'chunk'),
                    use_short_conv=True,
                    safe_gate=getattr(args, 'kda_safe_gate', True),
                    # lower_bound only participates in the safe_gate path; the
                    # Kimi-Linear negative-softplus path (safe_gate=False) is
                    # unbounded by construction.
                    lower_bound=(getattr(args, 'kda_lower_bound', -5.0)
                                 if getattr(args, 'kda_safe_gate', True) else None),
                    layer_idx=layer_id,
                    norm_eps=args.norm_eps,
                )
            else:
                self.gdn_attn = _GatedDeltaNet(
                    hidden_size=args.dim,
                    num_heads=n_gdn_heads,
                    head_dim=gdn_head_dim,
                    expand_v=gdn_v_expand,
                    conv_size=getattr(args, 'gdn_short_conv_kernel', 4),
                    mode=getattr(args, 'gdn_mode', 'chunk'),
                    use_gate=True,
                    use_short_conv=True,
                    layer_idx=layer_id,
                    norm_eps=args.norm_eps,
                )
        else:
            # Full attention (with gate if in GDN hybrid mode)
            use_gate = getattr(args, 'gdn_enabled', False)
            self.attention = Attention(args, use_gate=use_gate)

        # MoE: conditionally replace FeedForward with MoE module
        n_dense = getattr(args, 'moe_n_dense_layers', 0)
        n_tail_dense = getattr(args, 'moe_n_tail_dense_layers', 0)
        interleave = getattr(args, 'moe_interleave_step', 1)
        self.moe_enabled = (
            getattr(args, 'moe_enabled', False)
            and layer_id >= n_dense
            and layer_id < (args.n_layers - n_tail_dense)
            and (layer_id - n_dense) % interleave == 0
        )
        if self.moe_enabled:
            self.moe = MoE(args)
        else:
            self.feed_forward = FeedForward(
                dim=args.dim,
                inner_dim=args.inner_dim,
                dropout=args.dropout,
            )

        self.attention_norm = RMSNorm(args.dim, eps=args.norm_eps)
        self.ffn_norm = RMSNorm(args.dim, eps=args.norm_eps)
        self.use_activation_checkpointing = args.use_activation_checkpointing
        # Per-layer overrides, set by Transformer.__init__ (ac_skip_layers /
        # ac_input_offload / ac_offload). Defaults reproduce historical
        # behavior exactly: checkpoint iff the global flag, no offload.
        self.ac_checkpoint = args.use_activation_checkpointing
        # F9 op-level SAC (2026-08-17): selective policy inside the block
        # checkpoint — save named-expensive outputs (flex attention), recompute
        # the cheap tissue. None = plain checkpoint, byte-identical legacy.
        self.ac_sac_policy = getattr(args, 'ac_sac_policy', None)
        self._act_offloader = None

        # KEEL: Highway-style Post-LN configuration
        self.use_keel = getattr(args, 'use_keel', False)
        if self.use_keel:
            self.keel_alpha = getattr(args, 'keel_alpha', None) or (args.n_layers * 2)
            # Post-LN layers (only for layer_id > 0; first block stays Pre-LN)
            if layer_id > 0:
                self.post_attn_norm = RMSNorm(args.dim, eps=args.norm_eps)
                self.post_ffn_norm = RMSNorm(args.dim, eps=args.norm_eps)

    def _ffn(self, x):
        """Route through MoE or dense FeedForward."""
        return self.moe(x) if self.moe_enabled else self.feed_forward(x)

    def _attn(self, x, freqs_cos, freqs_sin, block_mask=None, doc_cu=None):
        """Route through GDN or softmax attention."""
        if self.use_gdn:
            if doc_cu is not None:
                # Doc-confined training: FLA varlen resets the recurrent state
                # (and the ShortConv receptive field) at every cu boundary —
                # the delta-rule analog of the softmax layers' BlockMask.
                # flash-varlen convention: batch flattened to [1, B*S].
                B, S, D = x.shape
                out, *_ = self.gdn_attn(x.reshape(1, B * S, D), cu_seqlens=doc_cu)
                return out.reshape(B, S, D)
            out, *_ = self.gdn_attn(x)
            return out
        if isinstance(block_mask, tuple):
            # SWA hybrid: (global_mask, local_mask) — pick by this layer's kind.
            block_mask = block_mask[1] if self.swa_local else block_mask[0]
        return self.attention(x, freqs_cos, freqs_sin, block_mask)

    def _forward_block(self, x, freqs_cos, freqs_sin, block_mask=None, doc_cu=None):
        """Inner forward for activation checkpointing."""
        if self.use_keel:
            if self.layer_id == 0:
                # First block: standard Pre-LN (no Post-LN, no alpha scaling)
                h = x + self._attn(self.attention_norm(x), freqs_cos, freqs_sin, block_mask, doc_cu)
                out = h + self._ffn(self.ffn_norm(h))
            else:
                # KEEL: x_{l+1} = LN(alpha * x_l + F_l(LN(x_l)))
                attn_out = self._attn(self.attention_norm(x), freqs_cos, freqs_sin, block_mask, doc_cu)
                h = self.post_attn_norm(self.keel_alpha * x + attn_out)
                ffn_out = self._ffn(self.ffn_norm(h))
                out = self.post_ffn_norm(self.keel_alpha * h + ffn_out)
        else:
            # Original Pre-LN path (unchanged)
            h = x + self._attn(self.attention_norm(x), freqs_cos, freqs_sin, block_mask, doc_cu)
            out = h + self._ffn(self.ffn_norm(h))
        return out

    def forward(self, x, freqs_cos, freqs_sin, block_mask=None, doc_cu=None):
        """
        TRAINING PATH - Uses activation checkpointing when enabled.
        """
        _off = self._act_offloader if self.training else None
        with (_off.hooks() if _off is not None else nullcontext()):
            if self.ac_checkpoint and self.training:
                # Non-reentrant checkpoint passes non-tensor args (BlockMask) through
                # to the recompute untouched; its int tensors need no grad tracking.
                # Under an offload context, only the checkpoint's boundary saves ride
                # to host (the checkpoint's own inner hooks are innermost and still
                # discard intermediates — Pool A).
                _sac_kw = {}
                if self.ac_sac_policy:
                    _sac_kw['context_fn'] = _make_sac_context_fn(self.ac_sac_policy)
                out = cp.checkpoint(self._forward_block, x, freqs_cos, freqs_sin, block_mask,
                                    doc_cu, use_reentrant=False, **_sac_kw)
            else:
                # No checkpoint: every saved intermediate hits the hooks (Pool B
                # when an offloader is attached; plain VRAM spend otherwise).
                out = self._forward_block(x, freqs_cos, freqs_sin, block_mask, doc_cu)
        return out

    def forward_with_cache(self, x, freqs_cos, freqs_sin, start_pos: int, fla_cache=None):
        """
        INFERENCE PATH - No checkpointing, uses KV cache.
        GDN/KDA layers thread their recurrent state through the shared FLA
        Cache owned by Transformer.generate_forward (audit 2026-07-11 #2: the
        old path discarded it, so every decode step after prefill saw a fresh
        length-1 sequence).
        """
        # Move input to this block's device (for multi-GPU sharded models)
        device = next(self.parameters()).device
        if x.device != device:
            x = x.to(device)

        if self.use_gdn:
            attn_out, _, _ = self.gdn_attn(
                self.attention_norm(x),
                past_key_values=fla_cache,
                use_cache=fla_cache is not None,
            )
        else:
            attn_out = self.attention.forward_with_cache(self.attention_norm(x), freqs_cos, freqs_sin, start_pos)

        if self.use_keel:
            if self.layer_id == 0:
                h = x + attn_out
                out = h + self._ffn(self.ffn_norm(h))
            else:
                h = self.post_attn_norm(self.keel_alpha * x + attn_out)
                ffn_out = self._ffn(self.ffn_norm(h))
                out = self.post_ffn_norm(self.keel_alpha * h + ffn_out)
        else:
            h = x + attn_out
            out = h + self._ffn(self.ffn_norm(h))
        return out

class MTPModule(nn.Module):
    """DeepSeek-V3-style sequential multi-token-prediction module (depth 1).

    Predicts t+2: h'_i = Block(proj([RMSNorm(h_i); RMSNorm(Emb(t_{i+1}))])),
    read out through the SHARED final norm + output head (the caller applies
    them). Trunk gradient flows through h_i (no detach) — that flow IS the
    training signal MTP exists to add. Never called at inference/generation;
    reusable later for speculative decoding."""

    def __init__(self, args: ModelArgs):
        super().__init__()
        self.h_norm = RMSNorm(args.dim, eps=args.norm_eps)
        self.emb_norm = RMSNorm(args.dim, eps=args.norm_eps)
        self.proj = nn.Linear(2 * args.dim, args.dim, bias=False)
        self.block = TransformerBlock(args.n_layers, args)
        # under SWA the MTP block rides the GLOBAL mask (it is a readout head;
        # capping its view at the window would starve the long-range signal)
        self.block.swa_local = False

    def forward(self, h, next_emb, freqs_cos, freqs_sin, block_mask=None):
        x = self.proj(torch.cat([self.h_norm(h), self.emb_norm(next_emb)], dim=-1))
        return self.block(x, freqs_cos, freqs_sin, block_mask)


class Transformer(nn.Module):
    """
    Dense Transformer with isolated training/inference paths.
    
    Training: model(tokens, targets=targets) -> (None, loss)
              model(tokens) -> (logits, None)
    
    Inference with KV cache:
        model.setup_caches(batch_size, max_seq_len)
        logits = model.generate_forward(prompt_tokens, start_pos=0)
        logits = model.generate_forward(next_token, start_pos=prompt_len)
        model.clear_caches()
    """
    last_loss: Optional[torch.Tensor]

    def __init__(self, params: ModelArgs):
        super().__init__()
        self.params = params
        # doc-mask/pos-reset/mtp-boundary-mask key on token==bos_token_id; the
        # -1 "not set" sentinel would silently never match (the feature becomes
        # a no-op), so refuse to build a model that needs the id but wasn't
        # given one.
        if (params.doc_attn_mask or params.doc_pos_reset
                or getattr(params, 'mtp_doc_boundary_mask', False)) \
                and params.bos_token_id < 0:
            raise ValueError(
                "doc_attn_mask/doc_pos_reset/mtp_doc_boundary_mask require an explicit "
                "bos_token_id (ModelArgs defaults to the -1 'not set' sentinel; there is "
                "no safe universal document-separator id — use tokenizer.bos_id)")
        self.vocab_size = params.vocab_size
        self.n_layers = params.n_layers
        
        self.tok_embeddings = nn.Embedding(params.vocab_size, params.dim)
        self.dropout = nn.Dropout(params.dropout)
        
        self.layers = nn.ModuleList()
        for layer_id in range(params.n_layers):
            self.layers.append(TransformerBlock(layer_id, params))

        # ── Activation-memory levers (see ModelArgs comments; default-off =
        # historical behavior). Skip set is evenly striped over depth so the
        # recompute relief and the offload traffic spread across the forward
        # instead of clumping. One shared offloader -> one copy stream per
        # device, so D2H/H2D traffic serializes in issue order. ──
        _n_skip = int(getattr(params, 'ac_skip_layers', 0) or 0)
        _in_off = bool(getattr(params, 'ac_input_offload', False))
        _sk_off = bool(getattr(params, 'ac_offload', False))
        if _n_skip > 0 or _in_off:
            _L = len(self.layers)
            if _n_skip > _L:
                raise ValueError(f"ac_skip_layers={_n_skip} > n_layers={_L}")
            _skip = set()
            if _n_skip > 0:
                _skip = {int(round((k + 0.5) * _L / _n_skip)) for k in range(_n_skip)}
                _skip = {min(i, _L - 1) for i in _skip}
            _off = _ActOffloader(int(getattr(params, 'ac_offload_min_mb', 8)) * 1024 * 1024) \
                if (_in_off or _sk_off) else None
            # Partial Pool A: stripe the offload over N blocks (0 = all).
            _n_inoff = int(getattr(params, 'ac_input_offload_layers', 0) or 0)
            if _n_inoff > _L:
                raise ValueError(f"ac_input_offload_layers={_n_inoff} > n_layers={_L}")
            _inoff_set = None
            if _in_off and _n_inoff > 0:
                _inoff_set = {min(int(round((k + 0.5) * _L / _n_inoff)), _L - 1)
                              for k in range(_n_inoff)}
            for _i, _blk in enumerate(self.layers):
                if _i in _skip:
                    _blk.ac_checkpoint = False
                    if _sk_off:
                        _blk._act_offloader = _off
                elif _in_off and _blk.ac_checkpoint and (_inoff_set is None or _i in _inoff_set):
                    _blk._act_offloader = _off
            self._ac_skip_set = _skip  # boot banner reads this
            self._ac_inoff_count = (len(_inoff_set) if _inoff_set is not None
                                    else (_L - len(_skip) if _in_off else 0))
        else:
            self._ac_skip_set = set()

        # Delta-rule decode state (audit 2026-07-11 #2): one shared FLA Cache
        # per generation threads each GDN/KDA layer's recurrent state across
        # decode steps. FLA's cache helpers index by layer_idx and assume DENSE
        # indices, but in the hybrid only some layers are delta-rule — so
        # rewrite each delta layer's layer_idx to a compacted 0..K-1 sequence.
        _fla_idx = 0
        for _blk in self.layers:
            if getattr(_blk, 'use_gdn', False):
                _blk.gdn_attn.layer_idx = _fla_idx
                _fla_idx += 1
        self._fla_cache = None      # created at prefill (start_pos == 0)
        self._fla_cache_pos = 0     # next position the recurrent state expects

        self.norm = RMSNorm(params.dim, eps=params.norm_eps)
        self.output = nn.Linear(params.dim, params.vocab_size, bias=False)

        # AttnRes: per-layer pseudo-queries + key norms live on Transformer (NOT
        # TransformerBlock) so they're outside the per-layer fully_shard() boundary.
        # The root fully_shard(reshard_after_forward=False) unshards them at the
        # start of forward(), making them available as plain tensors in the loop.
        self.attn_res_enabled = getattr(params, 'attn_res_enabled', False)
        if self.attn_res_enabled:
            self.attn_res_block_size = getattr(params, 'attn_res_block_size', 8)
            self.attn_res_queries = nn.ParameterList([
                nn.Parameter(torch.zeros(params.dim)) for _ in range(params.n_layers)
            ])
            self.attn_res_key_norms = nn.ModuleList([
                RMSNorm(params.dim, eps=params.norm_eps) for _ in range(params.n_layers)
            ])

        # Auxiliary prediction heads at configured intermediate depths.
        # Each tap reads block-output activations during forward and produces
        # its own next-token loss against shifted targets. Per-head weighting
        # is applied by the trainer, not the model.
        self.aux_head_layers: List[int] = sorted(set(getattr(params, 'aux_head_layers', []) or []))
        for _li in self.aux_head_layers:
            if _li < 0 or _li >= params.n_layers:
                raise ValueError(
                    f"aux_head_layers entry {_li} is out of range for n_layers={params.n_layers}"
                )
        self.aux_heads = nn.ModuleDict({
            str(li): AuxHead(params.dim, params.vocab_size, params.norm_eps)
            for li in self.aux_head_layers
        })
        # Set-form for O(1) membership tests inside the forward loop
        self._aux_head_layer_set: set = set(self.aux_head_layers)
        self._last_aux_loss_tensors: dict = {}
        # Z-loss (confidence penalty on logsumexp). Disabled by default; the
        # trainer sets self._zloss_fp32_accum post-build when settings.z_loss is
        # enabled:
        #     None  -> z-loss OFF; loss path byte-for-byte identical to baseline.
        #     False -> z-loss ON, backend='bf16'       (option D, bf16 recon).
        #     True  -> z-loss ON, backend='fp32_accum' (option D, fp32 accum).
        # Stashes mirror _last_aux_loss_tensors: per-head dicts for aux heads,
        # scalars for the main head. The trainer selects whichever matches the
        # live readout (main head normally, deepest aux tap under SCS scaffold).
        self._zloss_fp32_accum = None    # None=off | False=bf16 | True=fp32_accum
        self._last_zloss = None          # main-head raw zloss = mean(logZ**2)
        self._last_logz = None           # main-head logZ_mean = mean(logZ)
        self._last_logz_rms = None       # main-head logZ rms = sqrt(mean logZ**2)
        self._last_logz_p95 = None       # main-head logZ 95th pctile (tail)
        # dn4 Lever 2: deadband CENTERED z-loss (target='centered'). 'raw' = today's
        # mean(logZ**2); 'centered' = mean(relu(logZ_c - tau)**2), gauge-invariant.
        self._zloss_target = 'raw'       # 'raw' | 'centered'  (trainer sets post-build)
        self._zloss_tau = 0.0            # deadband ceiling on logZ_c (centered only)
        self._last_logZ_c = None         # centered log-partition mean (telemetry)
        self._last_h_mu = None            # common-mode gauge magnitude h.mu mean (telemetry)
        self._last_aux_zloss: dict = {}  # per-aux-head raw zloss
        self._last_aux_logz: dict = {}   # per-aux-head logZ_mean

        # Optional weight tying
        self.tie_word_embeddings = getattr(params, "tie_word_embeddings", True)
        if self.tie_word_embeddings:
            self.output.weight = self.tok_embeddings.weight

        # Precompute positional tables (mode-aware: rope | nope | envelope)
        freqs_cos, freqs_sin = compute_rope_tables(
            self.params.dim // self.params.n_heads,
            self.params.max_seq_len,
            self.params.rope_theta,
            getattr(self.params, 'rope_mode', 'rope'),
        )
        self.register_buffer("freqs_cos", freqs_cos, persistent=False)
        self.register_buffer("freqs_sin", freqs_sin, persistent=False)

        # Initialize weights
        self.apply(self._init_weights)
        output_std = 0.02 / math.sqrt(2 * params.n_layers)
        for pn, p in self.named_parameters():
            if pn.endswith('w3.weight') or pn.endswith('wo.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=output_std)
            # GDN output projection: scaled init like wo
            elif '.gdn_attn.o_proj.weight' in pn:
                torch.nn.init.normal_(p, mean=0.0, std=output_std)
            # Expert weights (3D nn.Parameter, no .weight suffix — not hit by _init_weights).
            # Convention matches THIS codebase's dense FFN (w1,w2 @ 0.02; only w3
            # depth-scaled), NOT torchtitan's (w2 AND w3 scaled). We originally copied
            # torchtitan's expert init verbatim, which under our dense convention left
            # routed experts ~sqrt(2*n_layers)x weaker at boot than the shared expert
            # in the same layer (audit 2026-07-11 #14; ruled unintentional 2026-07-13).
            elif '.experts.w1' in pn or '.experts.w2' in pn:
                torch.nn.init.normal_(p, mean=0.0, std=0.02)
            elif '.experts.w3' in pn:
                torch.nn.init.normal_(p, mean=0.0, std=output_std)

        # MTP (festival feature 3): constructed AND initialized LAST — strictly
        # after every trunk draw in BOTH init passes above — so adding/removing
        # MTP cannot shift trunk initialization on any construction path (the
        # paired-arm property, pinned by the t_mtp parity test). Module
        # CONSTRUCTION itself consumes RNG (Linear kaiming init), hence the
        # strict ordering rather than registration tricks. Loss weight lives
        # trainer-side (z-loss pattern).
        if getattr(params, 'mtp_enabled', False):
            # mtp × gdn SUPPORTED (2026-08-03, Wizard102 prep): the historical
            # blocker was the MTP block's layer_id (= n_layers) landing on the
            # GDN interleave and becoming recurrent. The gdn_n_tail_global
            # condition (use_gdn &= layer_id < n_layers - tail) can never be
            # true at layer_id = n_layers, so the readout head is structurally
            # full-attention now. The assert pins that invariant against any
            # future refactor of the interleave logic. NOTE: self-speculative
            # decode with a delta trunk needs recurrent-state snapshot/rollback
            # (spec rewinds; KDA state cannot seek) — handled in neo_common's
            # spec engine, gated there.
            self.mtp = MTPModule(params)
            assert not getattr(self.mtp.block, 'use_gdn', False), \
                "MTP readout block must be full attention (interleave refactor broke the tail-global invariant?)"
            self.mtp.apply(self._init_weights)
            for _pn, _p in self.mtp.named_parameters():
                if _pn.endswith('w3.weight') or _pn.endswith('wo.weight'):
                    torch.nn.init.normal_(_p, mean=0.0, std=output_std)
        else:
            self.mtp = None
        self._last_mtp_loss = None
        # Speculative decoding: number of positions materialized in the MTP
        # block's KV cache (maintained by the spec decode path; the trunk
        # ledger alone can't tell whether the mtp cache is in lockstep —
        # e.g. after a classic-decode turn on the same cache).
        self._mtp_cache_len = 0

        self.last_loss = None

        # doc-mask feature guards: fail at construction, not mid-training.
        if (self.params.doc_attn_mask or self.params.doc_pos_reset):
            if self.params.doc_attn_mask and flex_attention is None:
                raise RuntimeError(
                    "doc_attn_mask requires torch >= 2.5 (torch.nn.attention.flex_attention)")
            # doc_attn_mask × delta-rule is SUPPORTED (2026-07-31): KDA/GDN
            # layers reset state at BOS via FLA varlen cu_seqlens — see
            # docs/KDA_VARLEN_DOC_RESET.md. The trainer capability-gates the
            # installed fla kernels. doc_pos_reset stays fatal: per-doc RoPE
            # would apply to the softmax layers only, an asymmetric geometry
            # no arm has validated.
            if self.params.doc_pos_reset and getattr(self.params, 'gdn_enabled', False):
                raise RuntimeError(
                    "doc_pos_reset is not supported with gdn_enabled (per-doc RoPE "
                    "positions would affect only the hybrid's softmax layers)")
        if getattr(self.params, 'swa_enabled', False):
            if flex_attention is None:
                raise RuntimeError(
                    "swa_enabled requires torch >= 2.5 (torch.nn.attention.flex_attention)")
            if getattr(self.params, 'gdn_enabled', False):
                raise RuntimeError(
                    "swa_enabled is not supported with gdn_enabled (GDN layers have no "
                    "windowed-attention semantics; the hybrid patterns would collide)")
            if self.params.swa_window >= self.params.max_seq_len:
                logger_warn = f"swa_window ({self.params.swa_window}) >= max_seq_len " \
                              f"({self.params.max_seq_len}) — local layers degenerate to full causal"
                print(logger_warn)

    def _build_block_mask(self, tokens: torch.Tensor, doc: bool, window=None):
        """FlexAttention BlockMask: causal AND (same-document?) AND (window?).

        Head dim is broadcast (H=None) — one mask serves every layer and head of
        its kind. Runs eagerly (see forward). Cost is NOT free: eager
        create_block_mask materializes a full-resolution O(B·S²) boolean grid
        before reducing to block granularity (~500MB transient at B=12/S=2048,
        ~3-4ms, largely CPU-bound). Amortized over every layer sharing the mask
        it is well worth it, and the no-BOS dispatch in forward skips the
        doc-only variant entirely for boundary-free micro-batches — but do not
        call it per-layer."""
        doc_ids = doc_ids_from_tokens(tokens, self.params.bos_token_id) if doc else None

        def mask_mod(b, h, q_idx, kv_idx):
            m = q_idx >= kv_idx
            if window is not None:
                m = m & (q_idx - kv_idx < window)
            if doc_ids is not None:
                m = m & (doc_ids[b, q_idx] == doc_ids[b, kv_idx])
            return m

        B, S = tokens.shape
        return create_block_mask(mask_mod, B=B, H=None, Q_LEN=S, KV_LEN=S,
                                 device=tokens.device)

    def _build_doc_block_mask(self, tokens: torch.Tensor):
        """Back-compat wrapper: causal AND same-document (no window)."""
        return self._build_block_mask(tokens, doc=True, window=None)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def init_weights(self):
        """
        Initialize weights for meta-device workflow.

        Call this AFTER:
        1. Creating model on meta device
        2. Applying FSDP2 sharding (fully_shard)
        3. Materializing with to_empty(device)

        DTensor's RNG tracker will ensure consistent initialization
        across sharded ranks when using nn.init functions.
        """
        # Standard deviation for output projections (scaled by depth)
        output_std = 0.02 / math.sqrt(2 * self.params.n_layers)

        for name, module in self.named_modules():
            # MTP inits in its own pass BELOW — strictly after every trunk draw
            # (both walks), so the DTensor RNG tracker stream the trunk consumes
            # is identical with or without MTP (paired-arm property).
            if name == 'mtp' or name.startswith('mtp.'):
                continue
            if isinstance(module, nn.Linear):
                # Use trunc_normal_ like TorchTitan for better stability
                torch.nn.init.trunc_normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    torch.nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                torch.nn.init.trunc_normal_(module.weight, mean=0.0, std=0.02)
            elif isinstance(module, RMSNorm):
                # RMSNorm weight should be ones
                if hasattr(module, 'weight') and module.weight is not None:
                    torch.nn.init.ones_(module.weight)
            elif isinstance(module, GroupedExperts):
                # 3D expert weights — same convention as the dense FFN here
                # (w1,w2 @ 0.02; only w3 depth-scaled). See the construction-path
                # comment in __init__ for the torchtitan-lineage history.
                torch.nn.init.trunc_normal_(module.w1, mean=0.0, std=0.02)
                torch.nn.init.trunc_normal_(module.w2, mean=0.0, std=0.02)
                torch.nn.init.trunc_normal_(module.w3, mean=0.0, std=output_std)
            elif isinstance(module, MoE):
                # Re-init buffers on correct device after materialization
                if module.load_balance_coeff is not None:
                    module.expert_bias.zero_()
                module.tokens_per_expert.zero_()
            elif isinstance(module, nn.Conv1d):
                # GDN short-conv: FLA ShortConvolution subclasses Conv1d, and
                # the inherited kaiming-uniform reset IS its construction init.
                module.reset_parameters()
            elif _GatedDeltaNet is not None and isinstance(module, _GatedDeltaNet):
                # FLA VALUE-constructs A_log/dt_bias in __init__ (no nn.init),
                # so the meta-init to_empty() flow leaves them uninitialized-
                # garbage unless we replay the recipe here (RoPE-freqs bug
                # class; see the freqs comment below). Recipe verified against
                # fla/layers/gated_deltanet.py 2026-07-11:
                #   A_log   = log(U(0, 16))
                #   dt_bias = inv_softplus(clamp(exp(U(ln 1e-3, ln 1e-1)), min=1e-4))
                # Child Linears/convs/norms re-init via their own branches.
                with torch.no_grad():
                    module.A_log.uniform_(0.0, 16.0).log_()
                    _db = module.dt_bias
                    _db.uniform_(math.log(1e-3), math.log(1e-1)).exp_().clamp_(min=1e-4)
                    _db.add_(torch.log(-torch.expm1(-_db)))
            elif _KimiDeltaAttention is not None and isinstance(module, _KimiDeltaAttention):
                # KDA (gdn_impl='kda'): same to_empty() hazard class. Recipe
                # verified against fla/layers/kda.py 2026-07-30:
                #   safe_gate: A_log = zeros (per-head log-scale, learned)
                #   else:     A_log = log(U(1, 16))   (Kimi-Linear/GDN lineage)
                #   dt_bias  = inv_softplus(clamp(exp(U(ln 1e-3, ln 1e-1)), min=1e-4))
                # Child Linears/convs/norms re-init via their own branches.
                _safe = getattr(module, 'safe_gate', getattr(self.params, 'kda_safe_gate', True))
                with torch.no_grad():
                    if _safe:
                        module.A_log.zero_()
                    else:
                        module.A_log.uniform_(1.0, 16.0).log_()
                    _db = module.dt_bias
                    _db.uniform_(math.log(1e-3), math.log(1e-1)).exp_().clamp_(min=1e-4)
                    _db.add_(torch.log(-torch.expm1(-_db)))
            elif module.__class__.__name__ in ('FusedRMSNormGated', 'FusedRMSNormSwishGate') \
                    and hasattr(module, 'reset_parameters'):
                # GDN o_norm: FLA's gated norm class, invisible to the
                # isinstance(RMSNorm) branch above. Its reset_parameters is
                # ones-init.
                module.reset_parameters()
            elif isinstance(module, Attention) and hasattr(module, 'mask'):
                # torch<2.0 manual-path causal mask buffer (registered only when
                # neither flash-attn nor SDPA exists) — same to_empty() hazard
                # class as the RoPE freqs. Unreachable on any FSDP2-capable
                # torch; recomputed for meta-init completeness.
                with torch.no_grad():
                    _mask = torch.full_like(module.mask, float("-inf"))
                    module.mask.copy_(torch.triu(_mask, diagonal=1))

        # Apply scaled initialization to output projections (w3, wo, GDN o_proj)
        # These benefit from smaller init to prevent output explosion in deep nets
        for name, param in self.named_parameters():
            if name.startswith('mtp.'):
                continue  # mtp pass below (after ALL trunk draws)
            if name.endswith('w3.weight') or name.endswith('wo.weight'):
                torch.nn.init.trunc_normal_(param, mean=0.0, std=output_std)
            elif '.gdn_attn.o_proj.weight' in name:
                torch.nn.init.trunc_normal_(param, mean=0.0, std=output_std)

        # AttnRes pseudo-queries: ZERO by design (uniform depth-mixing at boot).
        # Plain Parameters in a ParameterList — no module branch above touches
        # them, so without this they'd stay to_empty() garbage. zeros_ draws no
        # RNG, so the paired-arm draw ordering is unaffected.
        if getattr(self, 'attn_res_enabled', False):
            for _q in self.attn_res_queries:
                torch.nn.init.zeros_(_q)

        # MTP pass — LAST, after every trunk draw in both loops above, so trunk
        # init is bit-identical with/without MTP under the same seed (any RNG
        # the mtp draws consume shifts only what comes after, which is nothing).
        if self.mtp is not None:
            for _name, _module in self.mtp.named_modules():
                if isinstance(_module, nn.Linear):
                    torch.nn.init.trunc_normal_(_module.weight, mean=0.0, std=0.02)
                    if _module.bias is not None:
                        torch.nn.init.zeros_(_module.bias)
                elif isinstance(_module, RMSNorm):
                    if hasattr(_module, 'weight') and _module.weight is not None:
                        torch.nn.init.ones_(_module.weight)
            for _pn, _p in self.mtp.named_parameters():
                if _pn.endswith('w3.weight') or _pn.endswith('wo.weight'):
                    torch.nn.init.trunc_normal_(_p, mean=0.0, std=output_std)

        # RoPE tables MUST be recomputed here (torchtitan pattern): in the meta-init
        # flow, to_empty() replaces the value-carrying freqs buffers with uninitialized
        # storage and — being persistent=False — nothing else ever refills them. The CUDA
        # caching allocator deterministically leaves cos=fresh-zero-pages and sin=the
        # recycled cos block, i.e. attention scores degrade to a separable cos-envelope
        # instead of relative rotation (found 2026-07-02; every prior meta-init FSDP2 run
        # trained under that corruption). copy_() is a no-op when values are already
        # correct, so non-meta construction paths are unaffected. Mode-aware:
        # a nope/envelope arm must refill its OWN intended tables here, not
        # standard RoPE (the [freqs-check] rail verifies against the same fn).
        fc, fs = compute_rope_tables(
            self.params.dim // self.params.n_heads,
            self.params.max_seq_len,
            self.params.rope_theta,
            getattr(self.params, 'rope_mode', 'rope'),
        )
        with torch.no_grad():
            self.freqs_cos.copy_(fc)
            self.freqs_sin.copy_(fs)

    # =========================================================================
    # KV Cache Management (for inference only)
    # =========================================================================
    
    def setup_caches(self, max_batch_size: int, max_seq_len: int, force: bool = False):
        """
        Allocate KV caches for all layers.
        Must be called before using generate_forward().

        Args:
            max_batch_size: Maximum batch size for generation
            max_seq_len: Maximum sequence length (prompt + generated tokens)
            force: If False (default) and caches are already allocated at a
                   size >= (max_batch_size, max_seq_len) AND on the expected
                   device/dtype, this is a no-op so the existing allocation
                   (and its contents) survive. This is what lets cross-turn
                   prefix reuse keep the same cache tensors across generations.
                   Pass force=True to always reallocate (zero-fresh caches).

        Reallocation (force=True, or growing, or a device/dtype change) resets
        the cache token ledger (`cache_token_ids`) because the prior contents
        no longer describe a known token sequence.

        Raises:
            ValueError: if max_seq_len exceeds the model's trained max_seq_len.
                The RoPE freqs tables (freqs_cos/freqs_sin) are precomputed to
                exactly params.max_seq_len; positions beyond that have no
                rotary embedding, so generating there would silently misalign
                RoPE. Callers must keep context_size <= params.max_seq_len.
        """
        trained_max = self.params.max_seq_len
        if max_seq_len > trained_max:
            raise ValueError(
                f"setup_caches(max_seq_len={max_seq_len}) exceeds the model's "
                f"trained max_seq_len={trained_max}. RoPE frequencies are only "
                f"precomputed to {trained_max} positions; generating beyond that "
                f"would silently misalign RoPE. Reduce context_size to "
                f"<= {trained_max}."
            )
        # SWA note: local layers use ROLLING WINDOW caches (min(max_seq_len, W)
        # slots) — long generation is training-faithful (windowed) at any length,
        # and local-layer decode cost is bounded by W. See the per-layer sizing
        # in the allocation loop below.

        n_kv_heads = self.params.n_heads if self.params.n_kv_heads is None else self.params.n_kv_heads
        head_dim = self.params.dim // self.params.n_heads

        # Idempotent fast path: keep the existing allocation if it's big enough
        # AND on the right device/dtype. The cached K/V contents (and the token
        # ledger that describes them) are preserved, which is what cross-turn
        # prefix reuse relies on. A device/dtype mismatch must NOT no-op: the
        # cached tensors would be unusable / wrong-precision.
        same_devdtype = False
        if not force and self.has_caches():
            cur_bsz, cur_len = self.cache_capacity()
            # Check device/dtype of the first non-GDN cache against weights.
            for layer in self.layers:
                if getattr(layer, 'use_gdn', False):
                    continue
                ck = layer.attention.cache_k
                w = layer.attention.wq.weight
                same_devdtype = (ck is not None and ck.device == w.device
                                 and ck.dtype == w.dtype)
                break
            if (cur_bsz is not None and cur_bsz >= max_batch_size
                    and cur_len >= max_seq_len and same_devdtype):
                return  # existing allocation already big enough → reuse as-is

        # Decide whether we can GROW IN PLACE while preserving contents. This is
        # the difference between a block-boundary crossing costing a cheap copy
        # vs. a full re-prefill of the whole conversation. Growth is safe to
        # preserve iff: not forced, a cache already exists, same device/dtype,
        # same batch size, and we are only EXTENDING the seq_len dimension. K/V
        # at positions [0, old_len) are position-absolute (RoPE baked in), so
        # copying them into the front of the larger buffer keeps them exact.
        can_preserve = False
        old_len = None
        if not force and self.has_caches() and same_devdtype:
            cur_bsz, cur_len = self.cache_capacity()
            if (cur_bsz is not None and cur_bsz == max_batch_size
                    and cur_len < max_seq_len):
                can_preserve = True
                old_len = cur_len

        _swa_on = getattr(self.params, 'swa_enabled', False)
        _swa_W = int(getattr(self.params, 'swa_window', 0)) if _swa_on else None
        for layer in self.layers:
            if getattr(layer, 'use_gdn', False):
                continue  # GDN layers have no KV cache
            # Get device and dtype from layer weights
            device = layer.attention.wq.weight.device
            dtype = layer.attention.wq.weight.dtype

            # SWA rolling cache: LOCAL layers hold min(max_seq_len, W) slots
            # (slot = pos % Lc); global layers keep the full length. This cuts
            # KV memory to ~1/3 on the wizard-era hybrids AND makes long
            # generation training-faithful (windowed) instead of full-cache OOD.
            _local = _swa_on and getattr(layer, 'swa_local', False)
            layer.attention.cache_window = _swa_W if _local else None
            len_l = min(max_seq_len, _swa_W) if _local else max_seq_len

            old_k = layer.attention.cache_k
            if can_preserve and old_k is not None and old_k.shape[1] == len_l:
                # same per-layer size (e.g. a LOCAL ring at W while max_seq_len
                # grows): slot = pos % Lc is invariant to max_seq_len, so the
                # buffer stays valid as-is.
                continue

            new_k = torch.zeros(
                (max_batch_size, len_l, n_kv_heads, head_dim),
                device=device, dtype=dtype
            )
            new_v = torch.zeros(
                (max_batch_size, len_l, n_kv_heads, head_dim),
                device=device, dtype=dtype
            )
            if can_preserve and old_k is not None:
                # Growing this layer's buffer: copy the still-valid prefix. For a
                # LOCAL layer this only happens when the old length was < W, i.e.
                # the ring never wrapped (tokens <= old max_seq_len < W), so
                # slot == pos and a prefix copy is exact.
                _copy = min(old_len, old_k.shape[1], len_l)
                new_k[:, :_copy] = old_k[:, :_copy]
                new_v[:, :_copy] = layer.attention.cache_v[:, :_copy]
            layer.attention.cache_k = new_k
            layer.attention.cache_v = new_v

        # MTP block cache (speculative decoding): the draft module runs its own
        # causal attention over the mtp input sequence, so it needs a KV cache
        # of its own — GLOBAL/full-length (mtp.block.swa_local is forced False).
        if getattr(self, 'mtp', None) is not None:
            _att = self.mtp.block.attention
            _att.cache_window = None
            _old = _att.cache_k
            if not (can_preserve and _old is not None and _old.shape[1] == max_seq_len):
                device = self.mtp.block.attention.wq.weight.device
                dtype = self.mtp.block.attention.wq.weight.dtype
                new_k = torch.zeros((max_batch_size, max_seq_len, n_kv_heads, head_dim),
                                    device=device, dtype=dtype)
                new_v = torch.zeros((max_batch_size, max_seq_len, n_kv_heads, head_dim),
                                    device=device, dtype=dtype)
                if can_preserve and _old is not None:
                    _copy = min(old_len, _old.shape[1], max_seq_len)
                    new_k[:, :_copy] = _old[:, :_copy]
                    new_v[:, :_copy] = _att.cache_v[:, :_copy]
                else:
                    self._mtp_cache_len = 0  # fresh mtp cache: nothing materialized
                _att.cache_k = new_k
                _att.cache_v = new_v

        # Capacity bookkeeping: with per-layer lengths, probing layer 0 (LOCAL
        # under SWA -> W slots) would misreport; store the logical capacity.
        self._cache_bsz = max_batch_size
        self._cache_msl = max_seq_len

        if can_preserve:
            # Contents [0, old_len) carried over → the ledger still describes
            # them correctly. Trim the ledger to old_len just in case it somehow
            # ran ahead of the physical capacity (it shouldn't), so it never
            # claims more than was copied.
            led = self.get_cache_ledger()
            if len(led) > old_len:
                self.set_cache_ledger(led[:old_len])
            # else: ledger already within [0, old_len] — keep it as-is.
        else:
            # Truly fresh allocation → any previously remembered ledger is stale.
            self.reset_cache_ledger()

    def clear_caches(self):
        """Free KV cache memory."""
        for layer in self.layers:
            if getattr(layer, 'use_gdn', False):
                continue  # GDN layers have no KV cache
            if layer.attention.cache_k is not None:
                del layer.attention.cache_k
            if layer.attention.cache_v is not None:
                del layer.attention.cache_v
            layer.attention.cache_k = None
            layer.attention.cache_v = None
        if getattr(self, 'mtp', None) is not None:
            self.mtp.block.attention.cache_k = None
            self.mtp.block.attention.cache_v = None
        self._mtp_cache_len = 0
        self._cache_bsz = None
        self._cache_msl = None
        # Delta-rule recurrent state shares the cache lifecycle.
        self._fla_cache = None
        self._fla_cache_pos = 0

    # ----- Delta-state snapshot/rollback (spec decode; KDA_SPEC_DECODE_ROLLBACK) --
    @staticmethod
    def _clone_state_tree(o):
        if torch.is_tensor(o):
            return o.detach().clone()
        if isinstance(o, dict):
            return {k: Transformer._clone_state_tree(v) for k, v in o.items()}
        if isinstance(o, (list, tuple)):
            t = [Transformer._clone_state_tree(v) for v in o]
            return t if isinstance(o, list) else tuple(t)
        return o

    def snapshot_delta_state(self):
        """Clone the delta layers' recurrent/conv states + position. Cheap
        (~1MB/layer) unlike KV. Returns None for non-delta models — callers
        can hook unconditionally."""
        if not getattr(self.params, 'gdn_enabled', False) or self._fla_cache is None:
            return None
        states = getattr(self._fla_cache, 'states', None)
        if states is None:
            raise RuntimeError(
                "FLA Cache has no .states — fla version changed its schema; "
                "update snapshot_delta_state before using spec decode on delta trunks.")
        return (self._clone_state_tree(list(states)), self._fla_cache_pos)

    def restore_delta_state(self, snap):
        """Restore a snapshot_delta_state() result. Re-clones, so one snap can
        restore multiple times. Resets the state position so the seek-guard
        accepts continuation from the snapshot point."""
        if snap is None:
            return
        states, pos = snap
        self._fla_cache.states[:] = self._clone_state_tree(states)
        self._fla_cache_pos = pos

        # The cache no longer exists → its token ledger is meaningless.
        self.reset_cache_ledger()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # ----- Cache token ledger -------------------------------------------------
    # The ledger records the exact token IDs physically materialized in cache
    # positions [0, len(ledger)). It lives on the MODEL, co-located with the
    # cache tensors it describes, so the two share one lifecycle and cannot
    # desync: any (re)allocation or clear resets it (see setup_caches /
    # clear_caches). Cross-turn prefix reuse reads/writes it via these helpers.

    def reset_cache_ledger(self):
        """Forget the cache token ledger (contents are unknown/stale)."""
        self._cache_token_ids: list[int] = []

    def get_cache_ledger(self) -> "list[int]":
        """Token IDs currently materialized in the KV cache, positions [0, N)."""
        return getattr(self, "_cache_token_ids", [])

    def set_cache_ledger(self, token_ids: "list[int]"):
        """Record the token IDs now materialized in the cache. Caller must pass
        EXACTLY the ids physically forwarded into the cache (not trimmed text)."""
        self._cache_token_ids = list(token_ids)

    def has_caches(self) -> bool:
        """Check if KV caches are currently allocated."""
        for layer in self.layers:
            if not getattr(layer, 'use_gdn', False):
                return layer.attention.cache_k is not None
        return False  # all layers are GDN

    def mtp_decode_chunk(self, h_pre: torch.Tensor, next_tokens: torch.Tensor,
                         start_pos: int) -> torch.Tensor:
        """Speculative-decoding draft pass: run the MTP module over a chunk at
        inference, using its own KV cache (allocated by setup_caches).

        Args:
            h_pre: [B, S, D] PRE-final-norm trunk states for absolute positions
                   [start_pos, start_pos+S) — from generate_forward(return_h_pre=True).
            next_tokens: [B, S] the token at position i+1 for each chunk row i
                   (the accepted/known continuation — training's teacher-forced input).
            start_pos: absolute position of the chunk's first row.

        Returns:
            logits [B, S, V]: row i predicts t_{i+2} (the DRAFT distribution) —
            identical readout path to training (shared final norm + output head).
        """
        assert self.mtp is not None, "mtp_decode_chunk requires an MTP checkpoint"
        S = h_pre.shape[1]
        next_emb = self.tok_embeddings(next_tokens)
        x = self.mtp.proj(torch.cat(
            [self.mtp.h_norm(h_pre), self.mtp.emb_norm(next_emb)], dim=-1))
        freqs_cos = self.freqs_cos[start_pos:start_pos + S]
        freqs_sin = self.freqs_sin[start_pos:start_pos + S]
        h = self.mtp.block.forward_with_cache(x, freqs_cos, freqs_sin, start_pos)
        return self.output(self.norm(h))

    def min_rolling_cache_len(self):
        """Smallest live SWA ring capacity, or None when no rolling caches exist.

        The cross-turn prefix-reuse contract is APPEND-ONLY once more tokens
        than this have been materialized: a wrapped ring's slots for evicted
        positions hold FUTURE-position K/V, so rewind re-entry behind the
        ledger's high-water mark would silently attend the wrong timeline.
        Callers (stream_generate_kv / verify_kv_reuse_parity) degrade a
        post-wrap rewind to a full re-prefill."""
        lens = [l.attention.cache_k.shape[1] for l in self.layers
                if not getattr(l, 'use_gdn', False)
                and l.attention.cache_window is not None
                and l.attention.cache_k is not None]
        return min(lens) if lens else None

    def cache_capacity(self):
        """Return (max_batch_size, max_seq_len) of the currently allocated KV
        cache, or (None, None) if no cache is allocated.

        Uses the stored logical capacity when available (with SWA rolling
        caches, per-layer buffer lengths differ — probing layer 0, which is
        LOCAL under SWA, would misreport W as the capacity). Falls back to
        probing the first non-GDN layer for pre-SWA cache states."""
        if getattr(self, '_cache_msl', None) is not None and self.has_caches():
            return self._cache_bsz, self._cache_msl
        for layer in self.layers:
            if getattr(layer, 'use_gdn', False):
                continue
            ck = layer.attention.cache_k
            if ck is None:
                return None, None
            return ck.shape[0], ck.shape[1]
        return None, None  # all layers are GDN

    # =========================================================================
    # Training Forward (identical to original model_v1.py)
    # =========================================================================
    
    def forward(
        self,
        tokens: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
        start_pos: Optional[int] = None,
        active_layers: Optional[int] = None,
        scaffold_mode: bool = False,
        cce_valids: Optional[dict] = None,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Unified forward - handles training, eval, and KV-cached inference.

        IMPORTANT: Do NOT override __call__ in nn.Module subclasses when using
        FSDP2, as it bypasses the module call hooks that FSDP2 relies on.

        Args:
            tokens: Input token IDs [B, S]
            targets: Target token IDs for training [B, S], or None for inference
            start_pos: Starting position for KV-cached inference, or None for training/eval
            active_layers: If set, only run the first N layers (progressive tail truncation).
                           In scaffold_mode this is the truncation depth for both
                           forward and backward; otherwise the final norm + output
                           head still fire on the layer-N output.
            scaffold_mode: Scaffolded Cascading Supervision (SCS) phase. When True,
                           the main LM head (self.norm + self.output + main-loss CCE)
                           is skipped entirely — the loss is the sum of weighted aux
                           head losses captured during the truncated block loop. The
                           deepest active aux head is doing the LM prediction. Caller
                           must pass active_layers = deepest_active_aux_tap + 1 to
                           ensure that aux head's tap fires.

        Returns:
            (logits, loss) where:
            - Training (targets given, scaffold_mode=False): (None, main_loss);
              aux losses are stashed in self._last_aux_loss_tensors as usual.
            - Training (targets given, scaffold_mode=True): (None, loss_or_None).
              The main LM head is skipped, so the only contribution to the
              returned loss is any MoE balance-loss sum from MoE layers in the
              active range; loss is None when there's no MoE in the active
              range (the common dense case). Aux losses are still stashed in
              self._last_aux_loss_tensors and the trainer aggregates them into
              the total objective.
            - Inference (no targets): (logits, None)
            - KV-cached (start_pos given): (logits, None)
        """
        # KV-cached inference path
        if start_pos is not None:
            logits = self.generate_forward(tokens, start_pos)
            return logits, None

        # Standard training/eval path
        B, S = tokens.shape
        h = self.tok_embeddings(tokens)
        h = self.dropout(h)

        # doc-mask: BlockMask + per-document RoPE positions for packed windows.
        # Built EAGERLY here (Transformer.forward is not compiled; only the
        # per-layer submodules are) once per micro-batch, shared by all layers.
        # The KV-cache generate path (start_pos) is single-document and never
        # reaches this code.
        block_mask = None
        _doc_on = self.params.doc_attn_mask or self.params.doc_pos_reset
        _has_boundary = bool((tokens[:, 1:] == self.params.bos_token_id).any()) if _doc_on else False
        # boundary-free dispatch: a micro-batch with no INTRA-ROW document boundary
        # has doc-mask == plain causal (bit-exact, see t_no_bos_parity /
        # t_leading_bos_fast_path) and per-doc positions == arange — so take the
        # SDPA fast path / shared tables. A BOS at column 0 opens the row's only
        # document (inclusive-cumsum doc ids: the whole row shares one id), so it
        # creates no boundary; only a BOS at column >= 1 does. This sidesteps
        # flex-causal's SM86 fwd+bwd penalty on boundary-free batches, which
        # dominate for long-doc groups (books mean ~98k tok/doc -> ~2% of windows
        # carry a boundary) AND single-document rows (eval/benchmark batches).
        if self.params.doc_attn_mask and _has_boundary:
            block_mask = self._build_doc_block_mask(tokens)
        # Delta-rule (KDA/GDN) layers get the SAME doc confinement via FLA
        # varlen state resets (docs/KDA_VARLEN_DOC_RESET.md). Boundary-free
        # batches skip it: per-row state in [B, S] mode is exactly the
        # row-boundary flattened form (a column-0 BOS coincides with the row
        # start and is deduplicated by doc_cu_seqlens anyway), so the fast path
        # is bit-equivalent, mirroring the flex dispatch above.
        doc_cu = None
        if self.params.doc_attn_mask and getattr(self.params, 'gdn_enabled', False) and _has_boundary:
            doc_cu = doc_cu_seqlens(tokens, self.params.bos_token_id)
        if self.params.doc_pos_reset and _has_boundary:
            pos = doc_position_ids(tokens, self.params.bos_token_id)  # [B, S]
            # gather ONCE and pre-cast to the activation dtype: the FSDP mp_policy would
            # otherwise cast the fp32 [B,S,D/2] gather per LAYER, pinning ~n_layers bf16
            # copies under activation checkpointing (~+300MB at dn4 shapes). Layers see
            # bf16 freqs either way (mp_policy casts the shared fp32 slices today too).
            freqs_cos = self.freqs_cos[pos].to(h.dtype)
            freqs_sin = self.freqs_sin[pos].to(h.dtype)
        else:
            freqs_cos = self.freqs_cos[:S]
            freqs_sin = self.freqs_sin[:S]

        # SWA: LOCAL layers always need a windowed mask (the window constraint is
        # active regardless of document boundaries; doc confinement composes in when
        # live). GLOBAL layers keep the mask computed above (doc mask or the SDPA
        # fast path). Threaded as a (global, local) pair; TransformerBlock._attn
        # resolves per its layer kind. Flags-off runs never see the tuple.
        if self.params.swa_enabled:
            bm_local = self._build_block_mask(
                tokens, doc=(self.params.doc_attn_mask and _has_boundary),
                window=self.params.swa_window)
            block_mask = (block_mask, bm_local)

        n_active = active_layers if (active_layers is not None and active_layers < len(self.layers)) else len(self.layers)

        # Aux head taps captured during the block loop (training only).
        # Keyed by layer index; value is the block's output activation (the
        # tensor that becomes the next block's input under the default path).
        # Skipped during eval/val (self.training is False) — val loss reflects
        # only the main task.
        aux_taps: dict = {}
        # Capture aux taps during training, or during eval-time scaffold (val
        # needs the deepest active aux head's CE as its effective loss).
        capture_aux = bool(self._aux_head_layer_set) and (targets is not None) and (self.training or scaffold_mode)

        if self.attn_res_enabled:
            # AttnRes: selective depth-wise retrieval via learned block attention
            blocks = [h]                           # b_0 = token embedding
            partial_block = torch.zeros_like(h)    # intra-block accumulator
            bs = self.attn_res_block_size

            for i, blk in enumerate(self.layers):
                if i >= n_active:
                    break
                # Selective retrieval from completed blocks + partial sum
                h = block_attn_res(blocks, partial_block, self.attn_res_queries[i],
                                   self.attn_res_key_norms[i].weight, self.attn_res_key_norms[i].eps)
                h_out = blk(h, freqs_cos, freqs_sin, block_mask, doc_cu)
                # Accumulate layer delta (sublayer output) into partial block
                partial_block = partial_block + (h_out - h)
                # Block boundary: store completed block, reset accumulator
                if (i + 1) % bs == 0 and (i + 1) < n_active:
                    blocks.append(partial_block)
                    partial_block = torch.zeros_like(h)
                h = h_out
                if capture_aux and i in self._aux_head_layer_set:
                    aux_taps[i] = h

        elif n_active < len(self.layers):
            # Truncated path — skip tail layers (progressive tail truncation)
            for i, blk in enumerate(self.layers):
                if i >= n_active:
                    break
                h = blk(h, freqs_cos, freqs_sin, block_mask, doc_cu)
                if capture_aux and i in self._aux_head_layer_set:
                    aux_taps[i] = h
        elif capture_aux:
            # Full-depth path with aux heads enabled: enumerate to capture taps.
            for i, blk in enumerate(self.layers):
                h = blk(h, freqs_cos, freqs_sin, block_mask, doc_cu)
                if i in self._aux_head_layer_set:
                    aux_taps[i] = h
        else:
            # Full-depth path — identical to original for torch.compile fast path
            for blk in self.layers:
                h = blk(h, freqs_cos, freqs_sin, block_mask, doc_cu)

        # In scaffold_mode the main LM head is intentionally skipped — the
        # partial network's "LM head" is the deepest active aux head, and
        # running self.norm + self.output would (a) waste compute, (b) touch
        # uninitialised tail params via the all-gather, and (c) produce a
        # garbage loss against untrained weights. The aux taps captured above
        # carry the supervision.
        # MTP: capture the pre-final-norm residual stream before self.norm
        # overwrites h (the sequential module reads the raw trunk state).
        h_pre_norm = h if (self.mtp is not None and targets is not None
                           and self.training and not scaffold_mode) else None

        if not scaffold_mode:
            h = self.norm(h)

        # ── TRAINING BRANCH ────────────────────────────────────────
        if targets is not None:
            pad_id = self.params.pad_id

            # Reset main-head z-loss stashes each forward so a stale value from
            # a prior full-depth forward (e.g. baseline val) can never be
            # reused by the trainer. Only compute z-loss during training — the
            # model is .eval() in validation, where the z stats are unused, so
            # gating on self.training also skips the extra z-loss work there
            # (and keeps val display-only / unchanged). _zloss_fp32_accum is
            # None when z-loss is off (then this branch is byte-identical to
            # baseline), else False=bf16 / True=fp32_accum backend.
            self._last_zloss = None
            self._last_logz = None
            self._last_logz_rms = None
            self._last_logz_p95 = None
            self._last_mtp_loss = None
            _want_zloss = (self._zloss_fp32_accum is not None) and self.training

            if scaffold_mode:
                # No main loss to compute. Aux losses below are still computed
                # from the captured taps and stashed for the trainer. Main-head
                # z-loss stays None — the deepest aux head carries it.
                loss = None
            else:
                # Flatten without materializing a masked copy of h
                h_flat = h.reshape(-1, h.size(-1))
                tgt_flat = targets.reshape(-1)

                # Ensure hidden states match output weight dtype (CCE Triton kernel requires same dtype)
                out_dtype = self.output.weight.dtype
                if h_flat.dtype != out_dtype:
                    h_flat = h_flat.to(out_dtype)

                accum_fp32 = out_dtype == torch.float32
                # Main LM loss is ALWAYS pure CE (reduction='mean'), identical
                # to baseline whether or not z-loss is on — the z term is a
                # SEPARATE stashed quantity the trainer adds to the objective.
                loss = cce_loss(
                    h_flat,
                    self.output.weight,
                    tgt_flat,
                    valids=(cce_valids.get('main', _CCE_VALIDS_ABSENT)
                            if cce_valids is not None else _CCE_VALIDS_ABSENT),
                    accum_e_fp32=accum_fp32,
                    accum_c_fp32=accum_fp32,
                    reduction="mean",
                    ignore_index=pad_id,
                )
                if _want_zloss:
                    if self._zloss_target == 'centered':
                        # dn4 Lever 2: deadband centered z-loss. Stash into _last_zloss
                        # so the trainer's alpha*z_sel path is UNCHANGED; the penalized
                        # quantity is mean(relu(logZ_c - tau)**2) (gauge-invariant).
                        (self._last_zloss, self._last_logZ_c,
                         self._last_h_mu) = _centered_zloss_deadband(
                            h_flat, self.output.weight, tgt_flat, pad_id,
                            self._zloss_tau, self._zloss_fp32_accum,
                        )
                        # Keep the legacy logZ diag fields populated (centered analogues)
                        # so the existing logger / zloss_diag don't see None. The clean
                        # centered telemetry is in _last_logZ_c / _last_h_mu + the val-
                        # cadence logZ_c logging.
                        self._last_logz = self._last_logZ_c
                        self._last_logz_rms = self._last_zloss.detach().clamp_min(0).sqrt()
                        self._last_logz_p95 = self._last_logZ_c
                    else:
                        # Option D: no [N,V] materialization. Backend bool selects
                        # CCE fp32 accumulation in its backward (see _zloss_optionD).
                        (self._last_zloss, self._last_logz,
                         self._last_logz_rms, self._last_logz_p95) = _zloss_optionD(
                            h_flat, self.output.weight, tgt_flat, pad_id,
                            self._zloss_fp32_accum,
                        )

                # MTP (DeepSeek-style sequential module): predict t+2 through the
                # extra block, SHARED final norm + output head. Stashed like z-loss —
                # the trainer adds lambda * _last_mtp_loss to the objective; headline
                # ls:/ppl/val stay PURE t+1 CE. Training only (h_pre_norm is None at
                # eval), so val remains a clean t+1 measurement across arms.
                # Target alignment: main head at position i predicts targets[i]
                # (= x_{i+1}); MTP at position i predicts x_{i+2} = targets[i+1],
                # so mtp targets = targets shifted left once, last column = pad
                # (ignored). Emb(t_{i+1}) = tok_embeddings(targets) teacher-forces
                # the intermediate token, per DeepSeek-V3.
                if h_pre_norm is not None:
                    next_emb = self.tok_embeddings(targets)
                    h_mtp = self.mtp(h_pre_norm, next_emb, freqs_cos, freqs_sin,
                                     block_mask)
                    h_mtp = self.norm(h_mtp)
                    mtp_tgt = torch.cat(
                        [targets[:, 1:],
                         targets.new_full((targets.shape[0], 1), pad_id)], dim=1)
                    if self.params.mtp_doc_boundary_mask:
                        # Row i conditions on emb(targets[i]); when targets[i]
                        # is BOS, mtp_tgt[i] is the NEXT document's first
                        # content token — cross-document supervision under
                        # doc-mask. Ignore those rows. (Predicting the BOS
                        # itself, targets[i+1]==BOS, stays trained — same as
                        # the main head.)
                        mtp_tgt = mtp_tgt.masked_fill(
                            targets == self.params.bos_token_id, pad_id)
                    hm_flat = h_mtp.reshape(-1, h_mtp.size(-1))
                    if hm_flat.dtype != out_dtype:
                        hm_flat = hm_flat.to(out_dtype)
                    self._last_mtp_loss = cce_loss(
                        hm_flat,
                        self.output.weight,
                        mtp_tgt.reshape(-1),
                        valids=(cce_valids.get('mtp', _CCE_VALIDS_ABSENT)
                                if cce_valids is not None else _CCE_VALIDS_ABSENT),
                        accum_e_fp32=accum_fp32,
                        accum_c_fp32=accum_fp32,
                        reduction="mean",
                        ignore_index=pad_id,
                    )

            # MoE balance losses: fold in from layers that actually ran this
            # forward. Crucially we scope the loop to the active range —
            # under scaffold the tail MoE layers didn't fire, so their
            # `_last_aux_loss` would be either None or a stale tensor from a
            # prior full-depth forward (e.g. baseline val). Adding the stale
            # tensor would attempt to backward through a graph that's
            # already been consumed → RuntimeError. Bound by n_active to
            # match the forward loop.
            for i, blk in enumerate(self.layers):
                if i >= n_active:
                    break
                if getattr(blk, 'moe_enabled', False):
                    al = blk.moe._last_aux_loss
                    if al is not None:
                        loss = al if loss is None else loss + al
                        blk.moe._last_aux_loss = None

            # Auxiliary prediction-head losses at captured tap points. The
            # trainer reads these tensors from self._last_aux_loss_tensors,
            # applies the per-head schedule weight at the current step, and
            # sums them into the main loss before calling .backward(). Keeping
            # the weighting in the trainer means the schedule lives in config,
            # not in the model.
            #
            # In scaffold_mode there's no main loss to combine with — the
            # trainer treats the aux head sum as the total objective directly.
            new_aux_losses: dict = {}
            new_aux_zloss: dict = {}
            new_aux_logz: dict = {}
            if aux_taps:
                _tgt_flat = targets.reshape(-1)
                # Only the aux head that is the live LM readout under SCS
                # scaffold actually needs z-loss, but the model can't know
                # which tap the trainer will pick (scs_deepest_tap lives in
                # the trainer), so every fired aux head stashes its z-loss
                # when enabled; the trainer selects the deepest one. Gated on
                # self.training so validation skips the extra z-loss work.
                # Pass the backend bool (False=bf16/True=fp32_accum) when on,
                # None when off (z-loss disabled, or eval).
                _aux_zfp32 = (self._zloss_fp32_accum if self.training else None)
                for li, h_tap in aux_taps.items():
                    # Call through __call__ so FSDP unshard/reshard hooks
                    # fire. AuxHead.forward signature is
                    # (h_tap, tgt_flat, pad_id, zloss_fp32_accum) — it does its
                    # own RMSNorm + CCE on the flattened (B*S,) target tensor
                    # and returns (loss, zloss, logz). zloss/logz are None
                    # unless zloss_fp32_accum is not None.
                    _l, _z, _lz = self.aux_heads[str(li)](
                        h_tap, _tgt_flat, pad_id, _aux_zfp32
                    )
                    new_aux_losses[li] = _l
                    if _z is not None:
                        new_aux_zloss[li] = _z
                        new_aux_logz[li] = _lz
            self._last_aux_loss_tensors = new_aux_losses
            self._last_aux_zloss = new_aux_zloss
            self._last_aux_logz = new_aux_logz

            self.last_loss = loss
            return None, loss

        # ── INFERENCE / EVAL BRANCH ────────────────────────────────
        if scaffold_mode:
            # Under scaffold the final self.norm was skipped above (the deepest
            # aux head owns the readout); projecting the UN-normalized stream
            # through the head yields silently-garbage logits (audit 2026-07-11).
            raise ValueError(
                "scaffold_mode=True requires targets — the eval branch has no "
                "normalized stream to project. Read the aux-head losses from a "
                "training forward instead.")
        logits = self.output(h)
        return logits, None

    # =========================================================================
    # Inference Forward with KV Caching
    # =========================================================================
    
    def generate_forward(
        self,
        tokens: torch.Tensor,
        start_pos: int = 0,
        return_h_pre: bool = False,
    ) -> torch.Tensor:
        """
        INFERENCE FORWARD with KV caching - for generation.
        
        Must call setup_caches() before using this method.
        
        Args:
            tokens: Input token IDs [B, S]
                   - For prefill: S = prompt length
                   - For decode: S = 1 (single new token)
            start_pos: Starting position in the sequence
                      - For prefill: 0
                      - For decode: current sequence length
                      
        Returns:
            logits: [B, S, vocab_size]
        """
        assert self.has_caches(), "Must call setup_caches() before generate_forward()"

        B, S = tokens.shape
        h = self.tok_embeddings(tokens)
        # No dropout during inference

        # Slice freqs for current position range
        freqs_cos = self.freqs_cos[start_pos:start_pos + S]
        freqs_sin = self.freqs_sin[start_pos:start_pos + S]

        # Delta-rule (GDN/KDA) recurrent state lifecycle. Unlike the KV cache,
        # the recurrent state is NOT position-addressable: it summarizes the
        # whole prefix, so it cannot be rolled back or partially reused. A
        # prefill (start_pos == 0) starts a fresh Cache; every later call must
        # continue EXACTLY where the state left off. Cross-turn prefix reuse
        # (suffix prefill at start_pos > 0 after a cache reset elsewhere) is
        # unsupported for delta-rule hybrids — re-prefill from 0 instead.
        _fla_cache = None
        if getattr(self.params, 'gdn_enabled', False):
            if start_pos == 0:
                self._fla_cache = _new_fla_cache()
                self._fla_cache_pos = 0
            elif self._fla_cache is None or start_pos != self._fla_cache_pos:
                raise RuntimeError(
                    f"generate_forward(start_pos={start_pos}) but the delta-rule "
                    f"recurrent state is at position "
                    f"{self._fla_cache_pos if self._fla_cache is not None else 'None'}. "
                    f"GDN/KDA state summarizes the whole prefix and cannot seek; "
                    f"re-prefill from start_pos=0.")
            _fla_cache = self._fla_cache

        if self.attn_res_enabled:
            blocks = [h]
            partial_block = torch.zeros_like(h)
            bs = self.attn_res_block_size
            for i, blk in enumerate(self.layers):
                h = block_attn_res(blocks, partial_block, self.attn_res_queries[i],
                                   self.attn_res_key_norms[i].weight, self.attn_res_key_norms[i].eps)
                h_out = blk.forward_with_cache(h, freqs_cos, freqs_sin, start_pos, fla_cache=_fla_cache)
                partial_block = partial_block + (h_out - h)
                if (i + 1) % bs == 0 and (i + 1) < len(self.layers):
                    blocks.append(partial_block)
                    partial_block = torch.zeros_like(h)
                h = h_out
        else:
            for blk in self.layers:
                h = blk.forward_with_cache(h, freqs_cos, freqs_sin, start_pos, fla_cache=_fla_cache)

        if _fla_cache is not None:
            self._fla_cache_pos = start_pos + S

        if return_h_pre:
            # speculative decoding needs the PRE-final-norm trunk state (the
            # exact tensor the MTP module consumed at training time)
            h_pre = h
            h = self.norm(h)
            return self.output(h), h_pre
        h = self.norm(h)
        logits = self.output(h)
        return logits
