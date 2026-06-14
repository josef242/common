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
from contextlib import nullcontext

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


@torch._dynamo.disable
def cce_loss(hidden, weight, targets, **kwargs):
    """Thin wrapper so CCE executes in eager mode."""
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


# ----------------------------------------------------------------------------
# Flash Attention (optional)
# ----------------------------------------------------------------------------
flash_attn_func = None  # Set to actual import if available


# ----------------------------------------------------------------------------
# Gated DeltaNet (FLA library, optional)
# ----------------------------------------------------------------------------
_GatedDeltaNet = None  # Lazy-loaded when gdn_enabled=True


def _try_import_gdn():
    global _GatedDeltaNet
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
    # QK-Norm Mode: None | "before_rope" | "after_rope_legacy"
    qk_norm_mode: Optional[str] = None
    # Tie input embeddings and output LM head weights
    tie_word_embeddings: bool = True
    # RoPE base frequency (higher = longer context support)
    rope_theta: float = 500000.0
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
    gdn_interleave_step: int = 4           # every Nth layer is full-attention, rest are GDN
    n_gdn_heads: Optional[int] = None      # GDN head count (None = same as n_heads)
    gdn_head_dim: Optional[int] = None     # GDN q/k head dim (None = 256, FLA default)
    gdn_v_expand: float = 2.0             # value expansion ratio (v_dim = head_dim * expand)
    gdn_short_conv_kernel: int = 4         # short convolution kernel size
    gdn_mode: str = 'chunk'                # FLA mode: 'chunk' (training) or 'fused_recurrent'
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
    # freqs_cos, freqs_sin: [S, D//2]
    
    # Split into even/odd (equivalent to real/imag in complex view)
    xq_r, xq_i = xq.float().reshape(*xq.shape[:-1], -1, 2).unbind(-1)
    xk_r, xk_i = xk.float().reshape(*xk.shape[:-1], -1, 2).unbind(-1)
    
    # Reshape freqs for broadcasting: [1, S, 1, D//2]
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

    def forward(self, x: torch.Tensor, freqs_cos: torch.Tensor, freqs_sin: torch.Tensor):
        """
        TRAINING PATH - Identical to original model_v1.py
        No caching, no start_pos, no branching on cache existence.
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

        if self.use_flashattn2:
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

        # ➎ Update KV cache        
        assert self.cache_k is not None, "Must call setup_caches() before forward_with_cache()"
        self.cache_k[:bsz, start_pos:start_pos + seqlen] = xk
        self.cache_v[:bsz, start_pos:start_pos + seqlen] = xv
        
        # Retrieve all cached K/V up to current position
        keys = self.cache_k[:bsz, :start_pos + seqlen]
        values = self.cache_v[:bsz, :start_pos + seqlen]

        # (keys are already normalized/scaled in-cache for after_rope modes)

        # ➐ Attention (no dropout during inference)
        xq = xq.transpose(1, 2)       # [B, Hq, S_q, D]
        k = keys.transpose(1, 2)      # [B, Hkv, S_kv, D]
        v = values.transpose(1, 2)        

        # For single-token decode (seqlen == 1) we attend to all past keys, no mask.
        # For prefill (seqlen > 1) we need a causal mask.
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
            self._tokens_dropped_accum += n_dropped
            if n_dropped > 0:
                # Unbiased rescaling: scale kept scores so expected value is preserved
                sum_all = top_scores.sum()
                top_scores = top_scores * keep_mask
                sum_keep = top_scores.sum()
                if sum_keep > 0:
                    scale = (sum_all / sum_keep).clamp_(max=10.0)
                    top_scores = top_scores * scale
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
            self.use_gdn = (layer_id % gdn_step != gdn_step - 1)

        if self.use_gdn:
            _try_import_gdn()
            n_gdn_heads = getattr(args, 'n_gdn_heads', None) or args.n_heads
            gdn_head_dim = getattr(args, 'gdn_head_dim', None) or 256
            self.gdn_attn = _GatedDeltaNet(
                hidden_size=args.dim,
                num_heads=n_gdn_heads,
                head_dim=gdn_head_dim,
                expand_v=getattr(args, 'gdn_v_expand', 2.0),
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

    def _attn(self, x, freqs_cos, freqs_sin):
        """Route through GDN or softmax attention."""
        if self.use_gdn:
            out, *_ = self.gdn_attn(x)
            return out
        return self.attention(x, freqs_cos, freqs_sin)

    def _forward_block(self, x, freqs_cos, freqs_sin):
        """Inner forward for activation checkpointing."""
        if self.use_keel:
            if self.layer_id == 0:
                # First block: standard Pre-LN (no Post-LN, no alpha scaling)
                h = x + self._attn(self.attention_norm(x), freqs_cos, freqs_sin)
                out = h + self._ffn(self.ffn_norm(h))
            else:
                # KEEL: x_{l+1} = LN(alpha * x_l + F_l(LN(x_l)))
                attn_out = self._attn(self.attention_norm(x), freqs_cos, freqs_sin)
                h = self.post_attn_norm(self.keel_alpha * x + attn_out)
                ffn_out = self._ffn(self.ffn_norm(h))
                out = self.post_ffn_norm(self.keel_alpha * h + ffn_out)
        else:
            # Original Pre-LN path (unchanged)
            h = x + self._attn(self.attention_norm(x), freqs_cos, freqs_sin)
            out = h + self._ffn(self.ffn_norm(h))
        return out

    def forward(self, x, freqs_cos, freqs_sin):
        """
        TRAINING PATH - Uses activation checkpointing when enabled.
        """
        if self.use_activation_checkpointing and self.training:
            out = cp.checkpoint(self._forward_block, x, freqs_cos, freqs_sin, use_reentrant=False)
        else:
            out = self._forward_block(x, freqs_cos, freqs_sin)
        return out

    def forward_with_cache(self, x, freqs_cos, freqs_sin, start_pos: int):
        """
        INFERENCE PATH - No checkpointing, uses KV cache.
        GDN layers use regular forward (no KV cache, recurrent state in FLA).
        """
        # Move input to this block's device (for multi-GPU sharded models)
        device = next(self.parameters()).device
        if x.device != device:
            x = x.to(device)

        # GDN: no KV cache, just forward pass
        if self.use_gdn:
            attn_out, *_ = self.gdn_attn(self.attention_norm(x))
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
        self.vocab_size = params.vocab_size
        self.n_layers = params.n_layers
        
        self.tok_embeddings = nn.Embedding(params.vocab_size, params.dim)
        self.dropout = nn.Dropout(params.dropout)
        
        self.layers = nn.ModuleList()
        for layer_id in range(params.n_layers):
            self.layers.append(TransformerBlock(layer_id, params))
        
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
        self._last_aux_zloss: dict = {}  # per-aux-head raw zloss
        self._last_aux_logz: dict = {}   # per-aux-head logZ_mean

        # Optional weight tying
        self.tie_word_embeddings = getattr(params, "tie_word_embeddings", True)
        if self.tie_word_embeddings:
            self.output.weight = self.tok_embeddings.weight

        # Precompute RoPE frequencies
        freqs_cos, freqs_sin = precompute_freqs_cis(
            self.params.dim // self.params.n_heads,
            self.params.max_seq_len,
            self.params.rope_theta
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
            # Expert weights (3D nn.Parameter, no .weight suffix — not hit by _init_weights)
            elif '.experts.w1' in pn:
                torch.nn.init.normal_(p, mean=0.0, std=0.02)
            elif '.experts.w2' in pn or '.experts.w3' in pn:
                torch.nn.init.normal_(p, mean=0.0, std=output_std)

        self.last_loss = None

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
                # 3D expert weights: w1 gets standard init, w2/w3 get scaled init
                torch.nn.init.trunc_normal_(module.w1, mean=0.0, std=0.02)
                torch.nn.init.trunc_normal_(module.w2, mean=0.0, std=output_std)
                torch.nn.init.trunc_normal_(module.w3, mean=0.0, std=output_std)
            elif isinstance(module, MoE):
                # Re-init buffers on correct device after materialization
                if module.load_balance_coeff is not None:
                    module.expert_bias.zero_()
                module.tokens_per_expert.zero_()

        # Apply scaled initialization to output projections (w3, wo, GDN o_proj)
        # These benefit from smaller init to prevent output explosion in deep nets
        for name, param in self.named_parameters():
            if name.endswith('w3.weight') or name.endswith('wo.weight'):
                torch.nn.init.trunc_normal_(param, mean=0.0, std=output_std)
            elif '.gdn_attn.o_proj.weight' in name:
                torch.nn.init.trunc_normal_(param, mean=0.0, std=output_std)

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

        for layer in self.layers:
            if getattr(layer, 'use_gdn', False):
                continue  # GDN layers have no KV cache
            # Get device and dtype from layer weights
            device = layer.attention.wq.weight.device
            dtype = layer.attention.wq.weight.dtype

            new_k = torch.zeros(
                (max_batch_size, max_seq_len, n_kv_heads, head_dim),
                device=device, dtype=dtype
            )
            new_v = torch.zeros(
                (max_batch_size, max_seq_len, n_kv_heads, head_dim),
                device=device, dtype=dtype
            )
            if can_preserve:
                # Copy the still-valid prefix K/V into the larger buffer.
                old_k = layer.attention.cache_k
                old_v = layer.attention.cache_v
                new_k[:, :old_len] = old_k[:, :old_len]
                new_v[:, :old_len] = old_v[:, :old_len]
            layer.attention.cache_k = new_k
            layer.attention.cache_v = new_v

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

    def cache_capacity(self):
        """Return (max_batch_size, max_seq_len) of the currently allocated KV
        cache, or (None, None) if no cache is allocated.

        Reads the first non-GDN layer's cache_k shape: [bsz, seq_len, n_kv, hd].
        """
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

        freqs_cos = self.freqs_cos[:S]
        freqs_sin = self.freqs_sin[:S]

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
                h_out = blk(h, freqs_cos, freqs_sin)
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
                h = blk(h, freqs_cos, freqs_sin)
                if capture_aux and i in self._aux_head_layer_set:
                    aux_taps[i] = h
        elif capture_aux:
            # Full-depth path with aux heads enabled: enumerate to capture taps.
            for i, blk in enumerate(self.layers):
                h = blk(h, freqs_cos, freqs_sin)
                if i in self._aux_head_layer_set:
                    aux_taps[i] = h
        else:
            # Full-depth path — identical to original for torch.compile fast path
            for blk in self.layers:
                h = blk(h, freqs_cos, freqs_sin)

        # In scaffold_mode the main LM head is intentionally skipped — the
        # partial network's "LM head" is the deepest active aux head, and
        # running self.norm + self.output would (a) waste compute, (b) touch
        # uninitialised tail params via the all-gather, and (c) produce a
        # garbage loss against untrained weights. The aux taps captured above
        # carry the supervision.
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
                    accum_e_fp32=accum_fp32,
                    accum_c_fp32=accum_fp32,
                    reduction="mean",
                    ignore_index=pad_id,
                )
                if _want_zloss:
                    # Option D: no [N,V] materialization. Backend bool selects
                    # CCE fp32 accumulation in its backward (see _zloss_optionD).
                    (self._last_zloss, self._last_logz,
                     self._last_logz_rms, self._last_logz_p95) = _zloss_optionD(
                        h_flat, self.output.weight, tgt_flat, pad_id,
                        self._zloss_fp32_accum,
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
        logits = self.output(h)
        return logits, None

    # =========================================================================
    # Inference Forward with KV Caching
    # =========================================================================
    
    def generate_forward(
        self,
        tokens: torch.Tensor,
        start_pos: int = 0
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
        
        if self.attn_res_enabled:
            blocks = [h]
            partial_block = torch.zeros_like(h)
            bs = self.attn_res_block_size
            for i, blk in enumerate(self.layers):
                h = block_attn_res(blocks, partial_block, self.attn_res_queries[i],
                                   self.attn_res_key_norms[i].weight, self.attn_res_key_norms[i].eps)
                h_out = blk.forward_with_cache(h, freqs_cos, freqs_sin, start_pos)
                partial_block = partial_block + (h_out - h)
                if (i + 1) % bs == 0 and (i + 1) < len(self.layers):
                    blocks.append(partial_block)
                    partial_block = torch.zeros_like(h)
                h = h_out
        else:
            for blk in self.layers:
                h = blk.forward_with_cache(h, freqs_cos, freqs_sin, start_pos)

        h = self.norm(h)
        logits = self.output(h)
        return logits
