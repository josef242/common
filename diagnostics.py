# diagnostics.py
"""
Layer-wise diagnostics for LLM training with FSDP support.

Collects weight norms, gradient norms, and gradient-to-weight ratios
per layer block (attention, ffn) for monitoring training health.

Supports tied embeddings: when tok_embeddings and output share the same weight,
the final norm's weight gradient is used as a proxy for output-path intensity
(hook-free, torch.compile safe). Track the norm_to_embed_ratio trend over time.

Usage:
    from diagnostics import LayerDiagnostics

    diag = LayerDiagnostics(model, ddp_rank, ddp_world_size)

    # After backward(), before optimizer.step():
    diag.capture_gradients()

    # At validation time:
    diag.log_diagnostics(step, log_dir)
"""

import json
import os
import torch
import torch.distributed as dist
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict


@dataclass
class BlockStats:
    """Statistics for a single block (attention or ffn)."""
    w_norm: float  # Weight L2 norm (Frobenius)
    g_norm: float  # Gradient L2 norm
    ratio: float   # g_norm / w_norm
    w_rms: Optional[float] = None              # w_norm / sqrt(numel) — per-element scale
    param_delta_norm: Optional[float] = None   # ||W_after - W_before||_F (post-optimizer delta)
    param_delta_ratio: Optional[float] = None  # param_delta_norm / w_norm_before (== update_rms / w_rms)
    update_rms: Optional[float] = None         # post-optimizer update RMS (param_delta_norm / sqrt(numel))
    # Spectral / row / column geometry (only populated for select blocks: tok_embeddings, output)
    col_mean: Optional[float] = None           # mean of per-column RMS (over rows)
    col_p95: Optional[float] = None
    col_max: Optional[float] = None
    row_mean: Optional[float] = None           # mean of per-row RMS (over columns)
    row_p95: Optional[float] = None
    row_max: Optional[float] = None
    top_singular: Optional[float] = None       # largest singular value (power iteration)
    spectral_concentration: Optional[float] = None  # top_singular / w_norm
    feedback_gain: Optional[float] = None      # rms(dL/dh_in) / rms(dL/dlogits) — output head only


@dataclass
class LayerStats:
    """Statistics for a single transformer layer."""
    idx: int
    attn: BlockStats
    ffn: BlockStats                                  # aggregate (kept for backward compat)
    moe_enabled: bool = False
    experts: Optional[BlockStats] = None             # expert params only (w1/w2/w3 3D)
    shared_experts: Optional[BlockStats] = None      # shared expert params only
    gate: Optional[BlockStats] = None                # router gate weight
    expert_spread: Optional[Dict[str, float]] = None # {w_min, w_max, w_std, g_min, g_max, g_std}
    # Subgroup splits (kv3 FFN-controller / QK-norm investigation). All optional, None unless
    # the diagnostics tracker fills them. attn_qk vs attn_vo tests whether attention is quiet
    # only because QK-norm makes wq/wk forward-scale-invariant; ffn_w1w3 vs ffn_w2 tells whether
    # to eventually control SwiGLU in-proj only, w2 only, or all FFN. See docs/KV3_CONTROLLER_DESIGN.md.
    attn_qk: Optional[BlockStats] = None             # wq/wk (QK-norm scale-invariant)
    attn_vo: Optional[BlockStats] = None             # wv/wo (NOT QK-normed)
    ffn_w1w3: Optional[BlockStats] = None            # SwiGLU in-proj (gate+up, dim->inner)
    ffn_w2: Optional[BlockStats] = None              # SwiGLU down-proj (inner->dim)


@dataclass
class DiagnosticsSnapshot:
    """Complete diagnostics snapshot for one step."""
    step: int
    total_tokens: int  # Total tokens processed at this step
    tok_embeddings: BlockStats
    output: Optional[BlockStats]  # None if tied
    layers: List[LayerStats]
    # Gradient flow proxy for tied embeddings (None if not tied)
    # Keys: norm_grad (output-path proxy), embed_grad (combined), norm_to_embed_ratio
    tied_grad_flow: Optional[Dict[str, float]] = None


class LayerDiagnostics:
    """
    Collects layer-wise diagnostics with FSDP support.

    For FSDP: Each rank computes local shard norms, then we all_reduce
    the squared norms and take sqrt on rank 0 for the true global norm.

    For tied embeddings: Uses the final norm's weight gradient as a proxy
    for output-path intensity (hook-free, torch.compile safe). The norm
    weight gradient scales with how hard the output loss pushes through
    the final norm, while embed_grad is the combined gradient on the
    shared embedding weight. Track the ratio trend over time.
    """

    def __init__(self, model, ddp_rank: int, ddp_world_size: int, ddp: bool = True,
                 track_subgroups: bool = False):
        """
        Args:
            model: The FSDP-wrapped model (or compiled model wrapping FSDP)
            ddp_rank: This process's rank
            ddp_world_size: Total number of processes
            ddp: Whether we're in distributed mode
            track_subgroups: also track wq/wk-vs-wv/wo and w1/w3-vs-w2 pdr subgroups (kv3
                QK-norm investigation). Opt-in: roughly doubles the transient snapshot VRAM
                at diagnostic cadence, so off by default.
        """
        self.model = model
        self.ddp_rank = ddp_rank
        self.ddp_world_size = ddp_world_size
        self.ddp = ddp
        self._track_subgroups = track_subgroups

        # Stashed gradient norms from last backward pass
        # Values are float (scalar norm²) or list[float] (per-expert norm²)
        self._grad_norms: Dict[str, Any] = {}
        # Update ratio tracking (populated by snapshot_weights / capture_updates)
        self._weight_snapshots: Dict[str, tuple] = {}    # key → (params_list, cloned_tensors)
        self._snapshot_w_norm_sq: Dict[str, float] = {}  # key → local w_norm²
        self._snapshot_numel: Dict[str, int] = {}        # key → total (global) numel for the block
        self._update_norms: Dict[str, tuple] = {}        # key → (param_delta_norm, param_delta_ratio, update_rms)
        # Feedback gain (set by compute_feedback_gain, consumed by compute_diagnostics)
        self._feedback_gain: Optional[float] = None

        # Get the underlying model (handle torch.compile wrapper)
        self._raw_model = model._orig_mod if hasattr(model, "_orig_mod") else model

        # Tied embedding detection — use Python identity first (survives FSDP2
        # DTensor wrapping).  Only fall back to data_ptr() when there's no
        # output module at all (which implies tying via matmul with tok_embeddings).
        # NOTE: data_ptr() can false-positive under FSDP2 because DTensor shards
        # for separate parameters may share the same underlying storage slab.
        raw = self._raw_model
        if hasattr(raw, 'output') and raw.output is not None:
            self._tied_embeddings = raw.tok_embeddings.weight is raw.output.weight
        else:
            self._tied_embeddings = True  # No output module = must be tied

        self._init_message = None
        if self._tied_embeddings and ddp_rank == 0:
            self._init_message = ("Tied embeddings detected — using norm weight gradient "
                                  "as output-path proxy (hook-free, torch.compile safe)")

    def _compute_block_norm_squared(self, params: List[torch.nn.Parameter],
                                     use_grad: bool = False) -> torch.Tensor:
        """
        Compute sum of squared norms for a list of parameters (or their gradients).
        Returns a scalar tensor on the current device.

        Each tensor's local contribution is divided by _replication_factor so
        that the caller's world all-reduce (_reduce_norm_squared) counts every
        element exactly once. Two sqrt(world_size) inflation bugs lived here:
        (1) summing the full DTensor (resolves to the GLOBAL sum) before the
        all-reduce — fixed 2026-07-12 via local shards; (2) FSDP2 ROOT-group
        params (tok_embeddings/output/final norm) never reshard and appear as
        identical plain FULL tensors on every rank, so their local "shard" IS
        the full tensor — fixed 2026-07-13 via the replication divisor.
        """
        total = torch.tensor(0.0, device='cuda' if torch.cuda.is_available() else 'cpu')

        for p in params:
            t = p.grad if use_grad else p
            if t is None:
                continue
            total += (self._get_local_data(t).detach().float().pow(2).sum()
                      / self._replication_factor(t))

        return total

    def _reduce_norm_squared(self, local_norm_sq: torch.Tensor) -> float:
        """
        All-reduce squared norms across ranks and return the global norm.
        Only rank 0 gets the meaningful result, but all ranks participate.
        """
        if self.ddp and self.ddp_world_size > 1:
            dist.all_reduce(local_norm_sq, op=dist.ReduceOp.SUM)

        return local_norm_sq.sqrt().item()

    @staticmethod
    def _get_local_data(p):
        """Get local tensor data, bypassing DTensor dispatch for direct shard access."""
        return p.to_local() if hasattr(p, 'to_local') else p.data

    def _replication_factor(self, t) -> float:
        """How many ranks hold each element of t's LOCAL view. Contributions to a
        world all-reduce must be divided by this so every element counts once.

        - sharded DTensor: each element lives on exactly one rank -> 1.0
          (times the size of any Replicate mesh dim)
        - plain tensor in a distributed run: FSDP2 ROOT-group params
          (tok_embeddings / output / final norm — the root fully_shard group
          never reshards after forward) appear as the identical FULL tensor on
          every rank -> world_size. This was sqrt(8) inflation mechanism #2
          (2026-07-13): mechanism #1 (DTensor global sum) was fixed a day
          earlier, which cured the per-layer norms while out/emb stayed
          inflated — the tell that exposed this one.
        """
        if not (self.ddp and self.ddp_world_size > 1):
            return 1.0
        if hasattr(t, 'placements'):  # DTensor
            factor = 1.0
            mesh = t.device_mesh
            for i, placement in enumerate(t.placements):
                if placement.is_replicate():
                    factor *= mesh.size(i)
            return factor
        return float(self.ddp_world_size)

    def _get_attention_params(self, layer) -> List[torch.nn.Parameter]:
        """Get all parameters for the attention block of a layer."""
        if getattr(layer, 'use_gdn', False):
            gdn = layer.gdn_attn

            def _wts(m):
                # KDA (gdn_impl='kda') parameterizes some projections as
                # low-rank Sequentials (g_proj = Linear->Linear; f_proj
                # likewise); GDN uses plain Linears. Yield every leaf weight
                # either way — `.weight` on a Sequential is the AttributeError
                # that killed the kda-nope arm at its first telemetry cadence
                # (2026-07-30, step 50).
                if m is None:
                    return []
                if hasattr(m, 'weight') and m.weight is not None:
                    return [m.weight]
                return [c.weight for c in m.modules()
                        if hasattr(c, 'weight') and c.weight is not None]

            params = [*_wts(gdn.q_proj), *_wts(gdn.k_proj),
                      *_wts(gdn.v_proj), *_wts(gdn.o_proj)]
            params += _wts(getattr(gdn, 'g_proj', None))
            # Delta-rule dynamics: GDN carries a_proj/b_proj; KDA f_proj/b_proj
            for attr in ('a_proj', 'b_proj', 'f_proj'):
                params += _wts(getattr(gdn, attr, None))
            for attr in ('q_conv1d', 'k_conv1d', 'v_conv1d'):
                params += _wts(getattr(gdn, attr, None))
            params += _wts(getattr(gdn, 'o_norm', None))
            return params

        attn = layer.attention
        params = [attn.wq.weight, attn.wk.weight, attn.wv.weight, attn.wo.weight]

        # Include QK norm weights if present
        if hasattr(attn, 'q_norm') and attn.q_norm is not None:
            params.append(attn.q_norm.weight)
        if hasattr(attn, 'k_norm') and attn.k_norm is not None:
            params.append(attn.k_norm.weight)

        return params

    def _get_ffn_params(self, layer) -> List[torch.nn.Parameter]:
        """Get all parameters for the FFN block of a layer (dense or MoE)."""
        if getattr(layer, 'moe_enabled', False):
            params = [layer.moe.experts.w1, layer.moe.experts.w2, layer.moe.experts.w3]
            if layer.moe.shared_experts is not None:
                se = layer.moe.shared_experts
                params.extend([se.w1.weight, se.w2.weight, se.w3.weight])
            return params
        ff = layer.feed_forward
        return [ff.w1.weight, ff.w2.weight, ff.w3.weight]

    # --- Subgroup param lists (kv3 FFN-controller / QK-norm investigation) ---
    # Strict subsets of the attn/ffn aggregate blocks above (GDN small params / norms are
    # excluded — they belong to neither subgroup). Used to log per-subgroup pdr.
    def _get_attn_qk_params(self, layer) -> List[torch.nn.Parameter]:
        """Q/K projections only (forward scale-invariant under QK-norm)."""
        if getattr(layer, 'use_gdn', False):
            gdn = layer.gdn_attn
            return [gdn.q_proj.weight, gdn.k_proj.weight]
        attn = layer.attention
        return [attn.wq.weight, attn.wk.weight]

    def _get_attn_vo_params(self, layer) -> List[torch.nn.Parameter]:
        """V/O projections only (NOT covered by QK-norm)."""
        if getattr(layer, 'use_gdn', False):
            gdn = layer.gdn_attn
            return [gdn.v_proj.weight, gdn.o_proj.weight]
        attn = layer.attention
        return [attn.wv.weight, attn.wo.weight]

    def _get_ffn_w1w3_params(self, layer) -> List[torch.nn.Parameter]:
        """SwiGLU in-projections w1 (gate) + w3 (up), dim->inner_dim."""
        if getattr(layer, 'moe_enabled', False):
            params = [layer.moe.experts.w1, layer.moe.experts.w3]
            if layer.moe.shared_experts is not None:
                se = layer.moe.shared_experts
                params.extend([se.w1.weight, se.w3.weight])
            return params
        ff = layer.feed_forward
        return [ff.w1.weight, ff.w3.weight]

    def _get_ffn_w2_params(self, layer) -> List[torch.nn.Parameter]:
        """SwiGLU down-projection w2, inner_dim->dim."""
        if getattr(layer, 'moe_enabled', False):
            params = [layer.moe.experts.w2]
            if layer.moe.shared_experts is not None:
                params.append(layer.moe.shared_experts.w2.weight)
            return params
        ff = layer.feed_forward
        return [ff.w2.weight]

    def _get_expert_params(self, layer) -> List[torch.nn.Parameter]:
        """Get expert 3D params (w1/w2/w3) for a MoE layer."""
        return [layer.moe.experts.w1, layer.moe.experts.w2, layer.moe.experts.w3]

    def _get_shared_expert_params(self, layer) -> Optional[List[torch.nn.Parameter]]:
        """Get shared expert params, or None if no shared experts."""
        if layer.moe.shared_experts is None:
            return None
        se = layer.moe.shared_experts
        return [se.w1.weight, se.w2.weight, se.w3.weight]

    def _get_gate_params(self, layer) -> List[torch.nn.Parameter]:
        """Get router gate weight."""
        return [layer.moe.router.gate.weight]

    def _compute_per_expert_norms_squared(self, layer, use_grad: bool = False) -> List[torch.Tensor]:
        """Compute per-expert norm² for each local expert. Returns list of scalar tensors.

        With efsdp=1 (EP=world_size), each rank fully owns its local experts so
        norms are exact.  With efsdp>1, norms are partial shards — still
        directionally useful for spread (relative differences).
        """
        experts = layer.moe.experts
        num_local = experts.num_experts  # GroupedExperts.num_experts = num_local_experts
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        per_expert = []
        for i in range(num_local):
            total = torch.tensor(0.0, device=device)
            for p in [experts.w1, experts.w2, experts.w3]:
                data = p.grad if use_grad else p
                if data is None:
                    continue
                # Slice expert i from 3D tensor (handles DTensor via local data)
                local = data._local_tensor if hasattr(data, '_local_tensor') else data
                total += local[i].detach().float().pow(2).sum()
            per_expert.append(total)
        return per_expert

    def _compute_expert_spread(self, layer, layer_idx: int, device: str) -> Dict[str, float]:
        """Compute per-expert weight/gradient norm spread as summary stats.

        Uses all_gather (no all_reduce) to collect per-local-expert norms from
        all ranks.  With efsdp=1 (our EP setup), each rank contributes unique
        experts.  With efsdp>1, norms are partial shards — spread is approximate.
        """
        # Weight norms (local experts)
        per_expert_w = self._compute_per_expert_norms_squared(layer, use_grad=False)
        w_local = torch.tensor([t.item() for t in per_expert_w], device=device)

        # Gradient norms (from stashed per-expert values)
        per_g_list = self._grad_norms.get(f'layer_{layer_idx}_per_expert_g', [])
        g_local = torch.tensor(per_g_list, device=device) if per_g_list else torch.zeros_like(w_local)

        # Gather across all ranks (each contributes num_local_experts values)
        if self.ddp and self.ddp_world_size > 1:
            w_gathered = [torch.zeros_like(w_local) for _ in range(self.ddp_world_size)]
            g_gathered = [torch.zeros_like(g_local) for _ in range(self.ddp_world_size)]
            dist.all_gather(w_gathered, w_local)
            dist.all_gather(g_gathered, g_local)
            # With EP (efsdp=1): each rank has unique experts → concat = all experts
            # With EP (efsdp>1): duplicates exist → take first num_experts
            num_experts = layer.moe.num_experts
            w_all = torch.cat(w_gathered).sqrt()[:num_experts]
            g_all = torch.cat(g_gathered).sqrt()[:num_experts]
        else:
            w_all = w_local.sqrt()
            g_all = g_local.sqrt()

        return {
            'w_min': w_all.min().item(), 'w_max': w_all.max().item(),
            'w_std': w_all.std().item() if len(w_all) > 1 else 0.0,
            'g_min': g_all.min().item(), 'g_max': g_all.max().item(),
            'g_std': g_all.std().item() if len(g_all) > 1 else 0.0,
        }

    @staticmethod
    def _block_numel(params: List[torch.nn.Parameter]) -> int:
        """Total (global, logical) element count across a block of params.

        Note: torch.nn.Parameter wrapping a DTensor exposes the global logical
        shape via .numel(), so this is correct under FSDP2.
        """
        return sum(int(p.numel()) for p in params)

    @staticmethod
    def _to_full_tensor(p: torch.nn.Parameter) -> torch.Tensor:
        """Materialize a parameter to its full logical shape on every rank.

        For FSDP2 DTensors this issues an all_gather collective; must be called
        on all ranks. Returns a regular tensor (float32) with grad disabled.
        """
        if hasattr(p, 'full_tensor'):
            return p.full_tensor().detach().float()
        return p.detach().float()

    @torch.no_grad()
    def _compute_spectral_geometry(self, p: torch.nn.Parameter) -> Dict[str, float]:
        """Per-row, per-column, and top-singular geometry on a 2D weight.

        Materializes the full weight via full_tensor() (collective). Computation
        is deterministic so all ranks compute identical values; only rank 0 logs.
        Returns {} if the parameter is not 2D.
        """
        W = self._to_full_tensor(p)
        try:
            if W.dim() != 2:
                return {}

            # Per-column / per-row RMS
            col_rms = W.pow(2).mean(dim=0).sqrt()  # [cols]
            row_rms = W.pow(2).mean(dim=1).sqrt()  # [rows]

            fro = W.norm().item()

            # Top singular value via 5-iter power iteration. Fixed seed for
            # cross-rank determinism (full_tensor returns identical W on all ranks).
            n_rows, n_cols = W.shape
            gen = torch.Generator(device=W.device).manual_seed(0xABCDEF)
            u = torch.randn(n_cols, generator=gen, device=W.device, dtype=W.dtype)
            u = u / u.norm().clamp_min(1e-30)
            for _ in range(5):
                v = W @ u
                v = v / v.norm().clamp_min(1e-30)
                u = W.T @ v
                u = u / u.norm().clamp_min(1e-30)
            top = (W @ u).norm().item()

            return {
                'col_mean': col_rms.mean().item(),
                'col_p95': col_rms.quantile(0.95).item(),
                'col_max': col_rms.max().item(),
                'row_mean': row_rms.mean().item(),
                'row_p95': row_rms.quantile(0.95).item(),
                'row_max': row_rms.max().item(),
                'top_singular': top,
                'spectral_concentration': top / fro if fro > 1e-30 else 0.0,
            }
        finally:
            del W  # free transient full-tensor materialization on every rank

    def compute_feedback_gain(self, x: torch.Tensor, y: torch.Tensor) -> Optional[float]:
        """Empirical feedback gain through the output head: rms(dL/dh) / rms(dL/dlogits).

        The training path uses fused linear+cross-entropy (CCE), which has no
        explicit `output(h)` forward — backward hooks on the head don't fire.
        We instead run the model's eval branch (which uses explicit `self.output(h)`),
        capture `h` via a forward hook on `model.norm`, and compute autograd.grad
        on `[h, logits]` against a standard cross_entropy loss.

        Does NOT pollute model.grad (uses torch.autograd.grad on a fresh graph).
        Stashes the result on `self._feedback_gain` for `compute_diagnostics` to
        pick up on the next call. Returns the value or None on failure.

        Should be called once per val-cadence diagnostic step on all ranks
        (it issues a collective all_reduce internally).
        """
        raw = self._raw_model
        captured: Dict[str, torch.Tensor] = {}

        def _hook(_module, _input, output):
            captured['h'] = output

        handle = raw.norm.register_forward_hook(_hook)
        try:
            with torch.enable_grad():
                # Eval branch: targets=None → returns (logits, None) using explicit self.output(h)
                logits, _ = raw(x)

                if 'h' not in captured or logits is None:
                    self._feedback_gain = None
                    return None

                h = captured['h']
                pad_id = raw.params.pad_id
                loss = torch.nn.functional.cross_entropy(
                    logits.reshape(-1, logits.size(-1)),
                    y.reshape(-1),
                    ignore_index=pad_id,
                )

                # Grads on intermediate tensors only — does not touch model.grad
                grad_h, grad_logits = torch.autograd.grad(
                    outputs=loss,
                    inputs=[h, logits],
                    retain_graph=False,
                )

                # Reduce sum-of-squares + element counts across ranks for global RMS
                device = grad_h.device
                stack = torch.stack([
                    grad_h.detach().float().pow(2).sum(),
                    torch.tensor(float(grad_h.numel()), device=device),
                    grad_logits.detach().float().pow(2).sum(),
                    torch.tensor(float(grad_logits.numel()), device=device),
                ])
                if self.ddp and self.ddp_world_size > 1:
                    dist.all_reduce(stack, op=dist.ReduceOp.SUM)
                h_sq, h_n, l_sq, l_n = stack.tolist()

                h_rms = (h_sq / h_n) ** 0.5 if h_n > 0 else 0.0
                l_rms = (l_sq / l_n) ** 0.5 if l_n > 0 else 0.0
                gain = h_rms / l_rms if l_rms > 1e-30 else 0.0
                self._feedback_gain = gain
                return gain
        finally:
            handle.remove()
            captured.clear()

    def _compute_block_stats(self, params: List[torch.nn.Parameter],
                             grad_norm_sq: Optional[torch.Tensor] = None) -> BlockStats:
        """
        Compute BlockStats for a set of parameters.

        If grad_norm_sq is provided, use it. Otherwise compute from current gradients.
        """
        # Weight norm
        w_norm_sq = self._compute_block_norm_squared(params, use_grad=False)
        w_norm = self._reduce_norm_squared(w_norm_sq)

        # Gradient norm
        if grad_norm_sq is not None:
            g_norm = self._reduce_norm_squared(grad_norm_sq)
        else:
            g_norm_sq = self._compute_block_norm_squared(params, use_grad=True)
            g_norm = self._reduce_norm_squared(g_norm_sq)

        # Ratio (avoid division by zero)
        ratio = g_norm / w_norm if w_norm > 1e-10 else 0.0

        # Per-element scale (RMS) — w_norm is Frobenius == sqrt(sum sq)
        numel = self._block_numel(params)
        w_rms = w_norm / (numel ** 0.5) if numel > 0 else 0.0

        return BlockStats(w_norm=w_norm, g_norm=g_norm, ratio=ratio, w_rms=w_rms)

    def capture_gradients(self):
        """
        Capture gradient norms right after backward(), before optimizer.step().

        Call this immediately after loss.backward() completes.
        Stashes the squared norms for later use at validation time.

        IMPORTANT: Must be called every step for accurate gradient snapshots.
        """
        self._grad_norms.clear()

        model = self._raw_model

        # tok_embeddings (combined gradient when tied — this is the actual optimizer signal)
        if hasattr(model, 'tok_embeddings'):
            g_sq = self._compute_block_norm_squared([model.tok_embeddings.weight], use_grad=True)
            self._grad_norms['tok_embeddings'] = g_sq.item()

        # output path proxy: final norm weight gradient scales with output-path intensity
        if self._tied_embeddings:
            if hasattr(model, 'norm') and hasattr(model.norm, 'weight'):
                g_sq = self._compute_block_norm_squared([model.norm.weight], use_grad=True)
                self._grad_norms['norm_weight'] = g_sq.item()
        elif hasattr(model, 'output') and model.output is not None:
            g_sq = self._compute_block_norm_squared([model.output.weight], use_grad=True)
            self._grad_norms['output'] = g_sq.item()

        # Per-layer attention and FFN
        if hasattr(model, 'layers'):
            for i, layer in enumerate(model.layers):
                # Attention block
                attn_params = self._get_attention_params(layer)
                g_sq = self._compute_block_norm_squared(attn_params, use_grad=True)
                self._grad_norms[f'layer_{i}_attn'] = g_sq.item()

                # FFN block (aggregate — backward compat)
                ffn_params = self._get_ffn_params(layer)
                g_sq = self._compute_block_norm_squared(ffn_params, use_grad=True)
                self._grad_norms[f'layer_{i}_ffn'] = g_sq.item()

                # Subgroup splits (kv3): wq/wk vs wv/wo, w1/w3 vs w2 (opt-in)
                if self._track_subgroups:
                    for _sg, _fn in (('attn_qk', self._get_attn_qk_params),
                                     ('attn_vo', self._get_attn_vo_params),
                                     ('ffn_w1w3', self._get_ffn_w1w3_params),
                                     ('ffn_w2', self._get_ffn_w2_params)):
                        self._grad_norms[f'layer_{i}_{_sg}'] = \
                            self._compute_block_norm_squared(_fn(layer), use_grad=True).item()

                # MoE component breakdown
                if getattr(layer, 'moe_enabled', False):
                    # Expert params only
                    expert_params = self._get_expert_params(layer)
                    g_sq = self._compute_block_norm_squared(expert_params, use_grad=True)
                    self._grad_norms[f'layer_{i}_experts'] = g_sq.item()

                    # Shared expert params
                    shared_params = self._get_shared_expert_params(layer)
                    if shared_params:
                        g_sq = self._compute_block_norm_squared(shared_params, use_grad=True)
                        self._grad_norms[f'layer_{i}_shared_experts'] = g_sq.item()

                    # Router gate
                    gate_params = self._get_gate_params(layer)
                    g_sq = self._compute_block_norm_squared(gate_params, use_grad=True)
                    self._grad_norms[f'layer_{i}_gate'] = g_sq.item()

                    # Per-expert gradient norms (for spread)
                    per_expert_g = self._compute_per_expert_norms_squared(layer, use_grad=True)
                    self._grad_norms[f'layer_{i}_per_expert_g'] = [t.item() for t in per_expert_g]

    def snapshot_weights(self):
        """
        Clone tracked parameters before optimizer.step() for update ratio computation.

        Call immediately before optimizer.step() on diagnostic steps only.
        Snapshots are freed after capture_updates().
        """
        self._weight_snapshots.clear()
        self._snapshot_w_norm_sq.clear()

        model = self._raw_model
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

        # tok_embeddings
        if hasattr(model, 'tok_embeddings'):
            self._snapshot_block('tok_embeddings', [model.tok_embeddings.weight], device)

        # output (only if not tied)
        if not self._tied_embeddings and hasattr(model, 'output') and model.output is not None:
            self._snapshot_block('output', [model.output.weight], device)

        # Per-layer attn and ffn
        if hasattr(model, 'layers'):
            for i, layer in enumerate(model.layers):
                self._snapshot_block(f'layer_{i}_attn', self._get_attention_params(layer), device)
                self._snapshot_block(f'layer_{i}_ffn', self._get_ffn_params(layer), device)
                # Subgroup splits (kv3, opt-in): capture_updates is key-driven, so these
                # auto-produce _update_norms[layer_{i}_{attn_qk|attn_vo|ffn_w1w3|ffn_w2}].
                if self._track_subgroups:
                    self._snapshot_block(f'layer_{i}_attn_qk', self._get_attn_qk_params(layer), device)
                    self._snapshot_block(f'layer_{i}_attn_vo', self._get_attn_vo_params(layer), device)
                    self._snapshot_block(f'layer_{i}_ffn_w1w3', self._get_ffn_w1w3_params(layer), device)
                    self._snapshot_block(f'layer_{i}_ffn_w2', self._get_ffn_w2_params(layer), device)

    def _snapshot_block(self, key: str, params: List[torch.nn.Parameter], device: str):
        """Clone param data and record local w_norm² + global numel for a single block.

        Contributions are divided by _replication_factor. NOTE: at this call
        point (post-backward, pre-step) ALL params — including the root group —
        are sharded DTensors, so the factor is 1 and historical
        param_delta_norm/update_rms were always correct (verified 2026-07-13
        via pdn/pdr reconstruction). The divisor is kept as defense: it engages
        only if a plain replicated tensor ever reaches this path."""
        clones = []
        w_norm_sq = torch.tensor(0.0, device=device)
        for p in params:
            local = self._get_local_data(p)
            clones.append(local.detach().clone())
            w_norm_sq += local.detach().float().pow(2).sum() / self._replication_factor(p)
        self._weight_snapshots[key] = (params, clones)
        self._snapshot_w_norm_sq[key] = w_norm_sq
        self._snapshot_numel[key] = self._block_numel(params)

    def capture_updates(self):
        """
        Compute update norms by comparing current params to pre-step snapshots.

        Call immediately after optimizer.step() on diagnostic steps.
        Frees snapshots after computation.
        """
        if not self._weight_snapshots:
            return

        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self._update_norms.clear()
        keys = list(self._weight_snapshots.keys())
        n = len(keys)

        # Batch all local values for a single all_reduce: [delta_sq_0, w_sq_0, ...]
        local_vals = torch.zeros(n * 2, device=device)
        for i, key in enumerate(keys):
            params, clones = self._weight_snapshots[key]
            delta_sq = torch.tensor(0.0, device=device)
            for p, clone in zip(params, clones):
                local = self._get_local_data(p)
                delta_sq += ((local.detach().float() - clone.float()).pow(2).sum()
                             / self._replication_factor(p))
            local_vals[i * 2] = delta_sq
            local_vals[i * 2 + 1] = self._snapshot_w_norm_sq[key]

        # Single all_reduce for all blocks
        if self.ddp and self.ddp_world_size > 1:
            dist.all_reduce(local_vals, op=dist.ReduceOp.SUM)

        # Unpack global norms
        for i, key in enumerate(keys):
            delta_norm = local_vals[i * 2].sqrt().item()
            w_norm_before = local_vals[i * 2 + 1].sqrt().item()
            update_ratio = delta_norm / w_norm_before if w_norm_before > 1e-10 else 0.0
            numel = self._snapshot_numel.get(key, 0)
            update_rms = delta_norm / (numel ** 0.5) if numel > 0 else 0.0
            self._update_norms[key] = (delta_norm, update_ratio, update_rms)

        # Free snapshots
        self._weight_snapshots.clear()
        self._snapshot_w_norm_sq.clear()
        self._snapshot_numel.clear()

    def compute_diagnostics(self, step: int, total_tokens: int = 0) -> DiagnosticsSnapshot:
        """
        Compute full diagnostics snapshot.

        Uses stashed gradient norms from capture_gradients() and
        computes current weight norms.

        Args:
            step: Current training step
            total_tokens: Total tokens processed so far
        """
        model = self._raw_model
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

        # tok_embeddings (combined weight gradient when tied)
        tok_emb_params = [model.tok_embeddings.weight]
        tok_emb_g_sq = torch.tensor(self._grad_norms.get('tok_embeddings', 0.0), device=device)
        tok_emb_stats = self._compute_block_stats(tok_emb_params, tok_emb_g_sq)

        # Spectral / col / row geometry for the embedding weight.
        # Always compute (the embedding is the output head when tied, so this
        # covers both regimes from a single materialization).
        emb_spec = self._compute_spectral_geometry(model.tok_embeddings.weight)
        for k, v in emb_spec.items():
            setattr(tok_emb_stats, k, v)

        # output and tied gradient flow
        output_stats = None
        tied_grad_flow = None

        if self._tied_embeddings:
            # Proxy: final norm weight gradient as output-path intensity indicator
            # The tok_embeddings gradient is the combined signal (output + embed paths).
            # The norm weight gradient scales with output-path intensity only.
            norm_g_sq = torch.tensor(
                self._grad_norms.get('norm_weight', 0.0), device=device)
            tok_g_sq = torch.tensor(
                self._grad_norms.get('tok_embeddings', 0.0), device=device)

            norm_g = self._reduce_norm_squared(norm_g_sq)
            tok_g = self._reduce_norm_squared(tok_g_sq)

            tied_grad_flow = {
                'norm_grad': norm_g,        # Proxy for output-path intensity
                'embed_grad': tok_g,        # Combined gradient on shared weight
                'norm_to_embed_ratio': norm_g / tok_g if tok_g > 1e-10 else 0.0,
            }

        elif hasattr(model, 'output') and model.output is not None:
            output_params = [model.output.weight]
            output_g_sq = torch.tensor(self._grad_norms.get('output', 0.0), device=device)
            output_stats = self._compute_block_stats(output_params, output_g_sq)
            # Spectral / col / row geometry for the output head
            out_spec = self._compute_spectral_geometry(model.output.weight)
            for k, v in out_spec.items():
                setattr(output_stats, k, v)

        # Per-layer stats
        layer_stats = []
        if hasattr(model, 'layers'):
            for i, layer in enumerate(model.layers):
                # Attention
                attn_params = self._get_attention_params(layer)
                attn_g_sq = torch.tensor(self._grad_norms.get(f'layer_{i}_attn', 0.0), device=device)
                attn_stats = self._compute_block_stats(attn_params, attn_g_sq)

                # FFN
                ffn_params = self._get_ffn_params(layer)
                ffn_g_sq = torch.tensor(self._grad_norms.get(f'layer_{i}_ffn', 0.0), device=device)
                ffn_stats = self._compute_block_stats(ffn_params, ffn_g_sq)

                # Subgroup splits (kv3, opt-in): wq/wk vs wv/wo, w1/w3 vs w2
                attn_qk_stats = attn_vo_stats = ffn_w1w3_stats = ffn_w2_stats = None
                if self._track_subgroups:
                    def _sg_stats(_sg, _params, _i=i):   # bind i at def-time (not a late closure)
                        _g = torch.tensor(self._grad_norms.get(f'layer_{_i}_{_sg}', 0.0), device=device)
                        return self._compute_block_stats(_params, _g)
                    attn_qk_stats = _sg_stats('attn_qk', self._get_attn_qk_params(layer))
                    attn_vo_stats = _sg_stats('attn_vo', self._get_attn_vo_params(layer))
                    ffn_w1w3_stats = _sg_stats('ffn_w1w3', self._get_ffn_w1w3_params(layer))
                    ffn_w2_stats = _sg_stats('ffn_w2', self._get_ffn_w2_params(layer))

                # MoE component breakdown
                moe_enabled = getattr(layer, 'moe_enabled', False)
                expert_stats = None
                shared_stats = None
                gate_stats = None
                expert_spread = None

                if moe_enabled:
                    # Experts
                    expert_params = self._get_expert_params(layer)
                    expert_g_sq = torch.tensor(
                        self._grad_norms.get(f'layer_{i}_experts', 0.0), device=device)
                    expert_stats = self._compute_block_stats(expert_params, expert_g_sq)

                    # Shared experts
                    shared_params = self._get_shared_expert_params(layer)
                    if shared_params:
                        shared_g_sq = torch.tensor(
                            self._grad_norms.get(f'layer_{i}_shared_experts', 0.0), device=device)
                        shared_stats = self._compute_block_stats(shared_params, shared_g_sq)

                    # Gate
                    gate_params = self._get_gate_params(layer)
                    gate_g_sq = torch.tensor(
                        self._grad_norms.get(f'layer_{i}_gate', 0.0), device=device)
                    gate_stats = self._compute_block_stats(gate_params, gate_g_sq)

                    # Per-expert spread
                    expert_spread = self._compute_expert_spread(layer, i, device)

                layer_stats.append(LayerStats(
                    idx=i, attn=attn_stats, ffn=ffn_stats,
                    moe_enabled=moe_enabled, experts=expert_stats,
                    shared_experts=shared_stats, gate=gate_stats,
                    expert_spread=expert_spread,
                    attn_qk=attn_qk_stats, attn_vo=attn_vo_stats,
                    ffn_w1w3=ffn_w1w3_stats, ffn_w2=ffn_w2_stats,
                ))

        snapshot = DiagnosticsSnapshot(
            step=step,
            total_tokens=total_tokens,
            tok_embeddings=tok_emb_stats,
            output=output_stats,
            layers=layer_stats,
            tied_grad_flow=tied_grad_flow,
        )

        # Overlay update ratio data from snapshot_weights / capture_updates
        if self._update_norms:
            self._overlay_update_data(snapshot)
            self._update_norms.clear()

        # Overlay feedback gain (set by compute_feedback_gain at val cadence).
        # Attach to output stats when untied; to tok_embeddings when tied.
        if self._feedback_gain is not None:
            target = snapshot.output if snapshot.output is not None else snapshot.tok_embeddings
            if target is not None:
                target.feedback_gain = self._feedback_gain
            self._feedback_gain = None  # consume

        return snapshot

    def _overlay_update_data(self, snapshot: DiagnosticsSnapshot):
        """Populate param_delta_norm / param_delta_ratio / update_rms fields from captured update data."""
        def _apply(stats: Optional[BlockStats], key: str):
            if stats is not None and key in self._update_norms:
                pdn, pdr, urms = self._update_norms[key]
                stats.param_delta_norm = pdn
                stats.param_delta_ratio = pdr
                stats.update_rms = urms

        _apply(snapshot.tok_embeddings, 'tok_embeddings')
        _apply(snapshot.output, 'output')
        for ls in snapshot.layers:
            _apply(ls.attn, f'layer_{ls.idx}_attn')
            _apply(ls.ffn, f'layer_{ls.idx}_ffn')
            # Subgroup splits (kv3): param_delta_ratio for wq/wk, wv/wo, w1/w3, w2
            _apply(ls.attn_qk, f'layer_{ls.idx}_attn_qk')
            _apply(ls.attn_vo, f'layer_{ls.idx}_attn_vo')
            _apply(ls.ffn_w1w3, f'layer_{ls.idx}_ffn_w1w3')
            _apply(ls.ffn_w2, f'layer_{ls.idx}_ffn_w2')

    @staticmethod
    def _block_to_dict(bs: BlockStats) -> dict:
        """Convert BlockStats to dict, omitting None values for clean JSONL."""
        return {k: v for k, v in asdict(bs).items() if v is not None}

    @staticmethod
    def _layer_to_dict(ls: LayerStats) -> dict:
        """Convert a LayerStats to a JSON-serializable dict with optional MoE fields."""
        bd = LayerDiagnostics._block_to_dict
        d = {'idx': ls.idx, 'attn': bd(ls.attn), 'ffn': bd(ls.ffn)}
        # Subgroup splits (kv3): emit when present (None unless the tracker filled them)
        for _name, _bs in (('attn_qk', ls.attn_qk), ('attn_vo', ls.attn_vo),
                           ('ffn_w1w3', ls.ffn_w1w3), ('ffn_w2', ls.ffn_w2)):
            if _bs is not None:
                d[_name] = bd(_bs)
        if ls.moe_enabled:
            d['moe_enabled'] = True
            if ls.experts is not None:
                d['experts'] = bd(ls.experts)
            if ls.shared_experts is not None:
                d['shared_experts'] = bd(ls.shared_experts)
            if ls.gate is not None:
                d['gate'] = bd(ls.gate)
            if ls.expert_spread is not None:
                d['expert_spread'] = ls.expert_spread
        return d

    def log_diagnostics(self, step: int, log_dir: str, total_tokens: int = 0,
                         filename: str = "diagnostics.jsonl",
                         awd_data: dict = None, moe_data: dict = None,
                         activation_data: dict = None, zloss_data: dict = None,
                         rc_data: dict = None, cg_data: dict = None,
                         mtp_data: dict = None):
        """
        Compute diagnostics and write to JSONL file.
        Only rank 0 writes to the file.

        Args:
            step: Current training step
            log_dir: Directory to write the log file
            total_tokens: Total tokens processed so far
            filename: Name of the JSONL file
            awd_data: Optional AWD per-component state
            moe_data: Optional MoE balance stats
            activation_data: Optional forward-activation RMS profile produced by
                ActivationProbe.detach_and_get(). Schema:
                  {'layers_by_idx': {idx: {h_in_rms, attn_out_rms, h_mid_rms,
                                            ffn_out_rms, h_out_rms}},
                   'final_norm_in_rms': float,
                   'final_norm_out_rms': float}
                Per-layer entries are folded into each layer record under 'act';
                top-level entries become top-level fields.
        """
        snapshot = self.compute_diagnostics(step, total_tokens)

        # Only rank 0 writes
        if self.ddp_rank == 0:
            # Convert to JSON-serializable dict
            data = {
                'step': snapshot.step,
                # norm_v 3 = replication-aware reductions everywhere: v2 fixed the
                # DTensor global-sum double-count (layer norms) but missed FSDP2
                # ROOT-group params (out/emb are unsharded full tensors on every
                # rank), which kept w_norm/param_delta_norm/update_rms inflated by
                # sqrt(world_size) for exactly those blocks. Unstamped and v2 lines
                # are retro-rescaled by tools/fix_diagnostics_norms.py.
                'norm_v': 3,
                'total_tokens': snapshot.total_tokens,
                'tok_embeddings': self._block_to_dict(snapshot.tok_embeddings),
                'output': self._block_to_dict(snapshot.output) if snapshot.output else None,
                'layers': [
                    self._layer_to_dict(ls)
                    for ls in snapshot.layers
                ]
            }

            # Include tied gradient flow decomposition when applicable
            if snapshot.tied_grad_flow is not None:
                data['tied_grad_flow'] = snapshot.tied_grad_flow

            # Include AWD state when provided
            if awd_data is not None:
                data['awd'] = awd_data

            # Include MoE balance stats when provided
            if moe_data is not None:
                # Convert tuples to dicts for clean JSONL
                data['moe'] = {
                    'avg_cv': moe_data['avg_cv'],
                    'per_layer': [
                        {'layer': lid, 'pct': pct, 'cv': cv, 'bias': bias}
                        for lid, pct, cv, bias in moe_data['per_layer']
                    ]
                }

            # Include z-loss stats when provided (z-loss enabled). Snapshot of
            # the latest per-step values, recorded at this val cadence so the
            # dashboard gets a structured time series alongside the other
            # diagnostics. Per-step resolution still lives in train_log.txt.
            if zloss_data is not None:
                data['z_loss'] = zloss_data

            # Include MTP stats when provided (mtp enabled). Same val-cadence
            # snapshot pattern as z_loss: loss = all-reduced t+2 CE this step,
            # gap = mtp_loss - main_loss (the plan-ahead difficulty margin —
            # should stay positive and slowly NARROW as look-ahead circuitry
            # forms), w_norm = aggregate MTP module weight norm (module health).
            if mtp_data is not None:
                data['mtp'] = mtp_data

            # Include row-center (gauge subtraction) stats when provided. Same
            # val-cadence snapshot pattern as z_loss: muW_pre is the per-step
            # gauge regrowth rate (the real diagnostic), muW_post ~0 confirms the
            # projection took, m_bar is the 1st-moment gauge, proj_ratio is the
            # gauge's relative size in the head.
            if rc_data is not None:
                data['row_center'] = rc_data

            # Include centered head-geometry health metrics when provided (Item B):
            # the canonical post-row-centering head-health series (||W_c||, s1_c,
            # spectral_concentration_c, effective_rank_c, small-sigma percentiles)
            # and the dn1-collapse early-warning. val-cadence snapshot.
            if cg_data is not None:
                data['centered_geom'] = cg_data

            # Merge forward-activation RMS profile when provided.
            if activation_data is not None:
                layers_by_idx = activation_data.get('layers_by_idx') or {}
                for layer_dict in data['layers']:
                    act = layers_by_idx.get(layer_dict['idx'])
                    if act:
                        layer_dict['act'] = act
                if 'final_norm_in_rms' in activation_data:
                    data['final_norm_in_rms'] = activation_data['final_norm_in_rms']
                if 'final_norm_out_rms' in activation_data:
                    data['final_norm_out_rms'] = activation_data['final_norm_out_rms']

            # Append to JSONL file
            filepath = os.path.join(log_dir, filename)
            with open(filepath, 'a') as f:
                f.write(json.dumps(data) + '\n')

        return snapshot

    def print_summary(self, snapshot: DiagnosticsSnapshot, logger=None,
                       awd_data: dict = None, moe_data: dict = None):
        """
        Print a human-readable summary of the diagnostics.
        Only rank 0 prints.

        Args:
            snapshot: Diagnostics snapshot to summarize
            logger: Optional logger with print_and_log method
            awd_data: Optional AWD per-component state from AdaptiveWD.get_diagnostics_data()
            moe_data: Optional MoE balance stats from _update_expert_bias hook
        """
        if self.ddp_rank != 0:
            return

        def log(msg):
            if logger:
                logger.print_and_log(msg)
            else:
                print(msg)

        def _mean(vals):
            return sum(vals) / len(vals)

        log(f"=== Diagnostics @ step {snapshot.step} ===")
        te = snapshot.tok_embeddings
        te_line = (f"  tok_embeddings: w={te.w_norm:.4f} "
                   f"g={te.g_norm:.6f} ratio={te.ratio:.6f}")
        if te.param_delta_ratio is not None:
            te_line += f" update={te.param_delta_ratio:.6f}"
        if te.update_rms is not None:
            te_line += f" urms={te.update_rms:.2e}"
        log(te_line)
        if te.top_singular is not None:
            log(f"    geom: col={te.col_mean:.4f}/{te.col_p95:.4f}/{te.col_max:.4f} "
                f"row={te.row_mean:.4f}/{te.row_p95:.4f}/{te.row_max:.4f} "
                f"sigma1={te.top_singular:.4f} concentration={te.spectral_concentration:.4f}")
        if te.feedback_gain is not None:  # tied case
            log(f"    feedback_gain={te.feedback_gain:.4f}")

        if snapshot.output:
            out = snapshot.output
            out_line = (f"  output:         w={out.w_norm:.4f} "
                        f"g={out.g_norm:.6f} ratio={out.ratio:.6f}")
            if out.param_delta_ratio is not None:
                out_line += f" update={out.param_delta_ratio:.6f}"
            if out.update_rms is not None:
                out_line += f" urms={out.update_rms:.2e}"
            log(out_line)
            if out.top_singular is not None:
                log(f"    geom: col={out.col_mean:.4f}/{out.col_p95:.4f}/{out.col_max:.4f} "
                    f"row={out.row_mean:.4f}/{out.row_p95:.4f}/{out.row_max:.4f} "
                    f"sigma1={out.top_singular:.4f} concentration={out.spectral_concentration:.4f}")
            if out.feedback_gain is not None:
                log(f"    feedback_gain={out.feedback_gain:.4f}")

        if snapshot.tied_grad_flow:
            tgf = snapshot.tied_grad_flow
            log(f"  tied_grad_flow: norm_grad={tgf['norm_grad']:.6f} "
                f"embed_grad={tgf['embed_grad']:.6f} "
                f"ratio={tgf['norm_to_embed_ratio']:.4f}")

        # Summary stats across layers — separate MoE vs dense
        if snapshot.layers:
            moe_layers = [l for l in snapshot.layers if l.moe_enabled]
            dense_layers = [l for l in snapshot.layers if not l.moe_enabled]
            n_moe = len(moe_layers)

            label = f"layers (n={len(snapshot.layers)}"
            if n_moe > 0:
                label += f", {n_moe} MoE, {len(dense_layers)} dense"
            label += "):"
            log(f"  {label}")

            # Attention (all layers)
            attn_ratios = [l.attn.ratio for l in snapshot.layers]
            log(f"    attn ratio:    min={min(attn_ratios):.6f} "
                f"max={max(attn_ratios):.6f} "
                f"mean={_mean(attn_ratios):.6f}")

            attn_updates = [l.attn.param_delta_ratio for l in snapshot.layers if l.attn.param_delta_ratio is not None]
            if attn_updates:
                log(f"    attn update:   min={min(attn_updates):.6f} "
                    f"max={max(attn_updates):.6f} "
                    f"mean={_mean(attn_updates):.6f}")

            # Dense FFN
            if dense_layers:
                ffn_ratios = [l.ffn.ratio for l in dense_layers]
                log(f"    ffn ratio:     min={min(ffn_ratios):.6f} "
                    f"max={max(ffn_ratios):.6f} "
                    f"mean={_mean(ffn_ratios):.6f}")

                ffn_updates = [l.ffn.param_delta_ratio for l in dense_layers if l.ffn.param_delta_ratio is not None]
                if ffn_updates:
                    log(f"    ffn update:    min={min(ffn_updates):.6f} "
                        f"max={max(ffn_updates):.6f} "
                        f"mean={_mean(ffn_updates):.6f}")

            # Subgroup pdr (kv3 QK-norm / FFN-split investigation) — only when tracked, so kv2
            # and other runs' stdout is unchanged. One line with the four subgroup medians.
            if snapshot.layers and snapshot.layers[0].ffn_w1w3 is not None:
                def _sg_med(field):
                    vals = sorted(getattr(l, field).param_delta_ratio for l in snapshot.layers
                                  if getattr(l, field) is not None
                                  and getattr(l, field).param_delta_ratio is not None)
                    return vals[len(vals) // 2] if vals else float('nan')
                log(f"    [subgroup pdr] attn qk={_sg_med('attn_qk'):.3e} vo={_sg_med('attn_vo'):.3e}"
                    f" | ffn w1w3={_sg_med('ffn_w1w3'):.3e} w2={_sg_med('ffn_w2'):.3e}")

            # MoE components
            if moe_layers:
                expert_ratios = [l.experts.ratio for l in moe_layers if l.experts]
                shared_ratios = [l.shared_experts.ratio for l in moe_layers if l.shared_experts]
                gate_ratios = [l.gate.ratio for l in moe_layers if l.gate]

                if expert_ratios:
                    log(f"    expert ratio:  min={min(expert_ratios):.6f} "
                        f"max={max(expert_ratios):.6f} "
                        f"mean={_mean(expert_ratios):.6f}")
                if shared_ratios:
                    log(f"    shared ratio:  min={min(shared_ratios):.6f} "
                        f"max={max(shared_ratios):.6f} "
                        f"mean={_mean(shared_ratios):.6f}")
                if gate_ratios:
                    log(f"    gate ratio:    min={min(gate_ratios):.6f} "
                        f"max={max(gate_ratios):.6f} "
                        f"mean={_mean(gate_ratios):.6f}")

                # Expert weight spread
                spreads = [l.expert_spread for l in moe_layers if l.expert_spread]
                if spreads:
                    avg_w_std = _mean([s['w_std'] for s in spreads])
                    avg_g_std = _mean([s['g_std'] for s in spreads])
                    log(f"    expert spread: w_std={avg_w_std:.4f} g_std={avg_g_std:.4f}")

        # MoE balance stats
        if moe_data:
            log(f"  moe balance: avg_cv={moe_data['avg_cv']:.4f}")

        # AWD summary
        if awd_data:
            active = {k: v for k, v in awd_data.items() if v['mult'] != 1.0}
            if active:
                log(f"  awd ({len(active)} active / {len(awd_data)} tracked):")
                for name, info in active.items():
                    metrics_str = ", ".join(f"{k}={v:.4f}" for k, v in info['metrics'].items())
                    log(f"    {name}: mult={info['mult']:.2f}x wd={info['eff_wd']:.4f} [{metrics_str}]")
            else:
                log(f"  awd ({len(awd_data)} tracked, all nominal)")
