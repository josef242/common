# adamc_optimizer.py
"""
AdamC optimizer variants for FSDP2 training.

Includes:
- AdamC: Standard 32-bit implementation with corrected weight decay
- AdamC8bitTorchAO: torchao-based 8-bit (FSDP2/DTensor compatible)
"""
import torch
from torch.optim import Optimizer
import math

try:
    # Try stable API first (torchao >= 0.4)
    from torchao.optim import AdamW8bit as TorchAOAdamW8bit
    HAS_TORCHAO = True
except ImportError:
    try:
        # Fall back to prototype API (older torchao versions)
        from torchao.prototype.low_bit_optim import AdamW8bit as TorchAOAdamW8bit
        HAS_TORCHAO = True
    except ImportError:
        HAS_TORCHAO = False


class AdamC(Optimizer):
    """
    AdamC optimizer - Adam with Corrected weight decay for normalized layers.

    Based on "Why Gradients Rapidly Increase Near the End of Training" (2025)
    by Aaron Defazio at Meta FAIR.

    Args:
        params: iterable of parameters to optimize or dicts defining parameter groups
        lr: learning rate (default: 1e-3)
        betas: coefficients used for computing running averages of gradient and its square (default: (0.9, 0.999))
        eps: term added to the denominator to improve numerical stability (default: 1e-8)
        weight_decay: weight decay coefficient (default: 0.01)
    """

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=0.01):
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

        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay,
                       is_normalized=False)  # default: not a normalized layer
        super(AdamC, self).__init__(params, defaults)

        # Side-dict for per-param WD overrides (keyed by id(param))
        self.wd_overrides = {}

    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step."""
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError('AdamC does not support sparse gradients')

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    # Exponential moving average of gradient values
                    state['exp_avg'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    # Exponential moving average of squared gradient values
                    state['exp_avg_sq'] = torch.zeros_like(p, memory_format=torch.preserve_format)

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                beta1, beta2 = group['betas']

                state['step'] += 1
                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']

                # Decay the first and second moment running average coefficient
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                # Compute bias-corrected moments
                exp_avg_corrected = exp_avg / bias_correction1
                exp_avg_sq_corrected = exp_avg_sq / bias_correction2

                # Compute the Adam update
                denom = exp_avg_sq_corrected.sqrt().add_(group['eps'])
                step_size = group['lr']

                # Apply the update
                p.addcdiv_(exp_avg_corrected, denom, value=-step_size)

                wd = self.wd_overrides.get(id(p), group['weight_decay'])
                if wd != 0:
                    if group.get('is_normalized', False):
                        # AdamC: weight decay scales with lr²/max_lr
                        effective_decay = wd * (group['lr']**2 / self.max_lr)
                        p.mul_(1 - effective_decay)
                    else:
                        # Standard AdamW weight decay
                        p.mul_(1 - group['lr'] * wd)

        return loss


class AdamC8bitTorchAO:
    """
    8-bit version of AdamC using torchao.

    This optimizer is FSDP2/DTensor compatible and should be used for
    distributed training with FSDP2.

    Wraps torchao's AdamW8bit and applies AdamC's corrected weight decay
    for normalized layers after the base optimizer step.

    Args:
        params: iterable of parameters to optimize or dicts defining parameter groups
        lr: learning rate (default: 1e-3)
        betas: coefficients used for computing running averages of gradient and its square (default: (0.9, 0.999))
        eps: term added to the denominator to improve numerical stability (default: 1e-8)
        weight_decay: weight decay coefficient (default: 0.01)
    """

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01):
        if not HAS_TORCHAO:
            raise ImportError(
                "torchao is required for AdamC8bitTorchAO. "
                "Install with: pip install torchao"
            )

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
            # Check if it's a list of param_groups or a list of parameters
            if len(param_groups) > 0 and not isinstance(param_groups[0], dict):
                # It's a list of parameters, wrap it
                param_groups = [{'params': param_groups}]

        # Store our extra metadata (is_normalized, our weight_decay) per group
        self._group_metadata = []
        base_params = []
        for group in param_groups:
            # Store metadata we need for AdamC weight decay correction
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
                'weight_decay': 0,  # We handle weight decay ourselves
            }
            base_params.append(base_group)

        # Create the underlying torchao optimizer
        self._base_optimizer = TorchAOAdamW8bit(base_params)

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
        # We iterate over base optimizer's param_groups (for params/lr) and our metadata (for is_normalized/weight_decay)
        for group, meta in zip(self._base_optimizer.param_groups, self._group_metadata):
            group_wd = meta['weight_decay']
            # Get LR - may be a tensor (torchao) or float
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


# Legacy alias for backward compatibility
AdamC8bit = AdamC8bitTorchAO
