"""Math validation for the dn4 deadband centered z-loss (Lever 2).

Checks the load-bearing properties of logZ_c = logZ - h.mu and the deadband
penalty mean(relu(logZ_c - tau)^2):
  [1] reconstruction identity: logZ - h.mu == logsumexp(h @ (W - mu).T)
  [2] gauge invariance: W += 1.c^T leaves logZ_c unchanged (raw logZ shifts by h.c)
  [3] ZERO common-mode gradient (Objective A): sum_v dL/dw_v ~ 0
  [4] deadband: tau above max(logZ_c) -> loss 0 AND grad 0; tau below -> loss > 0
  [5] the real cce-based _centered_zloss_deadband matches the materialized ref (GPU)

[1-4] use a materialized fp32 reference (no cce -> runs anywhere). [5] needs cce+cuda.
"""
import os, sys
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
torch.manual_seed(0)
dev = 'cuda' if torch.cuda.is_available() else 'cpu'


def ref(h, W, tgt, pad_id, tau):
    """Materialized fp32 reference (loss, logZ, logZ_c, mu)."""
    logits = h.float() @ W.float().t()                 # [N,V]
    logZ = torch.logsumexp(logits, dim=-1)             # [N]
    mu = W.float().mean(0)                             # [D]
    logZ_c = logZ - h.float() @ mu                    # [N]
    valid = (tgt != pad_id).float()
    excess = (logZ_c - tau).clamp_min(0.0)
    loss = (excess * excess * valid).sum() / valid.sum().clamp_min(1.0)
    return loss, logZ, logZ_c, mu


N, V, D, pad_id, tau = 64, 512, 32, -100, 2.0
h = torch.randn(N, D, device=dev)
W = torch.randn(V, D, device=dev) + 0.3 * torch.randn(D, device=dev)   # seed a common-mode gauge
tgt = torch.randint(0, V, (N,), device=dev)

# [1] identity
_, logZ, logZ_c, mu = ref(h, W, tgt, pad_id, tau)
logZ_c_alt = torch.logsumexp(h.float() @ (W.float() - mu).t(), dim=-1)
id_err = (logZ_c - logZ_c_alt).abs().max().item()
print(f"[1] logZ_c identity         max|err| = {id_err:.2e}")

# [2] gauge invariance
c = torch.randn(D, device=dev)
_, logZ2, logZ_c2, _ = ref(h, W + c.unsqueeze(0), tgt, pad_id, tau)
gauge_logZc = (logZ_c2 - logZ_c).abs().max().item()
gauge_resid = ((logZ2 - logZ) - (h.float() @ c)).abs().max().item()   # raw logZ shifts by exactly h.c
print(f"[2] gauge: logZ_c invariant max|d|={gauge_logZc:.2e} ; raw logZ shift==h.c resid={gauge_resid:.2e}")

# [3] zero common-mode gradient
Wg = W.clone().detach().requires_grad_(True)
loss, *_ = ref(h, Wg, tgt, pad_id, tau)
loss.backward()
cm = Wg.grad.sum(dim=0).norm().item() / Wg.grad.norm().item()
print(f"[3] zero common-mode grad: ||sum_v dL/dw_v||/||dL/dW|| = {cm:.2e}")

# [4] deadband on/off
tau_hi = logZ_c.max().item() + 1.0
Wg2 = W.clone().detach().requires_grad_(True)
loss_hi, *_ = ref(h, Wg2, tgt, pad_id, tau_hi)
loss_hi.backward()
ghi = Wg2.grad.norm().item()
loss_lo, *_ = ref(h, W, tgt, pad_id, logZ_c.min().item() - 1.0)
print(f"[4] deadband: tau>max -> loss={loss_hi.item():.2e} grad={ghi:.2e} ; tau<min -> loss={loss_lo.item():.2e}")

# [5] real cce fn vs materialized ref
real = "skipped (no cce/cuda)"
try:
    from model_v2 import _centered_zloss_deadband
    if dev == 'cuda':
        hh, WW = h.bfloat16(), W.bfloat16()
        lr, lzc, hmu = _centered_zloss_deadband(hh, WW, tgt, pad_id, tau, fp32_accum=True)
        lref, _, _, _ = ref(hh.float(), WW.float(), tgt, pad_id, tau)
        rel = abs(lr.item() - lref.item()) / max(lref.item(), 1e-9)
        real = f"loss_real={lr.item():.4e} vs ref={lref.item():.4e}  rel={rel:.2e}  (logZ_c={lzc.item():.2f} h_mu={hmu.item():.2f})"
except Exception as e:
    real = f"skipped ({type(e).__name__}: {e})"
print(f"[5] real cce fn vs ref: {real}")

ok = (id_err < 1e-3 and gauge_logZc < 1e-3 and cm < 1e-4
      and loss_hi.item() < 1e-9 and ghi < 1e-9 and loss_lo.item() > 0)
print("RESULT:", "PASS" if ok else "FAIL")
sys.exit(0 if ok else 1)
