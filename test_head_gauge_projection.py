"""Deterministic fp32 unit test for the dn4 head applied-update gauge projection.

Validates the theorem  P(W - eta*U) = P(W) - eta*P(U):  projecting the vocab-row
mean out of the head's APPLIED Adam update each step keeps the CENTERED head
bit-close to an unprojected arm, while the raw heads differ only by a pure
common-row gauge. Also exercises the decoupled-WD induction (WD maps gauge->gauge,
so the centered heads still match with wd>0).

Single-device, fp32, no distributed -- exercises the REAL code path
(_project_head_update_gauge_ + adam_update + row_center helpers) on plain tensors
(_global_row_mean/_subtract_row_mean_ fall back to local ops when not a DTensor).
The bf16/FSDP/SR equivalence (spec "Step 2") is an integration test for later.
"""
import os, sys
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from muon_fsdp2 import adam_update, _project_head_update_gauge_
from row_center import _global_row_mean

torch.manual_seed(0)


def centered(W):
    mu, _ = _global_row_mean(W, vocab_dim=0)
    return W - mu.unsqueeze(0)


def gauge_residual(Delta):
    """||Delta - 1*mean_v(Delta)^T|| / ||Delta|| -> ~0 iff Delta is a pure common-row gauge."""
    mu, _ = _global_row_mean(Delta, vocab_dim=0)
    non_gauge = Delta - mu.unsqueeze(0)
    return non_gauge.norm().item() / max(Delta.norm().item(), 1e-12)


V, Dim, T, lr, wd = 1024, 128, 300, 3e-4, 0.02
betas, eps = (0.9, 0.95), 1e-10

W0 = torch.randn(V, Dim, dtype=torch.float32)
W0 = centered(W0)                                  # gauge-free start: any final gauge is accumulated
bias = 0.5 * torch.randn(Dim)                      # a common-row drift injected into every grad, so
                                                   # arm A's update carries a persistent gauge to accrue
WA, WB = W0.clone(), W0.clone()
mA, vA = torch.zeros_like(WA), torch.zeros_like(WA)
mB, vB = torch.zeros_like(WB), torch.zeros_like(WB)

ubar_pre_first = ubar_post_last = None
for t in range(1, T + 1):
    g = torch.randn(V, Dim, dtype=torch.float32) + bias.unsqueeze(0)   # SAME biased grad, both arms
    # arm A: ordinary Adam (gauge from the biased update accumulates)
    UA = adam_update(g, mA, vA, t, betas, eps)
    WA.mul_(1 - lr * wd); WA.add_(UA, alpha=-lr)
    # arm B: gauge-projected Adam (the REAL hook)
    UB = adam_update(g, mB, vB, t, betas, eps)
    pre, post = _project_head_update_gauge_(UB, verify=True)
    if ubar_pre_first is None:
        ubar_pre_first = pre
    ubar_post_last = post
    WB.mul_(1 - lr * wd); WB.add_(UB, alpha=-lr)

cgap = (centered(WA) - centered(WB)).norm().item() / centered(WA).norm().item()
nonleak = (centered(WA) - centered(WB)).norm().item() / max(centered(WA).norm().item(), 1e-12)  # == cgap
gpure = gauge_residual(WA - WB)              # context only (normed by the tiny gap; not an assertion)
rawgap = (WA - WB).norm().item() / WA.norm().item()
muA = _global_row_mean(WA, 0)[0].norm().item()
muB = _global_row_mean(WB, 0)[0].norm().item()

print(f"steps={T}  lr={lr}  wd={wd}  V={V} D={Dim}")
print(f"[1] centered-head gap ||P(WA)-P(WB)||/||P(WA)|| = {cgap:.3e}   THEOREM (expect ~fp32 floor)")
print(f"[2] accumulated head gauge ||mu(W)||  A={muA:.3e}  B={muB:.3e}   (expect B << A)")
print(f"[3] raw-head gap      ||WA-WB||/||WA||          = {rawgap:.3e}   (expect >0: projection acted)")
print(f"    ||Ubar|| projected: first-step={ubar_pre_first:.3e}  last-step post-write={ubar_post_last:.3e}")
print(f"    (gauge residual of raw gap = {gpure:.3e} -- normed by the tiny gap, so == [1]x||W||/||gap||)")

ok = (cgap < 1e-5) and (muB < 0.05 * muA) and (rawgap > 1e-4)
print("RESULT:", "PASS" if ok else "FAIL")
sys.exit(0 if ok else 1)
