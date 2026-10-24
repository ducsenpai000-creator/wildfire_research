"""Re-implementation of DSA (Yao et al., "DSA: Discriminative Scatter Analysis for Early Smoke Segmentation",
ECCV 2024, DOI 10.1007/978-3-031-72784-9_26). No official code was available (checked 2026-10-17), so this
follows the paper text (Sec. 3, Eq. 1-9; Sec. 4.4):

  * K learnable proxies per class (K = 5, Sec. 4.4), D = embedding dimension.
  * Pixel-proxy matching (P2M, Eq. 1-3): entropy-regularised optimal transport solved with 3 Sinkhorn
    iterations, equipartition over the K proxies of the pixel's own class (no gradient through M).
  * IOS: within-class scatter Xi_w (Eq. 4, pixel vs. matched proxy) and non-parametric between-class scatter
    Xi_b (Eq. 5, pixel vs. nearest proxy of the other class).
  * OOS: within-class proxy scatter Xi_p (Eq. 6, proxy vs. nearest other proxy of the same class, weighted by
    the class pixel fraction N_wi / N).
  * L_DSA = [Tr((Xi_w + gamma I)^-1 Xi_b) + Tr((Xi_w + gamma I)^-1 Xi_p)]^-1, gamma = 1e-5 (Eq. 8).
  * Total loss L = lambda * L_DSA + L_base, lambda = 0.4 (Eq. 9, Sec. 4.4).

Choices NOT specified in the paper (documented assumptions):
  * Sinkhorn temperature eps = 0.05 (SwAV / ProtoSeg default; the paper cites SwAV for Eq. 1).
  * Pixel embeddings and proxies are L2-normalised before matching and scatter computation (ProtoSeg practice).
  * Embedding layer for PIDNet: the input of the final segmentation head (DFM output, 128-d for PIDNet-S, 1/8 res);
    labels are downsampled to that resolution with nearest-neighbour interpolation.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


@torch.no_grad()
def sinkhorn_assign(x, p, eps=0.05, iters=3):
    """x: [N, D] pixel embeddings of one class, p: [K, D] its proxies -> hard assignment index [N]."""
    q = torch.exp((p @ x.t()) / eps)  # [K, N]
    q = q / q.sum()
    k, n = q.shape
    for _ in range(iters):
        q = q / q.sum(dim=1, keepdim=True) / k   # each proxy gets N/K pixels (equipartition)
        q = q / q.sum(dim=0, keepdim=True) / n   # each pixel is a distribution over proxies
    return q.argmax(dim=0)


class DSALoss(nn.Module):
    def __init__(self, dim, k=5, eps=0.05, gamma=1e-5, sinkhorn_iters=3):
        super().__init__()
        self.k, self.eps, self.gamma, self.iters = k, eps, gamma, sinkhorn_iters
        self.proxies = nn.Parameter(F.normalize(torch.randn(2, k, dim), dim=-1))

    def forward(self, feats, labels):
        """feats: [B, D, h, w] embeddings; labels: [B, 1, H, W] in {0,1} (float)."""
        b, d, h, w = feats.shape
        lab = F.interpolate(labels.float(), size=(h, w), mode="nearest").reshape(-1).long()
        x = F.normalize(feats.permute(0, 2, 3, 1).reshape(-1, d).float(), dim=1)
        p = F.normalize(self.proxies.float(), dim=-1)  # [2, K, D]
        n = x.shape[0]
        eye = torch.eye(d, device=x.device)
        xi_w = torch.zeros(d, d, device=x.device)
        xi_b = torch.zeros(d, d, device=x.device)
        xi_p = torch.zeros(d, d, device=x.device)
        for c in (0, 1):
            xc = x[lab == c]
            nc = xc.shape[0]
            if nc == 0:
                continue
            idx = sinkhorn_assign(xc.detach(), p[c].detach(), self.eps, self.iters)
            r_w = xc - p[c][idx]                                   # Eq. 4
            xi_w = xi_w + r_w.t() @ r_w / n
            other = p[1 - c]                                       # Eq. 5: nearest proxy of the other class
            nn_o = torch.cdist(xc.detach(), other.detach()).argmin(dim=1)
            r_b = xc - other[nn_o]
            xi_b = xi_b + r_b.t() @ r_b / n
            dist_pp = torch.cdist(p[c], p[c]) + torch.eye(self.k, device=x.device) * 1e9   # Eq. 6
            r_p = p[c] - p[c][dist_pp.detach().argmin(dim=1)]
            xi_p = xi_p + (nc / n) * (r_p.t() @ r_p)
        a = xi_w + self.gamma * eye
        j = torch.trace(torch.linalg.solve(a, xi_b)) + torch.trace(torch.linalg.solve(a, xi_p))
        return 1.0 / j                                             # Eq. 8
