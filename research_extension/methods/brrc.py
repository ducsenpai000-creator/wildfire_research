"""BRRC-Loss (Boundary-Reliability and Recall-Calibrated loss) for the WACV-2025 PIDNet student.
Design pre-registered in reports/BRRC_DESIGN_PREREGISTRATION.md (2026-10-18), before any BRRC training.

BRRCFullModel subclasses the upstream FullModel and reproduces its forward pass and loss *exactly*
(0.4*l0 + 20*l1 + 1*l2 + 1*l3, BCE with pos_weight, BAS threshold 0.8), with two optional changes:
  BR: per-pixel weights on the two segmentation BCE terms (l0, l1): background pixels within a scale-aware
      chessboard distance r = clip(rho*sqrt(|M|), r_min, r_max) of the SAM mask M get weight
      w_min + (1-w_min)*d/r; all other pixels weight 1. Weighted BCE is normalised by the sum of weights.
      Labels are never modified.
  RC: + lambda_T * mean_i [1 - (TP+1)/(TP + alpha*FP + beta*FN + 1)] on the main-head probabilities (soft Tversky).
Parameter names are unchanged (state_dict compatible with the upstream FullModel / evaluator).
"""
import math

import torch
import torch.nn.functional as F

from pidnet_utils.utils import FullModel  # upstream (read-only)


@torch.no_grad()
def boundary_reliability_weights(labels, rho, r_min=3, r_max=40, w_min=0.0):
    """labels: [B,1,H,W] in {0,1}. Returns weights [B,1,H,W] (outer background band down-weighted)."""
    fg = labels > 0.5
    b = labels.shape[0]
    area = fg.flatten(1).sum(1).float()
    r = torch.clamp(rho * torch.sqrt(area), r_min, r_max)            # per-image radius
    r = torch.where(area > 0, r, torch.zeros_like(r)).view(b, 1, 1, 1)
    dist = torch.full_like(labels, float("inf"))
    cur = fg.float()
    for k in range(1, r_max + 1):                                     # chessboard distance via 3x3 dilation
        nxt = F.max_pool2d(cur, kernel_size=3, stride=1, padding=1)
        newly = (nxt > 0) & (cur == 0)
        dist[newly] = float(k)
        cur = nxt
    w = torch.ones_like(labels)
    band = (~fg) & (dist <= r) & (r > 0)
    w[band] = (w_min + (1 - w_min) * dist / r.clamp(min=1e-6))[band]
    return w


def soft_tversky(prob, labels, alpha=0.3, beta=0.7):
    p, y = prob.flatten(1), labels.flatten(1)
    tp = (p * y).sum(1)
    fp = (p * (1 - y)).sum(1)
    fn = ((1 - p) * y).sum(1)
    return (1 - (tp + 1) / (tp + alpha * fp + beta * fn + 1)).mean()


class BRRCFullModel(FullModel):
    def __init__(self, model, sem_loss, bd_loss, br_rho=0.0, br_rmin=3, br_rmax=40, br_wmin=0.0,
                 rc_lambda=0.0, rc_alpha=0.3, rc_beta=0.7):
        super().__init__(model, sem_loss, bd_loss)
        self.br = dict(rho=br_rho, r_min=br_rmin, r_max=br_rmax, w_min=br_wmin)
        self.rc = dict(lam=rc_lambda, alpha=rc_alpha, beta=rc_beta)
        self.last_terms = {}

    def _bce(self, logits, labels, weight):
        if weight is None:
            return self.sem_loss(logits, labels)                     # upstream: mean over all pixels
        pw = self.sem_loss.pos_weight
        return F.binary_cross_entropy_with_logits(logits, labels, weight=weight, pos_weight=pw,
                                                  reduction="sum") / weight.sum()

    def forward(self, inputs, labels=None, bd_gt=None, id=None, box=None, lab_img=None, plot_outputs=False,
                *args, **kwargs):
        outputs = self.model(inputs, *args, **kwargs)
        h, w = inputs.size(-2), inputs.size(-1)
        ph, pw = outputs[0].size(2), outputs[0].size(3)
        if ph != h or pw != w:
            for i in range(len(outputs)):
                outputs[i] = F.interpolate(outputs[i], size=(h, w), mode="bilinear", align_corners=True)
        if plot_outputs:
            return outputs
        if len(torch.nonzero(labels)) < 1:
            acc = torch.Tensor([1])
        else:
            acc = self.iou(preds=torch.sigmoid(outputs[1]), target=labels)
        weight = None
        if self.training and self.br["rho"] > 0:
            weight = boundary_reliability_weights(labels, self.br["rho"], self.br["r_min"], self.br["r_max"],
                                                  self.br["w_min"])
        loss_s0 = self._bce(outputs[0], labels, weight)
        loss_s = self._bce(outputs[1], labels, weight)
        loss_b = self.bd_loss(outputs[-1], bd_gt)
        filler = torch.zeros_like(labels)
        bd_label = torch.where(torch.sigmoid(outputs[-1]) > 0.8, labels, filler)
        bd_output = torch.where(torch.sigmoid(outputs[-1]) > 0.8, outputs[1], filler)
        loss_sb = self.sem_loss(bd_output, bd_label)
        loss = 0.4 * loss_s0 + 20 * loss_s + 1 * loss_b + 1 * loss_sb
        if self.training and self.rc["lam"] > 0:
            l_t = soft_tversky(torch.sigmoid(outputs[1]), labels, self.rc["alpha"], self.rc["beta"])
            loss = loss + self.rc["lam"] * l_t
            self.last_terms["tversky"] = float(l_t)
        if weight is not None:
            self.last_terms["mean_w"] = float(weight.mean())
        return torch.unsqueeze(loss, 0), outputs[:-1], acc, [loss_s, loss_b]
