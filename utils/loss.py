# YOLOv5 🚀 by Ultralytics, GPL-3.0 license
import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.metrics import bbox_iou
from utils.torch_utils import is_parallel


def smooth_BCE(eps=0.1):
    return 1.0 - 0.5 * eps, 0.5 * eps


class BCEBlurWithLogitsLoss(nn.Module):
    def __init__(self, alpha=0.05):
        super().__init__()
        self.loss_fcn = nn.BCEWithLogitsLoss(reduction='none')
        self.alpha = alpha

    def forward(self, pred, true):
        loss = self.loss_fcn(pred, true)
        pred = torch.sigmoid(pred)
        dx = pred - true
        alpha_factor = 1 - torch.exp((dx - 1) / (self.alpha + 1e-4))
        return loss * alpha_factor


class FocalLoss(nn.Module):
    def __init__(self, loss_fcn, gamma=1.5, alpha=0.25):
        super().__init__()
        self.loss_fcn = loss_fcn
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = loss_fcn.reduction
        self.loss_fcn.reduction = 'none'

    def forward(self, pred, true):
        loss = self.loss_fcn(pred, true)
        pred_prob = torch.sigmoid(pred)
        p_t = true * pred_prob + (1 - true) * (1 - pred_prob)
        alpha_factor = true * self.alpha + (1 - true) * (1 - self.alpha)
        modulating_factor = (1 - p_t) ** self.gamma
        loss *= alpha_factor * modulating_factor
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss


class ComputeLoss:
    def __init__(self, model, autobalance=False):
        device = next(model.parameters()).device
        h = model.hyp

        self.BCEcls = nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor([h['cls_pw']], device=device))
        self.BCEobj = nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor([h['obj_pw']], device=device))

        self.cp, self.cn = smooth_BCE(eps=h.get('label_smoothing', 0.0))

        if h['fl_gamma'] > 0:
            self.BCEcls = FocalLoss(self.BCEcls, h['fl_gamma'])
            self.BCEobj = FocalLoss(self.BCEobj, h['fl_gamma'])

        det = model.module.model[-1] if is_parallel(model) else model.model[-1]

        self.na = det.na
        self.nc = det.nc
        self.nl = det.nl
        self.anchors = det.anchors

        self.balance = {3: [4.0, 1.0, 0.4]}.get(
            det.nl, [4.0, 1.0, 0.25, 0.06, 0.02])
        self.ssi = list(det.stride).index(16) if autobalance else 0

        self.autobalance = autobalance
        self.gr = getattr(model, 'gr', 1.0)
        self.hyp = h

    def __call__(self, p, targets):
        device = targets.device
        lcls = torch.zeros(1, device=device)
        lbox = torch.zeros(1, device=device)
        lobj = torch.zeros(1, device=device)

        tcls, tbox, indices, anchors = self.build_targets(p, targets)

        for i, pi in enumerate(p):
            b, a, gj, gi = indices[i]
            tobj = torch.zeros_like(pi[..., 0], device=device)

            n = b.shape[0]
            if n:
                ps = pi[b, a, gj, gi]

                pxy = ps[:, :2].sigmoid() * 2 - 0.5
                pwh = (ps[:, 2:4].sigmoid() * 2) ** 2 * anchors[i]
                pbox = torch.cat((pxy, pwh), 1)

                iou = bbox_iou(pbox.T, tbox[i], x1y1x2y2=False, CIoU=True)
                lbox += (1.0 - iou).mean()

                tobj[b, a, gj, gi] = ((1.0 - self.gr) + self.gr * iou.detach().clamp(0)).to(tobj.dtype)


                if self.nc > 1:
                    t = torch.full_like(ps[:, 5:], self.cn, device=device)
                    t[range(n), tcls[i]] = self.cp
                    lcls += self.BCEcls(ps[:, 5:], t)

            obji = self.BCEobj(pi[..., 4], tobj)

            if tobj.sum() > 0:
                lobj += obji.mean() * self.balance[i]

            if self.autobalance and tobj.sum() > 0:
                self.balance[i] = self.balance[i] * 0.9999 + \
                                  0.0001 / obji.detach().mean().item()

        lbox = torch.nan_to_num(lbox)
        lobj = torch.nan_to_num(lobj)
        lcls = torch.nan_to_num(lcls)

        bs = tobj.shape[0]
        return (lbox + lobj + lcls) * bs, torch.cat((lbox, lobj, lcls)).detach()

    def build_targets(self, p, targets):
        na, nt = self.na, targets.shape[0]
        tcls, tbox, indices, anch = [], [], [], []
        device = targets.device

        gain = torch.ones(7, device=device)
        ai = torch.arange(na, device=device).float().view(na, 1).repeat(1, nt)
        targets = torch.cat((targets.repeat(na, 1, 1), ai[:, :, None]), 2)

        g = 0.5
        off = torch.tensor(
            [[0, 0], [1, 0], [0, 1], [-1, 0], [0, -1]],
            device=device
        ).float() * g

        for i in range(self.nl):
            anchors = self.anchors[i]
            gain[2:6] = torch.tensor(p[i].shape, device=device)[[3, 2, 3, 2]]
            t = targets * gain

            if nt:
                r = t[:, :, 4:6] / anchors[:, None]
                j = torch.max(r, 1 / r).max(2)[0] < self.hyp['anchor_t']
                t = t[j]

                if t.shape[0]:
                    gxy = t[:, 2:4]
                    gxi = gain[[2, 3]] - gxy
                    j, k = ((gxy % 1 < g) & (gxy > 1)).T
                    l, m = ((gxi % 1 < g) & (gxi > 1)).T
                    t = torch.cat((t, t[j], t[k], t[l], t[m]), 0)
                    offsets = off.repeat(t.shape[0] // off.shape[0], 1)
                else:
                    offsets = torch.zeros((0, 2), device=device)
            else:
                t = targets[0]
                offsets = torch.zeros((0, 2), device=device)

            if t.shape[0]:
                bc, gxy, gwh, a = t[:, :2].long().T, t[:, 2:4], t[:, 4:6], t[:, 6].long()
                gij = (gxy - offsets).long()
                gi, gj = gij.T

                indices.append((
                    bc[0],
                    a,
                    gj.clamp_(0, gain[3] - 1),
                    gi.clamp_(0, gain[2] - 1)
                ))
                tbox.append(torch.cat((gxy - gij, gwh), 1))
                anch.append(anchors[a])
                tcls.append(bc[1])
            else:
                # ===== 关键：空占位，不能不 append =====
                indices.append((
                    torch.zeros(0, dtype=torch.long, device=device),
                    torch.zeros(0, dtype=torch.long, device=device),
                    torch.zeros(0, dtype=torch.long, device=device),
                    torch.zeros(0, dtype=torch.long, device=device),
                ))
                tbox.append(torch.zeros((0, 4), device=device))
                anch.append(torch.zeros((0, 2), device=device))
                tcls.append(torch.zeros(0, dtype=torch.long, device=device))

        return tcls, tbox, indices, anch

