# -*- coding: utf-8 -*-

import torch
import torch.nn.functional as F


def focal_loss(logits, labels, gamma=2):
    r"""
    focal loss for multi classification

    `https://arxiv.org/pdf/1708.02002.pdf`

    FL(p_t)=-alpha(1-p_t)^{gamma}ln(p_t)
    """

    ce_loss = F.cross_entropy(logits, labels, reduction="none")
    log_pt = -ce_loss
    pt = torch.exp(log_pt)
    weights = (1 - pt) ** gamma
    fl = weights * ce_loss
    fl = fl.mean()
    return fl


def balanced_focal_loss(logits, labels, alpha=0.25, gamma=2):
    return alpha * focal_loss(logits, labels, gamma)


def focal_lossv1(logits, labels, gamma=2):
    r"""
    FL(p_t)=-alpha(1-p_t)^{gamma}ln(p_t)
    """

    log_pt = F.log_softmax(logits, dim=-1)
    pt = torch.exp(log_pt)
    labels = labels.view(-1, 1)
    pt = pt.gather(1, labels)
    ce_loss = -torch.log(pt)
    weights = (1 - pt) ** gamma
    fl = weights * ce_loss
    fl = fl.mean()
    return fl


if __name__ == "__main__":
    logits = torch.tensor([[0.3, 0.6, 0.9, 1], [0.6, 0.4, 0.9, 0.5]])
    labels = torch.tensor([1, 3])
    print(focal_loss(logits, labels))
    print(focal_lossv1(logits, labels))
    print(balanced_focal_loss(logits, labels))