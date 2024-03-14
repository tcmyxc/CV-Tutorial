import torch
import torch.nn.functional as F


def cal_loss(logits, labels, reduction="mean"):
    log_pt = F.log_softmax(logits, dim=-1)  # 这里相当于 CE loss
    pt = torch.exp(log_pt)  # 通过 softmax 函数后打的分
    true_pt = pt.gather(1, labels.view(-1, 1))  # 挑选出真实值对应的 softmax 打分，也可以使用独热编码实现
    ce_loss = -torch.log(true_pt)

    pred_labels = torch.argmax(logits, dim=-1)

    # 预测对的，取top2，作为额外的惩罚项
    pred_true = (pred_labels == labels)

    # 预测错了，取top1作为惩罚
    pred_false = (pred_labels != labels)

    # 使用topk函数获取每行的前两个最大值的值和索引
    _, top_indices = torch.topk(logits, k=2, dim=-1)
    top1_pred_labels = top_indices[:, 0]
    top2_pred_labels = top_indices[:, 1]

    top1_pred_labels = top1_pred_labels.type_as(labels)
    top2_pred_labels = top2_pred_labels.type_as(labels)
    # print("top2_pred_labels:", top2_pred_labels)
    calibration_loss = (pt.gather(1, top2_pred_labels.view(-1, 1)) * pred_true.view(-1, 1)
                        + pt.gather(1, top1_pred_labels.view(-1, 1)) * pred_false.view(-1, 1))
    # print(calibration_loss)
    # calibration_loss = torch.exp(calibration_loss)
    calibration_loss = torch.log(calibration_loss + 1)

    fl = ce_loss + calibration_loss

    if reduction == "sum":
        fl = fl.sum()
    elif reduction == "mean":
        fl = fl.mean()
    else:
        raise ValueError(f"reduction '{reduction}' is not valid")
    return fl


if __name__ == "__main__":
    logits = torch.tensor([[0.3, 0.6, 0.1, 1], [0.6, 0.4, 0.9, 0.5]])
    labels = torch.tensor([3, 3])
    print(cal_loss(logits, labels))
    print(F.cross_entropy(logits, labels))
