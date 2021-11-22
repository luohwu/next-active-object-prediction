import torch
def IOU(bboxes1, bboxes2):
    assert bboxes1.shape[0] == bboxes2.shape[0], f' # of predicts != # of targets'
    w1 = (bboxes1[:, 2] - bboxes1[:, 0])
    h1 = (bboxes1[:, 3] - bboxes1[:, 1])
    w2 = (bboxes2[:, 2] - bboxes2[:, 0])
    h2 = (bboxes2[:, 3] - bboxes2[:, 1])
    area1 = w1 * h1
    area2 = w2 * h2

    inter_l = torch.max(bboxes1[:, 0], bboxes2[:, 0])
    inter_r = torch.min(bboxes1[:, 2], bboxes2[:, 2])
    inter_t = torch.max(bboxes1[:, 1], bboxes2[:, 1])
    inter_b = torch.min(bboxes1[:, 3], bboxes2[:, 3])
    inter_area = torch.clamp((inter_r - inter_l), min=0) * torch.clamp((inter_b - inter_t), min=0)

    union = area1 + area2 - inter_area
    iou = inter_area / union
    return (iou)