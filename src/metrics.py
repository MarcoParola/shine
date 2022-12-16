import torch
import torchvision.ops.boxes as bops
from src.utils import from_xywh_to_x1y1x2y2_bbox


def compute_iou(bbox1, bbox2):
    box1 = from_xywh_to_x1y1x2y2_bbox(bbox1)
    box1 = torch.tensor([box1], dtype=torch.float)
    box2 = from_xywh_to_x1y1x2y2_bbox(bbox2)
    box2 = torch.tensor([box2], dtype=torch.float)
    iou = bops.box_iou(box1, box2)
    return iou