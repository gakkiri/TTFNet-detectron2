import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models.resnet as resnet
from detectron2.layers import ShapeSpec

# from centernet.network.backbone import Backbone
from detectron2.modeling import Backbone
from detectron2.modeling.backbone import build_backbone
from detectron2.modeling.backbone.build import BACKBONE_REGISTRY
from detectron2.modeling.meta_arch.build import META_ARCH_REGISTRY
from detectron2.structures import Boxes, ImageList, Instances

from .centernet_decode import CenterNetDecoder, gather_feature
from .centernet_deconv import CenternetDeconv
from .centernet_gt import CenterNetGT
from .centernet_head import CenternetHead

__all__ = ["CenterNet"]

_resnet_mapper = {18: resnet.resnet18, 50: resnet.resnet50, 101: resnet.resnet101}


class ResnetBackbone(Backbone):
    def __init__(self, cfg, input_shape=None, pretrained=True):
        super().__init__()
        depth = cfg.MODEL.RESNETS.DEPTH
        backbone = _resnet_mapper[depth](pretrained=pretrained)
        self.stage0 = nn.Sequential(backbone.conv1, backbone.bn1, backbone.relu, backbone.maxpool)
        self.stage1 = backbone.layer1
        self.stage2 = backbone.layer2
        self.stage3 = backbone.layer3
        self.stage4 = backbone.layer4

    def forward(self, x):
        x = self.stage0(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        return x


def reg_l1_loss(output, mask, index, target):
    pred = gather_feature(output, index, use_transform=True)
    mask = mask.unsqueeze(dim=2).expand_as(pred).float()
    # loss = F.l1_loss(pred * mask, target * mask, reduction='elementwise_mean')
    loss = F.l1_loss(pred * mask, target * mask, reduction="sum")
    loss = loss / (mask.sum() + 1e-4)
    return loss


def ct_focal_loss(pred, gt, gamma=2.0):
    """
    Focal loss used in CornerNet & CenterNet. Note that the values in gt (label) are in [0, 1] since
    gaussian is used to reduce the punishment and we treat [0, 1) as neg example.

    Args:
        pred: tensor, any shape.
        gt: tensor, same as pred.
        gamma: gamma in focal loss.

    Returns:

    """
    pos_inds = gt.eq(1).float()
    neg_inds = gt.lt(1).float()

    neg_weights = torch.pow(1 - gt, 4)  # reduce punishment
    pos_loss = -torch.log(pred) * torch.pow(1 - pred, gamma) * pos_inds
    neg_loss = -torch.log(1 - pred) * torch.pow(pred, gamma) * neg_weights * neg_inds

    num_pos = pos_inds.float().sum()
    pos_loss = pos_loss.sum()
    neg_loss = neg_loss.sum()

    if num_pos == 0:
        return neg_loss
    return (pos_loss + neg_loss) / num_pos


def giou_loss(pred,
              target,
              weight,
              avg_factor=None):
    """GIoU loss.
    Computing the GIoU loss between a set of predicted bboxes and target bboxes.
    """
    pos_mask = weight > 0
    weight = weight[pos_mask].float()
    if avg_factor is None:
        avg_factor = torch.sum(pos_mask).float().item() + 1e-6
    bboxes1 = pred[pos_mask].view(-1, 4)
    bboxes2 = target[pos_mask].view(-1, 4)

    lt = torch.max(bboxes1[:, :2], bboxes2[:, :2])  # [rows, 2]
    rb = torch.min(bboxes1[:, 2:], bboxes2[:, 2:])  # [rows, 2]
    wh = (rb - lt + 1).clamp(min=0)  # [rows, 2]
    enclose_x1y1 = torch.min(bboxes1[:, :2], bboxes2[:, :2])
    enclose_x2y2 = torch.max(bboxes1[:, 2:], bboxes2[:, 2:])
    enclose_wh = (enclose_x2y2 - enclose_x1y1 + 1).clamp(min=0)

    overlap = wh[:, 0] * wh[:, 1]
    ap = (bboxes1[:, 2] - bboxes1[:, 0] + 1) * (bboxes1[:, 3] - bboxes1[:, 1] + 1)
    ag = (bboxes2[:, 2] - bboxes2[:, 0] + 1) * (bboxes2[:, 3] - bboxes2[:, 1] + 1)
    ious = overlap / (ap + ag - overlap)

    enclose_area = enclose_wh[:, 0] * enclose_wh[:, 1]  # i.e. C in paper
    u = ap + ag - overlap
    gious = ious - (enclose_area - u) / enclose_area
    iou_distances = 1 - gious
    return torch.sum(iou_distances * weight) / avg_factor


@BACKBONE_REGISTRY.register()
def build_torch_backbone(cfg, input_shape=None):
    """
    Build a backbone.

    Returns:
        an instance of :class:`Backbone`
    """
    if input_shape is None:
        input_shape = ShapeSpec(channels=len(cfg.MODEL.PIXEL_MEAN))

    backbone = ResnetBackbone(cfg, input_shape)
    assert isinstance(backbone, Backbone)
    return backbone


@META_ARCH_REGISTRY.register()
class CenterNet(nn.Module):
    """
    Implement CenterNet (https://arxiv.org/abs/1904.07850).
    """

    def __init__(self, cfg):
        super().__init__()

        self.device = torch.device(cfg.MODEL.DEVICE)
        self.cfg = cfg

        # fmt: off
        self.num_classes = cfg.MODEL.CENTERNET.NUM_CLASSES
        # Loss parameters:
        # Inference parameters:
        self.max_detections_per_image = cfg.TEST.DETECTIONS_PER_IMAGE
        # fmt: on
        self.backbone = build_backbone(
            cfg, input_shape=ShapeSpec(channels=len(cfg.MODEL.PIXEL_MEAN))
        )
        self.upsample = CenternetDeconv(cfg, self.backbone.output_shape())
        self.head = CenternetHead(cfg)

        self.mean, self.std = cfg.MODEL.PIXEL_MEAN, cfg.MODEL.PIXEL_STD
        pixel_mean = torch.Tensor(cfg.MODEL.PIXEL_MEAN).to(self.device).view(3, 1, 1)
        pixel_std = torch.Tensor(cfg.MODEL.PIXEL_STD).to(self.device).view(3, 1, 1)
        self.normalizer = lambda x: (x - pixel_mean) / pixel_std

        self.to(self.device)

    def forward(self, batched_inputs):
        """
        Args:
            batched_inputs(list): batched outputs of :class:`DatasetMapper` .
                Each item in the list contains the inputs for one image.
                For now, each item in the list is a dict that contains:

                * image: Tensor, image in (C, H, W) format.
                * instances: Instances

                Other information that's included in the original dicts, such as:

                * "height", "width" (int): the output resolution of the model, used in inference.
                    See :meth:`postprocess` for details.
        Returns:
            dict[str: Tensor]:
        """
        images = self.preprocess_image(batched_inputs)

        if not self.training:
            return self.inference(images)

        features = self.backbone(images.tensor)
        # features = features[self.cfg.MODEL.RESNETS.OUT_FEATURES[0]]
        up_fmap = self.upsample(features)
        pred_dict = self.head(up_fmap)

        gt_dict = self.get_ground_truth(batched_inputs)
        return self.losses(pred_dict, gt_dict)

    def losses(self, pred_dict, gt_dict):
        # hm loss
        pred_score = pred_dict["cls"]
        cur_device = pred_score.device
        for k in gt_dict:
            gt_dict[k] = gt_dict[k].to(cur_device)

        pred_score = torch.clamp(pred_score, min=1e-4, max=1 - 1e-4)  # sigmoid in forward pipeline
        loss_cls = ct_focal_loss(pred_score, gt_dict["gt_heatmap"])

        # wh loss
        H, W = pred_score.shape[2:]
        wh_weight = gt_dict['gt_reg_weight']
        mask = wh_weight.view(-1, H, W)
        avg_factor = mask.sum() + 1e-4

        base_step = self.cfg.MODEL.CENTERNET.DOWN_SCALE  # 4
        shifts_x = torch.arange(0, (W - 1) * base_step + 1, base_step,
                                dtype=torch.float32, device=cur_device)
        shifts_y = torch.arange(0, (H - 1) * base_step + 1, base_step,
                                dtype=torch.float32, device=cur_device)
        shift_y, shift_x = torch.meshgrid(shifts_y, shifts_x)
        base_loc = torch.stack((shift_x, shift_y), dim=0)  # (2, h, w)

        pred_wh = pred_dict['wh']
        # (batch, h, w, 4)
        pred_boxes = torch.cat((base_loc - pred_wh[:, [0, 1]],
                                base_loc + pred_wh[:, [2, 3]]), dim=1).permute(0, 2, 3, 1)
        # (batch, h, w, 4)
        boxes = gt_dict['gt_box'].permute(0, 2, 3, 1)
        loss_wh = giou_loss(pred_boxes, boxes, mask, avg_factor=avg_factor)

        loss_cls *= self.cfg.MODEL.CENTERNET.LOSS.CLS_WEIGHT
        loss_wh *= self.cfg.MODEL.CENTERNET.LOSS.WH_WEIGHT

        loss = {"loss_cls": loss_cls, "loss_box_wh": loss_wh}
        return loss

    @torch.no_grad()
    def get_ground_truth(self, batched_inputs):
        return CenterNetGT.generate(self.cfg, batched_inputs)

    @torch.no_grad()
    def inference(self, images):
        """
        image(tensor): ImageList in detectron2.structures
        """
        n, c, h, w = images.tensor.shape
        new_h, new_w = (h | 31) + 1, (w | 31) + 1
        center_wh = np.array([w // 2, h // 2], dtype=np.float32)
        size_wh = np.array([new_w, new_h], dtype=np.float32)
        down_scale = self.cfg.MODEL.CENTERNET.DOWN_SCALE
        img_info = dict(
            center=center_wh, size=size_wh, height=new_h // down_scale, width=new_w // down_scale
        )

        pad_value = [-x / y for x, y in zip(self.mean, self.std)]
        aligned_img = torch.Tensor(pad_value).reshape((1, -1, 1, 1)).expand(n, c, new_h, new_w)
        aligned_img = aligned_img.to(images.tensor.device)

        pad_w, pad_h = math.ceil((new_w - w) / 2), math.ceil((new_h - h) / 2)
        aligned_img[..., pad_h : h + pad_h, pad_w : w + pad_w] = images.tensor

        features = self.backbone(aligned_img)
        # features = features[self.cfg.MODEL.RESNETS.OUT_FEATURES[0]]
        up_fmap = self.upsample(features)
        pred_dict = self.head(up_fmap)
        results = self.decode_prediction(pred_dict, img_info)

        ori_w, ori_h = img_info["center"] * 2
        det_instance = Instances((int(ori_h), int(ori_w)), **results)

        return [{"instances": det_instance}]

    def decode_prediction(self, pred_dict, img_info):
        """
        Args:
            pred_dict(dict): a dict contains all information of prediction
            img_info(dict): a dict contains needed information of origin image
        """
        fmap = pred_dict["cls"]
        reg = None
        if 'reg' in pred_dict:
            reg = pred_dict["reg"]
        wh = pred_dict["wh"]

        boxes, scores, classes = CenterNetDecoder.decode(fmap, wh, reg)
        # boxes = Boxes(boxes.reshape(boxes.shape[-2:]))
        scores = scores.reshape(-1)
        classes = classes.reshape(-1).to(torch.int64)

        # dets = CenterNetDecoder.decode(fmap, wh, reg)
        boxes = CenterNetDecoder.transform_boxes(boxes, img_info)
        boxes = Boxes(boxes)
        return dict(pred_boxes=boxes, scores=scores, pred_classes=classes)

    def preprocess_image(self, batched_inputs):
        """
        Normalize, pad and batch the input images.
        """
        images = [x["image"].to(self.device) for x in batched_inputs]
        images = [self.normalizer(img/255.) for img in images]
        images = ImageList.from_tensors(images, self.backbone.size_divisibility)
        return images


def build_model(cfg):

    model = CenterNet(cfg)
    return model
