from torch import nn
import torch.nn.functional as F

from ssd.modeling import registry
from ssd.modeling.anchors.prior_box import PriorBox
from ssd.modeling.box_head.box_predictor import make_box_predictor
from ssd.utils import box_utils
from .inference import PostProcessor
from .loss import MultiBoxLoss


@registry.BOX_HEADS.register('SSDBoxHead')
class SSDBoxHead(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.predictor = make_box_predictor(cfg)
        self.loss_evaluator = MultiBoxLoss(neg_pos_ratio=cfg.MODEL.NEG_POS_RATIO)
        self.post_processor = PostProcessor(cfg)
        self.priors = None
        self.l1loss = nn.L1Loss()

    def forward(self, features, targets=None, discrep=False):
        cls_logits, bbox_pred = self.predictor(features)
        if self.training:
            return self._forward_train(cls_logits, bbox_pred, targets, discrep=discrep)
        else:
            return self._forward_test(cls_logits, bbox_pred)

    def _forward_train(self, cls_logits, bbox_pred, targets, discrep=False):
        gt_boxes, gt_labels = targets['boxes'], targets['labels']
        reg_loss, cls_loss = self.loss_evaluator(cls_logits, bbox_pred, gt_labels, gt_boxes)
        detections = (cls_logits, bbox_pred)
        loss_dict = dict(
            reg_loss=reg_loss,
            cls_loss=cls_loss,
        )
        if discrep:
            x = cls_logits.view(-1, cls_logits.size(-1))
            x = x.softmax(1)
            discrep_loss= (-x * (x).log())
            discrep_loss[discrep_loss.isnan()] = 0
            loss_dict['discrep_loss'] = discrep_loss.sum(-1).mean()
            
        return detections, loss_dict

    def _forward_test(self, cls_logits, bbox_pred):
        if self.priors is None:
            self.priors = PriorBox(self.cfg)().to(bbox_pred.device)
        scores = F.softmax(cls_logits, dim=2)
        boxes = box_utils.convert_locations_to_boxes(
            bbox_pred, self.priors, self.cfg.MODEL.CENTER_VARIANCE, self.cfg.MODEL.SIZE_VARIANCE
        )
        boxes = box_utils.center_form_to_corner_form(boxes)
        detections = (scores, boxes)
        detections = self.post_processor(detections)
        return detections, {}
