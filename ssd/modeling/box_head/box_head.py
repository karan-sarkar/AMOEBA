from torch import nn
import torch.nn.functional as F
import torch

from ssd.modeling import registry
from ssd.modeling.anchors.prior_box import PriorBox
from ssd.modeling.box_head.box_predictor import make_box_predictor
from ssd.utils import box_utils
from .inference import PostProcessor
from .loss import MultiBoxLoss
from ssd.utils import box_utils


class GradReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg()

def grad_reverse(x):
    return GradReverse.apply(x)


@registry.BOX_HEADS.register('SSDBoxHead')
class SSDBoxHead(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.predictor = make_box_predictor(cfg)
        self.predictor2 = make_box_predictor(cfg)
        self.loss_evaluator = MultiBoxLoss(neg_pos_ratio=cfg.MODEL.NEG_POS_RATIO)
        self.post_processor = PostProcessor(cfg)
        self.priors = None
        self.l1loss = nn.L1Loss()
        self.celoss = nn.CrossEntropyLoss()

    def forward(self, features, targets=None, discrep=False):
        if discrep:
            features = tuple([grad_reverse(f) for f in features])
        cls_logits, bbox_pred = self.predictor(features)
        cls_logits2, bbox_pred2 = self.predictor2(features)
        if self.training:
            return self._forward_train(cls_logits, bbox_pred, cls_logits2, bbox_pred2, targets, discrep=discrep)
        else:
            return self._forward_test(cls_logits, bbox_pred)

    def _forward_train(self, cls_logits, bbox_pred, cls_logits2, bbox_pred2, targets, discrep=False, double=False):
        gt_boxes, gt_labels = targets['boxes'], targets['labels']
        reg_loss, cls_loss = self.loss_evaluator(cls_logits, bbox_pred, gt_labels, gt_boxes)
        reg_loss2, cls_loss2 = self.loss_evaluator(cls_logits2, bbox_pred2, gt_labels, gt_boxes)
        if double:
            reg_loss += reg_loss2
            cls_loss += cls_loss2
        detections = (cls_logits, bbox_pred)
        loss_dict = dict(
            reg_loss=reg_loss,
            cls_loss=cls_loss,
        )
        if discrep:
            x = cls_logits.view(-1, cls_logits.size(-1)).softmax(-1)
            y = cls_logits2.view(-1, cls_logits.size(-1)).softmax(-1)
            #mx = F.one_hot(x.argmax(1), x.size(1))
            #my = F.one_hot(y.argmax(1), y.size(1))
            #discrep_loss = (x - my).abs() + (y - mx).abs()
            z = (x + y) / 2
            loss_dict['discrep_loss'] = (x * (x / z).log()).sum() / x.size(0) + (y * (y / z).log()).sum() / x.size(0)
            

            #del loss_dict['reg_loss']
            #del loss_dict['cls_loss']
            
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
