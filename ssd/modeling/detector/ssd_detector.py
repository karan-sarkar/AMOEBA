from torch import nn

from ssd.modeling.backbone import build_backbone
from ssd.modeling.box_head import build_box_head

class C(nn.Module):
    def __init__(self):
        super(C, self).__init__()
        
        self.fc1 = nn.Linear(8732 * 14, 3072)
        self.bn1_fc = nn.BatchNorm1d(3072)
        self.fc2 = nn.Linear(3072, 512)
        self.bn2_fc = nn.BatchNorm1d(512)
        self.fc3 = nn.Linear(512, 1)
    def forward(self, a, b):
        x = torch.cat([a.softmax(-1), b], -1)
        x = x.view(-1, 8732 * 14)
        x = F.relu(self.bn1_fc(self.fc1(x)))
        x = F.relu(self.bn2_fc(self.fc2(x)))
        x = self.fc3(x)
        return x


class SSDDetector(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.backbone = build_backbone(cfg)
        self.box_head = build_box_head(cfg)
        self.c = C()

    def forward(self, images, targets=None, discrep=False):
        features = self.backbone(images)
        detections, detector_losses = self.box_head(features, targets)
        if discrep:
            cls, bbox = detections
            detector_losses['discrep'] = self.c(cls, bbox)
        
        if self.training:
            return detector_losses
        return detections
