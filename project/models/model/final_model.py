from torch.nn import Module
from yacs.config import CfgNode
from project.models.detection import Detector
from project.models.backbone import YoloBackbone
from project.models.transformers import IndividualTransformer
from project.utils import LoggerTool
import torch
class DualTransMPPE(Module):
    def __init__(
            self,
            cfg:CfgNode,
            logger:LoggerTool
    ):
        super().__init__()
        detector_in_channels=cfg.MODEL.DETECTOR.IN_CHANNELS
        self.anchors=cfg.MODEL.DETECTOR.ANCHORS
        self.anchors = [torch.tensor(self.anchors[idx], dtype=torch.float32) / i for idx, i in
                        enumerate([8, 16, 32, 64])]
        self.backbone=YoloBackbone(
            cfg=cfg,
            logger=logger
        )
        self.detector=Detector(
            anchors=self.anchors,
            in_channels=detector_in_channels,
            logger=logger
        )
        self.individual_trans=IndividualTransformer(cfg=cfg)
        logger.log_sth("init success!")

    def forward(
            self,
            x
    ):
        # 首先进入backbone,然后进入detector
        # 结果是四个尺度的features map
        features=self.backbone(x["imgs"])
        # 结果是四个尺度的目标检测结果
        detection=self.detector(features)
        # 实际上框的话可以直接使用ground truth的框，之后推理的时候使用detector的后处理内容
        mask=x["mask"]
        feature_map=features[0]

        return detection