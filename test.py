from project import LoggerTool,ConfigTool,ParserTool,CocoDataset,YoloBackbone
from pathlib import Path
import numpy as np
import cv2
import time
import torch
if __name__=="__main__":
    parser_tool=ParserTool()
    args=parser_tool.get_args()

    config_tool=ConfigTool()
    config_tool.merge_from_file(Path(args.cfg))
    config_tool.merge_from_list(args.opt)
    cfg=config_tool.get_cfg()

    logger=LoggerTool()
    logger.setup_logger(
        output_dir=Path(args.output),
        mode="train"
    )

    yolo_backbone=YoloBackbone(
        cfg=cfg,
        logger=logger
    )

    input_tensor=torch.rand((16,3,512,512))
    output_tensor=yolo_backbone(input_tensor)
    print("stop")
    # coco_loader=CocoDataset(
    #     mode="train",
    #     cfg=cfg
    # ).create_dataloader()

    # for img,heatmap,weight_mask,center_heatmap,center_weight_mask,offset_map,weight_map in coco_loader:
    #     print(img.shape)
#



