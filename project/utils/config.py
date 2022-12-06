from yacs.config import CfgNode
from pathlib import Path
from typing import Optional
class ConfigTool:
    def __init__(self):
        # 在这儿给一个初始化的CfgNode
        self.cfg=CfgNode(new_allowed=True)
        self.cfg.CFG_NAME=''
        self.cfg.OUTPUT_DIR=''
        self.cfg.LOG_DIR=''
        self.cfg.GPUS='0'
        # 后台打印频率
        self.cfg.PRINT_FREQ=20
        self.cfg.AUTO_RESUME=False
        self.cfg.PIN_MEMORY=True
        self.cfg.RANK=0
        self.cfg.VERBOSE=True
        # 这个配置干嘛的---
        self.cfg.DDP=True

        # cudnn配置
        self.cfg.CUDNN=CfgNode(new_allowed=True)
        self.cfg.CUDNN.BENCHMARK = True
        self.cfg.CUDNN.DETERMINISTIC = False
        self.cfg.CUDNN.ENABLED = True

        # 模型相关配置
        self.cfg.MODEL=CfgNode(new_allowed=True)
        self.cfg.MODEL.NAME = 'drnet'
        self.cfg.MODEL.INIT_WEIGHTS = True
        self.cfg.MODEL.PRETRAINED = ''
        self.cfg.MODEL.SYNC_BN = False

        # refine相关
        self.cfg.REFINE=CfgNode(new_allowed=True)
        self.cfg.REFINE.USE_REFINE = False
        self.cfg.REFINE.NUM_LAYERS = 1
        self.cfg.REFINE.MAX_PROPOSAL = 120

        # loss相关
        self.cfg.LOSS=CfgNode(new_allowed=True)
        self.cfg.LOSS.WITH_HEATMAP_LOSS = True
        self.cfg.LOSS.HEATMAP_LOSS_FACTOR = 1.0
        self.cfg.LOSS.WITH_OFFSET_LOSS = True
        self.cfg.LOSS.OFFSET_LOSS_FACTOR = 1.0
        self.cfg.LOSS.WITH_REFINE_LOSS = False
        self.cfg.LOSS.REFINE_LOSS_FACTOR = 1.0

        # dataset相关
        self.cfg.DATASET=CfgNode(new_allowed=True)
        self.cfg.DATASET.ROOT = 'data'
        self.cfg.DATASET.DATASET = 'ochuman'
        self.cfg.DATASET.NUM_JOINTS = 17
        self.cfg.DATASET.MAX_NUM_PEOPLE = 30
        self.cfg.DATASET.TRAIN = 'val'
        self.cfg.DATASET.TEST = 'test'
        self.cfg.DATASET.OFFSET_RADIUS = 4.0
        self.cfg.DATASET.BG_WEIGHT = 1.0
        self.cfg.DATASET.DET_TYPE = 'center'
        self.cfg.DATASET.DETKPT_NAME = ['center']
        self.cfg.DATASET.DETKPT_IDXS = [list(range(17))]
        self.cfg.DATASET.INPUT_SIZE = 512
        self.cfg.DATASET.OUTPUT_SIZE = 128

        # heatmap相关
        self.cfg.DATASET.SIGMA = 2.0
        self.cfg.DATASET.DETKPT_SIGMA = 4.0
        self.cfg.DATASET.BASE_SIZE = 256.0
        self.cfg.DATASET.BASE_SIGMA = 2.0
        self.cfg.DATASET.MIN_SIGMA = 1

        #train相关
        self.cfg.TRAIN=CfgNode(new_allowed=True)
        self.cfg.TRAIN.LR_FACTOR = 0.1
        self.cfg.TRAIN.LR_STEP = [90, 120]
        self.cfg.TRAIN.LR = 0.001
        self.cfg.TRAIN.OPTIMIZER = 'adam'
        self.cfg.TRAIN.MOMENTUM = 0.9
        self.cfg.TRAIN.WD = 0.0001
        self.cfg.TRAIN.NESTEROV = False
        self.cfg.TRAIN.GAMMA1 = 0.99
        self.cfg.TRAIN.GAMMA2 = 0.0
        self.cfg.TRAIN.BEGIN_EPOCH = 0
        self.cfg.TRAIN.END_EPOCH = 140
        self.cfg.TRAIN.RESUME = False
        self.cfg.TRAIN.CHECKPOINT = ''
        self.cfg.TRAIN.IMAGES_PER_GPU = 10
        self.cfg.TRAIN.SHUFFLE = True

        #测试相关
        self.cfg.TEST=CfgNode(new_allowed=True)
        self.cfg.TEST.IMAGES_PER_GPU = 1
        self.cfg.TEST.FLIP_TEST = True
        self.cfg.TEST.SCALE_FACTOR = [1]
        self.cfg.TEST.MODEL_FILE = ''
        self.cfg.TEST.OKS_SCORE = 0.7
        self.cfg.TEST.OKS_SIGMAS = []
        self.cfg.TEST.KEYPOINT_THRESHOLD = 0.01
        self.cfg.TEST.POOL_THRESHOLD1 = 300
        self.cfg.TEST.POOL_THRESHOLD2 = 200
        self.cfg.TEST.LOG_PROGRESS = True
    def merge_from_file(
            self,
            cfg_path:Path
    ):
        assert cfg_path.exists()
        self.cfg.defrost()
        self.cfg.merge_from_file(cfg_path)
        self.cfg.freeze()
        return self
    def merge_from_list(
            self,
            cfg_list:Optional[list]
    ):
        self.cfg.defrost()
        self.cfg.merge_from_list(cfg_list)
        self.cfg.freeze()
        return self
    def get_cfg(
            self
    ):
        return self.cfg