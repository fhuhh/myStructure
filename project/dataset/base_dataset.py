from torch.utils.data import Dataset
from torch.utils.data.dataset import T_co
from yacs.config import CfgNode
from . import CustomTransformer
class BaseDataset(Dataset):
    def __getitem__(self, index) -> T_co:
        pass

    def __init__(
            self,
            mode:str,
            cfg:CfgNode
    ):
        self.dataset_name=cfg.DATASET.DATASET
        self.root = cfg.DATASET.ROOT
        self.num_joints = cfg.DATASET.NUM_JOINTS
        self.det_type = cfg.DATASET.DET_TYPE
        self.det_kpt = cfg.DATASET.DETKPT_NAME
        self.det_kpt_ids = cfg.DATASET.DETKPT_IDXS
        self.input_size=cfg.DATASET.INPUT_SIZE
        self.output_size=cfg.DATASET.OUTPUT_SIZE
        self.offset_radius=cfg.DATASET.OFFSET_RADIUS
        self.pin_memory=cfg.PIN_MEMORY
        self.det_kpt_sigma=None
        self.sigma=None
        self.bg_weight=None
        self.heatmap_generator=None
        self.det_kpt_heatmap_generator=None
        self.offset_generator=None
        if mode.__eq__("train"):
            # 训练模式需要打乱
            self.shuffle=True
            self.det_kpt_sigma=cfg.DATASET.DETKPT_SIGMA
            self.sigma=cfg.DATASET.SIGMA
            self.bg_weight=cfg.DATASET.BG_WEIGHT
            self.train=True
        else:
            self.shuffle=False
            self.train=False
        self.batch_size=cfg.TRAIN.BATCH_SIZE
        self.transforms=CustomTransformer(
            cfg=cfg,
            mode=mode
        )
    def get_name(self):
        return self.dataset_name
    def __len__(self):
        pass