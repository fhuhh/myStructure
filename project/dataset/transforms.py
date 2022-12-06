from yacs.config import CfgNode
from torchvision.transforms import ToTensor,Normalize
from torch.nn import Module
import numpy as np
import cv2
from typing import Optional
import torch
# ocpose的关键点数量是14,应该是和crowdpose一样
FLIP_CONFIG = {
    'COCO': [
        0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15
    ],
    'COCO_WITH_DETKPT': [
        0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15, 17
    ],
    'CROWDPOSE': [
        1, 0, 3, 2, 5, 4, 7, 6, 9, 8, 11, 10, 12, 13
    ],
    'CROWDPOSE_WITH_DETKPT': [
        1, 0, 3, 2, 5, 4, 7, 6, 9, 8, 11, 10, 12, 13, 14
    ],
    'OCHUMAN': [
        0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15
    ],
    'OCHUMAN_WITH_DETKPT': [
        0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15, 17
    ],
    'OCPOSE':[
        1, 0, 3, 2, 5, 4, 7, 6, 9, 8, 11, 10, 12, 13
    ],
    'OCPOSE_WITH_DETKPT':[
        1, 0, 3, 2, 5, 4, 7, 6, 9, 8, 11, 10, 12, 13, 14
    ]
}


class AffineOp(Module):
    def __init__(
            self,
            input_size,
            output_size,
            flip:bool,
            flip_index:Optional[list]
    ):
        super().__init__()
        self.input_size=input_size
        self.output_size=output_size
        self.flip=flip
        self.flip_index=flip_index

    def forward(
            self,
            image: np.ndarray,
            joints:np.ndarray,
            area:np.ndarray,
            bbox:np.ndarray
    ):
        # 这里image是cv2读进来的,image转化为input_size,joints转化为output_size
        height, width = image.shape[:2]
        # joints = np.array(joints)
        #这个地方resize，但是关节点的变换需要确定
        #应该可以通过ratio来计算
        #首先填充一下，防止变换后人物形变
        h=max(height,width)
        ratio=self.input_size/h
        #
        if height==h:
            diff=h-width
            direction="right"
        else:
            diff=h-height
            direction="down"
        # 首先确定一下会不会左右翻转
        if self.flip and np.random.random()<0.5:
            #翻转
            image=cv2.flip(image,flipCode=1)
            joints=joints[:,self.flip_index]
            joints[:,:,0]=width-joints[:,:,0]-1
            bbox[:,0]=width-bbox[:,0]-1
        if direction.__eq__("right"):
            image=cv2.copyMakeBorder(image,0,0,0,diff,cv2.BORDER_CONSTANT,value=(0,0,0))
        else:
            image=cv2.copyMakeBorder(image,0,diff,0,0,cv2.BORDER_CONSTANT,value=(0,0,0))
        #resize一下大小
        image=cv2.resize(image,dsize=(self.input_size,self.input_size))
        #处理一下joints
        joints[:,:,0:2]=joints[:,:,0:2]*ratio
        bbox=bbox*ratio
        #面积应该也处理到output_size
        area=area*ratio**2
        return image,joints,area,bbox

class ToTensorOp(Module):
    def __init__(self):
        super().__init__()
        self.to_tensor=ToTensor()
    def forward(
            self,
            image,
            joints,
            area,
            bbox
    ):
        return self.to_tensor(image),torch.from_numpy(joints),torch.from_numpy(area),torch.from_numpy(bbox)
class NormalizeOp(Module):
    def __init__(self):
        super().__init__()
        self.normalize=Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    def forward(
            self,
            image,
            joints,
            area,
            bbox
    ):
        return self.normalize(image),joints,area,bbox
class ComposeOp(Module):
    def __init__(self,
            transforms:Optional[list]
        ):
        super().__init__()
        self.transforms=transforms
    def forward(
            self,
            image,
            joints,
            area,
            bbox
    ):
        for t in self.transforms:
            image,joints,area,bbox=t(image,joints,area,bbox)
        return image,joints,area,bbox


class CustomTransformer(Module):
    def __init__(
            self,
            cfg: CfgNode,
            mode:str
    ):
        # self.max_rotation=cfg.DATASET.MAX_ROTATION
        # self.min_scale = cfg.DATASET.MIN_SCALE
        # self.max_scale = cfg.DATASET.MAX_SCALE
        # self.max_translate = cfg.DATASET.MAX_TRANSLATE
        super().__init__()
        self.input_size = cfg.DATASET.INPUT_SIZE
        self.output_size = cfg.DATASET.OUTPUT_SIZE
        self.flip = cfg.DATASET.FLIP

        if 'coco' in cfg.DATASET.DATASET:
            self.dataset_name = 'COCO'
        elif 'crowdpose' in cfg.DATASET.DATASET:
            self.dataset_name = 'CROWDPOSE'
        elif 'ochuman' in cfg.DATASET.DATASET:
            self.dataset_name = 'OCHUMAN'
        elif 'ocpose' in cfg.DATASET.DATASET:
            self.dataset_name="OCPOSE"
        else:
            raise ValueError('Please implement flip_index for new dataset: %s.' % cfg.DATASET.DATASET)
        self.flip_index=FLIP_CONFIG[self.dataset_name]

        #直接在这里建立transforms
        if mode.__eq__("train"):
            self.transforms = ComposeOp(
                [
                    AffineOp(
                        input_size=self.input_size,
                        output_size=self.output_size,
                        flip=self.flip,
                        flip_index=self.flip_index
                    ),
                    ToTensorOp(),
                    NormalizeOp()
                ]
            )
        else:
            self.transforms=ComposeOp(
                [
                    ToTensorOp(),
                    NormalizeOp()
                ]
            )

    def forward(
            self,
            image,
            joints,
            area,
            bbox
    ):
        return self.transforms(
            image,
            joints,
            area,
            bbox
        )



