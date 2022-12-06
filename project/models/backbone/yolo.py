import torch
from torch.nn import Module
from . import Conv,Contract,Bottleneck,Concat,TransConv
from typing import Optional
from yacs.config import CfgNode
from torch import nn
from project.utils import LoggerTool
from project.models.feature_fuse import WaterfallFuser
class Focus(Module):
    def __init__(
            self,
            in_c:int,
            out_c:int,
            k:int,
            stride:int,
            padding:Optional[int]=None,
            group:int=1,
            act=True
    ):
        super().__init__()
        self.contract=Contract()
        self.conv=Conv(
            in_c=in_c*4,
            out_c=out_c,
            k=k,
            stride=stride,
            padding=padding,
            group=group,
            act=act
        )
    def forward(
            self,
            x
    ):
        return self.conv(
            self.contract(x)
        )

class C3(Module):
    def __init__(
            self,
            in_c: int,
            out_c: int,
            n:int,
            shortcut:bool=True,
            group:int=1,
            expansion:float=0.5,
            act:bool=True
    ):
        super().__init__()
        hidden=int(out_c*expansion)
        self.cv1=Conv(
            in_c=in_c,
            out_c=hidden,
            k=1,
            stride=1,
            act=act
        )
        self.cv2=Conv(
            in_c=in_c,
            out_c=hidden,
            k=1,
            stride=1,
            act=act
        )
        self.cv3=Conv(
            in_c=2*hidden,
            out_c=out_c,
            k=1,
            stride=1,
            act=act
        )
        self.bottlenecks=nn.Sequential(
            *[
                Bottleneck(
                    in_c=hidden,
                    out_c=hidden,
                    shortcut=shortcut,
                    group=group,
                    expansion=1.0,
                    act=act
                ) for _ in range(n)
            ]
        )
    def forward(
            self,
            x
    ):
        return self.cv3(
            torch.cat(
                (
                    self.bottlenecks(
                        self.cv1(x)
                    ),
                    self.cv2(x)
                ),
                dim=1
            )
        )
class SPP(Module):
    def __init__(
            self,
            in_c:int,
            out_c:int,
            k:Optional[list]
    ):
        super().__init__()
        if k is None:
            k = [3, 3, 3]
        hidden=out_c//2
        self.cv1=Conv(
            in_c=in_c,
            out_c=hidden,
            k=1,
            stride=1
        )
        self.cv2=Conv(
            in_c=hidden*(len(k)+1),
            out_c=out_c,
            k=1,
            stride=1
        )
        self.max_pools=nn.ModuleList()
        for pool_kernel in k:
            num_3x3_pool=(pool_kernel-3)//2+1
            self.max_pools.append(
                nn.Sequential(
                    *[
                        nn.MaxPool2d(
                            kernel_size=3,
                            stride=1,
                            padding=1
                        ) for _ in range(num_3x3_pool)
                    ]
                )
            )
    def forward(
            self,
            x
    ):
        x=self.cv1(x)
        return self.cv2(
            torch.cat(
                [x]+[max_pool(x) for max_pool in self.max_pools],
                dim=1
            )
        )

class YoloBackbone(Module):
    def __init__(
            self,
            cfg:CfgNode,
            logger:LoggerTool
    ):
        super().__init__()
        #首先获取配置
        yolo_backbone=cfg.MODEL.YOLO_BACKBONE
        self.from_list=yolo_backbone.FROM
        self.model_list=yolo_backbone.MODULE
        self.out_channel_list=yolo_backbone.OUT_CHANNEL
        self.block_list=yolo_backbone.BLOCK_NUM
        self.channel_list=[3]
        self.layers=[]
        self.head_len=yolo_backbone.HEAD_LEN
        logger.log_sth("{:<10}{:<20}{:<20}{:<10}".format("idx","name","in channel","out channel"))
        for i,(module,f,block_num,out_c) in enumerate(zip(self.model_list,self.from_list,self.block_list,self.out_channel_list)):
            if module=='FOCUS':
                in_c=self.channel_list[-1]
                self.layers.append(
                    Focus(
                        in_c=in_c,
                        out_c=out_c,
                        k=3,
                        stride=1,
                    )
                )
            elif module=='C3':
                in_c=self.channel_list[-1]
                self.layers.append(
                    C3(
                        in_c=in_c,
                        out_c=out_c,
                        n=block_num,
                        shortcut=False
                    )
                )
            elif module=='SPP':
                in_c=self.channel_list[-1]
                self.layers.append(
                    SPP(
                        in_c=in_c,
                        out_c=out_c,
                        k=[3,5,7]
                    )
                )
            elif module=='CONV':
                in_c=self.channel_list[-1]
                self.layers.append(
                    Conv(
                        in_c=in_c,
                        out_c=out_c,
                        k=3,
                        stride=2
                    )
                )
            elif module=='WATERFALL':
                # 只有这里才会有两个from
                in_c=[self.channel_list[f_idx] for f_idx in f]
                self.layers.append(
                    WaterfallFuser(
                        in_c=in_c[0],
                        out_c=out_c,
                        cfg=cfg
                    )
                )
            else:
                in_c=self.channel_list[-1]
                # 这里是一个TransConv
                self.layers.append(
                    TransConv(
                        in_c=in_c,
                        out_c=out_c,
                        k=3
                    )
                )
            logger.log_sth("{:<10}{:<20}{:<20}{:<10}".format(i,module,str(in_c),out_c))
            self.channel_list.append(out_c)

        self.layers=nn.ModuleList(self.layers)

    def forward(
            self,
            x
    ):
        # 返回的是-1,-3,-5,-7
        res_list=[x]

        for idx,module in enumerate(self.layers):
            if isinstance(module,WaterfallFuser):
                x=module(
                    [res_list[f_idx] for f_idx in self.from_list[idx]]
                )
            else:
                x=module(res_list[-1])
            res_list.append(x)
        return [res_list[idx] for idx in [-7,-5,-3,-1]]