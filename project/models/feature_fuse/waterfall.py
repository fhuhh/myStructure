import torch
from torch import nn
from torch.nn import Module
from typing import Optional
from yacs.config import CfgNode
class SepConv2d(Module):
    def __init__(
            self,
            in_c:int,
            out_c:int,
            k:int=3,
            stride:int=1,
            padding:Optional[int]=None,
            dilation:int=1,
            bias:bool=True,
            depth_mul:float=2.,
            act:bool=True
    ):
        super().__init__()
        hidden=int(in_c*depth_mul)
        self.depth_conv=nn.Conv2d(
            in_channels=in_c,
            out_channels=hidden,
            kernel_size=k,
            stride=stride,
            padding=k//2+dilation-1 if padding is None else padding,
            dilation=dilation,
            groups=in_c,
            bias=bias
        )

        self.point_conv=nn.Conv2d(
            in_channels=hidden,
            out_channels=out_c,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=bias,
        )

        if act:
            self.act=nn.SiLU(inplace=True)
        else:
            self.act=nn.ReLU(inplace=True)

    def forward(
            self,
            x
    ):
        x=self.depth_conv(x)
        x=self.act(x)
        x=self.point_conv(x)
        return x

class AstModule(Module):
    def __init__(
            self,
            in_c:int,
            out_c:int,
            k:int,
            dilation: int,
            padding:Optional[int]=None,
            act:bool=True
    ):
        super().__init__()
        self.ast_conv=SepConv2d(
            in_c=in_c,
            out_c=out_c,
            k=k,
            dilation=dilation,
            padding=padding,
            bias=False
        )

        self.bn=nn.BatchNorm2d(out_c)
        if act:
            self.act=nn.SiLU(inplace=True)
        else:
            self.act=nn.ReLU(inplace=True)

    def forward(
            self,
            x
    ):
        x=self.ast_conv(x)

        return self.act(
            self.bn(x)
        )

class WaterfallFuser(Module):
    def __init__(
            self,
            in_c:int,
            out_c:int,
            cfg:CfgNode,
            act:bool = True
    ):
        super().__init__()
        reduction=out_c//8
        self.dilation_rates=cfg.MODEL.WATERFALL_FUSER.DILATION_RATES
        ast_channels=[in_c]+[out_c]*len(self.dilation_rates)
        self.ast_modules=nn.ModuleList(
            [
               AstModule(
                   in_c=ast_channels[idx],
                   out_c=ast_channels[idx+1],
                   k=1 if idx==0 else 3,
                   dilation=dilation_rate,
                   padding=None,
                   act=act
               ) for idx,dilation_rate in enumerate(self.dilation_rates)
            ]
        )

        if act:
            self.act=nn.SiLU(inplace=True)
        else:
            self.act=nn.ReLU(inplace=True)

        self.global_avg_pool=nn.Sequential(
            nn.AvgPool2d(
                kernel_size=3,
                stride=1,
                padding=1
            ),
            nn.Conv2d(
                in_channels=in_c,
                out_channels=out_c,
                kernel_size=1,
                stride=1,
                bias=False
            ),
            nn.BatchNorm2d(
                num_features=out_c
            ),
            self.act
        )

        self.conv1=nn.Conv2d(
            in_channels=(len(self.dilation_rates)+1)*out_c,
            out_channels=out_c,
            kernel_size=1,
            bias=False
        )
        self.bn1=nn.BatchNorm2d(num_features=out_c)

        self.conv2=nn.Conv2d(
            in_channels=in_c,
            out_channels=reduction,
            kernel_size=1,
            bias=False
        )
        self.bn2=nn.BatchNorm2d(reduction)

        self.last_conv=nn.Sequential(
            nn.Conv2d(
                in_channels=out_c+reduction,
                out_channels=out_c,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False
            ),
            nn.BatchNorm2d(
                num_features=out_c
            ),
            self.act,
            nn.Conv2d(
                in_channels=out_c,
                out_channels=out_c,
                kernel_size=1,
                stride=1,
                bias=False
            ),
        )

    def forward(
            self,
            x:list
    ):
        high_features=x[0]
        low_features=x[1]
        ast_res=[high_features]
        for ast_module in self.ast_modules:
            ast_res.append(
                ast_module(
                    ast_res[-1]
                )
            )
        ast_res.append(
            self.global_avg_pool(
                high_features
            )
        )

        high_res=torch.cat(
            ast_res[1:],
            dim=1
        )

        high_res=self.act(
            self.bn1(
                self.conv1(high_res)
            )
        )

        low_res=self.act(
            self.bn2(
                self.conv2(low_features)
            )
        )

        return self.last_conv(
            torch.cat(
                [high_res,low_res],
                dim=1
            )
        )
