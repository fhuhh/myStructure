import torch
from torch.nn import Module
from typing import Optional
import torch.nn as nn
from torch import Tensor
class Conv(Module):

    def __init__(
            self,
            in_c:int,
            out_c:int,
            k:int,
            stride:int,
            padding:Optional[int]=None,
            group:int=1,
            act:bool=True
    ):
        super().__init__()
        self.conv=nn.Conv2d(
            in_channels=in_c,
            out_channels=out_c,
            kernel_size=k,
            stride=stride,
            padding=k//2 if padding is None else padding,
            groups=group,
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
        res=self.act(
            self.bn(
                self.conv(x)
            )
        )
        return res

class TransConv(Module):
    def __init__(
            self,
            in_c:int,
            out_c:int,
            k:int,
            stride:int=2,
            padding:Optional[int]=None,
            out_padding:Optional[int]=None,
            act:bool=True
    ):
        super().__init__()
        self.trans_conv=nn.ConvTranspose2d(
            in_channels=in_c,
            out_channels=out_c,
            kernel_size=k,
            stride=stride,
            padding=k//2 if padding is None else padding,
            output_padding=1 if out_padding is None else out_padding
        )
        self.bn=nn.BatchNorm2d(num_features=out_c)
        self.act=nn.SiLU(inplace=True) if act else nn.ReLU(inplace=True)

    def forward(
            self,
            x
    ):
        return self.act(
            self.bn(
                self.trans_conv(x)
            )
        )

class Contract(Module):
    # 这个模块相当于把宽高维的信息转化到channel去
    def __init__(
            self,
            gain=2
    ):
        super().__init__()
        self.gain=gain

    def forward(
            self,
            x:Tensor=torch.Tensor()
    ):
        n,c,h,w=x.size()
        x=x.view(
            n,
            c,
            h//self.gain,
            self.gain,
            w//self.gain,
            self.gain
        )
        x=x.permute(0,3,5,1,2,4)

        return x.contiguous().view(
            n,
            c*self.gain*self.gain,
            h//self.gain,
            w//self.gain
        )
class Bottleneck(Module):
    def __init__(
            self,
            in_c: int,
            out_c: int,
            shortcut: bool,
            group: int,
            expansion: float,
            act: bool
    ):
        super().__init__()
        hidden=int(expansion*out_c)
        self.cv1=Conv(
            in_c=in_c,
            out_c=hidden,
            k=1,
            stride=1,
            act=act
        )
        self.cv2=Conv(
            in_c=hidden,
            out_c=out_c,
            k=3,
            stride=1,
            group=group,
            act=act
        )

        self.add=shortcut and in_c==out_c

    def forward(
            self,
            x
    ):
        if self.add:
            return x+self.cv2(
                self.cv1(x)
            )
        else:
            return self.cv2(
                self.cv1(x)
            )
class Concat(Module):
    def __init__(
            self,
            dim=1
    ):
        super().__init__()
        self.dim=dim
    def forward(
            self,
            x
    ):
        return torch.cat(x,dim=self.dim)