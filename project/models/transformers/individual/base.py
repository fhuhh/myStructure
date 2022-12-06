import torch
from torch.nn import Module
def drop_path(
        x:torch.Tensor,
        drop_prob:float,
        training:bool=True
):
    if drop_prob==0 or not training:
        return x

    keep_prob=1-drop_prob
    prob_shape=[x.shape[0]]+(len(x.shape)-1)*[1]
    # rand结果在(0,drop_prob)区间内的话就会抛弃，否则被保留
    random_tensor=keep_prob+torch.rand(
        prob_shape,
        dtype=x.dtype,
        device=x.device
    )
    # 二值化
    random_tensor.floor_()
    return torch.div(x,keep_prob)*random_tensor

class DropPath(Module):
    def __init__(
            self,
            drop_prob:float=0
    ):
        super().__init__()
        self.drop_prob=drop_prob

    def forward(
            self,
            x
    ):
        return drop_path(
            x=x,
            drop_prob=self.drop_prob,
        )