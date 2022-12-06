from yacs.config import CfgNode
from torch.nn import Module,MultiheadAttention,LayerNorm,Linear
from project.utils import patch_embedding
from project.models.transformers import DropPath
import torch


class EncoderBlock(Module):
    def __init__(
            self,
            in_dim,
            out_dim,
            n_head,
            in_channel,
            patch_size,
    ):
        super().__init__()
        self.in_dim=in_dim
        self.out_dim=out_dim
        self.n_head=n_head
        self.in_channel=in_channel
        self.patch_size=patch_size
        # 假定输入会先经过norm
        self.norm1=LayerNorm(
            normalized_shape=self.in_dim
        )

        # norm2用于输出的使用
        self.norm2=LayerNorm(
            normalized_shape=self.out_dim
        )
        # 假定这里先不使用drop out
        self.k_proj=Linear(
            in_features=self.in_dim,
            out_features=self.out_dim,
            bias=False
        )
        self.q_proj=Linear(
            in_features=self.in_dim,
            out_features=self.out_dim,
            bias=False
        )
        self.v_proj=Linear(
            in_features=self.in_dim,
            out_features=self.out_dim,
            bias=False
        )

        self.out_proj=Linear(
            in_features=self.out_dim,
            out_features=self.out_dim,
            bias=True
        )

    def forward(
            self,
            x:dict
    ):
        q=x["query"]
        k=x["key"]
        v=x["value"]
        full_mask=x["full_mask"]
        person_mask=x["person_mask"]
        # 对full_mask进行变形

        batch_size,_,height,width=full_mask.shape








class IndividualTransformer(Module):
    def __init__(
            self,
            cfg:CfgNode
    ):
        super().__init__()
        trans_conf=cfg.MODEL.TRANSFORMER_1
        encoder_conf = cfg.MODEL.TRANSFORMER_1.ENCODER
        self.n_encoder=trans_conf.ENCODER_NUM
        self.n_decoder=trans_conf.DECODER_NUM
        self.patch_size=encoder_conf.PATCH_SIZE
        self.in_channel=encoder_conf.IN_CHANNEL
        self.in_dim=self.patch_size*self.patch_size*self.in_channel
        # 这里可以修改,先设置成维度不变
        self.out_dim=self.in_dim
        self.n_head=encoder_conf.N_HEAD
        # 特征图的大小需要知道
        self.in_size=trans_conf.IN_SIZE
        self.encoder_block=EncoderBlock(
            in_dim=self.in_dim,
            out_dim=self.out_dim,
            n_head=self.n_head,
            in_channel=self.in_channel,
            patch_size=self.patch_size
        )

    def forward(
            self,
            x
    ):
        # TODO:完成feature map的patch embedding
        # step1:reshape
        # step2:padding
        # step3:transpose
        patch_features=patch_embedding(
            feature_map=x,
            patch_size=self.patch_size
        )





