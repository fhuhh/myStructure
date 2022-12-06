import torch
import torch.nn.functional as F
from torch.nn import Embedding

def patch_embedding(
    feature_map:torch.Tensor,
    patch_size:int
):
    _,_,height,width=feature_map.shape
    # 然后padding
    height_padding=patch_size-height%patch_size
    width_padding=patch_size-width%patch_size
    features=F.pad(
        feature_map,
        pad=(width_padding//2,width_padding//2,height_padding//2,height_padding//2)
    )
    features = torch.permute(
        features,
        dims=(0, 2, 3, 1)
    )
    batch_size,height,width,channel_num=features.shape
    height_patch,width_patch=height//patch_size,width//patch_size
    # reshape为patch
    features=torch.reshape(
        features,
        shape=(batch_size,height_patch,patch_size,width_patch,patch_size,channel_num)
    )
    features=torch.permute(
        features,
        dims=(0,1,3,2,4,5)
    )
    # 合并最后三维
    features=torch.reshape(
        features,
        shape=(batch_size,height_patch*width_patch,patch_size*patch_size*channel_num)
    )
    return features

def get_full_mask(
    bbox:list,
    height:int,
    width:int
):
    """
    :param bbox:[[[x,y,w,h]...[x,y,w,h]],[[x,y,w,h]...[x,y,w,h]],...,[[x,y,w,h]...[x,y,w,h]]]
    :param height:mask的高
    :param width:mask的宽
    :return:B*1*H*W的mask
    """

def get_person_mask(
    bbox:list,
    height:int,
    width:int,
    max_person:int
):
    """
    :param bbox: [[[x,y,w,h]...[x,y,w,h]],[[x,y,w,h]...[x,y,w,h]],...,[[x,y,w,h]...[x,y,w,h]]]
    :param height: mask的高
    :param width: mask的宽
    :param max_person: 每张图中的最大人数
    :return: B*max_person*H*W的mask
    """

def get_pos_embedding(
    patch_size:int,
    height:int,
    width:int,
    channel:int,
    batch_size:int,
    embed:Embedding
):
    """
    注意生成策略一定要对,先生成(H/P*W/P)*(P*P*C)
    :param patch_size:即P
    :param height: 即H
    :param width: 即W
    :param channel: 即C
    :param batch_size: 即B
    :param embed: nn.Embedding
    :return: B*(H/P*W/P)*(P*P*C)的位置编码
    """

def get_person_embedding(
    patch_size:int,
    height:int,
    width:int,
    channel:int,
    batch_size:int,
    embed:Embedding,
    person_mask:torch.Tensor
):
    """
    注意生成策略一定要对,先生成max_person*C,max_person通过传入的person_mask获得
    :param person_mask: B*max_person*H*W
    :param patch_size:即P
    :param height: 即H
    :param width: 即W
    :param channel: 即C
    :param batch_size: 即B
    :param embed: nn.Embedding
    :return: B*(H/P*W/P)*(P*P*C)的位置编码
    """