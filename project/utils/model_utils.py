import torch
import torch.nn.functional as F

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