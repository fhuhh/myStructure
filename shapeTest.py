import numpy as np
import torch
from torch import nn

def change_m(m, c):
    '''
    改变mask数组值
    :param m: 原mask数组 2维
    :param c: 传入要改变的位置 [x,y,w,h]
    :return: 修改后的mask数组
    '''
    for j in range(c[1], c[1] + c[3]):    # 循环遍历修改不为1的值
        for i in range(c[0], c[0] + c[2]):
            if(m[j][i] != 1):
                m[j][i] = 1
    return m                              # 返回修改后的mask


def gen_mask(b, W, H):
    '''
    输入一组bbox数组，输出一组mask
    :param b: batch*[[x,y,w,h]]
    :param W: 样本宽
    :param H: 样本高
    :return: batch * 1 * H * W 即 batch * H * W
    '''
    batch = np.shape(b)[0]          # 得到batch信息
    mask = np.zeros((batch, W, H), dtype=float, order='C')  #生成原始mask组
    p_num = np.shape(b)[1]  # 得到每个batch中包含的样本数
    for i in range(0, batch):
        for j in range(0, p_num):   # 根据同个batch中的每个样本修改原始mask
            mask[i] = change_m(mask[i], b[i][j])
    return mask                     #返回修改后的mask batch * 1 * H * W

def gen_per_mask(b,  W, H, max_p = 10):
    '''
    输入一组bbox数组，输出一组与person一一对应的mask
    :param b: batch*[[x,y,w,h]]
    :param W,H: 生成mask的宽和高
    :param max_p: 一个实例中的最大人数
    :return: batch * max_p * H * W
    '''
    batch = np.shape(b)[0]              #batch信息
    mask = np.zeros((batch, max_p, W, H), dtype=float, order='C')
    p_num = np.shape(b)[1]              # 每个样本中的实例个数
    for i in range(0, batch):
       for j in range(0, p_num):
            mask[i][j] = change_m(mask[i][j], b[i][j])   # 为每个实例生成mask
    return mask

def per_mask_reshape(p_mask, c, patch):
    '''
    改变人物掩码的形态
    :param p_mask: 每个实例中每个人的mask数组 B * P_n * H * W
    :param c: channels
    :param patch: 切割的patch
    :return: P_n * B * ((H/p)*(W/p)) * (p*p*C) 形式的与person一一对应的mask组
    '''
    p = p_mask
    p = np.expand_dims(p, 1).repeat(c, axis=1)      #在第二个维度上赋值channel份，test正确
    p = p.transpose(2,0,3,4,1)                      #B * P_n * H * W ---> P_n*B*H*W*C,test正确
    shape = np.shape(p)
    p1 = p.reshape(shape[0], shape[1], shape[2] , shape[3] // patch, patch * shape[4])  #因为reshape默认按行分组，所以必须先竖着归类
    p1 = p1.transpose(0,1,3,2,4)
    p2 = p1.reshape(shape[0], shape[1], shape[2] // patch, shape[3]//patch, patch * patch * shape[4])
    p2 = p2.transpose(0,1,3,2,4)        #P_n*B*H*W*C--->P_n*B*(H/p)*(W/p)*(p*p*C)，test正确
    p3 = p2.reshape(shape[0], shape[1], (shape[2] // patch) * (shape[3]//patch), patch * patch * shape[4])
    return p3                           # P_n*B*(H/p)*(W/p)*(p*p*C)--->P_n*B*((H/p)*(W/p))*(p*p*C) 四维 ，test正确

def person_embedding(c, p_mask, patch, max_p = 10):
    '''
    生成每个batch的person信息码并嵌入mask,最后整合
    :param c: 通道数
    :param p_mask: 最初生成的每个实例中每个人的mask数组 B * P_n * H * W
    :param patch:  patch数
    :param max=p: 最大人数
    :return: B * ((H/p)*(W/p)) * (p*p*C) 形式的与batch一一对应的整合起来的person信息编码
             re_id 生成的person信息编码 p_max * 1
    '''
    p = p_mask
    shape = np.shape(p)
    embedding = nn.Embedding(max_p, 1)          # 生成embedding信息
    p_id = np.zeros(max_p, dtype=float, order='C')
    for i in range(0, max_p):                   # 生成嵌入embedding信息的矩阵
        p_id[i] = i
    p_id = embedding(torch.LongTensor(p_id))    #嵌入信息
    p_id = p_id.detach().numpy()                #转换为numpy数组
    re_id = p_id                                #保存转换后的数组
    re_id = re_id.reshape(max_p)                #转为一维
    p_id = p_id.repeat(patch * patch * c, 1)    #P-n --> P_n * (p * p * C)
    p_id = np.expand_dims(p_id, 1).repeat(shape[0], axis=1)                              #P_n * (p * p * C) --> P_n * B * (p*p*C)
    p_id = np.expand_dims(p_id, 2).repeat(shape[2] * shape[3] // patch // patch, axis=2) #P_n * B * (p*p*C) --> P_n*B*((H/p)*(W/p))*(p*p*C)  test正确
    p_m = per_mask_reshape(p_mask, c, patch)    #生成与每个人对应的mask
    p = p_id * p_m                              #将person信息嵌入mask
    for i in range(1,max_p):                    #将同一batch的所有person的信息整合
        p[0] = p[0] + p[i]
    return p[0],re_id                               #返回整合后的嵌入信息


