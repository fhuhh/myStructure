import torch
from yacs.config import CfgNode
from torch.nn import Module,BCEWithLogitsLoss
import math
def bbox_iou(box1, box2, x1y1x2y2=True, GIoU=False, DIoU=False, CIoU=False, eps=1e-7):
    # Returns the IoU of box1 to box2. box1 is 4, box2 is nx4
    box2 = box2.T

    # Get the coordinates of bounding boxes
    if x1y1x2y2:  # x1, y1, x2, y2 = box1
        b1_x1, b1_y1, b1_x2, b1_y2 = box1[0], box1[1], box1[2], box1[3]
        b2_x1, b2_y1, b2_x2, b2_y2 = box2[0], box2[1], box2[2], box2[3]
    else:  # transform from xywh to xyxy
        b1_x1, b1_x2 = box1[0] - box1[2] / 2, box1[0] + box1[2] / 2
        b1_y1, b1_y2 = box1[1] - box1[3] / 2, box1[1] + box1[3] / 2
        b2_x1, b2_x2 = box2[0] - box2[2] / 2, box2[0] + box2[2] / 2
        b2_y1, b2_y2 = box2[1] - box2[3] / 2, box2[1] + box2[3] / 2

    # Intersection area
    inter = (torch.min(b1_x2, b2_x2) - torch.max(b1_x1, b2_x1)).clamp(0) * \
            (torch.min(b1_y2, b2_y2) - torch.max(b1_y1, b2_y1)).clamp(0)

    # Union Area
    w1, h1 = b1_x2 - b1_x1, b1_y2 - b1_y1 + eps
    w2, h2 = b2_x2 - b2_x1, b2_y2 - b2_y1 + eps
    union = w1 * h1 + w2 * h2 - inter + eps

    iou = inter / union
    if GIoU or DIoU or CIoU:
        cw = torch.max(b1_x2, b2_x2) - torch.min(b1_x1, b2_x1)  # convex (smallest enclosing box) width
        ch = torch.max(b1_y2, b2_y2) - torch.min(b1_y1, b2_y1)  # convex height
        if CIoU or DIoU:  # Distance or Complete IoU https://arxiv.org/abs/1911.08287v1
            c2 = cw ** 2 + ch ** 2 + eps  # convex diagonal squared
            rho2 = ((b2_x1 + b2_x2 - b1_x1 - b1_x2) ** 2 +
                    (b2_y1 + b2_y2 - b1_y1 - b1_y2) ** 2) / 4  # center distance squared
            if DIoU:
                return iou - rho2 / c2  # DIoU
            elif CIoU:  # https://github.com/Zzh-tju/DIoU-SSD-pytorch/blob/master/utils/box/box_utils.py#L47
                v = (4 / math.pi ** 2) * torch.pow(torch.atan(w2 / h2) - torch.atan(w1 / h1), 2)
                with torch.no_grad():
                    alpha = v / (v - iou + (1 + eps))
                return iou - (rho2 / c2 + v * alpha)  # CIoU
        else:  # GIoU https://arxiv.org/pdf/1902.09630.pdf
            c_area = cw * ch + eps  # convex area
            return iou - (c_area - union) / c_area  # GIoU
    else:
        return iou  # IoU
class DetectorLoss(Module):
    def __init__(
            self,
            cfg: CfgNode
    ):
        # device会在初始化的时候确定
        super().__init__()
        self.device = torch.device(int(cfg.GPUS))
        bceClass=BCEWithLogitsLoss(
            pos_weight=torch.tensor(1.0,device=self.device),

        )
        bceObj=BCEWithLogitsLoss(
            pos_weight=torch.tensor(1.0,device=self.device)
        )
        self.bceClass=bceClass
        self.bceObj=bceObj
        self.nc=1
        self.anchors = cfg.MODEL.DETECTOR.ANCHORS
        self.na=len(self.anchors[0])//2
        self.nl=len(self.anchors)
        self.anchors=[torch.tensor(self.anchors[idx],dtype=torch.float32,device=self.device)/i for idx,i in enumerate([8,16,32,64])]
        # anchor需要重新弄成nl*na*2
        self.anchors=torch.cat(self.anchors,dim=0).reshape((self.nl,self.na,-1))
        self.balance=[4.0,1.0,0.25,0.06,0.02]
        self.box_weight=0.05
        self.cls_weight=0.5
        self.obj_weight=1.0
    def forward(
            self,
            x,
            targets
    ):
        l_cls,l_box,l_obj=torch.zeros(1,device=self.device),torch.zeros(1,device=self.device),torch.zeros(1,device=self.device)
        # 首先需要把targets转化为求loss的格式
        t_cls,t_box,indices,anchors=self.build_targets(x,targets)

        # 计算三种loss
        for i,pi in enumerate(x):
            idx_in_batch,anchor_idx,gj,gi=indices[i]
            t_obj=torch.zeros_like(pi[...,0],device=self.device)
            n_target=len(idx_in_batch)

            if n_target>0:
                p_selected=pi[idx_in_batch,anchor_idx,gj,gi]
                # regression
                # 预测的xy就是偏离当前坐标的offset
                pxy=p_selected[:,:2].sigmoid()*2-0.5
                # 预测的wh就是锚框的长宽系数
                pwh=(p_selected[:,2:4].sigmoid()*2)**2*anchors[i]
                pbox=torch.cat((pxy,pwh),dim=1)
                iou=bbox_iou(pbox.T,t_box[i],x1y1x2y2=False,CIoU=True)
                l_box+=(1.0-iou).mean()
                t_obj[idx_in_batch,anchor_idx,gj,gi]=iou.detach().clamp(0).type(torch.float32)
            l_obj+=self.bceObj(pi[...,4],t_obj)*self.balance[i]
        batch_size=x[0].shape[0]
        loss=l_box*self.box_weight+l_obj*self.obj_weight+l_cls*self.cls_weight
        return loss*batch_size,torch.cat((l_box,l_obj,l_cls)).detach()


    def build_targets(
            self,
            pred,
            targets
    ):
        """
        基本思路：各尺度锚框、四个方向的锚框都要考虑
        :param pred:
        :param targets:
        :return:
        """
        num_a,num_t=self.na,targets.shape[0]
        t_cls,t_obj,indices,anchors=[],[],[],[]
        # 控制尺度的array
        gain=torch.ones(7,device=self.device)
        anchor_idx=torch.arange(
            num_a,
            device=self.device,
            dtype=torch.float32
        ).view(num_a,-1).repeat(1,num_t)
        # 把锚框的编号加到最后一维去
        targets=torch.cat(
            (
                targets.repeat(num_a,1,1),
                anchor_idx[:,:,None]
            )
            ,dim=2
        )
        g=0.5

        off=torch.tensor(
            [
                [0,0],
                [1,0],
                [0,1],
                [-1,0],
                [0,-1]
            ],
            device=self.device
        )*g

        for i in range(self.nl):
            anchor=self.anchors[i]
            gain[2:6]=torch.tensor(pred[i].shape)[[3,2,3,2]]
            # 转移到相应的尺度
            t=targets*gain
            if num_t>0:
                # 除去与当前尺度锚框过于离谱的一些obj
                ratio_anchor=t[:,:,4:6]/anchor[:,None]
                ratio_filter=torch.max(ratio_anchor,1/ratio_anchor).max(2)[0]<4.0
                t=t[ratio_filter]
                # 从左到右，从上到下
                gxy=t[:,2:4]
                # 从右到左，从下到上
                gxi=gain[[2,3]]-gxy
                # 每一个点都需要把周围的点的anchor也考虑一下
                j,k=((gxy%1.<g)&(gxy>1.)).T
                l,m=((gxi%1.<g)&(gxi>1.)).T

                j=torch.stack(
                    (
                        torch.ones_like(j),
                        j,
                        k,
                        l,
                        m
                    )
                )
                t=t.repeat((5,1,1))[j]
                # 周围的点通过offsets来计算
                # 5*num*2,第一种offsets的第i个目标的实际坐标offset
                offsets=(torch.zeros_like(gxy)[None]+off[:,None])[j]
            else:
                t=targets[0]
                offsets=0
            idx_in_batch,obj_class=t[:,:2].long().T
            gxy=t[:,2:4]
            gwh=t[:,4:6]
            gij=(gxy-offsets).long()
            gi,gj=gij.T
            anchor_ids=t[:,-1].long()

            # 加入结果
            indices.append(
                (
                    idx_in_batch,
                    anchor_ids,
                    gj.clamp_(0,gain[3]-1),
                    gi.clamp_(0,gain[2]-1)
                )
            )

            t_obj.append(
                torch.cat((gxy-gij,gwh),dim=1)
            )
            anchors.append(anchor[anchor_ids])
            t_cls.append(obj_class)

        return t_cls,t_obj,indices,anchors



