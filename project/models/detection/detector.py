from torch.nn import Module,ModuleList,Conv2d
from typing import Optional
from project.utils.logger import LoggerTool
import torch

def non_max_suppression(
        prediction,
        conf_thres=0.25,
        iou_thres=0.45,
        classes=None,
        agnostic=False,
        multi_label=False,
        labels=(),
        max_det=300,
):
    """Non-Maximum Suppression (NMS) on inference results to reject overlapping detections

    Returns:
         list of detections, on (n,6) tensor per image [xyxy, conf, cls]
    """

    if isinstance(prediction, (list, tuple)):  # YOLOv5 model in validation model, output = (inference_out, loss_out)
        prediction = prediction[0]  # select only inference output

    device = prediction.device
    mps = 'mps' in device.type  # Apple MPS
    if mps:  # MPS not fully supported yet, convert tensors to CPU before NMS
        prediction = prediction.cpu()
    bs = prediction.shape[0]  # batch size
    nc = prediction.shape[2] - 5  # number of classes
    xc = prediction[..., 4] > conf_thres  # candidates

    # Checks
    assert 0 <= conf_thres <= 1, f'Invalid Confidence threshold {conf_thres}, valid values are between 0.0 and 1.0'
    assert 0 <= iou_thres <= 1, f'Invalid IoU {iou_thres}, valid values are between 0.0 and 1.0'

    # Settings
    # min_wh = 2  # (pixels) minimum box width and height
    max_wh = 7680  # (pixels) maximum box width and height
    max_nms = 30000  # maximum number of boxes into torchvision.ops.nms()
    time_limit = 0.5 + 0.05 * bs  # seconds to quit after
    redundant = True  # require redundant detections
    multi_label &= nc > 1  # multiple labels per box (adds 0.5ms/img)
    merge = False  # use merge-NMS

    t = time.time()
    output = [torch.zeros((0, 6), device=prediction.device)] * bs
    for xi, x in enumerate(prediction):  # image index, image inference
        # Apply constraints
        # x[((x[..., 2:4] < min_wh) | (x[..., 2:4] > max_wh)).any(1), 4] = 0  # width-height
        x = x[xc[xi]]  # confidence

        # Cat apriori labels if autolabelling
        if labels and len(labels[xi]):
            lb = labels[xi]
            v = torch.zeros((len(lb), nc + nm + 5), device=x.device)
            v[:, :4] = lb[:, 1:5]  # box
            v[:, 4] = 1.0  # conf
            v[range(len(lb)), lb[:, 0].long() + 5] = 1.0  # cls
            x = torch.cat((x, v), 0)

        # If none remain process next image
        if not x.shape[0]:
            continue

        # Compute conf
        x[:, 5:] *= x[:, 4:5]  # conf = obj_conf * cls_conf

        # Box/Mask
        box = xywh2xyxy(x[:, :4])  # center_x, center_y, width, height) to (x1, y1, x2, y2)

        # Detections matrix nx6 (xyxy, conf, cls)
        if multi_label:
            i, j = (x[:, 5:mi] > conf_thres).nonzero(as_tuple=False).T
            x = torch.cat((box[i], x[i, 5 + j, None], j[:, None].float(), mask[i]), 1)
        else:  # best class only
            conf, j = x[:, 5:mi].max(1, keepdim=True)
            x = torch.cat((box, conf, j.float()), 1)[conf.view(-1) > conf_thres]

        # Filter by class
        if classes is not None:
            x = x[(x[:, 5:6] == torch.tensor(classes, device=x.device)).any(1)]

        # Apply finite constraint
        # if not torch.isfinite(x).all():
        #     x = x[torch.isfinite(x).all(1)]

        # Check shape
        n = x.shape[0]  # number of boxes
        if not n:  # no boxes
            continue
        elif n > max_nms:  # excess boxes
            x = x[x[:, 4].argsort(descending=True)[:max_nms]]  # sort by confidence
        else:
            x = x[x[:, 4].argsort(descending=True)]  # sort by confidence

        # Batched NMS
        c = x[:, 5:6] * (0 if agnostic else max_wh)  # classes
        boxes, scores = x[:, :4] + c, x[:, 4]  # boxes (offset by class), scores
        # 利用torchvision的nms来选出好的框框
        i = torchvision.ops.nms(boxes, scores, iou_thres)  # NMS
        if i.shape[0] > max_det:  # limit detections
            i = i[:max_det]
        if merge and (1 < n < 3E3):  # Merge NMS (boxes merged using weighted mean)
            # update boxes as boxes(i,4) = weights(i,n) * boxes(n,4)
            iou = box_iou(boxes[i], boxes) > iou_thres  # iou matrix
            weights = iou * scores[None]  # box weights
            x[i, :4] = torch.mm(weights, x[:, :4]).float() / weights.sum(1, keepdim=True)  # merged boxes
            if redundant:
                i = i[iou.sum(1) > 1]  # require redundancy

        output[xi] = x[i]
        if mps:
            output[xi] = output[xi].to(device)
        if (time.time() - t) > time_limit:
            LOGGER.warning(f'WARNING ⚠️ NMS time limit {time_limit:.3f}s exceeded')
            break  # time limit exceeded

    return output

def make_grid(
        x_num:int=20,
        y_num:int=20
):
    yv,xv=torch.meshgrid(
            [
                torch.arange(y_num),
                torch.arange(x_num)
            ]
        )
    return torch.stack(
        [
            xv,
            yv
        ],
        dim=2
    ).view(1,1,ny,nx,2).float()
class Detector(Module):
    def __init__(
            self,
            num_class:int=1,
            anchors:Optional[list]=None,
            in_channels:Optional[list]=None,
            inplace:bool=True,
            logger:LoggerTool=None
    ):
        super().__init__()

        self.nc=num_class
        # x,y,w,h,box_conf,class_conf
        self.n_box_out=self.nc+5
        self.n_layer=len(anchors)
        self.n_anchor=len(anchors[0])//2
        # grid---
        self.grid=[torch.zeros(1)]*self.n_layer
        self.stride=[8,16,32,64]
        anchors_tensor=torch.cat(anchors,dim=0).view(self.n_layer,-1,2)
        self.register_buffer(
            "anchors",
            anchors_tensor
        )
        # 注册anchor grid
        self.register_buffer(
            "anchor_grid",
            anchors_tensor.clone().view(self.n_layer,1,-1,1,1,2)
        )
        # 卷积层
        self.conv_layers=ModuleList(
            [Conv2d(ch,self.n_box_out*self.n_anchor,1) for ch in in_channels]
        )

        self.inplace=inplace
        if logger:
            logger.log_sth("detector is created!")
            logger.log_sth("{:<20}{:<20}".format("in channels","anchors"))
            for i,(ch,anchor) in enumerate(zip(in_channels,anchors_tensor)):
                anchor_str=""
                for a in anchor:
                    anchor_str+="({},{})".format(a[0],a[1])
                logger.log_sth("{:<20}{:<20}".format(ch,anchor_str))

    def forward(
            self,
            x
    ):
        # 需要知道框怎么找到的
        for i in range(self.n_layer):
            x[i]=self.conv_layers[i](x[i])
            B,_,H,W=x[i].size()
            x[i]=x[i].view(B,self.n_anchor,self.n_box_out,H,W)\
                .permute(0,1,3,4,2)\
                .contiguous()
        return x

    def inference(
            self,
            x
    ):
        # 假设输入的还是4个尺度的特征图
        # 首先进入forward得到结果
        x_det=self(x)
        # 输出的最后一维分别是(x,y,w,h,box_conf,class_conf)
        infer_res = []
        for i in range(self.n_layer):
            B,_,H,W=x[i].shape
            if self.grid[i].shape[2:4]!=x[i].shape[2:4]:
                self.grid[i]=make_grid(W,H)
            # 对结果进行一个sigmoid
            y = x_det[i].sigmoid()
            xy = (y[..., 0:2] * 2. - 0.5 + self.grid[i])*self.stride[i]
            wh = (y[..., 2:4] * 2) ** 2 * self.anchor_grid[i].view(1, self.n_anchor, 1, 1, 2)
            infer_res.append(
                torch.cat(
                    [
                        xy,
                        wh,
                        y[...,4:]
                    ],
                    dim=-1
                ).view(B,-1,self.n_box_out)
            )
        # TODO:得到的结果必须经过后处理来得出框


        return torch.cat(infer_res,dim=1)


