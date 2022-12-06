import torch
from . import BaseDataset
from yacs.config import CfgNode
from pycocotools.coco import COCO
import os
import cv2
import numpy as np
from project.utils import HeatmapGenerator,OffsetGenerator
from torch.utils.data import DataLoader
def collate_fn(batch):
    img, heatmap, weight_mask, center_heatmap, center_weight_mask, offset_map, weight_map, bbox, joints=zip(*batch)
    # 需要组合一个40维的向量
    img=torch.stack(img,dim=0)
    heatmap=torch.stack(heatmap,dim=0)
    weight_mask=torch.stack(weight_mask,dim=0)
    center_heatmap=torch.stack(center_heatmap,dim=0)
    center_weight_mask=torch.stack(center_weight_mask,dim=0)
    offset_map=torch.stack(offset_map,dim=0)
    weight_map=torch.stack(weight_map,dim=0)
    joints=torch.cat(joints,dim=0)
    joint_coords=joints[:,:,0:2].contiguous().view(len(joints),-1)
    bbox_list=[]
    for i,boxes in enumerate(bbox):
        bbox_copy = torch.zeros((len(boxes), 6))
        bbox_copy[:,0]=i
        bbox_copy[:,1]=0
        bbox_copy[:,2:]=boxes[:,0:4]
        bbox_list.append(bbox_copy)
    bbox_copy=torch.cat(bbox_list,dim=0)
    return img,heatmap,weight_mask,center_heatmap,center_weight_mask,offset_map,weight_map,joints,torch.cat((bbox_copy,joint_coords),dim=1)


class CocoDataset(BaseDataset):
    def __init__(
            self,
            mode:str,
            cfg:CfgNode
    ):
        super(CocoDataset,self).__init__(mode,cfg)
        self.split=cfg.DATASET.TRAIN if self.train else cfg.DATASET.TEST
        self.anno_file_path=os.path.join(
            self.root,
            'annotations_trainval2017',
            "person_keypoints_{}.json".format(self.split)
        )
        self.imgs_path=os.path.join(
            self.root,
            'images',
            self.split
        )
        self.coco=COCO(self.anno_file_path)
        # image的ids
        self.ids=list(self.coco.imgs.keys())
        if mode.__eq__("train"):
            #需要对训练的样本进行一轮筛选
            self.ids=self.filter_img()
            self.heatmap_generator=HeatmapGenerator(
                output_size=self.output_size,
                num_joints=self.num_joints
            )
            self.det_kpt_heatmap_generator=HeatmapGenerator(
                output_size=self.output_size,
                num_joints=len(self.det_kpt)
            )
            self.offset_generator=OffsetGenerator(
                self.output_size,
                self.output_size,
                self.num_joints,
                len(self.det_kpt),
                self.offset_radius
            )


    def __len__(self):
        return len(self.ids)

    def __repr__(self):
        return "dataset : {}\ndataset size : {}\nroot path : {}".format(
            self.__class__.__name__,
            self.__len__(),
            self.root
        )

    def filter_img(self):
        #去除关键点太少的anno
        res_ids=[]
        for img_id in self.ids:
            anno_ids=self.coco.getAnnIds(img_id,iscrowd=False)
            anno_infos=self.coco.loadAnns(anno_ids)
            num_points=0
            for anno_info in anno_infos:
                num_points+=anno_info["num_keypoints"]
            if num_points>10:
                res_ids.append(img_id)
        return res_ids

    def __getitem__(self, item):
        img_id=self.ids[item]
        file_name=self.coco.loadImgs(img_id)[0]['file_name']
        img=cv2.imread(
            os.path.join(self.imgs_path,file_name),
            cv2.IMREAD_COLOR|cv2.IMREAD_IGNORE_ORIENTATION
        )
        img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        # 获取anno
        anno_ids=self.coco.getAnnIds(imgIds=img_id)
        annos=self.coco.loadAnns(ids=anno_ids)

        if self.train:
            return self.process_train(img, annos)
        else:
            return self.process_test(img,annos,img_id)

    def get_det_points(
            self,
            annos:list,
            joints:np.ndarray,
            det_joints:np.ndarray
    ):
        if len(self.det_kpt)>0:
            for pose_idx,anno in enumerate(annos):
                for det_idx,det_points in enumerate(self.det_kpt_ids):
                    selected_joints=joints[pose_idx,det_points,:2]
                    # 得到标出来的点
                    vis=(joints[pose_idx,det_points,2:3]>0).astype(np.float32)
                    # 得到当前块的标出点的横纵坐标和
                    joints_coordinate_sum=np.sum(selected_joints*vis,axis=0)
                    # 得到标出点的个数
                    vis_sum=np.sum(vis,axis=0)[0]
                    if vis_sum<=0:
                        # 当前的part没有被标出的点，那part的中心点自然不存在
                        det_joints[pose_idx,det_idx,2]=0
                    else:
                        det_joints[pose_idx,det_idx,:2]=joints_coordinate_sum/vis_sum
                        # 标记part的中心点为可见
                        det_joints[pose_idx,det_idx,2]=2
        return det_joints

    def process_train(
            self,
            img:np.ndarray,
            annos:list
    ):
        """
        这里需要学习yolo把ground搞成target
        :param img:
        :param annos:
        :return:
        """
        #把annos中的点为0的全部抛弃
        annos=[anno for anno in annos if anno['num_keypoints']>0]

        pose_num=len(annos)
        area=np.zeros((pose_num,1))
        bbox=np.zeros((pose_num,4))
        joints=np.zeros(
            (
                pose_num,
                self.num_joints,
                3
            )
        )
        det_joints=np.zeros(
            (
                pose_num,
                len(self.det_kpt),
                3
            )
        )

        #填好joints和area
        for pose_idx,anno in enumerate(annos):
            joints[pose_idx,:,:3]=np.array(anno['keypoints']).reshape((-1,3))
            area[pose_idx,0]=anno['bbox'][2]*anno['bbox'][3]
            bbox[pose_idx,0:4]=np.array(anno['bbox'])
            # 框需要改到中心点
            bbox[pose_idx,:2]+=bbox[pose_idx,2:]/2
        img,joints,area,bbox=self.transforms(
            img,
            joints,
            area,
            bbox
        )
        # 填好det_joints
        det_joints=self.get_det_points(
            annos=annos,
            joints=joints.numpy(),
            det_joints=det_joints
        )
        det_joints=torch.from_numpy(det_joints)
        # 生成heatmap,和背景weight_mask
        # 需要考虑输出图和输入图的大小不一样
        ratio=self.output_size/self.input_size
        out_joints=torch.zeros_like(joints)
        out_det_joints=torch.zeros_like(det_joints)
        out_joints[:,:,:2]=joints[:,:,:2]*ratio
        out_joints[:,:,2]=joints[:,:,2]
        out_det_joints[:,:,:2]=det_joints[:,:,:2]*ratio
        out_det_joints[:,:,2]=det_joints[:,:,2]
        heatmap,weight_mask=self.heatmap_generator(
            joints=out_joints.numpy(),
            sigma=self.sigma,
            bg_weight=self.bg_weight
        )
        # 生成part_center的heatmap和weight_mask
        center_heatmap,center_weight_mask=self.det_kpt_heatmap_generator(
            joints=out_det_joints.numpy(),
            sigma=self.det_kpt_sigma,
            bg_weight=self.bg_weight
        )
        # 生成offset_map
        offset_map,weight_map=self.offset_generator(
            joints=out_joints.numpy(),
            det_joints=out_det_joints.numpy(),
            area=area
        )

        joints[:,:,0:2]=joints[:,:,0:2]/self.input_size
        bbox/=self.input_size

        return img,torch.from_numpy(heatmap),\
               torch.from_numpy(weight_mask),torch.from_numpy(center_heatmap),\
               torch.from_numpy(center_weight_mask),torch.from_numpy(offset_map),\
               torch.from_numpy(weight_map),bbox,joints

    def process_test(
            self,
            img: np.ndarray,
            annos: list,
            img_id
    ):
        # 首先建立joints和area
        annos = [anno for anno in annos if anno['num_keypoints'] > 0]
        pose_num=len(annos)
        joints=np.zeros(
            (
                pose_num,
                self.num_joints,
                3
            )
        )
        area=np.zeros(
            (
                pose_num,
                1
            )
        )
        for pose_idx,anno in enumerate(annos):
            joints[pose_idx,:,:3]=np.array(anno["keypoints"]).reshape((-1,3))
            area[pose_idx,0]=anno["bbox"][2]*anno["bbox"][3]
        img,joints,area=self.transforms(
            img,
            joints,
            area
        )
        # 建立part_center的points
        det_joints=np.zeros(
            (
                pose_num,
                len(self.det_kpt),
                3
            )
        )
        det_joints=self.get_det_points(
            annos=annos,
            joints=joints,
            det_joints=det_joints
        )
        # 生成output_size的human_mask
        human_mask=np.zeros(
            (
                self.output_size,
                self.output_size
            )
        )
        ratio=self.output_size/self.input_size
        for _,anno in enumerate(annos):
            box=anno['bbox']
            top_left_x=int(box[0]*ratio-0.5)
            top_left_y=int(box[1]*ratio-0.5)
            bottom_right_x=int((box[0]+box[2])*ratio+0.5)
            bottom_right_y=int((box[1]+box[3])*ratio+0.5)
            human_mask[top_left_y:bottom_right_y,top_left_x:bottom_right_x]=1

        return img,img_id,joints,det_joints,human_mask,area

    def create_dataloader(self):
        return DataLoader(
            dataset=self,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            pin_memory=self.pin_memory,
            collate_fn=collate_fn
        )