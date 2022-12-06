import numpy as np

# 计算(x,y)处的高斯函数值
def get_heat_val(sigma,x,y,x0,y0):
    return np.exp(-((x-x0)**2+(y-y0)**2)/(2*sigma**2))

class HeatmapGenerator:
    def __init__(
            self,
            output_size,
            num_joints
    ):
        self.output_size=output_size
        self.num_joints=num_joints

    def __call__(
            self,
            joints:np.ndarray,
            sigma,
            bg_weight
    ):
        assert self.num_joints==joints.shape[1],\
        "num of joints should be {}".format(self.num_joints)
        heatmap=np.zeros(
            (
                self.num_joints,
                self.output_size,
                self.output_size
            )
        )
        # 这个map是干嘛的?---标注背景的位置
        ignored_heatmap=2*np.ones(
            (
                self.num_joints,
                self.output_size,
                self.output_size
            )
        )
        heatmap_list=[heatmap,ignored_heatmap]

        # 在每个joint点上赋值
        for pose in joints:
            for idx,point in enumerate(pose):
                if point[2]>0:
                    x,y=point[0],point[1]
                    # 点不能在图像外边
                    if x<0 or y<0 or x>=self.output_size or y>=self.output_size:
                        continue

                    up_left=(
                        int(np.floor(x-3*sigma-1)),
                        int(np.floor(y-3*sigma-1))
                    )

                    bottom_right=(
                        int(np.ceil(x+3*sigma+2)),
                        int(np.ceil(y+3*sigma+2))
                    )

                    cc,dd=max(0,up_left[0]),min(bottom_right[0],self.output_size)
                    aa,bb=max(0,up_left[1]),min(bottom_right[1],self.output_size)
                    # 创建承载heatmap的array
                    joint_region=np.zeros((bb-aa,dd-cc))

                    for row in range(aa,bb):
                        for col in range(cc,dd):
                            joint_region[row-aa,col-cc]=get_heat_val(
                                sigma=sigma,
                                x=col,
                                y=row,
                                x0=x,
                                y0=y
                            )
                    # 往heatmap去填写
                    # 实际上不同人的同个关节点会重合，所以一定要取最大
                    heatmap_list[0][idx,aa:bb,cc:dd]=np.maximum(
                        heatmap_list[0][idx,aa:bb,cc:dd],
                        joint_region
                    )
                    heatmap_list[1][idx,aa:bb,cc:dd]=1.

        # heatmap的背景地方需要修改为bg_weight
        heatmap_list[1][heatmap_list[1]==2]=bg_weight
        return heatmap_list

class OffsetGenerator:
    def __init__(self, output_h, output_w, num_joints, num_det_joints, radius):
        self.num_joints = num_joints
        self.output_w = output_w
        self.output_h = output_h
        # 这个是分块后的块数
        self.num_det_joints = num_det_joints
        self.radius = radius

    def __call__(
            self,
            joints:np.ndarray,
            det_joints:np.ndarray,
            area
    ):
        assert joints.shape[1]==self.num_joints,\
        "num of joints is {},but {} here!".format(self.num_joints,joints.shape[1])

        offset_map=np.zeros(
            (
                self.num_det_joints*self.num_joints*2,
                self.output_h,
                self.output_w
            )
        )

        weight_map=np.zeros(
            (
                self.num_det_joints*self.num_joints*2,
                self.output_h,
                self.output_w
            )
        )

        area_map=np.zeros(
            (
                self.num_det_joints,
                self.output_h,
                self.output_w
            )
        )
        for pose_id,(points,det_points) in enumerate(zip(joints,det_joints)):
                for det_idx,det_point in enumerate(det_points):
                    x,y,vis=det_point[0],det_point[1],det_point[2]

                    c_x=int(x)
                    c_y=int(y)
                    c_v=int(vis)
                    # c_v为2的时候才是可见点
                    if c_v<2 or c_x<0 or c_y<0 or c_x>=self.output_w or c_y>=self.output_h:
                        continue
                    # 每个点拥有两个map,一个代表offset-x,另一个代表offset-y
                    # 由于每个part中心都要生成一个pose,所以map总量为2*num_joints*part_center_num
                    start_map_idx=det_idx*self.num_joints*2
                    # 为当前的pose的每个点都建立基于det_idx的offset
                    for point_idx,point in enumerate(points):
                        if point[2]>0:
                            x,y=point[0],point[1]
                            # 出界的点不要
                            if x<0 or y<0 or x>=self.output_w or y>=self.output_h:
                                continue
                            # 以part-center为中心进行搜索
                            start_x=max(int(c_x-self.radius),0)
                            start_y=max(int(c_y-self.radius),0)
                            end_x=min(int(c_x+self.radius),self.output_w)
                            end_y=min(int(c_y+self.radius),self.output_h)

                            for pos_col in range(start_x,end_x):
                                for pos_row in range(start_y,end_y):
                                    # 计算偏移量
                                    offset_x=pos_col-x
                                    offset_y=pos_row-y
                                    if (offset_map[start_map_idx+point_idx*2,pos_row,pos_col]!=0 or
                                        offset_map[start_map_idx+point_idx*2+1,pos_row,pos_col]!=0):
                                            if area_map[det_idx,pos_row,pos_col]<area[pose_id]:
                                                # 面积大的优先级低，面积小的优先级高
                                                continue
                                    offset_map[start_map_idx+point_idx*2,pos_row,pos_col]=offset_x
                                    offset_map[start_map_idx+point_idx*2+1,pos_row,pos_col]=offset_y
                                    weight_map[start_map_idx+point_idx*2,pos_row,pos_col]=1./np.sqrt(area[pose_id])
                                    weight_map[start_map_idx+point_idx*2+1,pos_row,pos_col]=1./np.sqrt(area[pose_id])
                                    area_map[det_idx,pos_row,pos_col]=area[pose_id]
        return offset_map,weight_map

