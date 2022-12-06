import logging
from pathlib import Path
import time
from torch.utils.tensorboard import SummaryWriter
from torch import Tensor
import torch
class LoggerTool:
    def __init__(self):
        self.output_dir=None
        self.logger_format=logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        self.logger:logging.Logger=None
        self.tensor_writer:SummaryWriter=None

    def setup_logger(
            self,
            output_dir:Path,
            mode:str
    ):
        # 如果日志文件夹不存在则创建
        if not output_dir.exists():
            output_dir.mkdir(parents=True)
        # 决定文件的名字
        time_str=time.strftime("%Y-%m-%d-%H-%M")
        log_file_name="{}_{}.log".format(time_str,mode)
        # 创建logger
        self.logger=logging.getLogger(__name__)
        self.logger.setLevel(logging.DEBUG)
        file_handler=logging.FileHandler(output_dir / log_file_name)
        file_handler.setFormatter(self.logger_format)
        stream_handler=logging.StreamHandler()
        stream_handler.setFormatter(self.logger_format)
        self.logger.addHandler(file_handler)
        self.logger.addHandler(stream_handler)
        self.tensor_writer=SummaryWriter(log_dir=output_dir)
        self.output_dir=output_dir
        return self
    # 日志记录
    def log_sth(
            self,
            content:str
    ):
        self.logger.log(level=logging.INFO,msg=content)

    # tensor_log记录
    def tensor_log_sth(
            self,
            mode:str,
            tag:str,
            content:Tensor,
            x_val=None,
    ):
        if mode.__eq__("img"):
        #     记录图片
            self.tensor_writer.add_images(
                tag=tag,
                img_tensor=content,
                global_step=x_val,
                dataformats="NCHW"
            )
        else:
            # 记录数值
            self.tensor_writer.add_scalar(
                tag=tag,
                scalar_value=content,
                global_step=x_val
            )

    def save_checkpoint(
            self,
            model,
            file_name:str="model.pth"
    ):
        torch.save(model,self.output_dir/file_name)
        self.log_sth("model saved:{}".format(file_name))