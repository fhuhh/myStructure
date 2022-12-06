import torch

from project.utils import ConfigTool,LoggerTool,ParserTool
from project.models import DualTransMPPE
from project.dataset import CocoDataset
from pathlib import Path
from torch.optim import Adam,lr_scheduler
from project.loss import DetectorLoss
from tqdm import tqdm
def train_detector():
    torch.set_printoptions(precision=10)
    args=ParserTool().get_args()
    cfg=ConfigTool()\
        .merge_from_file(Path(args.cfg))\
        .merge_from_list(args.opt)\
        .get_cfg()
    device = torch.device(int(cfg.GPUS))
    logger=LoggerTool().setup_logger(Path(args.output),"train")

    dataloader=CocoDataset("train",cfg=cfg).create_dataloader()
    n_batch=len(dataloader)
    model=DualTransMPPE(cfg=cfg,logger=logger).to(device=device)
    lr=cfg.TRAIN.LR
    lr_factor=cfg.TRAIN.LR_FACTOR
    lr_step=cfg.TRAIN.LR_STEP
    momentum=cfg.TRAIN.MOMENTUM
    epochs=cfg.TRAIN.END_EPOCH
    optimizer=Adam(
        model.parameters(),
        lr=lr,
        betas=(momentum,0.999)
    )
    lr_lambda=lambda x:(1-x/(epochs-1))*(1-lr_factor)+lr_factor
    scheduler=lr_scheduler.LambdaLR(
        optimizer=optimizer,
        lr_lambda=lr_lambda
    )
    losser=DetectorLoss(
        cfg=cfg
    )
    for epoch in range(epochs):
        model.train()
        pbar=enumerate(dataloader)
        pbar=tqdm(pbar,total=n_batch)
        logger.log_sth("{:<10}{:<10}{:<10}{:<10}{:<10}".format("step","box","obj","cls","total"))
        optimizer.zero_grad()
        for i,(img,heatmap,weight_mask,center_heatmap,center_weight_mask,offset_map,weight_map,joints,targets) in pbar:
            ni=i+n_batch*epoch
            img=img.to(device,non_blocking=True)
            pred=model(img)
            targets=targets[:,0:6]
            loss,loss_items=losser(pred,targets.to(device))
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            logger.log_sth("{:<10.3f}{:<10.3f}{:<10.3f}{:<10.3f}{:<10.3f}".format(ni,loss_items[0].item(),loss_items[1].item(),loss_items[2].item(),loss.item()))
        scheduler.step(epoch=epoch)
        if (epoch+1)%50==0:
            logger.save_checkpoint(model,"model_{}.pth".format(epoch))




