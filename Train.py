import time
from torch.utils.data import DataLoader
from torch.utils.data.dataset import random_split
import Models
import BinarizationDataset
import os
import sys
import torch.nn as nn
from torch import optim
import logging
from tqdm import tqdm
from eval import *
import torch

from fcn import VGGNet, FCNs


def tensor_to_ndarray(t):
    cpu_tensor = t.cpu()
    res = cpu_tensor.detach().numpy()  # 转回numpy
    # print(res.shape)
    res = np.squeeze(res, 1)
    # res = np.swapaxes(res, 0, 2)
    # res = np.swapaxes(res, 0, 1)
    return res


def init_logger():
    '''
    初始化日志类
    :return: 日志类实例对象
    '''
    # 日志模块
    logger = logging.getLogger(__name__)
    logger.setLevel(level=logging.INFO)

    handler = logging.FileHandler(fr'logs/{time.strftime("%Y_%b_%d", time.localtime())}_log.txt')

    handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    # 输出到控制台
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    # 输出到日志
    logger.addHandler(handler)
    logger.addHandler(console)
    '''
    logger.info("Start print log") #一般信息
    logger.debug("Do something") #调试显示
    logger.warning("Something maybe fail.")#警告
    logger.info("Finish")
    '''
    return logger


logger = init_logger()
#图片存放路径
imgs_dir = r'D:\DIBCO_DATASET\data\dibco/'
#标注图像存放路径
masks_dir = r'D:\DIBCO_DATASET\data\dibco_thinner_gt/'
# imgs_dir='data/imgs-backup/'
# masks_dir = r'data/masks-backup/'
# 网络模型保存路径
dir_checkpoint = r'weights/'
auto_checkpoint = r'weights_backup/'


def train(net,
          device,
          epochs=1,
          batch_size=1,
          lr=0.001,
          val_percent=0.1,
          trained_epoch=0,
          save_cp=True):

    dataset = BinarizationDataset.BinDataset(imgs_dir, masks_dir)
    n_val = int(len(dataset) * val_percent)
    n_train = len(dataset) - n_val
    train, val = random_split(dataset, [n_train, n_val])
    train_loader = DataLoader(train, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True)
    val_loader = DataLoader(val, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True, drop_last=True)
    # print('数据集加载完毕')
    global_step = 0
    optimizer = optim.RMSprop(net.parameters(), lr=lr, weight_decay=1e-8, momentum=0.9)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=2)  # 'min' if net.n_classes > 1 else
    # if net.n_classes > 1:
    #     criterion = nn.CrossEntropyLoss()
    # else:
    criterion = nn.BCEWithLogitsLoss()
    logger.info('开始训练，即将读取epoch')
    p = trained_epoch + 1
    min_loss = 1.0
    start_time = time.localtime()
    try:
        for epoch in tqdm(range(epochs)):
            # logger.info('开始训练，即将读取epoch')
            # print('/r')
            logger.info(f'--------------------------第{p}轮训练开始--------------------------')

            net.train()
            epoch_loss = 0
            # print(f"第{epochs}轮开始训练")
            e_times = 1
            for batch in tqdm(train_loader):
                # logger.info(f'第{p}轮开始训练第{e_times}个batch')
                e_times = e_times + 1
                imgs = batch['image']
                true_masks = batch['mask']

                imgs = imgs.to(device=device, dtype=torch.float32)
                mask_type = torch.float32  # if net.n_classes == 1 else torch.long
                true_masks = true_masks.to(device=device, dtype=mask_type)
                imgs = imgs.transpose(3, 1).transpose(2, 3)
                true_masks = true_masks.unsqueeze(3)
                true_masks = true_masks.transpose(3, 1).transpose(2, 3)
                # print(imgs.size())
                masks_pred = net(imgs)
                # print(masks_pred.shape)
                loss = criterion(masks_pred, true_masks)
                # iou(masks_pred,true_masks)
                epoch_loss += loss.item()

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_value_(net.parameters(), 0.1)  # 梯度裁剪 防止梯度爆炸
                optimizer.step()
                global_step += 1
                loss_float = float(loss)
                logger.info(f'{p}轮第{e_times}个loss:' + str(loss_float))


            val_score = eval_net(net, val_loader, device)
            scheduler.step(val_score)
            if p % 200 == 0:  # 每五个EPOCH保存一次
                try:
                    os.mkdir(dir_checkpoint)
                except OSError:
                    pass
                torch.save(net.state_dict(),
                           auto_checkpoint + f'{time.strftime("%Y_%b_%d_%H_%M", time.localtime())}_{net.name}_AUTO{p}.pth')
                logger.info(
                    f'{device}下的{net.name}网络第{p}轮结果被保存为：' + f'{time.strftime("%Y_%b_%d_%H_%M", time.localtime())}_AUTO{p}.pth')
            # 保存最优
            if loss_float < min_loss:
                try:
                    os.mkdir(dir_checkpoint)
                except OSError:
                    pass
                torch.save(net.state_dict(),
                           auto_checkpoint + f'{time.strftime("%Y_%b_%d_%H", start_time)}_{net.name}_BestResult.pth')
                logger.info(
                    f'{device}下的{net.name}网络最优结果被保存为：' + f'{time.strftime("%Y_%b_%d_%H", start_time)}_BestResult.pth')
            p = p + 1
    except KeyboardInterrupt:
        torch.save(net.state_dict(), 'INTERRUPTED.pth')
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)

    if save_cp:

        try:
            os.mkdir(dir_checkpoint)
        except OSError:
            pass
        torch.save(net.state_dict(),
                   dir_checkpoint + f'{time.strftime("%Y_%b_%d_%H_%M", time.localtime())}_epoch{p}.pth')


if __name__ == '__main__':
    #指定设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # device = torch.device('cpu')
    # print(device)
    #加载网络
    net = Models.UNet()
    # net = UNET.MNet_3(1, n_channels=3)
    # net = Models.AttU_Net(3, 1)
    # net = Models.R2AttU_Net()
    # print(net.)
    # vgg_model = VGGNet(requires_grad=True, remove_fc=True)
    # net = FCNs(pretrained_net=vgg_model, n_class=1)
    logger.info(f'使用{net.name}网络结构，设备为：' + str(device))

    # net.load_state_dict(torch.load(r'E:\Unet_Binarization\weights\2020_Aug_20_13_05_AUTO100.pth'))

    net.to(device=device)
    # faster convolutions, but more memory
    # cudnn.benchmark = True

    try:
        train(net=net,
              epochs=15000,
              batch_size=8,
              lr=0.001,
              device=device,
              val_percent=0.1,
              trained_epoch=0
              )
    except KeyboardInterrupt:
        torch.save(net.state_dict(), 'INTERRUPTED.pth')
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)
