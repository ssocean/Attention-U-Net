import os
import numpy as np
import cv2


def _generate_rst_img(img_path, rst_dir, net, device, patch_size=256, mag_scale=1.0, is_bin=False):
    '''
        生成二值图
        :param img_path: 图片路径
        :param rst_dir: 结果图存放路径
        :param net: 网络模型实例
        :param patch_size: 切片大小，默认256，如果为0时不进行切分
        :param mag_scale: 单切片放缩比例，默认1.0
        :param device: 使用是被，默认为cpu
        :param is_bin: 是否输出二值图，默认否
        :return:
        '''

    fname = os.path.basename(img_path)
    fname = '.'.join(fname.split('.')[:-1])

    net = net
    net.to(device)

    img = cv2.imread(img_path)
    (h, w) = img.shape[:2]  # 保存图像的形状(h,w,c)


    n_w = (int(w / patch_size) + 1) * patch_size
    n_h = (int(h / patch_size) + 1) * patch_size
    count_w = int(w / patch_size) + 1
    count_h = int(h / patch_size) + 1
    img = cv2.resize(img, (n_w, n_h))
    pred_np = np.zeros((n_h, n_w), dtype=np.uint8)
    for i in range(0, count_w):
        for j in range(0, count_h):
            roi = img[j * patch_size:(j + 1) * patch_size,
                  i * patch_size:(i + 1) * patch_size]  # roi为每一个小切片，region of interest
            roi = cv2.resize(roi, (int(mag_scale * patch_size), int(mag_scale * patch_size)))  # 对roi进行放缩

            rst = cv2.threshold()
            # print(rst)
            rst = cv2.resize(rst, (patch_size, patch_size))  # 将roi缩放回原尺寸

            pred_np[j * patch_size:(j + 1) * patch_size,
            i * patch_size:(i + 1) * patch_size] = rst  # roi加入预先定义存放预测结果的numpy数组

    pred_np = cv2.resize(pred_np, (w, h))  # 预先定义存放预测结果的numpy数组放缩至图像原尺寸
    if is_bin:
        # pred_np = cv2.cvtColor(pred_np, cv2.COLOR_BGR2GRAY)
        _, pred_np = cv2.threshold(pred_np, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    if not os.path.exists(rst_dir):  # 如果目标路径不存在，创建该路径
        os.makedirs(rst_dir)
    file_name = f'{fname}_{patch_size}_{mag_scale}_{net.name}.png'
    retval_pth = os.path.join(rst_dir, file_name)

    retval = cv2.imwrite(retval_pth, pred_np)
    assert retval, r"预测结果保存失败"
