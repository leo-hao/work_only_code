# Obtained from: https://github.com/vikolss/DACS

from PIL import Image
import kornia
import numpy as np
import torch
import torch.nn as nn
import random
import os
import json
from .augmentations import *

IMG_MEAN = np.array((104.00698793, 116.66876762, 122.67891434), dtype=np.float32)
class rand_mixer():
    def __init__(self):
        # if dataset == "gta5":
        #     jpath = 'data/gta5_ids2path.json'
        #     self.resize = (1280, 720)
        #     input_size = (512, 512)
        #     self.data_aug = Image.composite([RandomCrop_gta(input_size)])
        # elif dataset == "cityscapes":
        #     jpath = 'data/cityscapes_ids2path.json'
        # else:
        #     print('rand_mixer {} unsupported'.format(dataset))
        #     return
        jpath = 'data/gta5_ids2path.json'
        self.resize = (1280, 720)
        input_size = (512, 512)
        self.data_aug = Compose([RandomCrop_gta(input_size)])        
        self.root = "/home/data/liuhao/experiments/DAFormer-master/data/gta"
        self.dataset = "gta5"
        self.class_map = {7: 0, 8: 1, 11: 2, 12: 3, 13: 4, 17: 5,
                     19: 6, 20: 7, 21: 8, 22: 9, 23: 10, 24: 11, 25: 12,
                     26: 13, 27: 14, 28: 15, 31: 16, 32: 17, 33: 18}
        with open(jpath, 'r') as load_f:
            self.ids2img_dict = json.load(load_f)

    def mix(self, in_img, in_lbl, classes):
        img_size = in_lbl.shape
        for i in classes:
            if self.dataset == "gta5":
                while(True):
                    name = random.sample(self.ids2img_dict[str(i)], 1)
                    img_path = os.path.join(self.root, "images/%s" % name[0])
                    label_path = os.path.join(self.root, "labels/%s" % name[0])
                    img = Image.open(img_path)
                    lbl = Image.open(label_path)
                    img = img.resize(self.resize, Image.BICUBIC)
                    lbl = lbl.resize(self.resize, Image.NEAREST)
                    img = np.array(img, dtype=np.uint8)
                    lbl = np.array(lbl, dtype=np.uint8)
                    img, lbl = self.data_aug(img, lbl) # random crop to input_size
                    img = np.asarray(img, np.float32)
                    lbl = np.asarray(lbl, np.float32)
                    label_copy = 255 * np.ones(lbl.shape, dtype=np.float32)
                    for k, v in self.class_map.items():
                        label_copy[lbl == k] = v
                    if i in label_copy:
                        lbl = label_copy.copy()
                        img = img[:, :, ::-1].copy()  # change to BGR
                        img -= IMG_MEAN
                        img = img.transpose((2, 0, 1))
                        break
                img = torch.Tensor(img).cuda()
                lbl = torch.Tensor(lbl).cuda()
                class_i = torch.Tensor([i]).type(torch.int64).cuda()
                MixMask = generate_class_mask(lbl, class_i)
                mixdata = torch.cat((img.unsqueeze(0), in_img.unsqueeze(0)))
                mixtarget = torch.cat((lbl.unsqueeze(0), in_lbl.unsqueeze(0)))
                data, target = one_mix(MixMask, data=mixdata, target=mixtarget)
                # 和模板混合，返回图片和标签
                return data, target

# 取消了flip
def strong_transform(param, data=None, target=None):
    assert ((data is not None) or (target is not None))
    data, target = one_mix(mask=param['mix'], data=data, target=target)
    data, target = color_jitter(
        color_jitter=param['color_jitter'],
        s=param['color_jitter_s'],
        p=param['color_jitter_p'],
        mean=param['mean'],
        std=param['std'],
        data=data,
        target=target)
    data, target = gaussian_blur(blur=param['blur'], data=data, target=target)
    return data, target

# 取消了one_mix
def strong_transform_ammend(param, data=None, target=None):
    assert ((data is not None) or (target is not None))
    data, target = color_jitter(
        color_jitter=param['color_jitter'],
        s=param['color_jitter_s'],
        p=param['color_jitter_p'],
        mean=param['mean'],
        std=param['std'],
        data=data,
        target=target)
    data, target = gaussian_blur(blur=param['blur'], data=data, target=target)
    return data, target

# 图片和模板图片混合，便签和模板标签混合
def strong_transform_class_mix(image1, image2, label1, label2, mask_img, mask_lbl, cls_mixer, cls_list, strong_parameters):
    inputs_, _ = one_mix(mask_img, data=torch.cat((image1.unsqueeze(0), image2.unsqueeze(0))))
    _, targets_ = one_mix(mask_lbl, target=torch.cat((label1.unsqueeze(0), label2.unsqueeze(0))))
    inputs, targets = cls_mixer.mix(inputs_.squeeze(0), targets_.squeeze(0), cls_list)
    out_img, out_lbl = strong_transform_ammend(strong_parameters, data=inputs_, target=targets_)
    return out_img, out_lbl

def get_mean_std(img_metas, dev):
    mean = [
        torch.as_tensor(img_metas[i]['img_norm_cfg']['mean'], device=dev)
        for i in range(len(img_metas))
    ]
    mean = torch.stack(mean).view(-1, 3, 1, 1)
    std = [
        torch.as_tensor(img_metas[i]['img_norm_cfg']['std'], device=dev)
        for i in range(len(img_metas))
    ]
    std = torch.stack(std).view(-1, 3, 1, 1)
    return mean, std


def denorm(img, mean, std):
    return img.mul(std).add(mean) / 255.0


def denorm_(img, mean, std):
    img.mul_(std).add_(mean).div_(255.0)


def renorm_(img, mean, std):
    img.mul_(255.0).sub_(mean).div_(std)


def color_jitter(color_jitter, mean, std, data=None, target=None, s=.25, p=.2):
    # s is the strength of colorjitter
    if not (data is None):
        if data.shape[1] == 3:
            if color_jitter > p:
                if isinstance(s, dict):
                    seq = nn.Sequential(kornia.augmentation.ColorJitter(**s))
                else:
                    seq = nn.Sequential(
                        kornia.augmentation.ColorJitter(
                            brightness=s, contrast=s, saturation=s, hue=s))
                denorm_(data, mean, std)
                data = seq(data)
                renorm_(data, mean, std)
    return data, target


def gaussian_blur(blur, data=None, target=None):
    if not (data is None):
        if data.shape[1] == 3:
            if blur > 0.5:
                sigma = np.random.uniform(0.15, 1.15)
                kernel_size_y = int(
                    np.floor(
                        np.ceil(0.1 * data.shape[2]) - 0.5 +
                        np.ceil(0.1 * data.shape[2]) % 2))
                kernel_size_x = int(
                    np.floor(
                        np.ceil(0.1 * data.shape[3]) - 0.5 +
                        np.ceil(0.1 * data.shape[3]) % 2))
                kernel_size = (kernel_size_y, kernel_size_x)
                seq = nn.Sequential(
                    kornia.filters.GaussianBlur2d(
                        kernel_size=kernel_size, sigma=(sigma, sigma)))
                data = seq(data)
    return data, target


def get_class_masks(labels):
    class_masks = []
    class_to_select = [12, 15, 16, 17, 18]
    cls_to_use = random.sample(class_to_select, 2)    
    for label in labels:
    #     classes = torch.unique(labels)
    #     nclasses = classes.shape[0]
    #     class_choice = np.random.choice(
    #         nclasses, int((nclasses + nclasses % 2) / 2), replace=False)
        cls_to_use = random.sample(class_to_select, 2)
        classes = classes[torch.Tensor(class_choice).long()]
        class_masks.append(generate_class_mask(label, classes).unsqueeze(0))
        # for i in  cls_to_use:
            # class_i = torch.Tensor([i]).type(torch.int64).cuda()
        class_masks.append(generate_class_mask(label, class_i).unsqueeze(0))
    return class_masks


def generate_class_mask(label, classes):
    label, classes = torch.broadcast_tensors(label,
                                             classes.unsqueeze(1).unsqueeze(2))
    class_mask = label.eq(classes).sum(0, keepdims=True)
    return class_mask


def one_mix(mask, data=None, target=None):
    if mask is None:
        return data, target
    if not (data is None):
        stackedMask0, _ = torch.broadcast_tensors(mask[0], data[0])
        data = (stackedMask0 * data[0] +
                (1 - stackedMask0) * data[1]).unsqueeze(0)
    if not (target is None):
        stackedMask0, _ = torch.broadcast_tensors(mask[0], target[0])
        target = (stackedMask0 * target[0] +
                  (1 - stackedMask0) * target[1]).unsqueeze(0)
    return data, target
