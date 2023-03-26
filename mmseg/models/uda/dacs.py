# The ema model and the domain-mixing are based on:
# https://github.com/vikolss/DACS
# 使用ImageNet特征距离的UDA自我训练的实现
import math
import os
import random
# 用copy模块下的deepcopy函数，防止元素被误改
from copy import deepcopy

import itertools
import torchdacs_transforms
import torch.nn.functional as F
import torch.nn as nn
import mmcv
import numpy as np
from matplotlib import pyplot as plt

#PyTorch Image Models (timm)是一个图像模型（models）、层（layers）、实用程序（utilities）、优化器（optimizers）、调度器（schedulers）、数据加载/增强（data-loaders / augmentations）和参考训练/验证脚本（reference training / validation scripts）的集合，目的是将各种SOTA模型组合在一起，从而能够重现ImageNet的训练结果。

from timm.models.layers import DropPath
from torch.nn.modules.dropout import _DropoutNd

from mmseg.core import add_prefix
from mmseg.models import UDA, build_segmentor
from mmseg.models.uda.uda_decorator import UDADecorator, get_module
from mmseg.models.utils.dacs_transforms import (denorm, generate_class_mask, generate_class_mask_soft, get_class_masks,
                                                get_mean_std, strong_transform, strong_transform_soft, one_mix)

from mmseg.models.utils.visualization import subplotimg
from mmseg.utils.utils import downscale_label_ratio
from mmseg.models.utils.augmentations import(RandomCrop_gta, Compose)
import json
import PIL.Image as Image

IMG_MEAN = np.array((104.00698793, 116.66876762, 122.67891434), dtype=np.float32)
class rand_mixer():
    def __init__(self, dataset):
        if dataset == "gta5":
            jpath = 'data/gta5_ids2path.json'
            self.resize = (1280, 720)
            input_size = (512, 512)
            self.data_aug = Compose([RandomCrop_gta(input_size)])
        elif dataset == "cityscapes":
            jpath = 'data/cityscapes_ids2path.json'
        else:
            print('rand_mixer {} unsupported'.format(dataset))
            return
        
        self.dataset = dataset
        self.class_map = {7: 0, 8: 1, 11: 2, 12: 3, 13: 4, 17: 5,
                     19: 6, 20: 7, 21: 8, 22: 9, 23: 10, 24: 11, 25: 12,
                     26: 13, 27: 14, 28: 15, 31: 16, 32: 17, 33: 18}
        with open(jpath, 'r') as load_f:
            self.ids2img_dict = json.load(load_f)

    def mix(self, in_img, in_lbl, classes):
        img_size = in_lbl.shape
        # in_img.shape: torch.Size([1, 3, 512, 512])
        # in_lbl.shape: torch.Size([1, 1, 512, 512])
        for i in classes:
            if self.dataset == "gta5":
                while(True):
                    name = random.sample(self.ids2img_dict[str(i)], 1)
                    img_path = os.path.join( "data/gta/images/%s" % name[0])
                    label_path = os.path.join("data/gta/labels/%s" % name[0])
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
                #-------测试shape---------
                # #################
                # img.shape: torch.Size([3, 512, 512])
                # lbl.shape: torch.Size([512, 512])
                # in_img.shape: torch.Size([1, 3, 512, 512])
                # in_lbl.shape: torch.Size([1, 1, 512, 512])
                # print('test 2')
                # 只是从两个少量类里面2选一的class，而不是随机一半
                class_i = torch.Tensor([i]).type(torch.int64).cuda()
                # mask 少了一维度所以要加上去,所以重写这个generate_class_mask函数为soft
                MixMask = generate_class_mask(lbl, class_i).unsqueeze(0)
                # MixMask.shape torch.Size([1, 1, 512, 512])
                # 照着将mixdata变成3维度
                if not (in_img is None):
                    mixdata = torch.stack((img, in_img[0])) 
                else:
                    mixdata = None
                # mixtarget 变成2维
                mixtarget = torch.stack((lbl, in_lbl[0][0]))
                # print('#################')
                # print('mixdata.shape:', mixdata.shape)
                # print('mixtarget.shape:', mixtarget.shape)
                # print ('MixMask.shape', MixMask.shape)
                # mixdata.shape: torch.Size([2, 3, 512, 512])
                # mixtarget.shape: torch.Size([2, 1, 512, 512])
                # print('#################')
                data, target = one_mix(MixMask, data=mixdata, target=mixtarget)
                # print('##############')
                # print('test 3')
                # print('data.shape:',data.shape)
                # print('target.shape:',target.shape)
                # test 3
                # data.shape: torch.Size([1, 3, 512, 512])
                # target.shape: torch.Size([1, 1, 512, 512])
                # print('##############')
                return data, target



# ema model 和 model 参数相等判断
def _params_equal(ema_model, model):
    for ema_param, param in zip(ema_model.named_parameters(),
                                model.named_parameters()):
        if not torch.equal(ema_param[1].data, param[1].data):
            # print("Difference in", ema_param[0])
            return False
    return True

# 求范数 默认2范数：平方和 的 开方
def calc_grad_magnitude(grads, norm_type=2.0):
    norm_type = float(norm_type)
    # 如果 是无穷范数 即 最大的 绝对值
    if norm_type == math.inf:
        norm = max(p.abs().max() for p in grads)
    else:
        # stack 沿着一个新维度对输入张量序列进行连接。 序列中所有的张量都应该为相同形状。
        norm = torch.norm(
            torch.stack([torch.norm(p, norm_type) for p in grads]), norm_type)

    return norm


def get_share_weight(domain_out, before_softmax,  class_temperature=10.0):
    before_softmax = before_softmax / class_temperature
    after_softmax = F.softmax(before_softmax, dim=1)
    
    entropy = torch.sum(- after_softmax * torch.log(after_softmax + 1e-10), dim=1, keepdim=True)
    entropy_norm = entropy / np.log(after_softmax.size(1))
    weight = entropy_norm - domain_out
    return weight.detach()

def normalize_weight(x):
    min_val = x.min()
    max_val = x.max()
    x = (x - min_val) / (max_val - min_val)
    return x.detach()

def soft_label_cross_entropy(pred, soft_label, pixel_weights=None):
    N, C, H, W = pred.shape
    loss = -soft_label.float()*torch.log(pred)
    if pixel_weights is None:
        return torch.mean(torch.sum(loss, dim=1))
    return torch.mean(pixel_weights*torch.sum(loss, dim=1))

def adjust_learning_rate(method, base_lr, iters, warmup_iters, warmup_ratio, max_iters, power=1.0):
    if method=='poly':
        if iters >= warmup_iters:
            lr = base_lr * ((1 - float(iters) / max_iters) ** (power))
        else:
            k = (1 - iters / warmup_iters) * (1 - warmup_ratio)
            lr = base_lr * (1 - k)
    else:
        raise NotImplementedError
    return lr

class PixelDiscriminator(nn.Module):
    def __init__(self, input_nc, ndf=512, num_classes=1):
        super(PixelDiscriminator, self).__init__()

        self.D = nn.Sequential(
            nn.Conv2d(input_nc, ndf, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv2d(ndf, ndf//2, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=0.2, inplace=True)
		)
        self.cls1 = nn.Conv2d(ndf//2, num_classes, kernel_size=3, stride=1, padding=1)
        self.cls2 = nn.Conv2d(ndf//2, num_classes, kernel_size=3, stride=1, padding=1)

    def forward(self, x, size=None):
        out = self.D(x)
        src_out = self.cls1(out)
        tgt_out = self.cls2(out)
        out = torch.cat((src_out, tgt_out), dim=1)
        if size is not None:
            out = F.interpolate(out, size=size, mode='bilinear', align_corners=True)
        return out

class PixelDiscriminator2(nn.Module):
    def __init__(self, input_nc, ndf=512, num_classes=1):
        super(PixelDiscriminator2, self).__init__()

        self.D = nn.Sequential(
            nn.Conv2d(input_nc, ndf, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv2d(ndf, ndf//2, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=0.2, inplace=True)
		)
        self.cls1 = nn.Conv2d(ndf//2, num_classes, kernel_size=3, stride=1, padding=1)
        self.sig = nn.Sigmoid()

    def forward(self, x, size=None):
        out = self.D(x)
        out = self.cls1(out)
        out = self.sig(out)
        if size is not None:
            out = F.interpolate(out, size=size, mode='bilinear', align_corners=True)
        return out    

@UDA.register_module()
class DACS(UDADecorator):

    def __init__(self, **cfg):
        super(DACS, self).__init__(**cfg)
        self.local_iter = 0
        self.max_iters = cfg['max_iters']
        self.alpha = cfg['alpha']
        # 伪标签阈值
        self.pseudo_threshold = cfg['pseudo_threshold']
        self.psweight_ignore_top = cfg['pseudo_weight_ignore_top']
        self.psweight_ignore_bottom = cfg['pseudo_weight_ignore_bottom']
        # imageNet 特征距离 λ
        self.fdist_lambda = cfg['imnet_feature_dist_lambda']
        self.fdist_classes = cfg['imnet_feature_dist_classes']
        self.fdist_scale_min_ratio = cfg['imnet_feature_dist_scale_min_ratio']
        self.enable_fdist = self.fdist_lambda > 0
        self.mix = cfg['mix']
        self.blur = cfg['blur']
        self.color_jitter_s = cfg['color_jitter_strength']
        self.color_jitter_p = cfg['color_jitter_probability']
        self.debug_img_interval = cfg['debug_img_interval']
        self.print_grad_magnitude = cfg['print_grad_magnitude']
        assert self.mix == 'class'

        self.debug_fdist_mask = None
        self.debug_gt_rescale = None

        self.class_probs = {}
        ema_cfg = deepcopy(cfg['model'])
        self.ema_model = build_segmentor(ema_cfg)

        # 使用 imagenet 模型距离
        if self.enable_fdist:
            self.imnet_model = build_segmentor(deepcopy(cfg['model']))
        else:
            self.imnet_model = None

        self.mmd_loss = cfg['mmd_loss']
        self.mix_losses_soft = cfg['mix_losses_soft']
        self.soft_paste  = cfg['soft_paste']

        device = torch.device('cuda:0')
        self.adv_loss = cfg['adv_loss']
        self.adv_loss_fada = cfg['adv_loss_fada']
        if self.adv_loss_fada:
            self.model_D = PixelDiscriminator(512, 512, num_classes=19)
            self.model_D.to(device) 
            self.optimizer_D = torch.optim.Adam(self.model_D.parameters(), lr=0.008, betas=(0.9, 0.99))
            self.optimizer_D.zero_grad()
            

        if self.adv_loss:
            self.model_D = PixelDiscriminator(512, 512, num_classes=19)
            self.model_D.to(device) 
            self.model_Dis = PixelDiscriminator2(512, 512, num_classes=1)
            self.model_Dis.to(device)

            self.optimizer_D = torch.optim.Adam(self.model_D.parameters(), lr=0.008, betas=(0.9, 0.99))
            self.optimizer_D.zero_grad()
            self.bce_loss = torch.nn.BCELoss(reduction='none')

        
        if self.soft_paste:
            self.gta5_cls_mixer = rand_mixer("gta5")
            self.class_to_select = [12, 15, 16, 17, 18]

    # 获得 ema 模型
    def get_ema_model(self):
        return get_module(self.ema_model)

    # 获得 imageNet 模型
    def get_imnet_model(self):
        return get_module(self.imnet_model)

    # 初始化ema权重
    def _init_ema_weights(self):
        for param in self.get_ema_model().parameters():
            # 在x->y->z传播中，如果我们对y进行detach()，梯度还是能正常传播的
            # 但如果我们对y进行detach_()，就把x->y->z切成两部分：x和y->z，x就无法接受到后面传过来的梯度
            # 截断反向传播的梯度流。
            param.detach_()

        # 获取两个模型的参数
        mp = list(self.get_model().parameters())
        mcp = list(self.get_ema_model().parameters())
        for i in range(0, len(mp)):
            # ？？
            if not mcp[i].data.shape:  # scalar tensor
                mcp[i].data = mp[i].data.clone()
            else:
                mcp[i].data[:] = mp[i].data[:].clone()

    def _update_ema(self, iter):
        alpha_teacher = min(1 - 1 / (iter + 1), self.alpha)
        for ema_param, param in zip(self.get_ema_model().parameters(),
                                    self.get_model().parameters()):
            if not param.data.shape:  # scalar tensor
                ema_param.data = \
                    alpha_teacher * ema_param.data + \
                    (1 - alpha_teacher) * param.data
            else:
                ema_param.data[:] = \
                    alpha_teacher * ema_param[:].data[:] + \
                    (1 - alpha_teacher) * param[:].data[:]

    # 只有训练迭代步骤，反向传播和优化更新在优化器钩子
    def train_step(self, data_batch, optimizer, **kwargs):
        """The iteration step during training.

        This method defines an iteration step during training, except for the
        back propagation and optimizer updating, which are done in an optimizer
        hook. Note that in some complicated cases or models, the whole process
        including back propagation and optimizer updating is also defined in
        this method, such as GAN.

        Args:
            data (dict): The output of dataloader.
            optimizer (:obj:`torch.optim.Optimizer` | dict): The optimizer of
                runner is passed to ``train_step()``. This argument is unused
                and reserved.

        Returns:
            dict: It should contain at least 3 keys: ``loss``, ``log_vars``,
                ``num_samples``.
                ``loss`` is a tensor for back propagation, which can be a
                weighted sum of multiple losses.
                ``log_vars`` contains all the variables to be sent to the
                logger.
                ``num_samples`` indicates the batch size (when the model is
                DDP, it means the batch size on each GPU), which is used for
                averaging the logs.
        """

        optimizer.zero_grad()
        log_vars = self(**data_batch)
        # print('--------------------------')
        # print("！！！！！！！DACS 里的数据构成！！！！！！")
        
        # img, lab, _, _ = data_batch
        # print(img.shape)
        # print('--------------------------')
        optimizer.step()

        log_vars.pop('loss', None)  # remove the unnecessary 'loss'
        outputs = dict(
            log_vars=log_vars, num_samples=len(data_batch['img_metas']))
        return outputs          

    def masked_feat_dist(self, f1, f2, mask=None):
        feat_diff = f1 - f2
        # mmcv.print_log(f'fdiff: {feat_diff.shape}', 'mmseg')
        pw_feat_dist = torch.norm(feat_diff, dim=1, p=2)
        # mmcv.print_log(f'pw_fdist: {pw_feat_dist.shape}', 'mmseg')
        if mask is not None:
            # mmcv.print_log(f'fd mask: {mask.shape}', 'mmseg')
            pw_feat_dist = pw_feat_dist[mask.squeeze(1)]
            # mmcv.print_log(f'fd masked: {pw_feat_dist.shape}', 'mmseg')
        return torch.mean(pw_feat_dist)

    def calc_feat_dist(self, img, gt, feat=None):
        assert self.enable_fdist
        with torch.no_grad():
            self.get_imnet_model().eval()
            feat_imnet = self.get_imnet_model().extract_feat(img)
            feat_imnet = [f.detach() for f in feat_imnet]
        lay = -1
        if self.fdist_classes is not None:
            fdclasses = torch.tensor(self.fdist_classes, device=gt.device)
            scale_factor = gt.shape[-1] // feat[lay].shape[-1]
            gt_rescaled = downscale_label_ratio(gt, scale_factor,
                                                self.fdist_scale_min_ratio,
                                                self.num_classes,
                                                255).long().detach()
            fdist_mask = torch.any(gt_rescaled[..., None] == fdclasses, -1)
            feat_dist = self.masked_feat_dist(feat[lay], feat_imnet[lay],
                                              fdist_mask)
            self.debug_fdist_mask = fdist_mask
            self.debug_gt_rescale = gt_rescaled
        else:
            feat_dist = self.masked_feat_dist(feat[lay], feat_imnet[lay])
        feat_dist = self.fdist_lambda * feat_dist
        feat_loss, feat_log = self._parse_losses(
            {'loss_imnet_feat_dist': feat_dist})
        feat_log.pop('loss', None)
        return feat_loss, feat_log

    def guassian_kernel(self,source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
        n_samples = int(source.size()[0])+int(target.size()[0])
        total = torch.cat([source, target], dim=0)
        # print('---------------------------------')
        # print('source.size',source.size)
        # print('source.size0',source.size(0))
        # print('source.size1',source.size(1))
        # print('source.shape',source.shape)
        # print('total.size',total.size)
        # print('total.shape',total.shape)
        # # 4,19,1,1
        # print('---------------------------------')    
        total0 = total.unsqueeze(0).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)), int(total.size(2)), int(total.size(2)))

        total1 = total.unsqueeze(1).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)), int(total.size(2)), int(total.size(2)))

        L2_distance = ((total0-total1)**2).sum(2)

        if fix_sigma:
            bandwidth = fix_sigma
        else:
            bandwidth = torch.sum(L2_distance.data) / (n_samples**2-n_samples)

        bandwidth /= kernel_mul ** (kernel_num // 2)
        bandwidth_list = [bandwidth * (kernel_mul**i) for i in range(kernel_num)]

        kernel_val = [torch.exp(-L2_distance / bandwidth_temp) for bandwidth_temp in bandwidth_list]

        return sum(kernel_val)

    # Maximum Mean Discrepancy (MMD) to align feature 
    def mmd_rbf(self,source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):

        batch_size = int(source.size()[0])
        kernels = self.guassian_kernel(source, target,
            kernel_mul=kernel_mul, kernel_num=kernel_num, fix_sigma=fix_sigma)

        XX = kernels[:batch_size, :batch_size]
        YY = kernels[batch_size:, batch_size:]
        XY = kernels[:batch_size, batch_size:]
        YX = kernels[batch_size:, :batch_size]
        loss = torch.mean(XX + YY - XY -YX)
        loss = loss * 0.05
        # 解析loss 、加入log
        mmd_loss, mmd_log = self._parse_losses(
            {'loss_mmd': loss})
        mmd_log.pop('loss', None)
        return mmd_loss, mmd_log



    def forward_train(self, img, img_metas, gt_semantic_seg, target_img,
                      target_img_metas):
        """Forward function for training.

        Args:
            img (Tensor): Input images.
            img_metas (list[dict]): List of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                `mmseg/datasets/pipelines/formatting.py:Collect`.
            gt_semantic_seg (Tensor): Semantic segmentation masks
                used if the architecture supports semantic segmentation task.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """
        log_vars = {}
        batch_size = img.shape[0]
        dev = img.device

        # Init/update ema model
        if self.local_iter == 0:
            self._init_ema_weights()
            # assert _params_equal(self.get_ema_model(), self.get_model())

        if self.local_iter > 0:
            self._update_ema(self.local_iter)
            # assert not _params_equal(self.get_ema_model(), self.get_model())
            # assert self.get_ema_model().training

        means, stds = get_mean_std(img_metas, dev)
        strong_parameters = {
            'mix': None,
            'color_jitter': random.uniform(0, 1),
            'color_jitter_s': self.color_jitter_s,
            'color_jitter_p': self.color_jitter_p,
            'blur': random.uniform(0, 1) if self.blur else 0,
            'mean': means[0].unsqueeze(0),  # assume same normalization
            'std': stds[0].unsqueeze(0)
        }



        # Train on source images
        clean_losses = self.get_model().forward_train(
            img, img_metas, gt_semantic_seg, return_feat=True)
        src_feat = clean_losses.pop('features')
        clean_loss, clean_log_vars = self._parse_losses(clean_losses)
        log_vars.update(clean_log_vars)
        clean_loss.backward(retain_graph=self.enable_fdist)








        if self.print_grad_magnitude:
            params = self.get_model().backbone.parameters()
            seg_grads = [
                p.grad.detach().clone() for p in params if p.grad is not None
            ]
            grad_mag = calc_grad_magnitude(seg_grads)
            mmcv.print_log(f'Seg. Grad.: {grad_mag}', 'mmseg')

        # ImageNet feature distance
        if self.enable_fdist:
            feat_loss, feat_log = self.calc_feat_dist(img, gt_semantic_seg,
                                                      src_feat)
            feat_loss.backward()
            log_vars.update(add_prefix(feat_log, 'src'))
            if self.print_grad_magnitude:
                params = self.get_model().backbone.parameters()
                fd_grads = [
                    p.grad.detach() for p in params if p.grad is not None
                ]
                fd_grads = [g2 - g1 for g1, g2 in zip(seg_grads, fd_grads)]
                grad_mag = calc_grad_magnitude(fd_grads)
                mmcv.print_log(f'Fdist Grad.: {grad_mag}', 'mmseg')

        # Generate pseudo-label
        for m in self.get_ema_model().modules():
            if isinstance(m, _DropoutNd):
                m.training = False
            if isinstance(m, DropPath):
                m.training = False
        ema_logits = self.get_ema_model().encode_decode(
            target_img, target_img_metas)

        ema_softmax = torch.softmax(ema_logits.detach(), dim=1)
        pseudo_prob, pseudo_label = torch.max(ema_softmax, dim=1)
        ps_large_p = pseudo_prob.ge(self.pseudo_threshold).long() == 1
        ps_size = np.size(np.array(pseudo_label.cpu()))
        pseudo_weight = torch.sum(ps_large_p).item() / ps_size
        pseudo_weight = pseudo_weight * torch.ones(
            pseudo_prob.shape, device=dev)

        if self.adv_loss_fada:
            self.model_D.train()

            current_lr_D = adjust_learning_rate('poly', 0.008, self.local_iter, 0, 0, self.max_iters, power=0.9)     
            for index in range(len(self.optimizer_D.param_groups)):
                self.optimizer_D.param_groups[index]['lr'] = current_lr_D       

            self.optimizer_D.zero_grad()
            # generate soft labels
            src_pred = self.get_model().encode_decode(img, img_metas)
            src_soft_label = F.softmax(src_pred, dim=1).detach()
            src_soft_label[src_soft_label>0.9] = 0.9

            tgt_pred_ema = ema_logits
            tgt_soft_label = F.softmax(tgt_pred_ema, dim=1).detach()
            tgt_soft_label[tgt_soft_label>0.9] = 0.9
        
            tgt_size = target_img.shape[-2:]
            tgt_fea_ema = self.get_ema_model().extract_feat(target_img)
            tgt_fea_ema = [f.detach() for f in tgt_fea_ema]
            tgt_D_pred = self.model_D(tgt_fea_ema[-1], tgt_size)
            loss_adv_tgt = 0.001 * soft_label_cross_entropy(tgt_D_pred, torch.cat((tgt_soft_label, torch.zeros_like(tgt_soft_label)), dim=1))
            adv_tgt_loss, adv_tgt_log = self._parse_losses(
                {'loss_adv_tgt': loss_adv_tgt})
            adv_tgt_log.pop('loss', None)
            adv_tgt_loss.backward()
            log_vars.update(add_prefix(adv_tgt_log, 'adv'))

            self.optimizer_D.zero_grad()
            # 0代表源域
            src_size = img.shape[-2:]
            src_fea = [f.detach() for f in src_feat]
            src_D_pred = self.model_D(src_fea[-1].detach(), src_size)
            loss_D_src = 0.5 * soft_label_cross_entropy(src_D_pred, torch.cat((src_soft_label, torch.zeros_like(src_soft_label)), dim=1))
            loss_D_src, D_src_log = self._parse_losses(
                {'loss_D_src': loss_D_src})
            D_src_log.pop('loss', None)
            loss_D_src.backward()
            log_vars.update(add_prefix(D_src_log, 'd_src'))

            # 源域放前面
            tgt_D_pred = self.model_D(tgt_fea_ema[-1].detach(), tgt_size)
            loss_D_tgt = 0.5*soft_label_cross_entropy(tgt_D_pred, torch.cat((torch.zeros_like(tgt_soft_label), tgt_soft_label), dim=1))
            loss_D_tgt, D_tgt_log = self._parse_losses(
                {'loss_D_tgt': loss_D_tgt})
            D_tgt_log.pop('loss', None)
            loss_D_tgt.backward()
            log_vars.update(add_prefix(D_tgt_log, 'd_tgt'))

            # torch.distributed.barrier()

            self.optimizer_D.step()            
        
        
        if self.adv_loss:
            
            self.model_D.train()
            self.model_Dis.train()
            current_lr_D = adjust_learning_rate('poly', 0.0001, self.local_iter, 0, 0, self.max_iters, power=0.9)
            for index in range(len(self.optimizer_D.param_groups)):
                self.optimizer_D.param_groups[index]['lr'] = current_lr_D

            self.optimizer_D.zero_grad()
            src_pred = self.get_model().encode_decode(img, img_metas)
            src_soft_label = F.softmax(src_pred, dim=1).detach()
            src_soft_label[src_soft_label>0.9] = 0.9
            src_size = img.shape[-2:]
            src_fea = [f.detach() for f in src_feat]
            # ??src_fea.shape
            #AttributeError: 'list' object has no attribute 'shape'

            src_fea_D = src_fea[-2]
            src_Dis_pred = self.model_Dis(src_fea_D.detach(), src_size)
            source_share_weight = get_share_weight(src_Dis_pred, src_pred, class_temperature=10.0)
            source_share_weight = normalize_weight(source_share_weight)
            src_D_pred = self.model_D(src_fea_D, src_size)
            loss_adv_src = 0.001*soft_label_cross_entropy(F.softmax(src_D_pred, dim=1).clamp(min=1e-7, max=1.0), torch.cat((src_soft_label,torch.zeros_like(src_soft_label)), dim=1),source_share_weight)
            adv_src_loss, adv_src_log = self._parse_losses(
                {'loss_adv_src': loss_adv_src})
            adv_src_log.pop('loss', None)
            adv_src_loss.backward()
            log_vars.update(add_prefix(adv_src_log, 'adv'))

            self.optimizer_D.zero_grad()

            tgt_fea_ema = self.get_ema_model().extract_feat(target_img)
            tgt_fea_ema = [f.detach() for f in tgt_fea_ema]
            tgt_size = target_img.shape[-2:]
            src_fea_D = src_fea[-2]
            # src_fea.shape??
            src_Dis_pred = self.model_Dis(src_fea_D.detach(), src_size)
            loss_Dis_src = 0.5 * self.bce_loss(src_Dis_pred, torch.ones_like(src_Dis_pred))
            Dis_src_loss, Dis_src_log = self._parse_losses(
                {'loss_Dis_src': loss_Dis_src})
            #AttributeError: 'Tensor' object has no attribute 'pop'
            #loss_Dis_src.pop('loss', None)
            Dis_src_log.pop('loss', None)
            Dis_src_loss.backward()
            log_vars.update(add_prefix(Dis_src_log, 'dis_src'))

            tgt_fea_D = tgt_fea_ema[-2]
            tgt_Dis_pred = self.model_Dis(tgt_fea_D.detach(), tgt_size)
            loss_Dis_tgt = 0.5 * self.bce_loss(tgt_Dis_pred, torch.zeros_like(tgt_Dis_pred))
            loss_Dis_tgt, Dis_tgt_log = self._parse_losses(
                {'loss_Dis_tgt': loss_Dis_tgt})
            Dis_tgt_log.pop('loss', None)
            loss_Dis_tgt.backward()
            log_vars.update(add_prefix(Dis_tgt_log, 'dis_tgt'))

            tgt_pred_ema = ema_logits
            source_share_weight = get_share_weight(src_Dis_pred, src_pred, class_temperature=10.0)
            source_share_weight = normalize_weight(source_share_weight)
            target_share_weight = -get_share_weight(tgt_Dis_pred, tgt_pred_ema, class_temperature=1.0)
            target_share_weight = normalize_weight(target_share_weight)

            src_D_pred = self.model_D(src_fea_D.detach(), src_size)
            loss_D_src = 0.5 * soft_label_cross_entropy(F.softmax(src_D_pred, dim=1).clamp(min=1e-7, max=1.0), torch.cat((torch.zeros_like(src_soft_label),src_soft_label), dim=1), source_share_weight)
            loss_D_src, D_src_log = self._parse_losses(
                {'loss_D_src': loss_D_src})
            D_src_log.pop('loss', None)
            loss_D_src.backward()
            log_vars.update(add_prefix(D_src_log, 'd_src'))


            tgt_soft_label = F.softmax(tgt_pred_ema, dim=1).detach()
            tgt_soft_label[tgt_soft_label>0.9] = 0.9

            tgt_D_pred = self.model_D(tgt_fea_D.detach(), tgt_size)
            loss_D_tgt = 0.5 * soft_label_cross_entropy(F.softmax(tgt_D_pred, dim=1).clamp(min=1e-7, max=1.0), torch.cat((tgt_soft_label,torch.zeros_like(tgt_soft_label)), dim=1), target_share_weight)
            loss_D_tgt, D_tgt_log = self._parse_losses(
                {'loss_D_tgt': loss_D_tgt})
            D_tgt_log.pop('loss', None)
            loss_D_tgt.backward()
            log_vars.update(add_prefix(D_tgt_log, 'd_tgt'))


            self.optimizer_D.step()


        if self.psweight_ignore_top > 0:
            # Don't trust pseudo-labels in regions with potential
            # rectification artifacts. This can lead to a pseudo-label
            # drift from sky towards building or traffic light.
            pseudo_weight[:, :self.psweight_ignore_top, :] = 0
        if self.psweight_ignore_bottom > 0:
            pseudo_weight[:, -self.psweight_ignore_bottom:, :] = 0
        gt_pixel_weight = torch.ones((pseudo_weight.shape), device=dev)

        # Apply mixing
        mixed_img, mixed_lbl = [None] * batch_size, [None] * batch_size
        mix_masks = get_class_masks(gt_semantic_seg)
        # print('#########测试mix_masks的shape##########')
        # print('mix_masks.shape', mix_masks.shape)
        # print('##########测试mix_masks结束#############')

        if self.soft_paste:
            cls_to_use = random.sample(self.class_to_select, 2)
        for i in range(batch_size):

            # img.shape torch.Size([2, 3, 512, 512])
            # target_img.shape torch.Size([2, 3, 512, 512])

            # img[i].shape torch.Size([3, 512, 512])
            # target_img[i].shape torch.Size([3, 512, 512])
            # gt_semantic_seg[i][0], pseudo_label[i] torch.Size([512, 512]) torch.Size([512, 512])
            # gt_semantic_seg, pseudo_label torch.Size([2, 1, 512, 512]) torch.Size([2, 512, 512])

            # tempdata.shape: torch.Size([2, 3, 512, 512])
            # temptarget.shape: torch.Size([2, 512, 512])
            
            
            strong_parameters_soft = strong_parameters
            strong_parameters_soft['mix'] = mix_masks[i] * 0.9
            strong_parameters['mix'] = mix_masks[i]
            # mix_masks[i].shape torch.Size([1, 1, 512, 512])

            # tempdata = torch.stack((img[i], target_img[i]))
            # temptarget = torch.stack((gt_semantic_seg[i][0], pseudo_label[i]))
            # print('tempdata.shape:',tempdata.shape)
            # print('temptarget.shape:',temptarget.shape)
            # tempdata.shape: torch.Size([2, 3, 512, 512])
            # temptarget.shape: torch.Size([2, 512, 512])
            # 开启软粘贴，从字典库中选取少量的类
            if self.soft_paste:
                mixed_img[i], mixed_lbl[i] = strong_transform_soft(
                    strong_parameters,
                    self.gta5_cls_mixer,
                    cls_to_use,
                    data=torch.stack((img[i], target_img[i])),
                    target=torch.stack((gt_semantic_seg[i][0], pseudo_label[i])))
                _, pseudo_weight[i] = strong_transform_soft(
                    strong_parameters,
                    self.gta5_cls_mixer,
                    cls_to_use,
                    target=torch.stack((gt_pixel_weight[i], pseudo_weight[i])))
            else:
                mixed_img[i], mixed_lbl[i] = strong_transform(
                strong_parameters_soft,
                data=torch.stack((img[i], target_img[i])),
                target=torch.stack((gt_semantic_seg[i][0], pseudo_label[i])))
                _, pseudo_weight[i] = strong_transform(
                    strong_parameters,
                    target=torch.stack((gt_pixel_weight[i], pseudo_weight[i])))
        mixed_img = torch.cat(mixed_img)
        mixed_lbl = torch.cat(mixed_lbl)

        # Train on mixed images
        # 相当于L_u
        if self.mix_losses_soft:
            pseudo_label0 = torch.unsqueeze(pseudo_label, 1)
            pseudo_weight0 = 0.1 * pseudo_weight
            mix_losses = self.get_model().forward_train(
                mixed_img, img_metas, mixed_lbl, pseudo_weight, return_feat=True)
            
            
            #------------
            # print('------------')
            # print('mixed_lbl.shape:',mixed_lbl.shape)
            # print('pseudo_label.shape:',pseudo_label.shape)
            # print('pseudo_label0.shape:',pseudo_label0.shape)
            # print('------------')
            #------------
            mix_losses_soft = self.get_ema_model().forward_train(target_img, target_img_metas, pseudo_label0,pseudo_weight0, return_feat=True)

            mix_feat = mix_losses.pop('features')
            
            mix_losses_soft.pop('features')

            mix_losses = add_prefix(mix_losses, 'mixs')
            mix_losses_soft = add_prefix(mix_losses_soft, 'mix_soft')

            mix_losses['mixs.decode.loss_seg'] = 0.9 * mix_losses['mixs.decode.loss_seg'] + 0.1 * mix_losses_soft['mix_soft.decode.loss_seg']
            mix_loss, mix_log_vars = self._parse_losses(mix_losses)
            log_vars.update(mix_log_vars)
            mix_loss.backward()
        else:
            mix_losses = self.get_model().forward_train(
            mixed_img, img_metas, mixed_lbl, pseudo_weight, return_feat=True)
            mix_feat = mix_losses.pop('features')
            mix_losses = add_prefix(mix_losses, 'mix')
            mix_loss, mix_log_vars = self._parse_losses(mix_losses)
            log_vars.update(mix_log_vars)
            mix_loss.backward()

        # target soft loss
        # ema->pred  
        # mix_losses_soft = self.get_ema_model().forward_train(
        #     target_img, target_img_metas, pseudo_label, return_feat=True)

        if self.mmd_loss:

            # src_feat = self.get_model.extract_feat(img)
            # mix_feat = self.get_model.extract_feat(mixed_img)
            
            ap = nn.AdaptiveAvgPool2d((1,1))
            # 提取的特征是list?没有size？无法ap,怎么解决的
            #gs = ap(src_feat)
            #gt = ap(mix_feat)
 
            src_feat = [f.detach() for f in src_feat]
            mix_feat = [f.detach() for f in mix_feat]
            #----------
            # print('-------------')
            # print('src_feat.shape:',src_feat.shape)
            # print('mix_feat.shape:',mix_feat.shape)
            # print('-------------')
            #----------
            # 试试取消-1看看
            gs = ap(src_feat[-1])
            gt = ap(mix_feat[-1])
            mmd_loss, mmd_log = self.mmd_rbf(gs, gt)
            mmd_loss = mmd_loss.requires_grad_()
            mmd_loss.backward()
            log_vars.update(add_prefix(mmd_log, 'mmd'))            

        if self.local_iter % self.debug_img_interval == 0:
            out_dir = os.path.join(self.train_cfg['work_dir'],
                                   'class_mix_debug')
            os.makedirs(out_dir, exist_ok=True)
            vis_img = torch.clamp(denorm(img, means, stds), 0, 1)
            vis_trg_img = torch.clamp(denorm(target_img, means, stds), 0, 1)
            vis_mixed_img = torch.clamp(denorm(mixed_img, means, stds), 0, 1)
            for j in range(batch_size):
                rows, cols = 2, 5
                fig, axs = plt.subplots(
                    rows,
                    cols,
                    figsize=(3 * cols, 3 * rows),
                    gridspec_kw={
                        'hspace': 0.1,
                        'wspace': 0,
                        'top': 0.95,
                        'bottom': 0,
                        'right': 1,
                        'left': 0
                    },
                )
                subplotimg(axs[0][0], vis_img[j], 'Source Image')
                subplotimg(axs[1][0], vis_trg_img[j], 'Target Image')
                subplotimg(
                    axs[0][1],
                    gt_semantic_seg[j],
                    'Source Seg GT',
                    cmap='cityscapes')
                subplotimg(
                    axs[1][1],
                    pseudo_label[j],
                    'Target Seg (Pseudo) GT',
                    cmap='cityscapes')
                subplotimg(axs[0][2], vis_mixed_img[j], 'Mixed Image')
                subplotimg(
                    axs[1][2], mix_masks[j][0], 'Domain Mask', cmap='gray')
                # subplotimg(axs[0][3], pred_u_s[j], "Seg Pred",
                #            cmap="cityscapes")
                subplotimg(
                    axs[1][3], mixed_lbl[j], 'Seg Targ', cmap='cityscapes')
                subplotimg(
                    axs[0][3], pseudo_weight[j], 'Pseudo W.', vmin=0, vmax=1)
                if self.debug_fdist_mask is not None:
                    subplotimg(
                        axs[0][4],
                        self.debug_fdist_mask[j][0],
                        'FDist Mask',
                        cmap='gray')
                if self.debug_gt_rescale is not None:
                    subplotimg(
                        axs[1][4],
                        self.debug_gt_rescale[j],
                        'Scaled GT',
                        cmap='cityscapes')
                for ax in axs.flat:
                    ax.axis('off')
                plt.savefig(
                    os.path.join(out_dir,
                                 f'{(self.local_iter + 1):06d}_{j}.png'))
                plt.close()
        self.local_iter += 1

        return log_vars
