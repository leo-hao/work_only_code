# The domain-mixing are based on:
# https://github.com/GaoLii/DSP
# https://github.com/vikolss/DACS
# https://github.com/lhoyer/DAFormer

import math
import os
import random
# 
from copy import deepcopy

import mmcv
import numpy as np
import torch
import torch.nn as nn
from matplotlib import pyplot as plt
from timm.models.layers import DropPath
from torch.nn.modules.dropout import _DropoutNd

from mmseg.core import add_prefix
from mmseg.models import UDA, build_segmentor
from mmseg.models.uda.uda_decorator import UDADecorator, get_module
from mmseg.models.utils.dsp_transforms import (denorm, get_class_masks,
                                                get_mean_std, strong_transform, rand_mixer,strong_transform_class_mix)
from mmseg.models.utils.visualization import subplotimg
from mmseg.utils.utils import downscale_label_ratio




def _params_equal(ema_model, model):
    for ema_param, param in zip(ema_model.named_parameters(),
                                model.named_parameters()):
        if not torch.equal(ema_param[1].data, param[1].data):
            # print("Difference in", ema_param[0])
            return False
    return True


def calc_grad_magnitude(grads, norm_type=2.0):
    norm_type = float(norm_type)
    if norm_type == math.inf:
        norm = max(p.abs().max() for p in grads)
    else:
        norm = torch.norm(
            torch.stack([torch.norm(p, norm_type) for p in grads]), norm_type)

    return norm


@UDA.register_module()
class DSP(UDADecorator):

    def __init__(self, **cfg):
        super(DSP, self).__init__(**cfg)
        self.local_iter = 0
        self.max_iters = cfg['max_iters']
        self.alpha = cfg['alpha']
        self.pseudo_threshold = cfg['pseudo_threshold']
        self.psweight_ignore_top = cfg['pseudo_weight_ignore_top']
        self.psweight_ignore_bottom = cfg['pseudo_weight_ignore_bottom']
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

        if self.enable_fdist:
            self.imnet_model = build_segmentor(deepcopy(cfg['model']))
        else:
            self.imnet_model = None

        # 增加选项 mmd Loss：
        # 默认不使用
        self.mmd_loss = cfg['mmd_loss']
        # 增加选项 source soft
        self.Ll_soft = cfg['Ll_soft']
        # 增加选项 target cons
        # 在没有找到下一个batch方法前，使用本来就有的mix_loss
        #gta_root='../../../data/gta/'
        # wrong
        #gta5_cls_mixer = rand_mixer(cfg['data']['train']['data_root'], "gta5")
        # class_to_select = [12, 15, 16, 17, 18]

        # 由于我暂时还不会, 在此处遍历下一个batch的图片，
        # 因此，我想到，要不我混个目标域到源域试试
        # data root wrong 
        #city_cla_mixer = rand_mixer(cfg['data']['val']['data_root'], "cityscapes")

    def get_ema_model(self):
        return get_module(self.ema_model)

    def get_imnet_model(self):
        return get_module(self.imnet_model)

    # 同dacs.trainUDA 89-122
    def _init_ema_weights(self):
        for param in self.get_ema_model().parameters():
            param.detach_()
        mp = list(self.get_model().parameters())
        mcp = list(self.get_ema_model().parameters())
        for i in range(0, len(mp)):
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
        optimizer.step()
        # print('--------------------------')
        # print("！！！！！！！dsp 里的数据构成！！！！！！")
        # for a, b in data_batch.items():
        #     print('a:', a)
        #     print('b:', b)
            
        
        #print(data_batch['img'])
        
        #print('--------------------------')
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
        # 解析loss 、加入log
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
        loss = loss * 0.005
        # 解析loss 、加入log
        mmd_loss, mmd_log = self._parse_losses(
            {'loss_mmd': loss})
        #mmd_log.pop('loss', None)
        return mmd_loss, mmd_log


    def forward_train(self, img, img_metas, gt_semantic_seg, img2, img_metas2, gt_semantic_seg2, target_img,
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
        # print('!!!!!!!!!!!!!!')
        # print(dev)
        # print('---------------')
        # gta5_cls_mixer = rand_mixer()
        # class_to_select = [12, 15, 16, 17, 18]
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
        # 我不知道train后返回了什么
        clean_losses = self.get_model().forward_train(
            img, img_metas, gt_semantic_seg, return_feat=True)
        # 弹出特征
        src_feat = clean_losses.pop('features')
        # 解析loss的log
        clean_loss, clean_log_vars = self._parse_losses(clean_losses)
        # 更新到log_vars
        log_vars.update(clean_log_vars)
        # 两个损失函数是截然不同的两类损失函数，因此我们可以通过代码：backward(retain_graph=True)在计算出第一个损失函数的梯度值后保存计算图用于继续计算第二个损失函数的梯度。
        # 同 L_l 
        clean_loss.backward(retain_graph=self.enable_fdist)
        # 默认是关闭的
        if self.print_grad_magnitude:
            params = self.get_model().backbone.parameters()
            seg_grads = [
                p.grad.detach().clone() for p in params if p.grad is not None
            ]
            grad_mag = calc_grad_magnitude(seg_grads)
            mmcv.print_log(f'Seg. Grad.: {grad_mag}', 'mmseg')

        # ImageNet feature distance
        # 如果用使用imgaegNet特征距离
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

        lam = 0.9
        # Generate pseudo-label
        for m in self.get_ema_model().modules():
            if isinstance(m, _DropoutNd):
                m.training = False
            if isinstance(m, DropPath):
                m.training = False
        # 目标的伪标签
        # 相当于 logits_u_w 但是又没有进行弱增强的那种。。。
        # target_img ----- images_remain ->> inputs_u_w  没进行弱增强，因为压根没加weakTransform这个代码
        # ！！！！！！！！！！！理由未知、、、可以去问问作者！！！！！！！
        ema_logits = self.get_ema_model().encode_decode(
            target_img, target_img_metas)

        # 同 pseudo_label
        ema_softmax = torch.softmax(ema_logits.detach(), dim=1)
        # 同 max_prbs, targets_u_w 目标标签
        pseudo_prob, pseudo_label = torch.max(ema_softmax, dim=1)
        # ge----大于等于 阈值
        ps_large_p = pseudo_prob.ge(self.pseudo_threshold).long() == 1
        ps_size = np.size(np.array(pseudo_label.cpu()))
        # 伪标签的权重
        pseudo_weight = torch.sum(ps_large_p).item() / ps_size
        pseudo_weight = pseudo_weight * torch.ones(
            pseudo_prob.shape, device=dev)
        

        if self.psweight_ignore_top > 0:
            # Don't trust pseudo-labels in regions with potential
            # rectification artifacts. This can lead to a pseudo-label
            # drift from sky towards building or traffic light.
            pseudo_weight[:, :self.psweight_ignore_top, :] = 0
        if self.psweight_ignore_bottom > 0:
            pseudo_weight[:, -self.psweight_ignore_bottom:, :] = 0

        pseudo_weight_s = pseudo_weight
        # 真实值的权重
        gt_pixel_weight = torch.ones((pseudo_weight.shape), device=dev)

        
        # Apply mixing
        # mix source -> Target 
        mixed_img, mixed_lbl = [None] * batch_size, [None] * batch_size
        # 从source2中取mask
        mix_masks = get_class_masks(gt_semantic_seg)
        #print('-----------------')
        #print(mix_masks.type)
        #print('------------------')
        mix_masks_lam = [lam * m for m in mix_masks]
        # 像DSP一样加上lam
        # mix_masks_lam = mix_masks * lam
        # 反向混合~~~
        # source->source
        if self.Ll_soft:
            mixed_img_s, mixed_lbl_s = [None] * batch_size, [None] * batch_size
            #mix_masks_s = get_class_masks(pseudo_label)
            # mix_masks_t_lam = mix_masks_t * lam

        #cls_to_use = random.sample(class_to_select, 2)
        # 遍历、 混合 图片、 标签、 权重
        for i in range(batch_size):
            strong_parameters['mix'] = mix_masks[i]
            # 把源域混到目标域
            # mixed_img[i], mixed_lbl[i] = strong_transform_class_mix(
            #     img2[i], target_img[i], gt_semantic_seg2[i][0], pseudo_label[i],
            #     mix_masks_lam, mix_masks, gta5_cls_mixer, cls_to_use, 
            #     strong_parameters)
            mixed_img[i], mixed_lbl[i] = strong_transform(
                strong_parameters,
                data=torch.stack((img[i], target_img[i])),
                target=torch.stack((gt_semantic_seg[i][0], pseudo_label[i])))

            _, pseudo_weight[i] = strong_transform(
                strong_parameters,
                target=torch.stack((gt_pixel_weight[i], pseudo_weight[i])))

            # 把目标域混到源域
            if self.Ll_soft:
                # mixed_img_s[i], mixed_lbl_s[i] = strong_transform_class_mix(
                #     img2[i], img[i], gt_semantic_seg2[i][0], gt_semantic_seg[i][0],
                #     mix_masks_lam, mix_masks, gta5_cls_mixer, cls_to_use, 
                #     strong_parameters)
                mixed_img_s[i], mixed_lbl_s[i] = strong_transform(
                    strong_parameters,
                    data=torch.stack((img[i], img2[i])),
                    target=torch.stack((gt_semantic_seg[i][0], gt_semantic_seg2[i][0])))
        
        # 
        
        # 相当于 inputs_u_s 融合后的输入 源 目标
        mixed_img = torch.cat(mixed_img)
        # 相当于 targets_u 融合后的标签
        mixed_lbl = torch.cat(mixed_lbl)
        # pseudo_weight = pseudo_weight * lam
        # 相当于 input_t , target_t 源源
        if self.Ll_soft:
            mixed_img_s = torch.cat(mixed_img_s)
            mixed_lbl_s = torch.cat(mixed_lbl_s)
            # pseudo_weight_s = pseudo_weight_s * (1 - lam)
        
        # Train on mixed images
        # 计算损失 相当于 L_cons 还缺 mmd和 L_soft
        # 问题，不知道从哪里来的图片
        # 从哪里载入的？模板原图片，
        # 看看mmcv吧
        # 如果开启双向混合，
        # 源混入目标 相当于L_u
        mix_losses = self.get_model().forward_train(
            mixed_img, img_metas, mixed_lbl, pseudo_weight, return_feat=True)
        mix_losses.pop('features')
        mix_losses = add_prefix(mix_losses, 'mix')
        # show mix_losses
        #print('---------------------------------')
        #print('show mix losses:', mix_losses.dtype)
        #print('before:',mix_losses['mix.decode.loss_seg'])
        # for a, b in mix_losses.items():
        #     print(a)
        #     print(b)
        #mix_losses['mix.decode.loss_seg'] = mix_losses['mix.decode.loss_seg'] * 0.9
        #print('After:',mix_losses['mix.decode.loss_seg'])
        #print('mix losses:', mix_losses.shape)
        #print('show mix losses:', mix_losses.dtype)
        #print('---------------------------------')
        mix_loss, mix_log_vars = self._parse_losses(mix_losses)
        log_vars.update(mix_log_vars)
        mix_loss.backward()
            

        #mmd_loss
        #del loss_feature
        if self.mmd_loss:
            # gs = self.get_model().extract_feat(
            #     mixed_img)
            #f_source = mix_masks_lam * p_logits
            #ap = nn.AdaptiveAvgPool2d((1,1))
            #gs = ap(p1)
            # print('---------------------------------')
            # # for a, b in gs.items():
            # #     print(a)
            # #     print(b)           

            # print(gs.shape)
            # print('---------------------------------') 


            # gt = self.get_model().extract_feat(mixed_img_s)
            #f_target = mix_masks_lam * pt_logits
            #gt = ap(p2)

            #loss_feature = mmd_rbf(f_source, f_target)
            # mmd_loss, mmd_log = self.mmd_rbf(gs, gt)

            #loss_mmd = 0.005 * loss_feature + 0.005 * loss_global
            #loss_mmd = 0.005 * loss_global
            #mmd_loss = add_prefix(mmd_loss, 'leo')
            #loss_mmd, mmd_log_vars = self._parse_losses(loss_mmd)
            
            p_logits = self.get_model().encode_decode(
                mixed_img, img_metas)
            #f_source = mix_masks_lam * p_logits
            ap = nn.AdaptiveAvgPool2d((1,1))
            gs = ap(p_logits)
            # print('---------------------------------')
            # # for a, b in gs.items():
            # #     print(a)
            # #     print(b)           

            # print(gs.shape)
            # print('---------------------------------') 


            pt_logits = self.get_model().encode_decode(mixed_img_s, img_metas)
            #f_target = mix_masks_lam * pt_logits
            gt = ap(pt_logits)

            #loss_feature = mmd_rbf(f_source, f_target)
            mmd_loss, mmd_log = self.mmd_rbf(gs, gt)
            mmd_loss.backward()
            log_vars.update(add_prefix(mmd_log, 'leommd'))

            


        # L_l soft
        # 源域混入了源
        # del soft
        if self.Ll_soft:
            Ll_soft_losses_1 = self.get_model().forward_train(mixed_img_s, img_metas, gt_semantic_seg)
            Ll_soft_losses_2 =  self.get_model().forward_train(mixed_img_s, img_metas, mixed_lbl_s)
            Ll_soft_losses_1 = add_prefix(Ll_soft_losses_1, 'Ll_soft_losses1')
            Ll_soft_losses_2 = add_prefix(Ll_soft_losses_2, 'Ll_soft_losses2')

            Ll_soft_losses_1['Ll_soft_losses1.decode.loss_seg'] = Ll_soft_losses_1['Ll_soft_losses1.decode.loss_seg'] * (1 - lam)
            Ll_soft_losses_2['Ll_soft_losses2.decode.loss_seg'] = Ll_soft_losses_2['Ll_soft_losses2.decode.loss_seg'] * lam 
            Ll_soft_losses =  Ll_soft_losses_2
            #Ll_soft_losses.pop('features')
            #Ll_soft_losses = add_prefix(Ll_soft_losses, 'Ll_soft_losses')
            #Ll_soft_losses2 = self.get_model().forward_train(mixed_img_s, img_metas, gt_semantic_seg)
            Ll_soft_loss1, Ll_soft_log_vars1 = self._parse_losses(Ll_soft_losses_1)
            log_vars.update(Ll_soft_log_vars1)
            Ll_soft_loss1.backward()

            Ll_soft_loss2, Ll_soft_log_vars2 = self._parse_losses(Ll_soft_losses_2)
            log_vars.update(Ll_soft_log_vars2)
            Ll_soft_loss2.backward()

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
