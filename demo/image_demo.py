# Obtained from: https://github.com/open-mmlab/mmsegmentation/tree/v0.16.0
# Modifications:
# - Config and checkpoint update
# - Saving instead of showing prediction

#代码报错：RuntimeError:Given groups=1, weight of size [64,3,7,7],expected input[1, 512,1024,3] to have 3 channels, but got 512 channels instead
请按以下要求，替我完成代码：
1. 基于mmsegmentation
2. 以segformer为主干
3. 使用cityscapes的验证集
4. 随机选择其中一张图的,获取它的特征
5. 推理这张图每个特征所代表的类别
6. 用tsne生成特征分布图，并用不同颜色表示不同类的特征

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from argparse import ArgumentParser

import mmcv
from tools.test import update_legacy_cfg

from mmseg.apis import inference_segmentor, init_segmentor
from mmseg.core.evaluation import get_classes, get_palette
import torch
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy as np
from mmcv import imread
def main():
    parser = ArgumentParser()
    parser.add_argument('img', help='Image file')
    parser.add_argument('config', help='Config file')
    parser.add_argument('checkpoint', help='Checkpoint file')
    parser.add_argument(
        '--device', default='cuda:0', help='Device used for inference')
    parser.add_argument(
        '--palette',
        default='cityscapes',
        help='Color palette used for segmentation map')
    parser.add_argument(
        '--opacity',
        type=float,
        default=1,
        help='Opacity of painted segmentation map. In (0, 1] range.')
    args = parser.parse_args()

    # build the model from a config file and a checkpoint file
    cfg = mmcv.Config.fromfile(args.config)
    cfg = update_legacy_cfg(cfg)
    model = init_segmentor(
        cfg,
        args.checkpoint,
        device=args.device,
        classes=get_classes(args.palette),
        palette=get_palette(args.palette),
        revise_checkpoint=[(r'^module\.', ''), ('model.', '')])
    img = imread(args.img)
    img = torch.from numpy(img).unsqueeze(0).to(args.device)

    # 获取特征
    with torch.no_grad():
        x = model.backbone(img)
        features = x['blocks'][-1].squeeze()

    # 使用TSNE将特征降维
    tsne = TSNE(n_components=2, init='pca', random_state=0)
    X_tsne = tsne.fit_transform(features.numpy().T)

    # 绘制特征分布图
    plt.figure(figsize=(10, 10))
    plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=np.arange(features.shape[0]), cmap=plt.cm.get_cmap('tab20', features.shape[0]))
    plt.colorbar()

    plt.show()

    # test a single image
    result = inference_segmentor(model, args.img)
    # show the results
    file, extension = os.path.splitext(args.img)
    pred_file = f'{file}_pred{extension}'
    assert pred_file != args.img
    model.show_result(
        args.img,
        result,
        palette=get_palette(args.palette),
        out_file=pred_file,
        show=False,
        opacity=args.opacity)
    print('Save prediction to', pred_file)


if __name__ == '__main__':
    main()
# Copyright (c) OpenMMLab. All rights reserved.
# from argparse import ArgumentParser

# from mmseg.apis import inference_segmentor, init_segmentor, show_result_pyplot
# from mmseg.core.evaluation import get_palette


# def main():
#     parser = ArgumentParser()
#     parser.add_argument('img', help='Image file')
#     parser.add_argument('config', help='Config file')
#     parser.add_argument('checkpoint', help='Checkpoint file')
#     parser.add_argument(
#         '--device', default='cuda:0', help='Device used for inference')
#     parser.add_argument(
#         '--palette',
#         default='cityscapes',
#         help='Color palette used for segmentation map')
#     parser.add_argument(
#         '--opacity',
#         type=float,
#         default=0.5,
#         help='Opacity of painted segmentation map. In (0, 1] range.')
#     args = parser.parse_args()

#     # build the model from a config file and a checkpoint file
#     model = init_segmentor(args.config, args.checkpoint, device=args.device)
#     # test a single image
#     result = inference_segmentor(model, args.img)
#     # show the results
#     show_result_pyplot(
#         model,
#         args.img,
#         result,
#         get_palette(args.palette),
#         opacity=args.opacity)


# if __name__ == '__main__':
#     main()
