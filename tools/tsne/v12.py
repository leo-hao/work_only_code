请按以下要求，替我完成代码：
1. 使用mmcv的方式
2. 基于mmsegmentation
3. 以segformer为主干
4. 使用dataloader读取cityscapes的验证集200张图片，及其标签
5. 用tsne生成cityscpaes的19个类的特征分布图
6. 不同类的特征用不同的颜色表示，图例为不同的颜色的
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from mmseg.apis import inference_segmentor, init_segmentor
from sklearn.manifold import TSNE
from mmseg.datasets.builder import DATASETS

# 加载Cityscapes验证集的配置文件
cfg = 'configs/segformer/segformer_mit-b0_512x1024_160k_cityscapes.py'

# 加载模型
model = init_segmentor(cfg, checkpoint='checkpoints/segformer_mit-b0_512x1024_160k_cityscapes_20210803_221228-4d235b23.pth')

# 加载Cityscapes数据集的验证集
cityscapes = DATASETS.get('CityscapesDataset')
data_root = 'data/cityscapes'
split = 'val'
cityscapes_val = cityscapes(data_root, split=split, mode='fine', **cityscapes.CLASSES)

# 保存所有图片的最后一层特征和标签
features = []
labels = []
for idx in range(len(cityscapes_val)):
    img_info = cityscapes_val[idx]
    result = inference_segmentor(model, img_info['img'])
    feature = result['seg_preds'][-1]
    feature = np.squeeze(feature)
    label = img_info['gt_label']
    feature = np.reshape(feature, (-1, feature.shape[2]))
    features.append(feature)
    labels.append(label)
features = np.concatenate(features, axis=0)
labels = np.concatenate(labels, axis=0)

# 用t-SNE算法生成特征分布图
tsne = TSNE(n_components=2, random_state=0)
features_tsne = tsne.fit_transform(features)

# 绘制特征分布图
colors = plt.cm.tab20(np.linspace(0, 1, 20))
plt.figure(figsize=(10, 10))
for i in range(cityscapes.CLASSES):
    plt.scatter(features_tsne[labels == i, 0], features_tsne[labels == i, 1], c=colors[i], label=cityscapes.CLASSES[i])
plt.legend()
plt.savefig('cityscapes_tsne.png')
