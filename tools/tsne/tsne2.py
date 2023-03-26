from mmcv import Config
from mmseg.apis import inference_segmentor, init_segmentor
from mmseg.datasets import build_dataloader, build_dataset
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy as np

# 1. 使用mmcv的方式
cfg = Config.fromfile('configs/segformer/segformer_mit-b0_512x512_160k_ade20k.py')

# 2. 基于mmsegmentation
model = init_segmentor(cfg, checkpoint='checkpoints/segformer_mit-b0_512x512_160k_ade20k_20210614_233347-2c4d3d9a.pth')

# 3. 以segformer为主干
backbone = model.backbone

# 4. 使用dataloader读取cityscapes的验证集200张图片
dataset = build_dataset(cfg.data.val)
dataloader = build_dataloader(
    dataset,
    samples_per_gpu=1,
    workers_per_gpu=1,
    dist=False,
    shuffle=False)

# 5. 用tsne生成cityscpaes的19个类的特征分布图
features = []
labels = []
for i in range(200):
    for data in dataloader:
        img = data['img'][0].cuda()
        label = data['gt_semantic_seg'][0].numpy()
        with torch.no_grad():
            feature = backbone(img).squeeze().cpu().numpy()
        features.append(feature)
        labels.append(label)
features = np.array(features)
labels = np.array(labels)
features = features.reshape(features.shape[0], -1)
tsne = TSNE(n_components=2, random_state=0)
features_tsne = tsne.fit_transform(features)

# 6. 不同类的特征用不同的颜色表示，图例为不同的颜色的
colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'w', 'orange', 'purple', 'brown', 'pink', 'gray', 'olive', 'navy', 'teal', 'gold', 'indigo', 'maroon']
for i in range(19):
    plt.scatter(features_tsne[labels==i, 0], features_tsne[labels==i, 1], c=colors[i], label=str(i))

plt.show()
