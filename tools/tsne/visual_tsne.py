import mmcv
import torch
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

# 加载模型和数据
cfg = mmcv.Config.fromfile('configs/segformer/segformer_mit-b0_512x512_80k_ade20k.py')
model = build_segmentor(cfg.model)
model.load_state_dict(torch.load('segformer_mit-b0_512x512_80k_ade20k.pth'))
dataset = build_dataset(cfg.data.val)

# 获取特征和标签
data_loader = build_dataloader(
    dataset,
    imgs_per_gpu=1,
    workers_per_gpu=cfg.data.workers_per_gpu,
    dist=False,
    shuffle=False)
model.eval()
features, labels = [], []
with torch.no_grad():
    for i, data in enumerate(data_loader):
        img = data['img'].cuda()
        label = data['gt_semantic_seg'].cuda()
        feat = model.extract_feat(img)
        features.append(feat.cpu().numpy())
        labels.append(label.cpu().numpy())
features = np.vstack(features)
labels = np.concatenate(labels)

# 对特征进行降维
tsne = TSNE(n_components=2)
features_tsne = tsne.fit_transform(features)

# 绘制特征分布图
num_classes = cfg.model.decode_head.num_classes
colors = plt.cm.hsv(np.linspace(0, 1, num_classes))
plt.figure(figsize=(10, 10))
for i in range(num_classes):
    plt.scatter(features_tsne[labels == i, 0], features_tsne[labels == i, 1], color=colors[i], s=10, label=str(i))
plt.legend()
plt.show()
