import torch
from mmseg.models import build_segmentor
from mmcv import Config
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy as np

# 加载配置文件
cfg = Config.fromfile('configs/segformer/segformer_mit-b0_512x1024_80k_cityscapes.py')

# 创建模型
model = build_segmentor(cfg.model)

# 加载权重
model.load_state_dict(torch.load('checkpoints/segformer_mit-b0_512x1024_80k_cityscapes_20211014_103745-9ba99c1e.pth', map_location='cpu')['state_dict'])

# 设置模型为评估模式
model.eval()

# 加载测试图片
img = 'tests/data/cityscapes/leftImg8bit/train/aachen/aachen_000000_000019_leftImg8bit.png'

# 将图片转换为tensor格式
img = torch.from_numpy(np.expand_dims(np.transpose(img, (2, 0, 1)), axis=0)).float()

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
