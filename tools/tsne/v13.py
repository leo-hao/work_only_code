import mmcv
import torch
import numpy as np
from mmseg.apis import inference_segmentor, init_segmentor
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

# 设置配置文件和权重文件路径
config_file = 'configs/segformer/segformer_mit-b0_512x512_160k_ade20k.py'
checkpoint_file = 'checkpoints/segformer_mit-b0_512x512_160k_ade20k-b023b197.pth'

# 初始化分割器
model = init_segmentor(config_file, checkpoint_file, device='cuda:0')

# 加载验证集
data_info = mmcv.load('data/cityscapes/annotations/val.json')
class_names = model.CLASSES
num_classes = len(class_names)
color_map = model.PALETTE

# 定义特征提取函数
def extract_features(img, model):
    with torch.no_grad():
        result = inference_segmentor(model, img)
    return result[0]

# 提取所有验证集图像的特征
features = []
for i in range(len(data_info['images'])):
    img_path = 'data/cityscapes/' + data_info['images'][i]['file_name']
    img = mmcv.imread(img_path)
    feature = extract_features(img, model)
    features.append(feature.flatten())
features = np.array(features)

# 对特征进行降维并可视化
tsne = TSNE(n_components=2, init='pca', random_state=0)
features_tsne = tsne.fit_transform(features)

# 创建图例和颜色列表
label2color = {i: color_map[i] for i in range(num_classes)}
colors = [label2color[label] for label in data_info['annotations'][0]['segmentation'][0]]

# 绘制特征分布图
fig, ax = plt.subplots(figsize=(10, 10))
for label in range(num_classes):
    idxs = [i for i in range(len(data_info['annotations'])) if data_info['annotations'][i]['category_id'] == label]
    color = label2color[label]
    ax.scatter(features_tsne[idxs, 0], features_tsne[idxs, 1], c=color, label=class_names[label])
ax.legend(loc='best', fontsize=12)
plt.show()
