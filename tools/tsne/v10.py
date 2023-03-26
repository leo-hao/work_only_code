以下代码报错：RuntimeError:Given groups = 1, weight of size[32,3,7,7],expected input[1,1024,2048,3] to have 3 channels , but got 1024 channels instead
import torch
from mmcv import Config
from mmseg.apis import inference_segmentor, init_segmentor
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

# 加载配置文件
cfg = Config.fromfile('configs/segformer/segformer_mit-b0_512x512_160k_cityscapes.py')
# 创建模型并加载预训练权重
model = init_segmentor(cfg, checkpoint='checkpoints/segformer_mit-b0_512x512_160k_cityscapes_20211026_195546-423a1695.pth', device='cuda:0')

# 加载Cityscapes验证集
val_data = cfg.data['val']
data_root = val_data['data_root']
ann_file = val_data['ann_file']
img_dir = f'{data_root}/{val_data["img_dir"]}'
with open(f'{data_root}/{ann_file}', 'r') as f:
    anns = f.readlines()

# 随机选择一张图像
rand_idx = torch.randint(low=0, high=len(anns), size=(1,))
img_info = anns[rand_idx]

# 获取图像路径
img_path = f'{img_dir}/{img_info.split()[0]}'
# 读取图像
img = Image.open(img_path).convert('RGB')
# 进行推理得到每个像素点的类别
result = inference_segmentor(model, img)

# 获取所有特征，并且将形状从 [C, H, W] 转换为 [H*W, C]
features = result[0].permute(1, 2, 0).reshape(-1, result.shape[1])

# 获取每个像素点的类别
class_ids = result[0].argmax(dim=0).flatten()

# 使用t-SNE算法将所有特征点映射到2D平面
tsne = TSNE(n_components=2, random_state=0)
features_tsne = tsne.fit_transform(features)

# 创建画布，并绘制所有特征点的散点图
fig, ax = plt.subplots(figsize=(8, 8))
scatter = ax.scatter(features_tsne[:, 0], features_tsne[:, 1], c=class_ids, cmap='tab20')

# 添加图例，表示每个颜色对应的类别
classes = cfg.data['classes']
legend1 = ax.legend(*scatter.legend_elements(),
                    loc="upper right", title="Classes", labels=[classes[i] for i in range(len(classes))])
ax.add_artist(legend1)
plt.show()
