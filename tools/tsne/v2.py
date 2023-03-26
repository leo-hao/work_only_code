import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from mmcv import Config
from mmcv.parallel import collate
from mmcv.runner import load_checkpoint, obj_from_dict

from mmseg.apis import inference_segmentor, init_segmentor

# 读取配置文件
cfg = Config.fromfile('configs/segformer/segformer_mit-b0_512x1024_80k_cityscapes.py')

# 构建模型
model = obj_from_dict(cfg.model, nn.Module)
checkpoint = load_checkpoint(model, 'checkpoint.pth') # 加载模型参数
model.CLASSES = checkpoint['meta']['CLASSES'] # 设置模型的类别数
model = model.cuda().eval()

# 初始化验证数据集
data_cfg = cfg.data.val
dataset_type = data_cfg.pop('type')
dataset = obj_from_dict(data_cfg, datasets)
data_loader = build_dataloader(dataset, samples_per_gpu=1, workers_per_gpu=cfg.data.workers_per_gpu, dist=False, shuffle=False)

# 读取10张图片的特征
features = []
labels = []
count = 0
for data in data_loader:
    with torch.no_grad():
        result = inference_segmentor(model, data['img'][0].cuda())
        features.append(result['feat'].cpu().numpy())
        labels.append(data['img_meta'][0]['ori_filename'])
        count += 1
        if count >= 10:
            break

features = np.concatenate(features)
labels = np.array(labels)

# 对特征进行t-SNE降维
tsne = TSNE(n_components=2, random_state=0)
features_tsne = tsne.fit_transform(features)

# 绘制散点图
plt.figure(figsize=(10, 10))
for i, filename in enumerate(labels):
    plt.scatter(features_tsne[i, 0], features_tsne[i, 1], label=filename, s=10)
plt.legend()
plt.show()
