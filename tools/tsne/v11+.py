#ValueError:Found array with 1 sample(s) (shape=(1, 2036161))while a minimum of 2 is requiered.

import torch
from mmseg.models import build_segmentor
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

# 加载模型配置文件并构建模型
cfg = torch.load('segformer_config.pth')
model = build_segmentor(cfg.model)

# 加载预训练权重
checkpoint = torch.load('segformer_checkpoint.pth')
model.load_state_dict(checkpoint['state_dict'])

# 将模型设置为评估模式并移至GPU
model.eval()
model.cuda()

# 加载图像并进行预处理
image = cv2.imread('image.jpg')
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
image = image.transpose(2, 0, 1)
image = torch.from_numpy(image).unsqueeze(0).float().cuda()

# 获取所有特征图和推理结果
with torch.no_grad():
    features = model.backbone(image)
    logits = model.decode_head.forward(*features)
    preds = logits.argmax(dim=1)

# 将所有特征图拼接在一起并降维到二维空间
features_concatenated = torch.cat([f.flatten(1) for f in features], dim=1)
features_embedded = TSNE(n_components=2).fit_transform(features_concatenated.cpu())

# 绘制特征分布图，每类特征用不同的颜色表示
plt.scatter(features_embedded[:, 0], features_embedded[:, 1], c=preds.squeeze().cpu())
plt.show()