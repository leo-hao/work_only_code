#在使用mmsegmentation进行测试时，以segformer为主干，cityscapes为数据集，怎么获取一张图片的特征，并通过tsne获取各类特征的分布图
'''
在使用mmsegmentation进行测试时，以Segformer为主干，Cityscapes为数据集，可以通过以下步骤获取一张图片的特征，并通过t-SNE获取各类特征的分布图：

加载数据集：首先需要加载Cityscapes数据集，可以使用mmseg.datasets.CityscapesDataset类加载数据集。
'''
from mmseg.datasets import CityscapesDataset

data_root = 'path/to/cityscapes'
img_dir = 'leftImg8bit'
ann_dir = 'gtFine'
split = 'val'

dataset = CityscapesDataset(
    data_root=data_root,
    img_dir=img_dir,
    ann_dir=ann_dir,
    split=split,
    pipeline=[],
    test_mode=True)
'''
加载模型：加载已经训练好的Segformer模型，可以使用mmseg.apis.load_checkpoint函数加载模型参数，然后使用mmseg.models.build_segmentor函数构建模型。
'''
from mmseg.apis import inference_segmentor, init_segmentor
from mmseg.models import build_segmentor

config_file = 'configs/segformer/segformer_mit-b0_512x1024_160k_cityscapes.py'
checkpoint_file = 'checkpoints/segformer_mit-b0_512x1024_160k_cityscapes_20210813_110619-d2152a97.pth'

cfg = mmcv.Config.fromfile(config_file)
cfg.data.test.pipeline = cfg.data.val.pipeline
cfg.data.test.test_mode = True

# build the model from a config file and a checkpoint file
model = build_segmentor(cfg.model)
_ = init_segmentor(model, checkpoint_file, device='cuda:0')

'''
加载测试图片：从Cityscapes数据集中选择一张测试图片，将其转换成模型需要的格式，通常为RGB格式的numpy数组。
'''
img_info = dataset.data_infos[0]
img = dataset.img_norm_cfg.to_tensor(
    dataset.pipeline(dict(img_info=img_info))['img']).unsqueeze(0).cuda()

'''
获取特征：在Segformer模型中，特征图通常是在Encoder部分的输出中获取的。可以通过注册Encoder中特定层的钩子函数来获取特征图，例如在Segformer-Mit-B0模型中，第4个Transformer
block的输出可以作为特征图，代码如下：
'''
features = {}

def get_features(name):
    def hook(model, input, output):
        features[name] = output.detach()
    return hook

model.encoder.layers[4].register_forward_hook(get_features('block4'))
with torch.no_grad():
    _ = model(img)
feature_map = features['block4'][0].cpu().numpy()

#用t-SNE可视化特征分布，及其特征所代表的类
#在获取特征之后，可以使用t-SNE对特征进行降维并可视化，
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

# Reshape feature map into (H*W, C) format
feature_map = feature_map.transpose(1, 2, 0)
feature_map = feature_map.reshape(-1, feature_map.shape[-1])

# Apply t-SNE to reduce feature dimensions to 2D
tsne = TSNE(n_components=2, verbose=1)
features_tsne = tsne.fit_transform(feature_map)

# Plot t-SNE visualization
fig, ax = plt.subplots(figsize=(10, 10))
for idx, class_name in enumerate(dataset.CLASSES):
    indices = dataset.get_cat_ids().index(idx)
    indices = dataset.get_cat_ids().get(indices)
    indices = np.where(dataset.img_ids == indices)[0]
    ax.scatter(features_tsne[indices, 0], features_tsne[indices, 1], label=class_name, s=5)
ax.legend()
plt.show()
