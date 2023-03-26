下面的代码在result = seg_model(return_loss=False, img=[img], img_metas=[{'ori_shape': img.shape处，报错：AttributeError:’list' object has no attribute 'shape'
请帮我改正
import mmcv
import numpy as np
from sklearn.manifold import TSNE

# Load the segformer model
from mmcv.runner import load_checkpoint
from mmseg.models import build_segmentor
config_file = 'configs/segformer/segformer_mit-b0_512x512_160k_ade20k.py'
checkpoint_file = 'checkpoints/segformer_mit-b0_512x512_160k_ade20k_20210726_215644-c8c8f000.pth'
cfg = mmcv.Config.formfile(config_file)
seg_model = build_segmentor(cfg.model, train_cfg=None, test_cfg=None)
checkpoint = load_checkpoint(seg_model, checkpoint_file)

# Load the Cityscapes validation dataset
from mmseg.datasets import build_dataset
dataset = build_dataset(cfg.data.val)

# Select a random image from the dataset
idx = np.random.randint(len(dataset))
img = dataset[idx]['img']

# Extract features from the image using the segformer model
with torch.no_grad():
    result = seg_model(return_loss=False, img=[img], img_metas=[{'ori_shape': img.shape, 'img_shape': img.shape,
                                                                  'pad_shape': img.shape, 'filename': ''}])
features = result[0]

# Use t-SNE to visualize the feature distribution
tsne = TSNE(n_components=2, random_state=0)
tsne_features = tsne.fit_transform(features.reshape(features.shape[0], -1))

# Plot the t-SNE features
import matplotlib.pyplot as plt
plt.scatter(tsne_features[:, 0], tsne_features[:, 1])
plt.show()
