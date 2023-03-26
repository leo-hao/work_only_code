from mmseg.datasets import CityscapesDataset
from mmcv import Config
from mmseg.apis import inference_segmentor, init_segmentor
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy as np

# Load the configuration file
cfg = Config.fromfile('configs/fcn/fcn_r50-d8_512x1024_80k_cityscapes.py')

# Load the model
model = init_segmentor(cfg, cfg.data.test['ann_file'])

# Load the dataset
dataset = CityscapesDataset(cfg.data.test)

# Get the features and labels
features = []
labels = []
for i in range(len(dataset)):
    data = dataset[i]
    result = inference_segmentor(model, data['img'])
    feature = result[0].cpu().numpy()
    feature = np.mean(feature, axis=(1, 2))
    features.append(feature)
    labels.append(data['gt_semantic_seg'].cpu().numpy().flatten())

# Convert the features and labels to numpy arrays
features = np.array(features)
labels = np.array(labels)

# Use t-SNE to reduce the dimensionality of the features
tsne = TSNE(n_components=2, random_state=0)
features_tsne = tsne.fit_transform(features)

# Plot the features and labels
plt.scatter(features_tsne[:, 0], features_tsne[:, 1], c=labels)
plt.colorbar()
plt.show()
