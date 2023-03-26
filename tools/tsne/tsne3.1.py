import mmcv
from mmseg.apis import inference_segmentor, init_segmentor
from sklearn.manifold import TSNE
import numpy as np
import matplotlib.pyplot as plt

# initialize the segmentor
config_file = 'configs/fcn/fcn_r50-d8_512x1024_80k_cityscapes.py'
checkpoint_file = 'checkpoints/fcn_r50-d8_512x1024_80k_cityscapes_20200605_094551-4dcb1394.pth'
segmentor = init_segmentor(config_file, checkpoint_file, device='cuda:0')

# load the dataset
data_dir = 'data/cityscapes/leftImg8bit/val'
img_infos = mmcv.scan_images(data_dir, recursive=True)

# randomly select 200 images
np.random.seed(0)
img_infos = np.random.choice(img_infos, size=200, replace=False)

# extract features from the images
features = []
for img_info in img_infos:
    img = mmcv.imread(img_info['filename'])
    result = inference_segmentor(segmentor, img)
    feature = result[0].cpu().numpy().flatten()
    features.append(feature)
features = np.array(features)

# apply t-SNE to reduce the dimensionality of the features
tsne = TSNE(n_components=2, perplexity=30, learning_rate=200, n_iter=1000, random_state=0)
features_tsne = tsne.fit_transform(features)

# plot the feature distribution
plt.scatter(features_tsne[:, 0], features_tsne[:, 1])
plt.show()

# free the GPU memory
segmentor = None
