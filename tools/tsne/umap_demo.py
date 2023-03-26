import mmcv
from mmseg.apis import inference_segmentor, init_segmentor
import numpy as np
import umap
import matplotlib.pyplot as plt

# initialize the model
config_file = 'configs/fcn/fcn_r50-d8_512x1024_80k_cityscapes.py'
checkpoint_file = 'checkpoints/fcn_r50-d8_512x1024_80k_cityscapes_20200605_094551-4dcb139a.pth'
model = init_segmentor(config_file, checkpoint_file, device='cuda:0')

# load an image
img = mmcv.imread('demo/demo.png')

# inference the image
result = inference_segmentor(model, img)

# get the feature map
feature_map = model.extract_feat(img)

# reshape the feature map
feature_map = feature_map.cpu().numpy()
feature_map = np.transpose(feature_map, (1, 2, 0))
feature_map = feature_map.reshape(-1, feature_map.shape[-1])

# reduce the dimensionality of the feature map
reducer = umap.UMAP(n_components=2)
embedding = reducer.fit_transform(feature_map)

# plot the embedding
plt.scatter(embedding[:, 0], embedding[:, 1], c=result[0], cmap='jet')
plt.colorbar()
plt.show()