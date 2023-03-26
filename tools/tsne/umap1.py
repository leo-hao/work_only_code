#在mmcv的基础上，以segformer为主干，用dataloader，用umap可视化cityscapes的200张验证集图片的特征分布，并且不同的类用不同颜色表示
from mmcv import Config
from mmseg.apis import inference_segmentor, init_segmentor
from mmseg.core.evaluation import get_palette
import numpy as np
import cv2
import umap
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# Load the configuration file
cfg = Config.fromfile('configs/segformer/segformer_mit-b0_512x512_160k_ade20k.py')

# Load the model
model = init_segmentor(cfg, checkpoint='checkpoints/segformer_mit-b0_512x512_160k_ade20k_20210615_190543-1c4f8f5e.pth')

# Load the validation dataset
dataset = model.cfg.data['val']
#palette = get_palette(dataset)
palette = get_palette('cityscapes')

# Load the validation data
data_root = dataset['data_root']
ann_file = dataset['ann_file']
img_dir = dataset['img_dir']
#seg_map_dir = dataset['seg_map_dir']
with open(ann_file, 'r') as f:
    anns = f.readlines()

# Select 200 images randomly
np.random.seed(0)
anns = np.random.choice(anns, 200)

# Extract features from the selected images
features = []
for ann in anns:
    ann = ann.strip().split()
    img_path = data_root + '/' + ann[0]
    img = cv2.imread(img_path)
    result = inference_segmentor(model, img)
    feature = result['feat'][0].cpu().numpy()
    features.append(feature.flatten())
features = np.array(features)

# Standardize the features
scaler = StandardScaler()
features = scaler.fit_transform(features)

# Reduce the dimensionality of the features
pca = PCA(n_components=50)
features = pca.fit_transform(features)

# Visualize the features using UMAP
reducer = umap.UMAP(n_components=2, random_state=0)
embedding = reducer.fit_transform(features)

# Visualize the embeddings
fig, ax = plt.subplots(figsize=(10, 10))
for i in range(len(anns)):
    ann = anns[i].strip().split()
    img_path = data_root + '/' + ann[0]
    img = cv2.imread(img_path)
    h, w, _ = img.shape
    embedding_i = embedding[i]
    label = int(ann[1])
    color = palette[label]
    ax.scatter(embedding_i[0], embedding_i[1], color=color, s=5)
ax.set_title('UMAP Visualization of Cityscapes Validation Set Features')
plt.show()
