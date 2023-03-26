# Welcome to Cursor


from mmseg.apis import inference_segmentor, init_segmentor
from mmcv import Config
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from PIL import Image

config_file = 'configs/segformer/segformer_mit-b0_512x512_160k_ade20k.py'
checkpoint_file = 'checkpoints/segformer_mit-b0_512x512_160k_ade20k_20210615_233955-9c7f4bd6.pth'
img_dir = 'data/cityscapes/leftImg8bit/val/'
ann_dir = 'data/cityscapes/annotations/val/'
img_list = 'data/cityscapes/lists/val.lst'
palette = np.array([
    [128, 64, 128],
    [244, 35, 232],
    [70, 70, 70],
    [102, 102, 156],
    [190, 153, 153],
    [153, 153, 153],
    [250, 170, 30],
    [220, 220, 0],
    [107, 142, 35],
    [152, 251, 152],
    [70, 130, 180],
    [220, 20, 60],
    [255, 0, 0],
    [0, 0, 142],
    [0, 0, 70],
    [0, 60, 100],
    [0, 80, 100],
    [0, 0, 230],
    [119, 11, 32]], dtype=np.uint8)

# build the model from a config file and a checkpoint file
cfg = Config.fromfile(config_file)
cfg.model.pretrained = None
model = init_segmentor(cfg, checkpoint_file, device='cuda:0')

# get the features of the 200 images
features = []
for i in range(200):
    img_path = img_dir + f'{i:06d}_leftImg8bit.png'
    ann_path = ann_dir + f'{i:06d}_gtFine_labelIds.png'
    result = inference_segmentor(model, img_path)
    feature = result[0].cpu().numpy()
    label = np.array(Image.open(ann_path))
    feature = feature[label !=255
# generate tsne plot
tsne = TSNE(n_components=2, random_state=0)
features_tsne = tsne.fit_transform(features)

# plot the tsne result
fig, ax = plt.subplots(figsize=(10, 10))
scatter = ax.scatter(features_tsne[:, 0], features_tsne[:, 1], c=label.flatten(), s=1, cmap='tab20')
legend1 = ax.legend(*scatter.legend_elements(), loc="lower left", title="Classes")
ax.add_artist(legend1)
plt.show()

# 1. Try generating with command K on a new line. Ask for a pytorch script of a feedforward neural network
# 2. Then, select the outputted code and hit chat. Ask if there's a bug. Ask how to improve.
# 3. Try selecting some code and hitting edit. Ask the bot to add residual layers.
# 4. To try out cursor on your own projects, go to the file menu (top left) and open a folder.

