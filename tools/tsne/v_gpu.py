import torch
from torchvision import transforms
from torch.utils.data import DataLoader
from mmseg.datasets import CityscapesDataset
from mmseg.models import build_segmentor
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy as np

# set device to cuda if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# load SegFormer model with pre-trained weights
config_file = 'configs/segformer/segformer_mit-b0_512x1024_80k_cityscapes.py'
checkpoint_file = 'checkpoints/segformer_mit-b0_512x1024_80k_cityscapes_20210817_200958-2f322e58.pth'
model = build_segmentor(config_file, test_cfg=None)
model.to(device)
model.eval()
model.load_state_dict(torch.load(checkpoint_file, map_location=device)['state_dict'])

# define data transform
data_transforms = transforms.Compose([
    transforms.Resize((512, 1024)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# load Cityscapes dataset
dataset = CityscapesDataset(
    data_root='data/cityscapes',
    ann_file='data/cityscapes/annotations/val.json',
    pipeline=[],
    img_prefix='data/cityscapes/leftImg8bit/val',
    seg_prefix='data/cityscapes/gtFine/val',
    data_cfg=dict(split='val'),
    test_mode=True
)

# define data loader
data_loader = DataLoader(dataset, batch_size=10, shuffle=False)

# get features and labels for each image
features_all = []
labels_all = []
with torch.no_grad():
    for data in data_loader:
        img = data['img'].to(device)
        labels = data['gt_semantic_seg'].to(device)
        _, features = model.extract_feat(img)
        features_all.append(features.cpu().numpy())
        labels_all.append(labels.cpu().numpy())
features_all = np.concatenate(features_all, axis=0)
labels_all = np.concatenate(labels_all, axis=0)

# reduce features dimension using t-SNE
tsne = TSNE(n_components=2, perplexity=30, n_iter=1000, learning_rate=200, init='pca')
features_tsne = tsne.fit_transform(features_all)

# plot features distribution
fig, ax = plt.subplots(figsize=(10, 10))
for i in range(19):
    idx = np.where(labels_all == i)
    ax.scatter(features_tsne[idx, 0], features_tsne[idx, 1], label=dataset.CLASSES[i], s=10)
ax.legend()
plt.show()
