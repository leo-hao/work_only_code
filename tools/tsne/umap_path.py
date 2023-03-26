import mmcv
from mmseg.apis import inference_segmentor, init_segmentor
from mmcv.parallel import collate, scatter
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import umap

# initialize the segformer model
config_file = 'configs/segformer/segformer_mit-b0_512x512_160k_ade20k.py'
checkpoint_file = 'checkpoints/segformer_mit-b0_512x512_160k_ade20k_20210615_155745-1c4f0fa5.pth'
model = init_segmentor(config_file, checkpoint_file, device='cuda:0')

# load the cityscapes validation dataset
data_root = 'data/cityscapes/'
ann_file = data_root + 'annotations/instancesonly_filtered_gtFine_val.json'
img_prefix = data_root + 'leftImg8bit/val/'
dataset = mmcv.load(ann_file)['images']
dataset = [dict(img_prefix=img_prefix, img_info=img_info) for img_info in dataset]
dataset = dataset[:200]  # only use the first 200 images for demonstration purposes
dataloader = DataLoader(dataset, batch_size=16, shuffle=False, num_workers=4, collate_fn=collate)

# extract neck features from the images
neck_features = []
for data in tqdm(dataloader):
    with torch.no_grad():
        result = inference_segmentor(model, data['img'][0])
        neck_feature = result['neck_feature'].cpu().numpy()
        neck_features.append(neck_feature)
neck_features = np.concatenate(neck_features, axis=0)

# use umap to visualize the feature distribution
reducer = umap.UMAP()
embedding = reducer.fit_transform(neck_features)
plt.scatter(embedding[:, 0], embedding[:, 1], c=np.arange(len(embedding)), cmap='rainbow')
plt.colorbar()
plt.show()
