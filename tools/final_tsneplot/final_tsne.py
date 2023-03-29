import torch
import mmcv
from mmcv import Config
from mmseg.apis import inference_segmentor, init_segmentor
from sklearn.manifold import TSNE
import numpy as np
import matplotlib.pyplot as plt
from mmseg.datasets import build_dataloader, build_dataset
from sklearn.decomposition import PCA
from MulticoreTSNE import MulticoreTSNE 

# initialize the segmentor
cfg = Config.fromfile('configs/segformer/segformer_mit-b0_8x1_1024x1024_160k_cityscapes.py')
checkpoint_file = 'checkpoint/segformer_mit-b0_8x1_1024x1024_160k_cityscapes_20211208_101857-e7f88502.pth'
model = init_segmentor(cfg, checkpoint_file, device='cuda:0')

# load the dataset
data_dir = 'data/cityscapes/leftImg8bit/val'
#img_infos = mmcv.scan_images(data_dir, recursive=True)
dataset = build_dataset(cfg.data.val)
dataloader = build_dataloader(
    dataset,
    samples_per_gpu=1,
    workers_per_gpu=1,
    dist=False,
    shuffle=False)

device = torch.device("cuda:0" if torch.cuda.is_available() else 'cpu')
model.eval()
features = {}
targets = {}
model.to(device)
feat_maps = np.empty([0, 19,1024,2048])
labels = np.empty([0, 1024, 2048])
#targets['train'] = np.empty([0, ]) 
for i, data in enumerate(dataloader):
  #if i == 1:
  #  break
  img_metas = data['img_metas'][0].data
  images = data['img'][0].data[0].unsqueeze(0).to(device)
  #print(images.shape)#torch.Size([1, 3, 1024, 1024])/val torch.Size([1, 3, 1024, 2048])
  #target = data['gt_semantic_seg'].data[0].to(device)
  #print(target)
  #print(target.shape)#torch.Size([1, 1, 1024, 1024])
  img_path = data['img_metas'][0].data[0][0].get('filename')
  #feat = model.extract_feat(images)
  #print(feat[0].shape)
  feat_map = model.encode_decode(images, img_metas)
  #print(feat_map.shape)#val:torch.Size([1, 19, 1024, 2048])
  #feat_map = out.squeeze()
  feat_map = feat_map.cpu().detach().numpy()
  #feat_map = (feat_map - np.mean(feat_map, axis=0)) / np.std(feat_map,axis=0)
  #feat_map = feat_map.reshape(19,-1)
  feat_maps = np.append(feat_maps, feat_map, axis=0)
  #print("out.shape:", out.shape)#out.shape: torch.Size([1, 19, 1024, 1024])
  #print("out:", out)
  #print('feat[0].shape:',feat[0].shape)#feat[0].shape: torch.Size([1, 32, 256, 256])
  #print('feat[1].shape:',feat[1].shape)#feat[1].shape: torch.Size([1, 64, 128, 128])
  #print('feat[2].shape:',feat[2].shape)#feat[2].shape: torch.Size([1, 160, 64, 64])
  #print('feat[3].shape:',feat[3].shape)#feat[3].shape: torch.Size([1, 256, 32, 32])
  #print(model.num_classes)
  label = inference_segmentor(model, img_path)
  labels = np.append(labels, label, axis=0)
  #print(output[0].shape)#(1024, 2048)
  #print(output)
  #print(output)
  #result_flat = output[0].flatten()
  # tsne = TSNE(n_components=2, random_state=0)
  # result_tsne = tsne.fit_transform(result_flat.reshape(-1, 1))
  # plt.scatter(result_tsne[:, 0], result_tsne[:, 1], c=result_flat, cmap='tab20')
  # plt.show()
  #print(output[0].shape)#(1024, 2048)
  #print(features['train'].shape)
  #features['train'] = np.append(features['train'],output[0], axis=0)
  #targets['train'] = np.append(targets['train'],target.cpu(), axis=0)
  #print(features['train'].shape)
  if i == 9:
    break

from skimage.transform import resize
from skimage.measure import block_reduce

# features_downsampled = np.zeros((feat_maps.shape[0], 19, 32, 64))
# for i in range(feat_maps.shape[0]):
#    for j in range(feat_maps.shape[1]):
#       features_downsampled[i][j] = resize(feat_maps[i, j], (32, 64),preserve_range=True)
features_downsampled = block_reduce(feat_maps, block_size=(1,1,32,32), func=np.mean)
#print(features_downsampled.shape)
#print(features_downsampled)

# labels_downsampled = np.zeros((labels.shape[0], 32, 64))
# for i in range(labels.shape[0]):
#    labels_downsampled[i] = resize(labels[i], (32, 64),preserve_range=True)
labels_downsampled = block_reduce(labels, block_size=(1,32,32), func=np.mean)
labels_downsampled =labels_downsampled.astype(np.int)
#print(labels_downsampled.shape)
#print(labels_downsampled)      
x= np.reshape(features_downsampled, (-1,19))

y = np.reshape(labels_downsampled, (-1,))
#x = x[1000000:1200000]
#y = y[1000000:1200000]
tsne = MulticoreTSNE(n_components=2,perplexity=10,learning_rate=100,n_jobs=4)
X_tsne = tsne.fit_transform(x)
PALETTE = np.array([[128, 64, 128], [244, 35, 232], [70, 70, 70], [102, 102, 156],
            [190, 153, 153], [153, 153, 153], [250, 170, 30], [220, 220, 0],
            [107, 142, 35], [152, 251, 152], [70, 130, 180], [220, 20, 60],
            [255, 0, 0], [0, 0, 142], [0, 0, 70], [0, 60, 100],
            [0, 80, 100], [0, 0, 230], [119, 11, 32]])
colors  = ['#%02x%02x%02x' % (r,g,b) for r,g,b in PALETTE]

fig, ax = plt.subplots(figsize=(19.20,10.80))
for i in range(19):
    indices = np.where(y == i)[0]
    ax.scatter(X_tsne[indices,0], X_tsne[indices,1], c=colors[i],label=str(i))
ax.legend()
plt.savefig('tsne.png',dpi=100)
plt.show()