首先用mmseg里面的segformer，
然后用dataloader读取200张cityscapes验证集图片，
接着将所有特征保存到feature中，
最后用umap绘制特征的分布图，其中不同的特征用不同的颜色表示