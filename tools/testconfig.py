from mmcv import Config
from mmseg.datasets import build_dataloader, build_dataset
def main():

    cfg = Config.fromfile('../configs/daformer/gta2cs_uda_warm_fdthings_rcs_croppl_a999_daformer_mitb5_s0.py')
    #print(f'Config:\n{cfg.pretty_text}')
    dataset = build_dataset(cfg.data.train)
    
    data_loaders = [
        build_dataloader(
            dataset,
            cfg.data.samples_per_gpu,
            0,
            # cfg.gpus will be ignored if distributed
            #len(cfg.gpu_ids),
            dist=False,
            seed=cfg.seed,
            drop_last=True) 
    ]
    data_loaders_iter = iter(data_loaders)
    data_batch = next(data_loaders_iter)
    images, labels, _, _ = data_batch
    print(images)
    print(labels)
    for a, b in data_batch.items():
        print('a:', a)
        print('b:', b)
if __name__ == '__main__':
    main()