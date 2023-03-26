from mmcv import Config
from mmseg.datasets import build_dataloader, build_dataset
from mmcv.runner.iter_based_runner import IterBasedRunner, IterLoader
def main():

    cfg = Config.fromfile('configs/daformer/gta2cs_uda_warm_fdthings_rcs_croppl_a999_daformer_mitb5_s0.py')
    #print(f'Config:\n{cfg.pretty_text}')
    #print(cfg.data.samples_per_gpu)
    dataset = [build_dataset(cfg.data.train)]
    
    data_loaders = [
        build_dataloader(
            ds,
            cfg.data.samples_per_gpu,
            0,#cfg.data.workers_per_gpu
            # cfg.gpus will be ignored if distributed
            1,#num_gpus
            dist=False,
            seed=cfg.seed,
            drop_last=True) for ds in dataset
    ]

    iter_loaders = [IterLoader(x) for x in data_loaders]
    data_loader= iter_loaders[0]
    data_batch = next(data_loader)
    for a, b in data_batch.items():
        print('a:', a)
        print('b:', b)

    # iter_loaders = [IterLoader(x) for x in data_loaders]
    # data_loader= iter_loaders[0]
    # for i, data_batch in enumerate(data_loader):
    #     for a, b in data_batch.items():
    #         print(a)
    #         print(b)

if __name__ == '__main__':
    main()