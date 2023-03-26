# 只用简单的数据增强的域适方法,没有FD和rare_sampling
# 用于final daformer的注释配置文件。
_base_ = [
    '../_base_/default_runtime.py',                             # 一些运行时的配置：log、resume、checkpoint
    # DAFormer Network Architecture
    '../_base_/models/upernet_beit.py',
    # GTA->Cityscapes Data Loading
    '../_base_/datasets/uda_gta_to_cityscapes_512x512.py',
    # Basic UDA Self-Training
    '../_base_/uda/dacs.py',
    # AdamW Optimizer
    '../_base_/schedules/adamw.py',
    # Linear Learning Rate Warmup with Subsequent Linear Decay
    '../_base_/schedules/poly10.py'
]
# Random Seed
seed = 0
# Modifications to Basic UDA


# Optimizer Hyperparameters
optimizer_config = None
optimizer = dict(
    lr=6e-05,
    paramwise_cfg=dict(
        custom_keys=dict(
            head=dict(lr_mult=10.0),
            pos_block=dict(decay_mult=0.0),
            norm=dict(decay_mult=0.0))))
n_gpus = 1
runner = dict(type='IterBasedRunner', max_iters=40000)
# Logging Configuration
checkpoint_config = dict(by_epoch=False, interval=40000, max_keep_ckpts=1)
#evaluation = dict(interval=4000, metric='mIoU')
evaluation = dict(interval=5000, metric='mIoU')
# Meta Information for Result Analysis
name = 'uper_beit_dacs'
exp = 'basic'
name_dataset = 'gta2cityscapes'
name_architecture = 'uper_fcn_beit'
name_encoder = 'beit'
name_decoder = 'uper_fcn'
name_uda = 'dacs_a99_cpl'
name_opt = 'adamw_6e-05_pmTrue_poly10_1x2_40k'
