# 用于final daformer的注释配置文件。
_base_ = [
    '../_base_/default_runtime.py',                             # 一些运行时的配置：log、resume、checkpoint
    # DAFormer Network Architecture
    '../_base_/models/daformer_sepaspp_convnext_B.py',
    # GTA->Cityscapes Data Loading
    '../_base_/datasets/uda_gta_to_cityscapes_512x512.py',
    # Basic UDA Self-Training
    '../_base_/uda/dacs.py',
    # AdamW Optimizer
    '../_base_/schedules/adamw_fp16.py',
    # Linear Learning Rate Warmup with Subsequent Linear Decay
    '../_base_/schedules/poly10warm.py'
]
# Random Seed
seed = 0
# Modifications to Basic UDA

uda = dict(
    # Increased Alpha
    alpha=0.999,
    # Thing-Class Feature Distance
    mix_losses_soft=True,
    soft_paste=True,
    mmd_loss=True,
    )
# Optimizer Hyperparameters
optimizer_config = None
optimizer = dict(
    lr=8e-05,
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
name = 'convnextB_uda_soft_paste_mmd'
exp = 'basic'
name_dataset = 'gta2cityscapes'
name_architecture = 'daformer_sepaspp_convnextB'
name_encoder = 'convnextB'
name_decoder = 'daformer_sepaspp'
name_uda = 'dacs_a999_fd_things_rcs0.01_cpl'
name_opt = 'adamw_e-04_fp16_pmTrue_poly10warm_1x2_40k'
