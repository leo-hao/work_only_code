# 只用简单的数据增强的域适方法,没有FD和rare_sampling
# 用于final daformer的注释配置文件。
_base_ = [
    '../_base_/default_runtime.py',                             # 一些运行时的配置：log、resume、checkpoint
    # DAFormer Network Architecture
    '../_base_/models/uper_swinB.py',
    # GTA->Cityscapes Data Loading
    '../_base_/datasets/uda_gta_to_cityscapes_512x512.py',
    # Basic UDA Self-Training
    '../_base_/uda/dacs.py',
    # AdamW Optimizer
    '../_base_/schedules/adamw_swin.py',
    # Linear Learning Rate Warmup with Subsequent Linear Decay
    '../_base_/schedules/poly10warm.py'
]
# Random Seed
seed = 0
# Modifications to Basic UDA
# uda = dict(
#     # Increased Alpha
#     alpha=0.999,
#     # Thing-Class Feature Distance
#     # mix_losses_soft=True,
#     # soft_paste=True,
#     # mmd_loss=True,
#     )

# Optimizer Hyperparameters
# AdamW optimizer, no weight decay for position embedding & layer norm
# in backbone
optimizer_config = None
optimizer = dict(
    _delete_=True,
    type='AdamW',
    lr=0.00006,
    betas=(0.9, 0.999),
    weight_decay=0.01,
    paramwise_cfg=dict(
        custom_keys={
            'absolute_pos_embed': dict(decay_mult=0.),
            'relative_position_bias_table': dict(decay_mult=0.),
            'norm': dict(decay_mult=0.)
        }))

lr_config = dict(
    _delete_=True,
    policy='poly',
    warmup='linear',
    warmup_iters=1500,
    warmup_ratio=1e-6,
    power=1.0,
    min_lr=0.0,
    by_epoch=False)
n_gpus = 1
runner = dict(type='IterBasedRunner', max_iters=40000)
# Logging Configuration
checkpoint_config = dict(by_epoch=False, interval=40000, max_keep_ckpts=1)
#evaluation = dict(interval=4000, metric='mIoU')
evaluation = dict(interval=5000, metric='mIoU')
# Meta Information for Result Analysis
name = 'uper_fcn_swinB_dacs'
exp = 'basic'
name_dataset = 'gta2cityscapes'
name_architecture = 'Uper_FCN_swinB'
name_encoder = 'swinB'
name_decoder = 'uper_fcn'
name_uda = 'dacs_a99_cpl'
name_opt = 'adamw_6e-05_pmTrue_poly10warm_1x2_40k'
