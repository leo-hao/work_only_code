# 只用简单的数据增强的域适方法,没有FD和rare_sampling
# 用于final daformer的注释配置文件。
_base_ = [
    '../_base_/default_runtime.py',                             # 一些运行时的配置：log、resume、checkpoint
    # DAFormer Network Architecture
    '../_base_/models/uper_convnextT.py',
    # GTA->Cityscapes Data Loading
    '../_base_/datasets/cityscapes_half_512x512.py',
    # Basic UDA Self-Training
    #'../_base_/uda/dacs.py',
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
optimizer = dict(
    constructor='LearningRateDecayOptimizerConstructor',
    _delete_=True,
    type='AdamW',
    lr=0.0001,
    betas=(0.9, 0.999),
    weight_decay=0.05,
    paramwise_cfg={
        'decay_rate': 0.9,
        'decay_type': 'stage_wise',
        'num_layers': 6
    })

lr_config = dict(
    _delete_=True,
    policy='poly',
    warmup='linear',
    warmup_iters=1500,
    warmup_ratio=1e-6,
    power=1.0,
    min_lr=0.0,
    by_epoch=False)

# By default, models are trained on 8 GPUs with 2 images per GPU

# fp16 settings
optimizer_config = dict(type='Fp16OptimizerHook', loss_scale='dynamic')
# fp16 placeholder
fp16 = dict()
n_gpus = 1
runner = dict(type='IterBasedRunner', max_iters=40000)
# Logging Configuration
checkpoint_config = dict(by_epoch=False, interval=40000, max_keep_ckpts=1)
#evaluation = dict(interval=4000, metric='mIoU')
evaluation = dict(interval=5000, metric='mIoU')
# Meta Information for Result Analysis
name = 'uper_fcn_convnextT_target'
exp = 'basic'
name_dataset = 'gta2cityscapes'
name_architecture = 'Uper_FCN_convnextT'
name_encoder = 'convnextT'
name_decoder = 'uper_fcn'
name_uda = 'dacs_a99_cpl'
name_opt = 'adamw_6e-05_pmTrue_poly10warm_1x2_40k'
