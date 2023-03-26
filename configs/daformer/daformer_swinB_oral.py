# 用于final daformer的注释配置文件。

# 直接用一半分辨率的目标域训练
_base_ = [
    '../_base_/default_runtime.py',                             # 一些运行时的配置：log、resume、checkpoint
    # DAFormer Network Architecture
    '../_base_/models/daformer_sepaspp_swinB.py',
    # GTA->Cityscapes Data Loading
    '../_base_/datasets/cityscapes_half_512x512.py',
    # Basic UDA Self-Training
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
name = 'daformer_swinB_oral'
exp = 'basic'
name_dataset = 'gta2cityscapes'
name_architecture = 'daformer_sepaspp_swinB'
name_encoder = 'swinB'
name_decoder = 'daformer_sepaspp'
name_opt = 'adamw_6e-05_pmTrue_poly10warm_1x2_40k'
