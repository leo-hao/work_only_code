# optimizer
# optimizer = dict(
#     constructor='LearningRateDecayOptimizerConstructor',
#     _delete_=True,
#     type='AdamW',
#     lr=0.0001,
#     betas=(0.9, 0.999),
#     weight_decay=0.05,
#     paramwise_cfg={
#         'decay_rate': 0.9,
#         'decay_type': 'stage_wise',
#         'num_layers': 12
#     })

optimizer = dict(
    type='AdamW', lr=0.00006, betas=(0.9, 0.999), weight_decay=0.01)
# fp16 settings
optimizer_config = dict(type='Fp16OptimizerHook', loss_scale='dynamic')
# fp16 placeholder
fp16 = dict()

