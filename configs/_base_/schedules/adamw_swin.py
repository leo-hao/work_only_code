# optimizer
# optimizer = dict(
#     type='AdamW', lr=0.00006, betas=(0.9, 0.999), weight_decay=0.01)
optimizer = dict(
    type='AdamW',
    lr=0.00006,
    betas=(0.9, 0.999),
    weight_decay=0.01,
    # no weight decay for position embedding & layer norm
    paramwise_cfg=dict(
        custom_keys={
            'absolute_pos_embed': dict(decay_mult=0.),
            'relative_position_bias_table': dict(decay_mult=0.),
            'norm': dict(decay_mult=0.)
        }))
optimizer_config = dict()
