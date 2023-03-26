_base_ = ['ConvNeXt.py']
crop_size = (512, 512)
model = dict(
    decode_head=dict(in_channels=[128, 256, 512, 1024], num_classes=19),
    auxiliary_head=dict(in_channels=512, num_classes=19),
    test_cfg=dict(mode='slide', crop_size=crop_size, stride=(341, 341)),
)