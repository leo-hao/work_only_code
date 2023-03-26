ConvNeXt L 224
pip install kornia==0.5.8 -i https://pypi.douban.com/simple/

CUDA_VISIBLE_DEVICES=1 python run_experiments.py --config configs/daformer/gta2cs_uda_warm_fdthings_rcs_croppl_a999_daformer_mitb5_s0.py


bash test.sh work_dirs/211108_1622_gta2cs_daformer_s0_7f24c

TEST ROOT work_dirs/211108_1622_gta2cs_daformer_s0_7f24c
Config File: work_dirs/211108_1622_gta2cs_daformer_s0_7f24c/211108_1622_gta2cs_daformer_s0_7f24c.json
Checkpoint File: work_dirs/211108_1622_gta2cs_daformer_s0_7f24c/latest.pth
Predictions Output Directory: work_dirs/211108_1622_gta2cs_daformer_s0_7f24c/preds/
2021-12-07 16:53:00,754 - mmseg - INFO - Loaded 500 images from data/cityscapes/leftImg8bit/val
/home/data/liuhao/experiments/DAFormer-master/mmseg/models/backbones/mix_transformer.py:214: UserWarning: DeprecationWarning: pretrained is a deprecated, please use "init_cfg" instead
  warnings.warn('DeprecationWarning: pretrained is a deprecated, '
Use load_from_local loader

[>>>>>>>>>>>>>>>>>>>>>>>>>>>>>] 500/500, 2.5 task/s, elapsed: 199s, ETA:     0sper class results:

+---------------+-------+-------+
|     Class     |  IoU  |  Acc  |
+---------------+-------+-------+
|      road     | 96.53 | 99.28 |
|    sidewalk   | 74.01 | 80.25 |
|    building   | 89.47 | 94.83 |
|      wall     | 53.35 | 63.74 |
|     fence     | 47.72 | 56.93 |
|      pole     | 50.64 | 61.25 |
| traffic light | 54.72 | 71.23 |
|  traffic sign | 63.57 | 73.51 |
|   vegetation  | 89.97 | 96.08 |
|    terrain    | 44.36 | 47.07 |
|      sky      | 92.56 | 98.75 |
|     person    | 71.79 | 84.35 |
|     rider     | 44.77 | 65.63 |
|      car      | 92.58 | 95.46 |
|     truck     | 77.82 | 89.24 |
|      bus      | 80.64 | 86.73 |
|     train     |  63.6 | 80.43 |
|   motorcycle  | 56.66 | 74.49 |
|    bicycle    |  63.4 | 79.11 |
+---------------+-------+-------+
Summary:

+-------+-------+-------+
|  aAcc |  mIoU |  mAcc |
+-------+-------+-------+
| 94.09 | 68.85 | 78.86 |
+-------+-------+-------+

soft：

    outputs = self.model.train_step(data_batch, self.optimizer, **kwargs)
  File "/cloud/leo/lib/python3.7/site-packages/mmcv/parallel/data_parallel.py", line 75, in train_step
    return self.module.train_step(*inputs[0], **kwargs[0])
  File "/home/featurize/work/DAFormer/mmseg/models/uda/dacs.py", line 159, in train_step
    log_vars = self(**data_batch)
  File "/cloud/leo/lib/python3.7/site-packages/torch/nn/modules/module.py", line 727, in _call_impl
    result = self.forward(*input, **kwargs)
  File "/cloud/leo/lib/python3.7/site-packages/mmcv/runner/fp16_utils.py", line 110, in new_func
    return old_func(*args, **kwargs)
  File "/home/featurize/work/DAFormer/mmseg/models/segmentors/base.py", line 110, in forward
    return self.forward_train(img, img_metas, **kwargs)
  File "/home/featurize/work/DAFormer/mmseg/models/uda/dacs.py", line 381, in forward_train
    target_img, target_img_metas, pseudo_label, return_feat=True)
  File "/home/featurize/work/DAFormer/mmseg/models/segmentors/encoder_decoder.py", line 162, in forward_train
    seg_weight)
  File "/home/featurize/work/DAFormer/mmseg/models/segmentors/encoder_decoder.py", line 95, in _decode_head_forward_train
    seg_weight)
  File "/home/featurize/work/DAFormer/mmseg/models/decode_heads/decode_head.py", line 194, in forward_train
    losses = self.losses(seg_logits, gt_semantic_seg, seg_weight)
  File "/cloud/leo/lib/python3.7/site-packages/mmcv/runner/fp16_utils.py", line 198, in new_func
    return old_func(*args, **kwargs)
  File "/home/featurize/work/DAFormer/mmseg/models/decode_heads/decode_head.py", line 229, in losses
    align_corners=self.align_corners)
  File "/home/featurize/work/DAFormer/mmseg/ops/wrappers.py", line 28, in resize
    return F.interpolate(input, size, scale_factor, mode, align_corners)
  File "/cloud/leo/lib/python3.7/site-packages/torch/nn/functional.py", line 3080, in interpolate
    'Input is {}D, size is {}'.format(dim, len(size)))
ValueError: size shape must match input shape. Input is 2D, size is 1









mmd:
2022-05-06 18:10:29,873 - mmseg - INFO - Checkpoints will be saved to /home/featurize/work/DAFormer/work_dirs/local-basic/220506_1810_uda_daformer_swinL224_mmd_soft_d005b by HardDiskBackend.
Traceback (most recent call last):
  File "run_experiments.py", line 102, in <module>
    train.main([config_files[i]])
  File "/home/featurize/work/DAFormer/tools/train.py", line 173, in main
    meta=meta)
  File "/home/featurize/work/DAFormer/mmseg/apis/train.py", line 131, in train_segmentor
    runner.run(data_loaders, cfg.workflow)
  File "/cloud/leo/lib/python3.7/site-packages/mmcv/runner/iter_based_runner.py", line 134, in run
    iter_runner(iter_loaders[i], **kwargs)
  File "/cloud/leo/lib/python3.7/site-packages/mmcv/runner/iter_based_runner.py", line 61, in train
    outputs = self.model.train_step(data_batch, self.optimizer, **kwargs)
  File "/cloud/leo/lib/python3.7/site-packages/mmcv/parallel/data_parallel.py", line 75, in train_step
    return self.module.train_step(*inputs[0], **kwargs[0])
  File "/home/featurize/work/DAFormer/mmseg/models/uda/dacs.py", line 160, in train_step
    log_vars = self(**data_batch)
  File "/cloud/leo/lib/python3.7/site-packages/torch/nn/modules/module.py", line 727, in _call_impl
    result = self.forward(*input, **kwargs)
  File "/cloud/leo/lib/python3.7/site-packages/mmcv/runner/fp16_utils.py", line 110, in new_func
    return old_func(*args, **kwargs)
  File "/home/featurize/work/DAFormer/mmseg/models/segmentors/base.py", line 110, in forward
    return self.forward_train(img, img_metas, **kwargs)
  File "/home/featurize/work/DAFormer/mmseg/models/uda/dacs.py", line 414, in forward_train
    gs = ap(src_feat)
  File "/cloud/leo/lib/python3.7/site-packages/torch/nn/modules/module.py", line 727, in _call_impl
    result = self.forward(*input, **kwargs)
  File "/cloud/leo/lib/python3.7/site-packages/torch/nn/modules/pooling.py", line 1107, in forward
    return F.adaptive_avg_pool2d(input, self.output_size)
  File "/cloud/leo/lib/python3.7/site-packages/torch/nn/functional.py", line 935, in adaptive_avg_pool2d
    _output_size = _list_with_default(output_size, input.size())
AttributeError: 'list' object has no attribute 'size'

swinS:
2022-05-06 18:30:46,908 - mmseg - INFO - Set random seed to 0, deterministic: False
/home/featurize/work/DAFormer/mmseg/models/backbones/swin.py:556: UserWarning: DeprecationWarning: pretrained is deprecated, please use "init_cfg" instead
  warnings.warn('DeprecationWarning: pretrained is deprecated, '
2022-05-06 18:30:48,686 - mmseg - INFO - load checkpoint from local path: pretrained/swin_small_patch4_window7_224.pth
Traceback (most recent call last):
  File "run_experiments.py", line 102, in <module>
    train.main([config_files[i]])
  File "/home/featurize/work/DAFormer/tools/train.py", line 145, in main
    model.init_weights()
  File "/cloud/leo/lib/python3.7/site-packages/mmcv/runner/base_module.py", line 116, in init_weights
    m.init_weights()
  File "/cloud/leo/lib/python3.7/site-packages/mmcv/runner/base_module.py", line 116, in init_weights
    m.init_weights()
  File "/home/featurize/work/DAFormer/mmseg/models/backbones/swin.py", line 720, in init_weights
    table_current = self.state_dict()[table_key]
KeyError: 'layers.0.blocks.0.attn.relative_position_bias_table'


swinB:
2022-05-06 18:35:02,265 - mmseg - INFO - Set random seed to 0, deterministic: False
/home/featurize/work/DAFormer/mmseg/models/backbones/swin.py:556: UserWarning: DeprecationWarning: pretrained is deprecated, please use "init_cfg" instead
  warnings.warn('DeprecationWarning: pretrained is deprecated, '
2022-05-06 18:35:04,910 - mmseg - INFO - load checkpoint from local path: pretrained/swin_base_patch4_window7_224_22k.pth
Traceback (most recent call last):
  File "run_experiments.py", line 102, in <module>
    train.main([config_files[i]])
  File "/home/featurize/work/DAFormer/tools/train.py", line 145, in main
    model.init_weights()
  File "/cloud/leo/lib/python3.7/site-packages/mmcv/runner/base_module.py", line 116, in init_weights
    m.init_weights()
  File "/cloud/leo/lib/python3.7/site-packages/mmcv/runner/base_module.py", line 116, in init_weights
    m.init_weights()
  File "/home/featurize/work/DAFormer/mmseg/models/backbones/swin.py", line 720, in init_weights
    table_current = self.state_dict()[table_key]
KeyError: 'layers.0.blocks.0.attn.relative_position_bias_table'
# 问题，base和small权重没有转换。。。


2022-05-07 18:07:20,127 - mmseg - INFO - Checkpoints will be saved to /home/featurize/work/DAFormer/work_dirs/local-basic/220507_1807_daformer_convnextB_dacs_9cf7f by HardDiskBackend.
Traceback (most recent call last):
  File "run_experiments.py", line 102, in <module>    train.main([config_files[i]])
  File "/home/featurize/work/DAFormer/tools/train.py", line 173, in main
    meta=meta)  File "/home/featurize/work/DAFormer/mmseg/apis/train.py", line 131, in train_segmentor
    runner.run(data_loaders, cfg.workflow)
  File "/cloud/leo/lib/python3.7/site-packages/mmcv/runner/iter_based_runner.py", line 134, in run    iter_runner(iter_loaders[i], **kwargs)
  File "/cloud/leo/lib/python3.7/site-packages/mmcv/runner/iter_based_runner.py", line 61, in train
    outputs = self.model.train_step(data_batch, self.optimizer, **kwargs)  
  File "/cloud/leo/lib/python3.7/site-packages/mmcv/parallel/data_parallel.py", line 75, in train_step
    return self.module.train_step(*inputs[0], **kwargs[0])  
  File "/home/featurize/work/DAFormer/mmseg/models/uda/dacs.py", line 159, in train_step    log_vars = self(**data_batch)
  File "/cloud/leo/lib/python3.7/site-packages/torch/nn/modules/module.py", line 727, in _call_impl
    result = self.forward(*input, **kwargs)  File "/cloud/leo/lib/python3.7/site-packages/mmcv/runner/fp16_utils.py", line 110, in new_func
    return old_func(*args, **kwargs)
  File "/home/featurize/work/DAFormer/mmseg/models/segmentors/base.py", line 110, in forward    return self.forward_train(img, img_metas, **kwargs)
  File "/home/featurize/work/DAFormer/mmseg/models/uda/dacs.py", line 384, in forward_train
    target_img, target_img_metas, pseudo_label0, return_feat=True)
  File "/home/featurize/work/DAFormer/mmseg/models/segmentors/encoder_decoder.py", line 162, in forward_train
    seg_weight)
  File "/home/featurize/work/DAFormer/mmseg/models/segmentors/encoder_decoder.py", line 95, in _decode_head_forward_train    seg_weight)
  File "/home/featurize/work/DAFormer/mmseg/models/decode_heads/decode_head.py", line 194, in forward_train
    losses = self.losses(seg_logits, gt_semantic_seg, seg_weight)
  File "/cloud/leo/lib/python3.7/site-packages/mmcv/runner/fp16_utils.py", line 198, in new_func
    return old_func(*args, **kwargs)
  File "/home/featurize/work/DAFormer/mmseg/models/decode_heads/decode_head.py", line 237, in losses
    ignore_index=self.ignore_index)
  File "/cloud/leo/lib/python3.7/site-packages/torch/nn/modules/module.py", line 727, in _call_impl
    result = self.forward(*input, **kwargs)
  File "/home/featurize/work/DAFormer/mmseg/models/losses/cross_entropy_loss.py", line 250, in forward
    **kwargs)
  File "/home/featurize/work/DAFormer/mmseg/models/losses/cross_entropy_loss.py", line 72, in cross_entropy
    ignore_index=ignore_index)
  File "/cloud/leo/lib/python3.7/site-packages/torch/nn/functional.py", line 2468, in cross_entropy
    return nll_loss(log_softmax(input, 1), target, weight, None, ignore_index, None, reduction)
  File "/cloud/leo/lib/python3.7/site-packages/torch/nn/functional.py", line 2262, in nll_loss
    .format(input.size(0), target.size(0)))
ValueError: Expected input batch_size (2) to match target batch_size (1).
# 通过input.shape和target.shape后查看清楚后，unsqueeze(1)改正了

Traceback (most recent call last):
  File "run_experiments.py", line 102, in <module>
    train.main([config_files[i]])
  File "/home/featurize/work/DAFormer/tools/train.py", line 173, in main
    meta=meta)
  File "/home/featurize/work/DAFormer/mmseg/apis/train.py", line 131, in train_segmentor
    runner.run(data_loaders, cfg.workflow)
  File "/cloud/leo/lib/python3.7/site-packages/mmcv/runner/iter_based_runner.py", line 134, in run
    iter_runner(iter_loaders[i], **kwargs)
  File "/cloud/leo/lib/python3.7/site-packages/mmcv/runner/iter_based_runner.py", line 61, in train
    outputs = self.model.train_step(data_batch, self.optimizer, **kwargs)
  File "/cloud/leo/lib/python3.7/site-packages/mmcv/parallel/data_parallel.py", line 75, in train_step
    return self.module.train_step(*inputs[0], **kwargs[0])
  File "/home/featurize/work/DAFormer/mmseg/models/uda/dacs.py", line 159, in train_step
    log_vars = self(**data_batch)
  File "/cloud/leo/lib/python3.7/site-packages/torch/nn/modules/module.py", line 727, in _call_impl
    result = self.forward(*input, **kwargs)
  File "/cloud/leo/lib/python3.7/site-packages/mmcv/runner/fp16_utils.py", line 110, in new_func
    return old_func(*args, **kwargs)
  File "/home/featurize/work/DAFormer/mmseg/models/segmentors/base.py", line 110, in forward
    return self.forward_train(img, img_metas, **kwargs)
  File "/home/featurize/work/DAFormer/mmseg/models/uda/dacs.py", line 426, in forward_train
    gs = ap(src_feat).flatten(1)
  File "/cloud/leo/lib/python3.7/site-packages/torch/nn/modules/module.py", line 727, in _call_impl
    result = self.forward(*input, **kwargs)
  File "/cloud/leo/lib/python3.7/site-packages/torch/nn/modules/pooling.py", line 1107, in forward
    return F.adaptive_avg_pool2d(input, self.output_size)
  File "/cloud/leo/lib/python3.7/site-packages/torch/nn/functional.py", line 935, in adaptive_avg_pool2d
    _output_size = _list_with_default(output_size, input.size())
AttributeError: 'tuple' object has no attribute 'size'




------------
Traceback (most recent call last):
  File "run_experiments.py", line 102, in <module>
    train.main([config_files[i]])
  File "/home/featurize/work/DAFormer/tools/train.py", line 173, in main
    meta=meta)
  File "/home/featurize/work/DAFormer/mmseg/apis/train.py", line 131, in train_segmentor
    runner.run(data_loaders, cfg.workflow)
  File "/cloud/leo/lib/python3.7/site-packages/mmcv/runner/iter_based_runner.py", line 134, in run
    iter_runner(iter_loaders[i], **kwargs)
  File "/cloud/leo/lib/python3.7/site-packages/mmcv/runner/iter_based_runner.py", line 61, in train
    outputs = self.model.train_step(data_batch, self.optimizer, **kwargs)
  File "/cloud/leo/lib/python3.7/site-packages/mmcv/parallel/data_parallel.py", line 75, in train_step
    return self.module.train_step(*inputs[0], **kwargs[0])
  File "/home/featurize/work/DAFormer/mmseg/models/uda/dacs.py", line 159, in train_step
    log_vars = self(**data_batch)
  File "/cloud/leo/lib/python3.7/site-packages/torch/nn/modules/module.py", line 727, in _call_impl
    result = self.forward(*input, **kwargs)
  File "/cloud/leo/lib/python3.7/site-packages/mmcv/runner/fp16_utils.py", line 110, in new_func
    return old_func(*args, **kwargs)
  File "/home/featurize/work/DAFormer/mmseg/models/segmentors/base.py", line 110, in forward
    return self.forward_train(img, img_metas, **kwargs)
  File "/home/featurize/work/DAFormer/mmseg/models/uda/dacs.py", line 435, in forward_train
    mmd_loss, mmd_log = self.mmd_rbf(gs, gt)
  File "/home/featurize/work/DAFormer/mmseg/models/uda/dacs.py", line 246, in mmd_rbf
    kernel_mul=kernel_mul, kernel_num=kernel_num, fix_sigma=fix_sigma)
  File "/home/featurize/work/DAFormer/mmseg/models/uda/dacs.py", line 223, in guassian_kernel
    total0 = total.unsqueeze(0).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)), int(total.size(2)), int(total.size(2)))
IndexError: Dimension out of range (expected to be in range of [-2, 1], but got 2)
(/cloud/leo) ➜  DAFormer 


Traceback (most recent call last):
  File "run_experiments.py", line 102, in <module>
    train.main([config_files[i]])
  File "/home/featurize/work/DAFormer/tools/train.py", line 173, in main
    meta=meta)
  File "/home/featurize/work/DAFormer/mmseg/apis/train.py", line 131, in train_segmentor
    runner.run(data_loaders, cfg.workflow)
  File "/cloud/leo/lib/python3.7/site-packages/mmcv/runner/iter_based_runner.py", line 134, in run
    iter_runner(iter_loaders[i], **kwargs)
  File "/cloud/leo/lib/python3.7/site-packages/mmcv/runner/iter_based_runner.py", line 61, in train
    outputs = self.model.train_step(data_batch, self.optimizer, **kwargs)
  File "/cloud/leo/lib/python3.7/site-packages/mmcv/parallel/data_parallel.py", line 75, in train_step
    return self.module.train_step(*inputs[0], **kwargs[0])
  File "/home/featurize/work/DAFormer/mmseg/models/uda/dacs.py", line 159, in train_step
    log_vars = self(**data_batch)
  File "/cloud/leo/lib/python3.7/site-packages/torch/nn/modules/module.py", line 727, in _call_impl
    result = self.forward(*input, **kwargs)
  File "/cloud/leo/lib/python3.7/site-packages/mmcv/runner/fp16_utils.py", line 110, in new_func
    return old_func(*args, **kwargs)
  File "/home/featurize/work/DAFormer/mmseg/models/segmentors/base.py", line 110, in forward
    return self.forward_train(img, img_metas, **kwargs)
  File "/home/featurize/work/DAFormer/mmseg/models/uda/dacs.py", line 437, in forward_train
    mmd_loss.backward()
  File "/cloud/leo/lib/python3.7/site-packages/torch/tensor.py", line 221, in backward
    torch.autograd.backward(self, gradient, retain_graph, create_graph)
  File "/cloud/leo/lib/python3.7/site-packages/torch/autograd/__init__.py", line 132, in backward
    allow_unreachable=True)  # allow_unreachable flag
RuntimeError: element 0 of tensors does not require grad and does not have a grad_fn


ValueError: size shape must match input shape. Input is 2D, size is 1
KeyError: 'layers.0.blocks.0.attn.relative_position_bias_table'
KeyError: 'data_time'
RuntimeError: element 0 of tensors does not require grad and does not have a grad_fn
AttributeError: 'list' object has no attribute 'size'




root@container-ccc41184ac-a777bc97:~/work# python run_experiments.py --config configs/daformer/uda_daformer_convnextB.py
Traceback (most recent call last):
  File "run_experiments.py", line 57, in <module>
    cfg = Config.fromfile(args.config)
  File "/root/miniconda3/lib/python3.8/site-packages/mmcv/utils/config.py", line 340, in fromfile
    import_modules_from_strings(**cfg_dict['custom_imports'])
  File "/root/miniconda3/lib/python3.8/site-packages/mmcv/utils/misc.py", line 73, in import_modules_from_strings
    imported_tmp = import_module(imp)
  File "/root/miniconda3/lib/python3.8/importlib/__init__.py", line 127, in import_module
    return _bootstrap._gcd_import(name[level:], package, level)
  File "<frozen importlib._bootstrap>", line 1014, in _gcd_import
  File "<frozen importlib._bootstrap>", line 991, in _find_and_load
  File "<frozen importlib._bootstrap>", line 975, in _find_and_load_unlocked
  File "<frozen importlib._bootstrap>", line 671, in _load_unlocked
  File "<frozen importlib._bootstrap_external>", line 848, in exec_module
  File "<frozen importlib._bootstrap>", line 219, in _call_with_frames_removed
  File "/root/miniconda3/lib/python3.8/site-packages/mmcls/models/__init__.py", line 2, in <module>
    from .backbones import *  # noqa: F401,F403
  File "/root/miniconda3/lib/python3.8/site-packages/mmcls/models/backbones/__init__.py", line 4, in <module>
    from .convnext import ConvNeXt
  File "/root/miniconda3/lib/python3.8/site-packages/mmcls/models/backbones/convnext.py", line 19, in <module>
    class LayerNorm2d(nn.LayerNorm):
  File "/root/miniconda3/lib/python3.8/site-packages/mmcv/utils/registry.py", line 315, in _register
    self._register_module(
  File "/root/miniconda3/lib/python3.8/site-packages/mmcv/utils/registry.py", line 249, in _register_module
    raise KeyError(f'{name} is already registered '
KeyError: 'LN2d is already registered in norm layer'
# force=True

Traceback (most recent call last):
  File "run_experiments.py", line 11, in <module>
    from tools import train
  File "/root/work/tools/train.py", line 19, in <module>
    from mmseg import __version__
  File "/root/work/mmseg/__init__.py", line 26, in <module>
    assert (mmcv_min_version <= mmcv_version <= mmcv_max_version), \
AssertionError: MMCV==1.5.0 is used but incompatible. Please install mmcv>=[1, 3, 7], <=[1, 4, 0].





Traceback (most recent call last):
  File "run_experiments.py", line 104, in <module>
    train.main([config_files[i]])
  File "/root/work/tools/train.py", line 143, in main
    model = build_train_model(
  File "/root/work/mmseg/models/builder.py", line 54, in build_train_model
    return UDA.build(
  File "/root/miniconda3/lib/python3.8/site-packages/mmcv/utils/registry.py", line 215, in build
    return self.build_func(*args, **kwargs, registry=self)
  File "/root/miniconda3/lib/python3.8/site-packages/mmcv/cnn/builder.py", line 27, in build_model_from_cfg
    return build_from_cfg(cfg, registry, default_args)
  File "/root/miniconda3/lib/python3.8/site-packages/mmcv/utils/registry.py", line 55, in build_from_cfg
    raise type(e)(f'{obj_cls.__name__}: {e}')
KeyError: 'DACS: "EncoderDecoder: \'SwinTransformer is not in the models registry\'"'


  File "run_experiments.py", line 104, in <module>
    train.main([config_files[i]])
  File "/root/work/tools/train.py", line 166, in main
    train_segmentor(
  File "/root/work/mmseg/apis/train.py", line 131, in train_segmentor
    runner.run(data_loaders, cfg.workflow)
  File "/root/miniconda3/lib/python3.8/site-packages/mmcv/runner/iter_based_runner.py", line 134, in run
    iter_runner(iter_loaders[i], **kwargs)
  File "/root/miniconda3/lib/python3.8/site-packages/mmcv/runner/iter_based_runner.py", line 61, in train
    outputs = self.model.train_step(data_batch, self.optimizer, **kwargs)
  File "/root/miniconda3/lib/python3.8/site-packages/mmcv/parallel/data_parallel.py", line 75, in train_step
    return self.module.train_step(*inputs[0], **kwargs[0])
  File "/root/work/mmseg/models/uda/dacs.py", line 236, in train_step
    log_vars = self(**data_batch)
  File "/root/miniconda3/lib/python3.8/site-packages/torch/nn/modules/module.py", line 727, in _call_impl
    result = self.forward(*input, **kwargs)
  File "/root/miniconda3/lib/python3.8/site-packages/mmcv/runner/fp16_utils.py", line 110, in new_func
    return old_func(*args, **kwargs)
  File "/root/work/mmseg/models/segmentors/base.py", line 110, in forward
    return self.forward_train(img, img_metas, **kwargs)
  File "/root/work/mmseg/models/uda/dacs.py", line 451, in forward_train
    _, pseudo_weight[i] = strong_transform_soft(
  File "/root/work/mmseg/models/utils/dacs_transforms.py", line 28, in strong_transform_soft
    print('data.shape:',data.shape)
AttributeError: 'NoneType' object has no attribute 'shape'
在运行训练文件时，出现了这样的问题：“AttributeError: ‘NoneType’ object has no attribute ‘shape’”。
后来参考了大神文章后发现是因为都的txt文件中有中文路径，改了文件名后运行没问题了。
还可能有以下问题：
原因： 1.图片不存在（路径不存在， 路径包含中文无法识别） 2.读取的图片内容和默认读取时参数匹配不匹配。（默认读取的是3通道的彩色图）例如读取到的图片是灰度图，就会返回None。3.也可能是路径中有中文
————————————————
版权声明：本文为CSDN博主「旅人_Eric」的原创文章，遵循CC 4.0 BY-SA版权协议，转载请附上原文出处链接及本声明。
原文链接：https://blog.csdn.net/qq_37099552/article/details/103909896


Traceback (most recent call last):
  File "run_experiments.py", line 104, in <module>
    train.main([config_files[i]])
  File "/root/work/tools/train.py", line 166, in main
    train_segmentor(
  File "/root/work/mmseg/apis/train.py", line 131, in train_segmentor
    runner.run(data_loaders, cfg.workflow)
  File "/root/miniconda3/lib/python3.8/site-packages/mmcv/runner/iter_based_runner.py", line 134, in run
    iter_runner(iter_loaders[i], **kwargs)
  File "/root/miniconda3/lib/python3.8/site-packages/mmcv/runner/iter_based_runner.py", line 61, in train
    outputs = self.model.train_step(data_batch, self.optimizer, **kwargs)
  File "/root/miniconda3/lib/python3.8/site-packages/mmcv/parallel/data_parallel.py", line 75, in train_step
    return self.module.train_step(*inputs[0], **kwargs[0])
  File "/root/work/mmseg/models/uda/dacs.py", line 253, in train_step
    log_vars = self(**data_batch)
  File "/root/miniconda3/lib/python3.8/site-packages/torch/nn/modules/module.py", line 727, in _call_impl
    result = self.forward(*input, **kwargs)
  File "/root/miniconda3/lib/python3.8/site-packages/mmcv/runner/fp16_utils.py", line 110, in new_func
    return old_func(*args, **kwargs)
  File "/root/work/mmseg/models/segmentors/base.py", line 110, in forward
    return self.forward_train(img, img_metas, **kwargs)
  File "/root/work/mmseg/models/uda/dacs.py", line 498, in forward_train
    mix_losses_soft = self.get_ema_model().forward_train(
  File "/root/work/mmseg/models/segmentors/encoder_decoder.py", line 160, in forward_train
    loss_decode = self._decode_head_forward_train(x, img_metas,
  File "/root/work/mmseg/models/segmentors/encoder_decoder.py", line 92, in _decode_head_forward_train
    loss_decode = self.decode_head.forward_train(x, img_metas,
  File "/root/work/mmseg/models/decode_heads/decode_head.py", line 194, in forward_train
    losses = self.losses(seg_logits, gt_semantic_seg, seg_weight)
  File "/root/miniconda3/lib/python3.8/site-packages/mmcv/runner/fp16_utils.py", line 198, in new_func
    return old_func(*args, **kwargs)
  File "/root/work/mmseg/models/decode_heads/decode_head.py", line 233, in losses
    loss['loss_seg'] = self.loss_decode(
  File "/root/miniconda3/lib/python3.8/site-packages/torch/nn/modules/module.py", line 727, in _call_impl
    result = self.forward(*input, **kwargs)
  File "/root/work/mmseg/models/losses/cross_entropy_loss.py", line 243, in forward
    loss_cls = self.loss_weight * self.cls_criterion(
  File "/root/work/mmseg/models/losses/cross_entropy_loss.py", line 67, in cross_entropy
    loss = F.cross_entropy(
  File "/root/miniconda3/lib/python3.8/site-packages/torch/nn/functional.py", line 2468, in cross_entropy
    return nll_loss(log_softmax(input, 1), target, weight, None, ignore_index, None, reduction)
  File "/root/miniconda3/lib/python3.8/site-packages/torch/nn/functional.py", line 2261, in nll_loss
    raise ValueError('Expected input batch_size ({}) to match target batch_size ({}).'
ValueError: Expected input batch_size (2) to match target batch_size (1).

  File "run_experiments.py", line 104, in <module>
    train.main([config_files[i]])
  File "/root/da/tools/train.py", line 166, in main
    train_segmentor(
  File "/root/da/mmseg/apis/train.py", line 131, in train_segmentor
    runner.run(data_loaders, cfg.workflow)
  File "/root/miniconda3/lib/python3.8/site-packages/mmcv/runner/iter_based_runner.py", line 134, in run
    iter_runner(iter_loaders[i], **kwargs)
  File "/root/miniconda3/lib/python3.8/site-packages/mmcv/runner/iter_based_runner.py", line 61, in train
    outputs = self.model.train_step(data_batch, self.optimizer, **kwargs)
  File "/root/miniconda3/lib/python3.8/site-packages/mmcv/parallel/data_parallel.py", line 75, in train_step
    return self.module.train_step(*inputs[0], **kwargs[0])
  File "/root/da/mmseg/models/uda/dacs.py", line 267, in train_step
    log_vars = self(**data_batch)
  File "/root/miniconda3/lib/python3.8/site-packages/torch/nn/modules/module.py", line 727, in _call_impl
    result = self.forward(*input, **kwargs)
  File "/root/miniconda3/lib/python3.8/site-packages/mmcv/runner/fp16_utils.py", line 110, in new_func
    return old_func(*args, **kwargs)
  File "/root/da/mmseg/models/segmentors/base.py", line 110, in forward
    return self.forward_train(img, img_metas, **kwargs)
  File "/root/da/mmseg/models/uda/dacs.py", line 517, in forward_train
    mix_losses = self.get_model().forward_train(
  File "/root/da/mmseg/models/segmentors/encoder_decoder.py", line 160, in forward_train
    loss_decode = self._decode_head_forward_train(x, img_metas,
  File "/root/da/mmseg/models/segmentors/encoder_decoder.py", line 92, in _decode_head_forward_train
    loss_decode = self.decode_head.forward_train(x, img_metas,
  File "/root/da/mmseg/models/decode_heads/decode_head.py", line 194, in forward_train
    losses = self.losses(seg_logits, gt_semantic_seg, seg_weight)
  File "/root/miniconda3/lib/python3.8/site-packages/mmcv/runner/fp16_utils.py", line 198, in new_func
    return old_func(*args, **kwargs)
  File "/root/da/mmseg/models/decode_heads/decode_head.py", line 233, in losses
    loss['loss_seg'] = self.loss_decode(
  File "/root/miniconda3/lib/python3.8/site-packages/torch/nn/modules/module.py", line 727, in _call_impl
    result = self.forward(*input, **kwargs)
  File "/root/da/mmseg/models/losses/cross_entropy_loss.py", line 243, in forward
    loss_cls = self.loss_weight * self.cls_criterion(
  File "/root/da/mmseg/models/losses/cross_entropy_loss.py", line 67, in cross_entropy
    loss = F.cross_entropy(
  File "/root/miniconda3/lib/python3.8/site-packages/torch/nn/functional.py", line 2468, in cross_entropy
    return nll_loss(log_softmax(input, 1), target, weight, None, ignore_index, None, reduction)
  File "/root/miniconda3/lib/python3.8/site-packages/torch/nn/functional.py", line 2266, in nll_loss
    ret = torch._C._nn.nll_loss2d(input, target, weight, _Reduction.get_enum(reduction), ignore_index)
RuntimeError: Expected object of scalar type Long but got scalar type Float for argument #2 'target' in call to _thnn_nll_loss2d_forward

修改的时候，直接在label后面加上.long()




Traceback (most recent call last):
  File "run_experiments.py", line 104, in <module>
    train.main([config_files[i]])
  File "/root/work/tools/train.py", line 166, in main
    train_segmentor(
  File "/root/work/mmseg/apis/train.py", line 131, in train_segmentor
    runner.run(data_loaders, cfg.workflow)
  File "/root/miniconda3/lib/python3.8/site-packages/mmcv/runner/iter_based_runner.py", line 134, in run
    iter_runner(iter_loaders[i], **kwargs)
  File "/root/miniconda3/lib/python3.8/site-packages/mmcv/runner/iter_based_runner.py", line 67, in train
    self.call_hook('after_train_iter')
  File "/root/miniconda3/lib/python3.8/site-packages/mmcv/runner/base_runner.py", line 309, in call_hook
    getattr(hook, fn_name)(self)
  File "/root/miniconda3/lib/python3.8/site-packages/mmcv/runner/hooks/optimizer.py", line 56, in after_train_iter
    runner.outputs['loss'].backward()
KeyError: 'loss'
加上了builder optimizers.....我的loss加了前缀