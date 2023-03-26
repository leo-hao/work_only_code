# Obtained from: https://github.com/open-mmlab/mmsegmentation/tree/v0.16.0
# Modifications:
# - Config and checkpoint update
# - Saving instead of showing prediction


import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from argparse import ArgumentParser

import mmcv
from tools.test import update_legacy_cfg

from mmseg.apis import inference_segmentor, init_segmentor
from mmseg.core.evaluation import get_classes, get_palette


def main():
    parser = ArgumentParser()
    parser.add_argument('img', help='Image file')
    parser.add_argument('config', help='Config file')
    parser.add_argument('checkpoint', help='Checkpoint file')
    parser.add_argument(
        '--device', default='cuda:0', help='Device used for inference')
    parser.add_argument(
        '--palette',
        default='cityscapes',
        help='Color palette used for segmentation map')
    parser.add_argument(
        '--opacity',
        type=float,
        default=1,
        help='Opacity of painted segmentation map. In (0, 1] range.')
    args = parser.parse_args()

    # build the model from a config file and a checkpoint file
    cfg = mmcv.Config.fromfile(args.config)
    cfg = update_legacy_cfg(cfg)
    model = init_segmentor(
        cfg,
        args.checkpoint,
        device=args.device,
        classes=get_classes(args.palette),
        palette=get_palette(args.palette),
        revise_checkpoint=[(r'^module\.', ''), ('model.', '')])
    # test a single image
    result = inference_segmentor(model, args.img)
    # show the results
    file, extension = os.path.splitext(args.img)
    pred_file = f'{file}_pred{extension}'
    assert pred_file != args.img
    model.show_result(
        args.img,
        result,
        palette=get_palette(args.palette),
        out_file=pred_file,
        show=False,
        opacity=args.opacity)
    print('Save prediction to', pred_file)


if __name__ == '__main__':
    main()
# Copyright (c) OpenMMLab. All rights reserved.
# from argparse import ArgumentParser

# from mmseg.apis import inference_segmentor, init_segmentor, show_result_pyplot
# from mmseg.core.evaluation import get_palette


# def main():
#     parser = ArgumentParser()
#     parser.add_argument('img', help='Image file')
#     parser.add_argument('config', help='Config file')
#     parser.add_argument('checkpoint', help='Checkpoint file')
#     parser.add_argument(
#         '--device', default='cuda:0', help='Device used for inference')
#     parser.add_argument(
#         '--palette',
#         default='cityscapes',
#         help='Color palette used for segmentation map')
#     parser.add_argument(
#         '--opacity',
#         type=float,
#         default=0.5,
#         help='Opacity of painted segmentation map. In (0, 1] range.')
#     args = parser.parse_args()

#     # build the model from a config file and a checkpoint file
#     model = init_segmentor(args.config, args.checkpoint, device=args.device)
#     # test a single image
#     result = inference_segmentor(model, args.img)
#     # show the results
#     show_result_pyplot(
#         model,
#         args.img,
#         result,
#         get_palette(args.palette),
#         opacity=args.opacity)


# if __name__ == '__main__':
#     main()
