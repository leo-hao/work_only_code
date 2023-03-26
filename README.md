
# rename 's/\.zip$//' *.zip

## DAFormer: Improving Network Architectures and Training Strategies for Domain-Adaptive Semantic Segmentation
# unzip autodl-tmp/cityscapes.zip -d autodl-tmp/data/cityscapes/
# unzip autodl-tmp/GTA5.zip -d autodl-tmp/data/gta/
# ln -s /root/dataset/cityscapes /root/work/data/cityscapes
# ln -s /root/dataset/gta5 /root/work/data/gta
ln -s /home/zkpk/Documents/GTA5 /home/zkpk/Documents/CLUDA/data/gta
ln -s /home/zkpk/Documents/cityscapes /home/zkpk/Documents/CLUDA/data/cityscapes
# CUDA_VISIBLE_DEVICES=0 python run_experiments.py --config configs/daformer/daformer_swinB_uda_soft.py

#  python run_experiments.py --config configs/daformer/daformer_swinB_uda_soft_paste.py
# python run_experiments.py --config configs/daformer/daformer_swinB_uda_soft_paste_mmd.py
# 
#  python run_experiments.py --config configs/daformer/daformer_convnextB_uda_soft_paste_mmd.py
# python run_experiments.py --config configs/daformer/daformer_swinB_uda_soft_paste_mmd_rare.py
# 
# python run_experiments.py --config configs/daformer/gta2cs_uda_warm_fdthings_rcs_croppl_a999_daformer_mitb5_s0.py
# python run_experiments.py --config configs/daformer/uda_daformer_swinL224.py
# python run_experiments.py --config configs/daformer/uda_daformer_swinL224_mmd_soft.py
# python run_experiments.py --config configs/daformer/uda_daformer_swinB_soft_mmd.py
# python run_experiments.py --config configs/daformer/uda_daformer_swinS.py
# python run_experiments.py --config configs/daformer/uda_daformer_convnextS.py
# python run_experiments.py --config configs/daformer/uda_daformer_convnextL.py
# python run_experiments.py --config configs/daformer/uda_daformer_convnextB.py
# python run_experiments.py --config configs/daformer/daformer_convnextB_src.py
# CUDA_VISIBLE_DEVICES=0 python run_experiments.py --config configs/daformer/daformer_convnextB_uda.py
# CUDA_VISIBLE_DEVICES=1 python run_experiments.py --config configs/daformer/daformer_convnextB_oral.py
# python run_experiments.py --config configs/daformer/daformer_swinB_src.py
# python run_experiments.py --config configs/daformer/daformer_swinB_uda.py
# python run_experiments.py --config configs/daformer/daformer_swinB_oral.py
# python run_experiments.py --config configs/daformer/uda_daformer_swinB_soft.py
# CUDA_VISIBLE_DEVICES=1 python run_experiments.py --config configs/daformer/uda_daformer_swinB_soft_paste.py
# python run_experiments.py --config configs/daformer/uda_daformer_swinB_soft_mmd.py
#  python run_experiments.py --config configs/daformer/uda_daformer_convnextB_soft_mmd.py
# python run_experiments.py --config configs/daformer/uda_daformer_convnextB_soft_mmd.py
# python run_experiments.py --config configs/daformer/uda_daformer_convnextB_soft_paste.py

# python  run_experiments.py --config configs/daformer/mitb_5_src.py
# python run_experiments.py --config configs/daformer/uper_fcn_swinB_uda_soft_paste_mmd_rare.py
# python tools/model_converters/beit2mmseg.py mae_pretrain_vit_base.pth mae_base.pth
# python tools/model_converters/beit2mmseg.py ${PRETRAIN_PATH} ${STORE_PATH}
# python run_experiments.py --config configs/daformer/uda_daformer_swinB_soft_paste_mmd.py

# python run_experiments.py --config configs/daformer/uper_mae_uda.py
# python run_experiments.py --config configs/daformer/uper_beit_uda.py
# python run_experiments.py --config configs/daformer/uper_mae_DAFormer.py
# python run_experiments.py --config configs/daformer/uper_beit_DAFomer.py

# python run_experiments.py --config configs/daformer/daformer_convnextB_uda_soft_paste_mmd_rare.py

**[[Arxiv]](https://arxiv.org/abs/2111.14887)**
**[[Paper]](https://arxiv.org/pdf/2111.14887.pdf)**

As acquiring pixel-wise annotations of real-world images for semantic
segmentation is a costly process, a model can instead be trained with more
accessible synthetic data and adapted to real images without requiring their
annotations. This process is studied in **Unsupervised Domain Adaptation (UDA)**.

Even though a large number of methods propose new UDA strategies, they
are mostly based on outdated network architectures. In this work, we
particularly study the influence of the network architecture on UDA performance
and propose **DAFormer**, a network architecture tailored for UDA. It consists of a
Transformer encoder and a multi-level context-aware feature fusion decoder.

DAFormer is enabled by three simple but crucial training strategies to stabilize the
training and to avoid overfitting the source domain: While the
**Rare Class Sampling** on the source domain improves the quality of pseudo-labels
by mitigating the confirmation bias of self-training towards common classes,
the **Thing-Class ImageNet Feature Distance** and a **Learning Rate Warmup** promote
feature transfer from ImageNet pretraining.

DAFormer significantly improves
the state-of-the-art performance **by 10.8 mIoU for GTA→Cityscapes**
and **by 5.4 mIoU for Synthia→Cityscapes** and enables learning even
difficult classes such as train, bus, and truck well.

![UDA over time](resources/uda_over_time.png)

The strengths of DAFormer, compared to the previous state-of-the-art UDA method
ProDA, can also be observed in qualitative examples from the Cityscapes
validation set.

![Demo](resources/demo.gif)
![Color Palette](resources/color_palette.png)

For more information on DAFormer, please check our
[[Paper]](https://arxiv.org/pdf/2111.14887.pdf).

If you find this project useful in your research, please consider citing:

```
@article{hoyer2021daformer,
  title={DAFormer: Improving Network Architectures and Training Strategies for Domain-Adaptive Semantic Segmentation},
  author={Hoyer, Lukas and Dai, Dengxin and Van Gool, Luc},
  journal={arXiv preprint arXiv:2111.14887},
  year={2021}
}
```

## Setup Environment

For this project, we used python 3.8.5. We recommend setting up a new virtual
environment:

```shell
python -m venv ~/venv/daformer
source ~/venv/daformer/bin/activate
```

In that environment, the requirements can be installed with:

```shell
pip install -r requirements.txt -f https://download.pytorch.org/whl/torch_stable.html
pip install mmcv-full==1.3.7  # requires the other packages to be installed first
```

Further, please download the MiT weights and a pretrained DAFormer using the
following script. If problems occur with the automatic download, please follow
the instructions for a manual download within the script.

```shell
sh tools/download_checkpoints.sh
```

All experiments were executed on a NVIDIA RTX 2080 Ti.

## Inference Demo

Already as this point, the provided DAFormer model (downloaded by
`tools/download_checkpoints.sh`) can be applied to a demo image:

```shell
python -m demo.image_demo demo/demo.png work_dirs/211108_1622_gta2cs_daformer_s0_7f24c/211108_1622_gta2cs_daformer_s0_7f24c.json work_dirs/211108_1622_gta2cs_daformer_s0_7f24c/latest.pth
```

When judging the predictions, please keep in mind that DAFormer had no access
to real-world labels during the training.

## Setup Datasets

**Cityscapes:** Please, download leftImg8bit_trainvaltest.zip and
gt_trainvaltest.zip from [here](https://www.cityscapes-dataset.com/downloads/)
and extract them to `data/cityscapes`.

**GTA:** Please, download all image and label packages from
[here](https://download.visinf.tu-darmstadt.de/data/from_games/) and extract
them to `data/gta`.

**Synthia:** Please, download SYNTHIA-RAND-CITYSCAPES from
[here](http://synthia-dataset.net/downloads/) and extract it to `data/synthia`.

The final folder structure should look like this:

```none
DAFormer
├── ...
├── data
│   ├── cityscapes
│   │   ├── leftImg8bit
│   │   │   ├── train
│   │   │   ├── val
│   │   ├── gtFine
│   │   │   ├── train
│   │   │   ├── val
│   ├── gta
│   │   ├── images
│   │   ├── labels
│   ├── synthia
│   │   ├── RGB
│   │   ├── GT
│   │   │   ├── LABELS
├── ...
```

**Data Preprocessing:** Finally, please run the following scripts to convert the label IDs to the
train IDs and to generate the class index for RCS:

```shell
python tools/convert_datasets/gta.py data/gta --nproc 8
python tools/convert_datasets/cityscapes.py data/cityscapes --nproc 8
python tools/convert_datasets/synthia.py data/synthia/ --nproc 8
```

## Training

For convenience, we provide an [annotated config file](configs/daformer/gta2cs_uda_warm_fdthings_rcs_croppl_a999_daformer_mitb5_s0.py) of the final DAFormer.
A training job can be launched using:

```shell
python run_experiments.py --config configs/daformer/gta2cs_uda_warm_fdthings_rcs_croppl_a999_daformer_mitb5_s0.py
```

For the experiments in our paper (e.g. network architecture comparison,
component ablations, ...), we use a system to automatically generate
and train the configs:

```shell
python run_experimenty.py --exp <ID>
```

More information about the available experiments and their assigned IDs, can be
found in [experiments.py](experiments.py). The generated configs will be stored
in `configs/generated/`.

## Testing & Predictions

The provided DAFormer checkpoint trained on GTA->Cityscapes
(already downloaded by `tools/download_checkpoints.sh`) can be tested on the
Cityscapes validation set using:

```shell
sh test.sh work_dirs/211108_1622_gta2cs_daformer_s0_7f24c
```

The predictions are saved for inspection to
`work_dirs/211108_1622_gta2cs_daformer_s0_7f24c/preds`
and the mIoU of the model is printed to the console. The provided checkpoint
should achieve 68.85 mIoU. Refer to the end of
`work_dirs/211108_1622_gta2cs_daformer_s0_7f24c/20211108_164105.log` for
more information such as the class-wise IoU.

Similarly, also other models can be tested after the training has finished:

```shell
sh test.sh path/to/checkpoint_directory
```

## Framework Structure

This project is based on [mmsegmentation version 0.16.0](https://github.com/open-mmlab/mmsegmentation/tree/v0.16.0).
For more information about the framework structure and the config system,
please refer to the [mmsegmentation documentation](https://mmsegmentation.readthedocs.io/en/latest/index.html)
and the [mmcv documentation](https://mmcv.readthedocs.ihttps://arxiv.org/abs/2007.08702o/en/v1.3.7/index.html).

The most relevant files for DAFormer are:

* [configs/daformer/gta2cs_uda_warm_fdthings_rcs_croppl_a999_daformer_mitb5_s0.py](configs/daformer/gta2cs_uda_warm_fdthings_rcs_croppl_a999_daformer_mitb5_s0.py):
  Annotated config file for the final DAFormer.
* [mmseg/models/uda/dacs.py](mmseg/models/uda/dacs.py):
  Implementation of UDA self-training with ImageNet Feature Distance.
* [mmseg/datasets/uda_dataset.py](mmseg/datasets/uda_dataset.py):
  Data loader for UDA with Rare Class Sampling.
* [mmseg/models/decode_heads/daformer_head.py](mmseg/models/decode_heads/daformer_head.py):
  Implementation of DAFormer decoder with context-aware feature fusion.
* [mmseg/models/backbones/mix_transformer.py](mmseg/models/backbones/mix_transformer.py):
  Implementation of Mix Transformer encoder (MiT).

## Acknowledgements

This project is based on the following open-source projects. We thank their
authors for making the source code publically available.

* [MMSegmentation](https://github.com/open-mmlab/mmsegmentation)
* [SegFormer](https://github.com/NVlabs/SegFormer)
* [DACS](https://github.com/vikolss/DACS)
#   w o r k _ o n l y _ c o d e  
 