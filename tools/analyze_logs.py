# Obtained from: https://github.com/open-mmlab/mmsegmentation/tree/v0.16.0
"""Modified from https://github.com/open-
mmlab/mmdetection/blob/master/tools/analysis_tools/analyze_logs.py."""
import argparse
import json
from collections import defaultdict

import matplotlib.pyplot as plt
import seaborn as sns

# python tools/analyze_logs.py work_dirs/local-basic/220708_1527_daformer_swinb_adv_78aa4/20220708_152737.log.json --keys   adv.loss_adv_src dis_src.loss_Dis_src  dis_tgt.loss_Dis_tgt d_src.loss_D_src  d_tgt.loss_D_tgt  --legend    adv.loss_adv_src dis_src.loss_Dis_src  dis_tgt.loss_Dis_tgt  d_src.loss_D_src  d_tgt.loss_D_tgt --out adv_loss_all.png
# python tools/analyze_logs.py work_dirs/local-basic/220708_1527_daformer_swinb_adv_78aa4/20220708_152737.log.json --keys   adv.loss_adv_src   --legend    adv.loss_adv_src  --out adv_loss_1.png
# python tools/analyze_logs.py work_dirs/local-basic/220708_1527_daformer_swinb_adv_78aa4/20220708_152737.log.json --keys   dis_src.loss_Dis_src   --legend     dis_src.loss_Dis_src   --out adv_loss_2.png
# python tools/analyze_logs.py work_dirs/local-basic/220708_1527_daformer_swinb_adv_78aa4/20220708_152737.log.json --keys   dis_tgt.loss_Dis_tgt   --legend      dis_tgt.loss_Dis_tgt  --out adv_loss_3.png
# python tools/analyze_logs.py work_dirs/local-basic/220708_1527_daformer_swinb_adv_78aa4/20220708_152737.log.json --keys    d_src.loss_D_src   --legend     d_src.loss_D_src  --out adv_loss_4.png
# python tools/analyze_logs.py work_dirs/local-basic/220708_1527_daformer_swinb_adv_78aa4/20220708_152737.log.json --keys    d_tgt.loss_D_tgt  --legend    d_tgt.loss_D_tgt --out adv_loss_5.png
# python tools/analyze_logs.py work_dirs/local-basic/220709_2158_daformer_swinb_adv_fada_604ee/20220709_215835.log.json --keys decode.loss_seg adv.loss_adv_tgt d_src.loss_D_src d_tgt.loss_D_tgt mix.decode.loss_seg   --legend decode.loss_seg adv.loss_adv_tgt d_src.loss_D_src d_tgt.loss_D_tgt mix.decode.loss_seg --out fada_adv_loss.png
# python tools/analyze_logs.py work_dirs/local-basic/220709_2158_daformer_swinb_adv_fada_604ee/20220709_215835.log.json --keys adv.loss_adv_tgt    --legend adv.loss_adv_tgt  --out fada_adv_loss.png
# python tools/analyze_logs.py work_dirs/local-basic/220709_2158_daformer_swinb_adv_fada_604ee/20220709_215835.log.json --keys  d_src.loss_D_src   --legend  d_src.loss_D_src  --out fada_d_src_loss.png
# python tools/analyze_logs.py work_dirs/local-basic/220709_2158_daformer_swinb_adv_fada_604ee/20220709_215835.log.json --keys d_tgt.loss_D_tgt    --legend d_tgt.loss_D_tgt  --out fada_d_tgt_loss.png
# python tools/analyze_logs.py work_dirs/local-basic/220709_2158_daformer_swinb_adv_fada_604ee/20220709_215835.log.json --keys src.loss_imnet_feat_dist    --legend src.loss_imnet_feat_dist  --out fada_src.loss_imnet_feat_dist.png
def plot_curve(log_dicts, args):
    if args.backend is not None:
        plt.switch_backend(args.backend)
    sns.set_style(args.style)
    # if legend is None, use {filename}_{key} as legend
    legend = args.legend
    if legend is None:
        legend = []
        for json_log in args.json_logs:
            for metric in args.keys:
                legend.append(f'{json_log}_{metric}')
    assert len(legend) == (len(args.json_logs) * len(args.keys))
    metrics = args.keys
    print(metrics)

    num_metrics = len(metrics)
    for i, log_dict in enumerate(log_dicts):
        epochs = list(log_dict.keys())
        for j, metric in enumerate(metrics):
            print(f'plot curve of {args.json_logs[i]}, metric is {metric}')
            plot_epochs = []
            plot_iters = []
            plot_values = []
            # In some log files, iters number is not correct, `pre_iter` is
            # used to prevent generate wrong lines.
            pre_iter = -1
            for epoch in epochs:
                epoch_logs = log_dict[epoch]
                if metric not in epoch_logs.keys():
                    continue
                if metric in [ 'mAcc', 'aAcc']:
                    plot_epochs.append(epoch)
                    plot_values.append(epoch_logs[metric][0])
                else:
                    for idx in range(len(epoch_logs[metric])):
                        if pre_iter > epoch_logs['iter'][idx]:
                            continue
                        pre_iter = epoch_logs['iter'][idx] 
                        plot_iters.append(epoch_logs['iter'][idx])
                        plot_values.append(epoch_logs[metric][idx])
            ax = plt.gca()
            label = legend[i * num_metrics + j]
            if metric in [ 'mAcc', 'aAcc']:
                ax.set_xticks(plot_epochs)
                plt.xlabel('epoch')
                plt.plot(plot_epochs, plot_values, label=label, marker='o')
            else:
                plt.xlabel('iter')
                plt.plot(plot_iters, plot_values, label=label, linewidth=0.5)
        plt.legend()
        if args.title is not None:
            plt.title(args.title)
    if args.out is None:
        plt.show()
    else:
        print(f'save curve to: {args.out}')
        plt.savefig(args.out)
        plt.cla()


def parse_args():
    parser = argparse.ArgumentParser(description='Analyze Json Log')
    parser.add_argument(
        'json_logs',
        type=str,
        nargs='+',
        help='path of train log in json format')
    parser.add_argument(
        '--keys',
        type=str,
        nargs='+',
        default=['mIoU'],
        help='the metric that you want to plot')
    parser.add_argument('--title', type=str, help='title of figure')
    parser.add_argument(
        '--legend',
        type=str,
        nargs='+',
        default=None,
        help='legend of each plot')
    parser.add_argument(
        '--backend', type=str, default=None, help='backend of plt')
    parser.add_argument(
        '--style', type=str, default='dark', help='style of plt')
    parser.add_argument('--out', type=str, default=None)
    args = parser.parse_args()
    return args


def load_json_logs(json_logs):
    # load and convert json_logs to log_dict, key is epoch, value is a sub dict
    # keys of sub dict is different metrics
    # value of sub dict is a list of corresponding values of all iterations
    log_dicts = [dict() for _ in json_logs]
    for json_log, log_dict in zip(json_logs, log_dicts):
        with open(json_log, 'r') as log_file:
            for line in log_file:
                log = json.loads(line.strip())
                # skip lines without `epoch` field
                if 'epoch' not in log:
                    continue
                epoch = log.pop('epoch')
                if epoch not in log_dict:
                    log_dict[epoch] = defaultdict(list)
                for k, v in log.items():
                    log_dict[epoch][k].append(v)
    return log_dicts


def main():
    args = parse_args()
    json_logs = args.json_logs
    for json_log in json_logs:
        assert json_log.endswith('.json')
    log_dicts = load_json_logs(json_logs)
    plot_curve(log_dicts, args)


if __name__ == '__main__':
    main()