import argparse
import torch
from torch.utils.data import DataLoader
import torch.optim as optim

import util.misc as utils
from datasets import build_dataset
from datasets import hake_meta

def get_args_parser():
    parser = argparse.ArgumentParser('Set transformer detector', add_help=False)

    parser.add_argument('--data_root', default='data/hake', type=str)
    parser.add_argument('--num_workers', default=0, type=int)
    parser.add_argument('--batch_size', default=4, type=int)

    parser.add_argument('--arch', default='ViT-B/16', type=str)
    parser.add_argument('--backbone_pretrained', default="pretrained/clip/ViT-B-16.new.pt", type=str)
    parser.add_argument('--backbone_freeze_layer', default=12, type=int)

    parser.add_argument('--lr', default=1e-3, type=float)
    parser.add_argument('--weight_decay', default=1e-5, type=float)
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--epochs', default=50, type=int)
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--eval', default=False, action='store_true')
    parser.add_argument('--vis_test', default=False, action='store_true')

    parser.add_argument('--device', default='cuda', type=str)
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--output_dir', default='./runs/',
                        help='path where to save, empty for no saving')
    return parser


def convert_to_hake_format(gt_labels, img_metas):
    hake_data = {
        "img_path": img_metas['img_path'].replace('data/', ''),
        "image_height": img_metas['image_height'],
        "image_width": img_metas['image_width'],
        "gt_verbs": [],
        "gt_pasta_foot": [],
        "gt_pasta_leg": [],
        "gt_pasta_hip": [],
        "gt_pasta_hand": [],
        "gt_pasta_arm": [],
        "gt_pasta_head": []
    }

    for part in ['verb', 'foot', 'leg', 'hip', 'hand', 'arm', 'head']:
        part_labels = gt_labels[part].numpy()
        gt_pasta_part = f"gt_pasta_{part}"

        if part == 'verb':
            for verb_tensor in part_labels:
                if verb_tensor.sum() == 0:
                    hake_data['gt_verbs'].append(hake_meta.ignore_idx['verb'])
                else:
                    hake_data['gt_verbs'].extend([i for i, v in enumerate(verb_tensor) if v == 1])
        else:
            if part_labels.sum() == 0:
                hake_data[gt_pasta_part].append(hake_meta.ignore_idx[part])
            else:
                for label in part_labels:
                    hake_data[gt_pasta_part].extend([i for i, v in enumerate(label) if v == 1])

    return hake_data


# 转换代码
def convert_indices_to_words(data, class_names):
    converted_data = {}

    for part in ['verb', 'foot', 'leg', 'hip', 'hand', 'arm', 'head']:
        key = f"gt_pasta_{part}" if part != 'verb' else 'gt_verbs'
        converted_data[key] = [class_names[part][i] for i in data[key]]

    return converted_data


def data_loader_test_demo(model, data_loader_train, device, epoch, output_dir, vis):
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'

    result = []
    for img, gt_labels, img_metas in metric_logger.log_every(data_loader_train, 500, header):

        img_meta_dict = img_metas[0]
        print(convert_to_hake_format(gt_labels, img_meta_dict))


def data_loader_test(args):

    dataset_train = build_dataset(split='train', args=args)
    dataset_val = build_dataset(split='val', args=args)

    from itertools import islice
    # Limit the train dataset to 1000 samples
    dataset_train_limited = torch.utils.data.Subset(dataset_train, range(1000))

    sampler_train = torch.utils.data.RandomSampler(dataset_train)
    sampler_val = torch.utils.data.SequentialSampler(dataset_val)
    batch_sampler_train = torch.utils.data.BatchSampler(
        sampler_train, args.batch_size, drop_last=True)
    data_loader_train = DataLoader(dataset_train, batch_sampler=batch_sampler_train,
                                   collate_fn=dataset_train.collate_fn, num_workers=args.num_workers)
    data_loader_val = DataLoader(dataset_val, batch_size=4, sampler=sampler_val,
                                 drop_last=False, collate_fn=dataset_val.collate_fn, num_workers=args.num_workers)

    return data_loader_train

if __name__ == '__main__':

    parser = argparse.ArgumentParser('DETR training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    dataset_train = build_dataset(split='train', args=args)
    dataset_val = build_dataset(split='val', args=args)
    sampler_train = torch.utils.data.RandomSampler(dataset_train)
    sampler_val = torch.utils.data.SequentialSampler(dataset_val)
    batch_sampler_train = torch.utils.data.BatchSampler(
        sampler_train, args.batch_size, drop_last=True)
    data_loader_train = DataLoader(dataset_train, batch_sampler=batch_sampler_train,
                                   collate_fn=dataset_train.collate_fn, num_workers=args.num_workers)
    data_loader_val = DataLoader(dataset_val, batch_size=1, sampler=sampler_val,
                                 drop_last=False, collate_fn=dataset_val.collate_fn, num_workers=args.num_workers)

    print(len(data_loader_test(args)))
