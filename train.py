import os, sys, time
import shutil
import yaml

import argparse
import torch
from torch.utils.data import dataloader
from torchvision import datasets



sys.path.append(os.path.dirname(__file__))

import source.yaml_units as yaml_utils


def load_models(config):
    gen_conf = config.models['generator']
    gen = yaml_utils.load_model(gen_conf['fn'], gen_conf['name'], gen_conf['args'])
    dis_conf = config.models['discriminator']
    dis = yaml_utils.load_model(dis_conf['fn'], dis_conf['name'], dis_conf['args'])
    return gen, dis
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', type=str, default='configs/base.yml', help='path to config file')
    parser.add_argument('--gpu', type=int, default=0, help='index of gpu to be used')
    parser.add_argument('--data_dir', type=str, default='./data/imagenet')
    parser.add_argument('--result_dir', type=str, default='./results/gans',
                        help='directory to save the results to')
    parser.add_argument('--inception_model_path', type=str, default='',
                        help='path to the inception model')
    parser.add_argument('--snapshot', type=str, default='',
                        help='path to the snapshot')
    parser.add_argument('--loaderjob', type=int,
                        help='number of parallel data ;oading processes')

    args = parser.parse_args()
    config = yaml_utils.Config(yaml.load(open(args.config_path)))
    gen, dis = load_models(config)


if __name__ == "__main__":
    main()
