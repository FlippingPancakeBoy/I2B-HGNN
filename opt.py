import datetime
import argparse
import random
import numpy as np
import torch
import math


dataset = 'ABIDE'
atlas = 'aal'
num_classes = 2
num_subjects = 871
sabp = 0.9
if atlas == 'aal':
    num_rois = 116
elif atlas == 'ho':
    num_rois = 111
# 一个被试的特征维数  ROI * 每个ROI的特征个数
# node_ftr_dim = math.ceil(num_rois * sabp) * 20
node_ftr_dim = 116 * 20
if dataset == 'ADNI':
    key = 'Image Data ID'
    labels = 'Group'
    ages = 'Age'
    genders = 'Sex'
    sites = 'Visit'
    variable = 'ROICorrelation'
elif dataset == 'ABIDE':
    key = 'SUB_ID'
    labels = 'DX_GROUP'
    ages = 'AGE_AT_SCAN'
    genders = 'SEX'
    sites = 'SITE_ID'
    variable = 'connectivity'
    handedness = 'HANDEDNESS_CATEGORY'


class OptInit:
    def __init__(self):
        parser = argparse.ArgumentParser(description='PyTorch implementation of HGCN')
        # train
        parser.add_argument('--train', default=1, type=int, help='train(default) or evaluate')
        parser.add_argument('--use_cpu', action='store_true', help='use cpu?')
        parser.add_argument('--seed', type=int, default=42, help='random state')
        parser.add_argument('--hgc', type=int, default=16, help='hidden units of gconv layer')
        parser.add_argument('--lg', type=int, default=4, help='number of gconv layers')
        parser.add_argument('--lr', default=0.01, type=float, help='initial learning rate')
        parser.add_argument('--wd', default=5e-3, type=float, help='weight decay')
        #mi_loss
        parser.add_argument('--mi_weight', default=0.08, type=int, help='use mi_loss or not')
        parser.add_argument('--mi_weight_eval', default=0.008, type=int, help='use mi_loss or not')
        #hg_loss
        parser.add_argument('--hg_weight', default=0.7, type=int, help='use hg_loss or not')
        parser.add_argument('--hg_weight_eval', default=0.1, type=int, help='use hg_loss or not')
        # epochs
        parser.add_argument('--num_iter', default=250, type=int, help='number of epochs for training')
        parser.add_argument('--folds', default=10, type=int, help='cross validation')
        parser.add_argument('--edropout', type=float, default=0.5, help='edge dropout rate')
        parser.add_argument('--dropout', default=0.55, type=float, help='ratio of dropout')
        parser.add_argument('--threshold', default=0.37, type=float, help='threshold of aff_score')
        parser.add_argument('--sabp', default=sabp, type=float, help='ratio of sabp')
        parser.add_argument('--node_ftr_dim', type=int, default=node_ftr_dim, help='dimension of node features')
        parser.add_argument('--log_save', type=bool, default=True, help='save log or not')
        parser.add_argument('--model_save', type=bool, default=True, help='save model or not')
        parser.add_argument('--dataset', default=dataset, type=str, help='name of dataset')
        parser.add_argument('--ckpt_path', type=str, default=rf'./save_model/{dataset}_{atlas}/',
                            help='checkpoint path to save trained models')
        parser.add_argument('--data_folder', default=rf'./data/{dataset}_{atlas}/', type=str, help='data folder')
        parser.add_argument('--atlas', type=str, default=atlas, help='atlas for network construction (node definition)')
        parser.add_argument('--num_rois', type=int, default=num_rois, help='number of brain regions')
        parser.add_argument('--num_classes', type=int, default=num_classes, help='number of classes')
        parser.add_argument('--num_subjects', type=int, default=num_subjects, help='number of subjects')
        parser.add_argument('--key', type=str, default=key, help='key values for image data')
        parser.add_argument('--labels', type=str, default=labels, help='the title of labels')

        parser.add_argument('--ages', type=str, default=ages, help='the title of ages')
        parser.add_argument('--genders', type=str, default=genders, help='the title of genders')
        parser.add_argument('--sites', type=str, default=sites, help='the title of sites')
        parser.add_argument('--handedness', type=str, default=handedness, help='the title of sites')

        parser.add_argument('--variable', type=str, default=variable, help='variable name of .mat file')

        args = parser.parse_args()

        args.time = datetime.datetime.now().strftime("%y%m%d")

        if args.use_cpu:
            args.device = torch.device('cpu')
            print(" Using CPU in torch")
        else:
            args.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
            print(" Using GPU in torch")

        self.args = args

    def print_args(self):
        # self.args.printer args
        print("==========       CONFIG      =============")
        for arg, content in self.args.__dict__.items():
            print("{}:{}".format(arg, content))
        print("==========     CONFIG END    =============")
        print("\n")
        phase = 'train' if self.args.train==1 else 'eval'
        print('===> Phase is {}.'.format(phase))

    def initialize(self):
        self.set_seed(123)
        self.print_args()
        return self.args

    def set_seed(self, seed=0):
        """固定系统种子随机值"""
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


