import random
from torch.utils.data import DataLoader
import datetime
import argparse

from sentence_encoder import *
from fewshot import *
from dataset import FSLDataset, SemcorDataset, fsl_pack, sup_pack

from preprocess.utils import *
from constant import *

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
from fewshot.amsl import AMSLoss


import warnings
warnings.filterwarnings("ignore", category=UserWarning)

def str_list(text):
    return tuple(text.split(','))


def int_list(text):
    return [int(x) for x in text.split(',')]


def parse_int_list(input_):
    if input_ == None:
        return []
    return list(map(int, input_.split(',')))


def parse_float_list(input_):
    if input_ == None:
        return []
    return list(map(float, input_.split(',')))


def one_or_list(parser):
    def parse_one_or_list(input_):
        output = parser(input_)
        if len(output) == 1:
            return output[0]
        else:
            return output

    return parse_one_or_list


def argument_parser():
    parser = argparse.ArgumentParser()
    # Training setting
    parser.add_argument('-m', '--model', default='proto', choices=fsl_class.keys())
    parser.add_argument('-e', '--encoder', default='bertgcn', choices=encoder_class.keys())
    parser.add_argument('-b', '--batch_size', default=2, type=int)
    parser.add_argument('--optimizer', default='sgd', type=str, choices=['adam', 'sgd', 'adadelta'])
    parser.add_argument('--lr', default=0.001, type=float)
    parser.add_argument('--lr_step_size', default=1000, type=int)
    parser.add_argument('--gpu', default='0')
    parser.add_argument('--local_rank', default='0')
    parser.add_argument('--ex', default='base')

    # Few-shot settings
    parser.add_argument('-d', '--dataset', default='rams', choices=['ace', 'rams', 'fed', 'debug'])
    parser.add_argument('--save', default='checkpoints', type=str)
    parser.add_argument('--seed', default=1234, type=int)
    parser.add_argument('--ams', default=0, type=float, help='Additive Margin')

    # Supervised setting
    parser.add_argument('--epoch', default=30, type=int)

    parser.add_argument('--train_way', default=10, type=int)
    parser.add_argument('--train_shot', default=10, type=int)
    parser.add_argument('-n', '--way', default=5, type=int)
    parser.add_argument('-k', '--shot', default=5, type=int)
    parser.add_argument('-q', '--query', default=4, type=int)
    parser.add_argument('-o', '--other', default=1, type=int)
    # Embedding
    parser.add_argument('--embedding', default='glove', type=str_list)
    parser.add_argument('--tune_embedding', default=False, action='store_true')
    parser.add_argument('--debug', default=False, action='store_true')
    parser.add_argument('--progress', default=False, action='store_true')
    parser.add_argument('--hidden_size', default=512, type=int)

    # Tree settings
    parser.add_argument('--tree', default='full', type=str, choices=['full', 'prune'])

    # BERT
    parser.add_argument('--bert_pretrained', default='bert-base-cased', type=str)
    parser.add_argument('--bert_layer', default=4, type=int)
    parser.add_argument('--bert_update', default=False, action='store_true')

    # CNN params
    parser.add_argument('--window', default=2, type=int)
    parser.add_argument('--dropout', default=0.5, type=float)

    # CNN, NCNN parameters
    parser.add_argument('--cnn_kernel_sizes', default=[2, 3, 4, 5], type=parse_int_list)
    parser.add_argument('--cnn_kernel_number', default=150, type=int)

    # RNN parameters (i.e, GRU, LSTM)
    parser.add_argument('--rnn_num_hidden_units', default=300, type=int)
    parser.add_argument('--rnn_pooling', default='pool_anchor',
                        choices=['pool_anchor', 'pool_max', 'pool_dynamic', 'pool_entity'])

    # GCNN params
    parser.add_argument('--num_rel_dep', default=50, type=int)
    parser.add_argument('--gcnn_kernel_numbers', default=[300, 300], type=parse_int_list)
    parser.add_argument('--gcnn_edge_patterns', default=[1, 0, 1], type=parse_int_list)
    parser.add_argument('--gcnn_pooling', default='pool_anchor',
                        choices=['pool_anchor', 'pool_max', 'pool_dynamic', 'pool_entity'])

    # Transformer model
    parser.add_argument('--wsd_model', default='none', type=str, choices=['none', 'gcn', 'bert'])

    parser.add_argument('--zeta', default=1, type=int)

    return parser


def main(args):
    random.seed(args.seed)
    torch.random.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    args.device = 'cuda'
    # torch.cuda.set_device(0)

    if args.dataset == 'ace':
        args.train_way = 18

    B = args.batch_size  # default = 4
    TN = args.train_way  # default = 20
    TK = args.train_shot  # default = 20
    N = args.way
    K = args.shot  # default = 5
    Q = args.query  # default = 5

    current_time = str(datetime.datetime.now().time())
    args.log_dir = 'logs/{}-{}-way-{}-shot-{}'.format(args.model, args.way, args.shot, current_time)

    args.max_length = dataset_constant[args.dataset]['max_length']
    args.n_class = dataset_constant[args.dataset]['n_class']

    print('Before load')

    feature_set = feature_map[args.encoder]
    train_dl = FSLDataset(TN, TK, Q,
                          features=feature_set,
                          length=1000000,
                          prefix='datasets/{}/fsl/train'.format(args.dataset))
    dev_dl = FSLDataset(N, K, Q,
                        features=feature_set,
                        length=500,
                        prefix='datasets/{}/fsl/dev'.format(args.dataset))
    test_dl = FSLDataset(N, K, Q,
                         features=feature_set,
                         length=500,
                         prefix='datasets/{}/fsl/test'.format(args.dataset))

    train_dl = DataLoader(train_dl, batch_size=B, num_workers=8, collate_fn=fsl_pack)
    dev_dl = DataLoader(dev_dl, batch_size=B, num_workers=8, collate_fn=fsl_pack, shuffle=False)
    test_dl = DataLoader(test_dl, batch_size=B, num_workers=8, collate_fn=fsl_pack, shuffle=False)

    print('-' * 80)
    for k, v in args.__dict__.items():
        print('{}\t{}'.format(k, v))
    print('-' * 80)

    _, vectors = load_pickle('datasets/vocab.pkl')
    encoder = encoder_class[args.encoder](vectors=vectors, args=args)
    encoder.init_weight()

    fsl_model = fsl_class[args.model](encoder, args)
    # print(fsl_model)
    # exit(0)
    fsl_model.init_weight()
    fsl_model.cuda()

    fsl_trainer = ReparameterizeTrainer(fsl_model, train_dl, dev_dl, test_dl, args)
    fsl_trainer.do_train()


if __name__ == '__main__':
    args = argument_parser().parse_args()
    main(args)
