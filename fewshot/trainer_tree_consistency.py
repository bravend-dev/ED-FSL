import torch
import numpy
import numpy as np

import torch.nn as nn
import tqdm
import json
import random
from sklearn.metrics import *
from fewshot.amsl import AMSLoss
import time
from preprocess.utils import *


class TreeConsistencyFSLTrainer(torch.nn.Module):

    def __init__(self, fsl_model, train_dl, dev_dl, test_dl, args):
        super(TreeConsistencyFSLTrainer, self).__init__()
        self.fsl_model = fsl_model

        self.TN = args.train_way
        self.TK = args.train_shot
        self.N = args.way
        self.K = args.shot
        self.Q = args.query
        self.B = args.batch_size
        self.O = args.other

        self.train_setting = (self.B, self.TN + self.O, self.TK, self.Q)
        self.eval_setting = (self.B, self.N + self.O, self.K, self.Q)

        self.train_dl = train_dl
        self.dev_dl = dev_dl
        self.test_dl = test_dl

        self.ignore_cuda_feature = ['token', 'label', 'target']

        self.args = args
        self.ce = nn.CrossEntropyLoss()

    def do_train(self):

        B, N, K, Q = self.train_setting

        params = [x for x in self.parameters() if x.requires_grad]

        print(self.train_setting)

        trainable = 0
        for p in params:
            trainable += p.numel()

        print('Trainable params: ', trainable)

        print('Optimizer: SGD')
        optimizer = torch.optim.SGD(params, lr=self.args.lr)

        self.fsl_model.train()
        pbar = tqdm.tqdm(self.train_dl, desc='Training', total=6000)
        for i, batch in enumerate(pbar):
            for k, v in batch.items():
                if k not in self.ignore_cuda_feature:
                    batch[k] = batch[k].cuda()

            # if i == 0:
            #     for k, v in batch.items():
            #         if isinstance(v, torch.Tensor):
            #             print(k, tuple(v.shape))
            #         elif isinstance(v, list):
            #             print(k, 'list of ', len(v), type(v[0]))
            optimizer.zero_grad()
            return_item = self.fsl_model(batch, self.train_setting)

            target = batch['target'].view(-1).cuda()
            logits = return_item['logit'].view(-1, N)

            # print('| target', tuple(target.shape))
            # print('| logits', tuple(logits.shape))

            loss_fsl = self.ce(logits, target)

            # print('Loss', tuple(loss.shape))

            batch2 = batch
            batch2['dep'] = batch['prune_dep']
            return_item2 = self.fsl_model(batch2, self.train_setting, interpolation=False)
            logits2 = return_item2['logit'].view(-1, N)
            #
            loss_tc = self.ce(logits2, target)

            l1 = self.get_detach_loss(loss_fsl)
            l2 = self.get_detach_loss(loss_tc)
            pbar.set_description(f'FSL={l1:.4f} TC={l2:.4f}')

            loss  = loss_fsl + loss_tc
            loss.backward()
            optimizer.step()

            # Evaluation
            if self.args.debug:

                train_log_iter = 5
                train_iter = 10
            else:
                train_log_iter = 400
                train_iter = 400

            if i % train_log_iter == 0 and i > 0:
                l = loss.detach().cpu().numpy()

                output_format = '| @ {} Loss={:.4f}'
                print(output_format.format(i, l))

            if i % train_iter == 0 and i > 0:
                self.do_eval('dev', i)
                self.do_eval('test', i)
                self.fsl_model.train()
            if i > 6000:
                return

    def get_detach_loss(self, loss):
        if loss:
            return loss.detach().cpu().numpy()
        else:
            return 0.0

    def do_eval(self, part='test', iteration=0):
        if part == 'dev':
            dl = self.dev_dl
        else:
            dl = self.test_dl
        B, N, K, Q = self.eval_setting
        self.fsl_model.eval()

        with torch.no_grad():
            predictions = []
            targets = []
            sample_indices = []
            for i, batch in enumerate(dl):
                for k, v in batch.items():
                    if k not in self.ignore_cuda_feature:
                        batch[k] = batch[k].cuda()
                return_item = self.fsl_model(batch, self.eval_setting, interpolation=False)
                logits = return_item['logit'].view(-1, N)
                _pred = torch.argmax(logits, dim=1).view(-1).cpu().numpy()
                _target = batch['target'].view(-1).numpy()

                predictions.append(_pred)
                targets.append(_target)
                sample_indices.append(batch['i'].tolist())

            p, r, f = self.metrics(targets, predictions)
            print('-> {:4s}: {:6.2f} {:6.2f} {:6.2f}'.format(part, p, r, f))

        # Save for further analysize

        predictions = np.stack(predictions).tolist()
        targets = np.stack(targets).tolist()
        item = {'iteration': iteration,
                'sample': sample_indices,
                'target': targets,
                'prediction': predictions}
        path = 'checkpoints/{}.{}.{}.{}.json'.format(self.args.model, self.args.encoder, part, iteration)
        save_json(item, path)

    def metrics(self, targets, predictions):

        def avg(x):
            return sum(x) / len(x)

        precisions, recalls, fscores = [], [], []
        labels = [x for x in range(self.O, self.N + self.O)]  # works for either self.O = O or 1
        # print('Evaluation label set: ', labels)
        for _t, _p in zip(targets, predictions):
            p = 100 * precision_score(_t, _p, labels=labels, average='micro', zero_division=0)
            r = 100 * recall_score(_t, _p, labels=labels, average='micro', zero_division=0)
            f = 100 * f1_score(_t, _p, labels=labels, average='micro', zero_division=0)
            precisions.append(p)
            recalls.append(r)
            fscores.append(f)
        return avg(precisions), avg(recalls), avg(fscores)

    def fscore(self, targets, predictions):
        targets = numpy.concatenate(targets)
        predictions = numpy.concatenate(predictions)
        zeros = np.zeros(predictions.shape, dtype='int')
        numPred = np.sum(np.not_equal(predictions, zeros))
        numKey = np.sum(np.not_equal(targets, zeros))
        predictedIds = np.nonzero(predictions)
        preds_eval = predictions[predictedIds]
        keys_eval = targets[predictedIds]
        correct = np.sum(np.equal(preds_eval, keys_eval))
        # print('correct : {}, numPred : {}, numKey : {}'.format(correct, numPred, numKey))
        precision = 100.0 * correct / numPred if numPred > 0 else 0.0
        recall = 100.0 * correct / numKey
        f1 = (2.0 * precision * recall) / (precision + recall) if (precision + recall) > 0. else 0.0
        return precision, recall, f1
