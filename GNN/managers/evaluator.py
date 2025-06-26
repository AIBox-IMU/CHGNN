import os
import numpy as np
from tqdm import tqdm
from sklearn import metrics
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import ipdb
import logging

class Evaluator():
    def __init__(self, params, graph_classifier, data):
        self.params = params
        self.graph_classifier = graph_classifier
        self.graph_classifier.params.device = params.device
        self.data = data

    def eval(self, save=False):
        pos_scores = []
        pos_labels = []
        neg_scores = []
        neg_labels = []
        hpy_pos_scores = []
        hpy_neg_scores = []
        dataloader = DataLoader(self.data, batch_size=self.params.batch_size, shuffle=False, num_workers=self.params.num_workers, collate_fn=self.params.collate_fn)

        self.graph_classifier.eval()
        with torch.no_grad():
            for b_idx, batch in enumerate(dataloader):

                pos_graph, pos_label, neg_graph, neg_label = self.params.move_batch_to_device(batch, self.params.device)
                # score_pos, hpy_score_pos = self.graph_classifier(pos_graph)
                # score_neg, hpy_score_neg = self.graph_classifier(neg_graph)

                score_pos = self.graph_classifier(pos_graph)
                score_neg = self.graph_classifier(neg_graph)

                pos_scores += score_pos.squeeze(1).detach().cpu().tolist()
                neg_scores += score_neg.squeeze(1).detach().cpu().tolist()
                pos_labels += pos_label.tolist()
                neg_labels += neg_label.tolist()

                hpy_pos_scores += hpy_score_pos.squeeze(1).detach().cpu().tolist()
                hpy_neg_scores += hpy_score_neg.squeeze(1).detach().cpu().tolist()
        auc = metrics.roc_auc_score(pos_labels + neg_labels, pos_scores + neg_scores)
        auc_pr = metrics.average_precision_score(pos_labels + neg_labels, pos_scores + neg_scores)
        acc = metrics.accuracy_score(y_true=pos_labels + neg_labels, y_pred=[1 if i>=0.5 else 0 for i in pos_scores+neg_scores])

        hpy_auc = metrics.roc_auc_score(pos_labels + neg_labels, hpy_pos_scores + hpy_neg_scores)
        hpy_auc_pr = metrics.average_precision_score(pos_labels + neg_labels, hpy_pos_scores + hpy_neg_scores)
        hpy_acc = metrics.accuracy_score(y_true=pos_labels + neg_labels, y_pred=[1 if i>=0.5 else 0 for i in hpy_pos_scores+hpy_neg_scores])
        eval_result = {'auc': auc, 'auc_pr': auc_pr, 'acc': acc}
        hpy_eval_result = {'hpy_auc': hpy_auc, 'hyp_auc_pr': hpy_auc_pr, 'hpy_acc': hpy_acc}
        logging.info(f'Hpy Eval Performance:{hpy_eval_result}')
        logging.info(f'Eval Performance:{eval_result}')
        # return {'auc': auc, 'auc_pr': auc_pr, 'acc': acc}
        return {'auc': 0.9*auc+0.1*hpy_auc, 'auc_pr': 0.9*auc_pr+0.1*hpy_auc_pr, 'acc': 0.9*acc+0.1*hpy_acc}