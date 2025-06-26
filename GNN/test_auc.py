import os
import argparse
import logging
import ipdb
from warnings import simplefilter
import torch
import time
import numpy as np
import dgl
import multiprocessing as mp

from datasets import SubgraphDataset, generate_subgraph_datasets
from managers.evaluator import Evaluator
from utils import *

set_rand_seed(822)


def main(params):
    simplefilter(action='ignore', category=UserWarning)

    if params.gpu >= 0:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(params.gpu)
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    model = torch.load(params.model_path, map_location=device)
    model.device = device
    model.params.device = device

    params.enclosing_sub_graph = model.params.enclosing_sub_graph
    if not hasattr(model.params, 'residual'):
        model.params.residual = params.residual

    if params.aug:
        params.db_path = os.path.join(params.main_dir, "..", f'data/{params.dataset}/subgraphs_aug_{params.aug}_enclose_{params.enclosing_sub_graph}_neg_{params.num_neg_samples_per_link}_hop_{params.hop}_unhop_{params.un_hop}')

    else:
        params.db_path = os.path.join(params.main_dir, "..", f'data/{params.dataset}/subgraphs_enclose_{params.enclosing_sub_graph}_neg_{params.num_neg_samples_per_link}_hop_{params.hop}')
    print(params.db_path)

    if not os.path.isdir(params.db_path):
        generate_subgraph_datasets(params,splits=['test'])

    start = time.time()

    test_data = SubgraphDataset(params.db_path, 'test', params.file_paths,
                            add_traspose_rels=params.add_traspose_rels,
                            num_neg_samples_per_link=params.num_neg_samples_per_link)
    test_evaluator = Evaluator(params, model, test_data)
    result = test_evaluator.eval(save=True)

    end = time.time()
    logger.info(f'Testing time used: {end - start} s')
    logging.info(f'{str(result)} ')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Testing script')

    # Experiment setup params
    parser.add_argument("--experiment_name", "-e", type=str, default="fb_v2_margin_loss",
                        help="Experiment name. Log file with this name will be created")
    parser.add_argument("--dataset", "-d", type=str, default="fb237_v2",
                        help="Path to dataset")
    parser.add_argument('--aug', type=bool, default=False,
                        help='whether to sample augmented subgraph')
    
    parser.add_argument("--hop", type=int, default=3,
                        help="How many hops to go while extracting subgraphs?")
    parser.add_argument("--un_hop", type=int, default=1,
                        help="How many hops to go while extracting augmented subgraphs? (only active when param 'aug' is True)")
    parser.add_argument("--gpu", type=int, default=0,
                        help="gpu id")
    parser.add_argument("--batch_size", "-bs", type=int, default=32,
                        help="test batch size")
    parser.add_argument('--add_traspose_rels', '-tr', type=bool, default=True,
                        help='Whether to append adj matrix list with symmetric relations?')
    parser.add_argument("--residual", "-res", type=bool, default=False)

    parser.add_argument("--num_neg_samples_per_link", '-neg', type=int, default=1,
                        help="Number of negative examples to sample per positive link")
    parser.add_argument("--max_links", type=int, default=1000000,
                    help="Set maximum number of train links (to fit into memory)")
    parser.add_argument('--constrained_neg_prob', '-cn', type=float, default=0.0,
                        help='with what probability to sample constrained heads/tails while neg sampling')
    parser.add_argument("--test_file", "-t", type=str, default="test",
                        help="Name of file containing test triplets")
    parser.add_argument("--num_workers", type=int, default=8,
                        help="Number of dataloading processes")
    params = parser.parse_args()
    logging.basicConfig(level=logging.INFO)

    params.main_dir = os.path.relpath(os.path.dirname(os.path.abspath(__file__)))
    params.file_paths = {
        'train': os.path.join(params.main_dir, '../data', params.dataset, 'train.txt'),
        'test': os.path.join(params.main_dir, '../data', params.dataset, 'test.txt')
    }
    if torch.cuda.is_available():
        params.device = torch.device('cuda')

    else:
        params.device = torch.device('cpu')
    params.collate_fn = collate_dgl
    params.move_batch_to_device = move_batch_to_device_dgl

    params.model_path = os.path.join(params.main_dir, 'experiments', params.experiment_name,
                                     'best_graph_classifier.pth')

    logger = logging.getLogger()

    file_handler = logging.FileHandler(
        os.path.join(params.main_dir, 'experiments', params.experiment_name, 'log_test_auc.txt'))
    logger.addHandler(file_handler)

    logger.info('============ Initialized logger ============')
    logger.info('\n'.join(f'{k}: {str(v)}' for k, v in sorted(dict(vars(params)).items())))

    logger.info('============================================')

    main(params)
