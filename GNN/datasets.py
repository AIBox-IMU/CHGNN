import os
import lmdb
import json
import torch
from torch.utils.data import Dataset

from utils import *
import time

def generate_subgraph_datasets(params, splits=None, saved_relation2id=None, max_label_value=None):

    if splits is None:
        splits = ['train', 'valid']
    testing = 'test' in splits
    adj_list, triplets, entity2id, relation2id, id2entity, id2relation = process_files(params.file_paths, saved_relation2id)
    data_path = os.path.join(params.main_dir, f'../data/{params.dataset}/relation2id.json')
    if not os.path.isdir(data_path) and not testing:
        with open(data_path, 'w') as f:
            json.dump(relation2id, f)

    graphs = {split_name: {'triplets': triplets[split_name], 'max_size': params.max_links} for split_name in splits}


    # Sample train and valid/test links
    for split in graphs.values():
        split['pos'], split['neg'] = sample_neg(adj_list, split['triplets'], params.num_neg_samples_per_link, max_size=split['max_size'], constrained_neg_prob=params.constrained_neg_prob)

    if testing:
        directory = os.path.join(params.main_dir, f'../data/{params.dataset}/')
        save_to_file(directory, f'neg_{params.test_file}_{params.constrained_neg_prob}.txt', graphs['test']['neg'], id2entity, id2relation)

    links2subgraphs(adj_list, graphs, params, max_label_value)


class SubgraphDataset(Dataset):
    """Extracted, labeled, subgraph dataset -- DGL Only"""

    def __init__(self, db_path, db_name, raw_data_paths, included_relations=None, add_traspose_rels=False, num_neg_samples_per_link=1):
        self.main_env = lmdb.open(db_path, readonly=True, max_dbs=3, lock=False)
        db_name_pos = f"{db_name}_pos"
        db_name_neg = f"{db_name}_neg"
        self.db_pos = self.main_env.open_db(db_name_pos.encode())
        self.db_neg = self.main_env.open_db(db_name_neg.encode())
        self.num_neg_samples_per_link = num_neg_samples_per_link
        ssp_graph, triplets, __, __, id2entity, id2relation = process_files(raw_data_paths, included_relations)
        
        self.add_traspose_rels = add_traspose_rels

        if add_traspose_rels:
            ssp_graph_t = [adj.T for adj in ssp_graph]
            ssp_graph += ssp_graph_t
        
        self.num_rels = len(ssp_graph)
        
        self.graph = ssp_multigraph_to_dgl(ssp_graph)
        self.triplets = triplets
        self.ssp_graph = ssp_graph
        self.id2entity = id2entity
        self.id2relation = id2relation
        with self.main_env.begin(db=self.db_pos) as txn:
            self.num_graphs_pos = int.from_bytes(txn.get('num_graphs'.encode()), byteorder='little')

        test_sample = self.__getitem__(0)

    def __getitem__(self, index):
        with self.main_env.begin(db=self.db_pos) as txn:
            str_id = '{:08}'.format(index).encode('ascii')

            subgraph_nodes_pos, subgraph_labels_pos, _ = deserialize(txn.get(str_id)).values()
            subgraph_pos = self._prepare_subgraphs(subgraph_nodes_pos, subgraph_labels_pos)
        
        subgraphs_neg_list = []
        with self.main_env.begin(db=self.db_neg) as txn:
            for i in range(self.num_neg_samples_per_link):
                str_id = '{:08}'.format(index + i * (self.num_graphs_pos)).encode('ascii')
                subgraph_nodes_neg, subgraph_labels_neg, _ = deserialize(txn.get(str_id)).values()
                subgraphs_neg_list.append(self._prepare_subgraphs(subgraph_nodes_neg, subgraph_labels_neg))

        return subgraph_pos, 1, subgraphs_neg_list, [0 for _ in range(len(subgraphs_neg_list))]

    def __len__(self):
        return self.num_graphs_pos

    def _prepare_subgraphs(self, subgraph_nodes, subgraph_labels):
        subgraph = self.graph.subgraph(subgraph_nodes)
        
        heads, tails, eids = subgraph.edges('all')

        # indicator for edges between the head and tail
        indicator1 = torch.logical_and(heads==0, tails==1)
        # indicator for edges to be predicted
        indicator2 = torch.logical_and(subgraph.edata['type'] == subgraph_labels, indicator1)

        if self.add_traspose_rels:
            # indicator for the transpose relation of the target edge
            indicator3 = torch.logical_and(heads==1, tails==0)
            indicator4 = torch.logical_and(subgraph.edata['type'] == (subgraph_labels + self.num_rels // 2), indicator3)
            indicator2 = torch.logical_or(indicator4, indicator2)
 
        subgraph.edata["target_edge"] = torch.zeros(subgraph.edata['type'].shape).type(torch.BoolTensor)

        if indicator2.sum() == 0:
            subgraph.add_edges(0, 1)
            subgraph.edata['type'][-1] = torch.tensor(subgraph_labels).type(torch.IntTensor)
            subgraph.edata['target_edge'][-1] = torch.tensor(1).type(torch.BoolTensor)
            if self.add_traspose_rels:
                subgraph.add_edges(1, 0)
                subgraph.edata['type'][-1] = torch.tensor(subgraph_labels + self.num_rels // 2).type(torch.IntTensor)
                subgraph.edata['target_edge'][-1] = torch.tensor(1).type(torch.BoolTensor)
        else:
            subgraph.edata['target_edge'][eids[indicator2]] = torch.ones(len(eids[indicator2])).type(torch.BoolTensor)
        
        subgraph.ndata["degree"] = subgraph.out_degrees() + 1
        return subgraph
    