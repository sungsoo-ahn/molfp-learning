import random
from rdkit import Chem
import torch
from torch.utils.data import DataLoader
from torch.utils.data.dataloader import default_collate
from torch_geometric.data import Data, Batch

from data.util import graph_data_obj_to_mol_simple


class SubBatch(Data):
    def __init__(self, batch=None, **kwargs):
        super(SubBatch, self).__init__(**kwargs)
        self.batch = batch

    @staticmethod
    def from_data_list(data_list, compute_true_target):
        keys = [set(data.keys) for data in data_list]
        keys = list(set.union(*keys))
        assert "batch" not in keys

        batch = SubBatch()

        for key in keys:
            batch[key] = []

        batch.batch = []
        batch.sub_batch = []
        batch.batch_num_nodes = []
        batch.sub_batch_num_nodes = []

        cumsum_node = 0
        sub_cumsum_node = 0

        for i, data in enumerate(data_list):
            num_nodes = data.num_nodes
            sub_num_nodes = data.sub_x.size(0)

            batch.batch.append(torch.full((num_nodes,), i, dtype=torch.long))
            batch.sub_batch.append(torch.full((sub_num_nodes,), i, dtype=torch.long))

            batch.batch_num_nodes.append(num_nodes)
            batch.sub_batch_num_nodes.append(sub_num_nodes)

            for key in data.keys:
                item = data[key]
                if key in ["edge_index"]:
                    item = item + cumsum_node
                elif key in ["sub_edge_index"]:
                    item = item + sub_cumsum_node

                batch[key].append(item)

            cumsum_node += num_nodes
            sub_cumsum_node += sub_num_nodes

        for key in keys:
            if key in ["smiles", "sub_smarts"]:
                continue

            batch[key] = torch.cat(batch[key], dim=data_list[0].__cat_dim__(key, batch[key][0]))

        batch.batch_size = len(data_list)
        batch.batch = torch.cat(batch.batch, dim=-1)
        batch.sub_batch = torch.cat(batch.sub_batch, dim=-1)
        batch.batch_num_nodes = torch.LongTensor(batch.batch_num_nodes)
        batch.sub_batch_num_nodes = torch.LongTensor(batch.sub_batch_num_nodes)

        # Generate negative samples
        neg_idxs = [idx for idx in range(batch.batch_size)]
        random.shuffle(neg_idxs)
        batch.neg_idxs = torch.LongTensor(neg_idxs)

        smarts_mol_list = [Chem.MolFromSmarts(sub_smarts) for sub_smarts in batch.sub_smarts]
        neg_smiles_list = [batch.smiles[idx] for idx in neg_idxs]
        neg_smiles_mol_list = [Chem.AllChem.MolFromSmiles(smiles) for smiles in neg_smiles_list]

        neg_targets = [
            smiles_mol.HasSubstructMatch(smarts_mol)
            for (smarts_mol, smiles_mol) in zip(smarts_mol_list, neg_smiles_mol_list)
        ]
        batch.neg_targets = torch.FloatTensor(neg_targets)

        return batch.contiguous()

    @property
    def num_graphs(self):
        """Returns the number of graphs in the batch."""
        return self.batch[-1].item() + 1


class SubDataLoader(DataLoader):
    r"""Data loader which merges data objects from a
    :class:`torch_geometric.data.dataset` to a mini-batch.
    Args:
        dataset (Dataset): The dataset from which to load the data.
        batch_size (int, optional): How may samples per batch to load.
            (default: :obj:`1`)
        shuffle (bool, optional): If set to :obj:`True`, the data will be
            reshuffled at every epoch (default: :obj:`True`)
    """

    def __init__(self, dataset, batch_size, shuffle, compute_true_target, **kwargs):
        super(SubDataLoader, self).__init__(
            dataset,
            batch_size,
            shuffle,
            collate_fn=lambda data_list: SubBatch.from_data_list(
                data_list, compute_true_target=compute_true_target
            ),
            **kwargs
        )
