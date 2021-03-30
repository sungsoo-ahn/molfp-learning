from collections import defaultdict
import argparse
from tqdm import tqdm
import numpy as np
from rdkit import Chem
import rdkit.Chem.AllChem as AllChem
from joblib import Parallel, delayed
import torch
import torch.nn.functional as F

from model import GraphEncoder, SubGraphEncoder, SubGraphDecoder
from data.dataset import MoleculeDataset
from data.dataloader import SearchDataLoader
from data.splitter import random_split

from data.util import (
    nx_to_graph_data_obj_simple,
    graph_data_obj_to_nx_simple,
    graph_data_obj_to_smarts,
)
from data.transform import reset_idxes


def get_graph_reps_list(loader, encoder, device):
    graph_reps_list = []
    ys_list = []
    for batch in tqdm(loader):
        ys_list.append(batch.y[: batch.batch_size])
        batch = batch.to(device)
        graph_reps = encoder(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
        graph_reps_list.append(graph_reps.cpu())

    return graph_reps_list, ys_list


def get_sub_graph_rep(graph_reps_list, ys_list, device):
    classifier = torch.nn.Linear(graph_reps_list[0].size(1), 1, bias=False).to(device)
    optim = torch.optim.Adam(classifier.parameters(), lr=1e-3)
    for _ in tqdm(range(10)):
        for graph_reps, ys in tqdm(zip(graph_reps_list, ys_list)):
            graph_reps = graph_reps.to(device)
            ys = ys.to(device)
            logits = classifier(graph_reps).squeeze(1)
            loss = F.binary_cross_entropy_with_logits(logits, (ys > 0).float())

            optim.zero_grad()
            loss.backward()
            optim.step()

    return classifier.weight.data.reshape(1, -1)


def get_smarts_candidates(graph_reps_list, sub_graph_rep, loader, encoder, sub_decoder, device):
    smarts_cnt = defaultdict(int)
    positive_cnt = 0
    total_substruct_preds = []
    for batch, graph_reps in tqdm(zip(loader, graph_reps_list)):
        batch = batch.to(device)
        graph_reps = graph_reps.to(device)
        sub_graph_reps = sub_graph_rep.repeat(batch.batch_size, 1)
        substruct_logits = torch.mean(graph_reps * sub_graph_reps, dim=1)
        substruct_preds = substruct_logits > 0
        total_substruct_preds.append(substruct_preds.cpu())
        substruct_preds = torch.split(substruct_preds, 1)
        recon_logits = sub_decoder(
            sub_graph_reps, batch.x, batch.edge_index, batch.edge_attr, batch.batch_num_nodes
        )
        recon_preds = recon_logits.reshape(-1) > 0
        recon_preds = torch.split(recon_preds, batch.batch_num_nodes.tolist())

        for idx, (substruct_pred, recon_pred) in enumerate(zip(substruct_preds, recon_preds)):
            if not substruct_pred.item():
                continue
            if not torch.any(recon_pred).item():
                continue

            data = batch.get_example(idx)
            nx_graph = graph_data_obj_to_nx_simple(data)
            inducing_nodes = torch.nonzero(recon_pred).squeeze(1).tolist()

            induced_nx_graph = nx_graph.subgraph(inducing_nodes)
            induced_nx_graph = reset_idxes(induced_nx_graph)
            subdata = nx_to_graph_data_obj_simple(induced_nx_graph)

            smarts = graph_data_obj_to_smarts(subdata.x, subdata.edge_index, subdata.edge_attr)

            positive_cnt += 1
            smarts_cnt[smarts] += 1

    sorted_keys = list(sorted(smarts_cnt.keys(), key=smarts_cnt.get, reverse=True))
    total_substruct_preds = torch.cat(total_substruct_preds, dim=0)
    print(positive_cnt)
    print({key: smarts_cnt[key] for key in sorted_keys[:5]})

    return total_substruct_preds, sorted_keys[:64]


def get_smarts_preds(smiles_list, smarts_list):
    pool = Parallel(16)

    def has_smarts_list(smarts):
        smarts_mol = Chem.MolFromSmarts(smarts)
        smiles_mol_list = [AllChem.MolFromSmiles(smiles) for smiles in smiles_list]

        return [smiles_mol.HasSubstructMatch(smarts_mol) for smiles_mol in smiles_mol_list]

    smarts_preds = pool(delayed(has_smarts_list)(smarts) for smarts in smarts_list)
    smarts_preds = torch.tensor(smarts_preds)
    return smarts_preds


def main():
    FRAC_TRAIN = 1.0
    FRAC_VALI = 0.0
    FRAC_TEST = 0.0
    BATCH_SIZE = 256
    NUM_WORKERS = 8
    DATASET_DIR = "../resource/dataset/"
    LOAD_DIR = "../resource/old_result/"
    DATASET = "clintox"

    # Training settings
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)

    args = parser.parse_args()

    torch.manual_seed(0)
    np.random.seed(0)
    device = torch.device(0)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(0)

    # set up dataset and transform function.
    dataset = MoleculeDataset(DATASET_DIR + DATASET, dataset=DATASET)
    dataset, _, _, smiles_list = random_split(
        dataset,
        null_value=0,
        frac_train=FRAC_TRAIN,
        frac_valid=FRAC_VALI,
        frac_test=FRAC_TEST,
        seed=args.seed,
        smiles_list=dataset.smiles_list,
    )
    smiles_list, _, _ = smiles_list
    smiles_list = [smiles[0] for smiles in smiles_list]
    loader = SearchDataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        compute_true_target=False,
        num_workers=NUM_WORKERS,
    )

    # set up encoder
    encoder = GraphEncoder().to(device)
    sub_encoder = SubGraphEncoder().to(device)
    sub_decoder = SubGraphDecoder().to(device)

    encoder.load_state_dict(torch.load(f"{LOAD_DIR}/encoder.pt"))
    sub_encoder.load_state_dict(torch.load(f"{LOAD_DIR}/sub_encoder.pt"))
    sub_decoder.load_state_dict(torch.load(f"{LOAD_DIR}/sub_decoder.pt"))

    encoder.eval()
    sub_encoder.eval()
    sub_decoder.eval()

    with torch.no_grad():
        graph_reps_list, ys_list = get_graph_reps_list(loader, encoder, device)

    # sub_graph_rep = torch.randn([1, 1024]).to(device)

    sub_graph_rep = get_sub_graph_rep(graph_reps_list, ys_list, device)
    sub_graph_rep /= torch.norm(sub_graph_rep)

    with torch.no_grad():
        substruct_preds, smarts_list = get_smarts_candidates(
            graph_reps_list, sub_graph_rep, loader, encoder, sub_decoder, device
        )

    smarts_preds = get_smarts_preds(smiles_list, smarts_list)

    substruct_acc = (smarts_preds == substruct_preds.view(1, -1)).float().mean(dim=1)
    print(substruct_acc)
    ys = torch.cat(ys_list, dim=0)
    old_acc = (substruct_preds == (ys > 0)).float().mean()
    print(old_acc)
    new_acc = (smarts_preds == (ys > 0).view(1, -1)).float().mean(dim=1)
    print(new_acc)


if __name__ == "__main__":
    main()
