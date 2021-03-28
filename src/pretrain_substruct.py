import argparse

from tqdm import tqdm
import numpy as np

import torch

from model import GNN
from data.dataset import MoleculeDataset
from data.dataloader import SubDataLoader
from data.splitter import random_split
from data.transform import AddRandomWalkSubstruct

import neptune

criterion = torch.nn.BCEWithLogitsLoss(reduction="none")

def train(model, sub_model, batch, model_optim, sub_model_optim, device):
    batch = batch.to(device)
    try:
        targets = batch.targets.reshape(-1).float()
    except:
        targets = torch.eye(batch.batch_size).reshape(-1).to(device)
    
    weights = targets / torch.sum(targets) + (1-targets) / torch.sum(1-targets)
    weights /= torch.sum(weights)
        
    graph_reps = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
    sub_graph_reps = sub_model(
        batch.sub_x, batch.sub_edge_index, batch.sub_edge_attr, batch.sub_batch
    )
    logits = torch.matmul(graph_reps, sub_graph_reps.T).reshape(-1)

    loss = torch.sum(weights * criterion(logits, targets))
    
    model_optim.zero_grad()
    sub_model_optim.zero_grad()
    loss.backward()
    model_optim.step()
    sub_model_optim.step()
       
    tp = torch.sum((torch.logical_and(logits > 0, targets > 0.5).float()))
    fp = torch.sum((torch.logical_and(logits > 0, targets < 0.5).float()))
    fn = torch.sum((torch.logical_and(logits < 0, targets > 0.5).float()))
    tn = torch.sum((torch.logical_and(logits < 0, targets < 0.5).float()))
    
    acc = (tp + tn) / (tp + fp + fn + tn)
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    
    num_nodes = batch.x.size(0) / batch.batch_size
    num_sub_nodes = batch.sub_x.size(0) / batch.batch_size
    
    return {
        "loss": loss,
        "acc": acc,
        "precision": precision,
        "recall": recall,
        "num_nodes": num_nodes,
        "num_sub_nodes": num_sub_nodes,
    }

def evaluate(model, sub_model, loader, device):
    model.eval()
    sub_model.eval()
    
    cum_loss = 0.0
    cum_tp = 0
    cum_zero_correct = 0
    cum_sample_cnt = 0
    cum_tp = 0
    cum_tn = 0
    cum_fp = 0
    cum_fn = 0
    cum_num_nodes = 0
    cum_num_sub_nodes = 0
    cum_cnt = 0
    
    for step, batch in enumerate(tqdm(loader)):
        batch = batch.to(device)
        targets = batch.targets.reshape(-1).float()        
        
        graph_reps = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
        sub_graph_reps = sub_model(
            batch.sub_x, batch.sub_edge_index, batch.sub_edge_attr, batch.sub_batch
        )
        logits = torch.matmul(graph_reps, sub_graph_reps.T).reshape(-1)

        cum_loss += torch.sum(criterion(logits, targets)).detach().cpu()
        cum_cnt += batch.batch_size
        cum_tp += torch.sum((torch.logical_and(logits > 0, targets > 0.5).float()))
        cum_fp += torch.sum((torch.logical_and(logits > 0, targets < 0.5).float()))
        cum_fn += torch.sum((torch.logical_and(logits < 0, targets > 0.5).float()))
        cum_tn += torch.sum((torch.logical_and(logits < 0, targets < 0.5).float()))
    
        cum_num_nodes += batch.x.size(0)
        cum_num_sub_nodes += batch.sub_x.size(0)
        
    avg_loss = cum_loss / (cum_tp + cum_fp + cum_fn + cum_tn)
    avg_acc = (cum_tp + cum_tn) / (cum_tp + cum_fp + cum_fn + cum_tn)
    avg_precision = cum_tp / (cum_tp + cum_fp)
    avg_recall = cum_tp / (cum_tp + cum_fn)
    avg_unintended_positives = (cum_tp - cum_cnt) / cum_tp
    avg_num_nodes = cum_num_nodes / cum_cnt
    avg_num_sub_nodes = cum_num_sub_nodes / cum_cnt
    
    model.train()
    sub_model.train()
    
    return {
        "loss": avg_loss,
        "acc": avg_acc,
        "precision": avg_precision,
        "recall": avg_recall,
        "unintended_positives": avg_unintended_positives,
        "num_nodes": avg_num_nodes,
        "num_sub_nodes": avg_num_sub_nodes
    }
    


def main():
    # Training settings
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--num_epochs", type=int, default=200)
    parser.add_argument("--dataset_dir", type=str, default="../resource/dataset/")
    parser.add_argument("--dataset", type=str, default="zinc_standard_agent")
    parser.add_argument("--output_model_file", type=str, default="")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--min_walk_length", type=int, default=10)
    parser.add_argument("--max_walk_length", type=int, default=40)
    parser.add_argument("--train_log_freq", type=int, default=100)
    parser.add_argument("--eval_log_freq", type=int, default=5000)
    args = parser.parse_args()

    torch.manual_seed(0)
    np.random.seed(0)
    device = (
        torch.device("cuda:" + str(args.device))
        if torch.cuda.is_available()
        else torch.device("cpu")
    )
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(0)

    # set up dataset and transform function.
    dataset = MoleculeDataset(
        args.dataset_dir + args.dataset,
        dataset=args.dataset,
        transform=AddRandomWalkSubstruct(
            min_walk_length=args.min_walk_length, max_walk_length=args.max_walk_length
        ),
    )
    train_dataset, valid_dataset, test_dataset = random_split(
        dataset, null_value=0, frac_train=0.95, frac_valid=0.05, frac_test=0.0, seed=args.seed
        )
    train_loader = SubDataLoader(
        train_dataset, batch_size=256, shuffle=True, compute_true_target=False, num_workers=8
        )
    vali_loader = SubDataLoader(
        valid_dataset, batch_size=32, shuffle=True, compute_true_target=True, num_workers=8
        )
    
    # set up model
    model = GNN().to(device)
    sub_model = GNN().to(device)

    # set up optimizer
    model_optim = torch.optim.Adam(model.parameters(), lr=1e-3)
    sub_model_optim = torch.optim.Adam(sub_model.parameters(), lr=1e-3)
    
    neptune.init(project_qualified_name="sungsahn0215/molfp-learning")
    neptune.create_experiment(name="molfp-embed", params=vars(args))

    step = 0
    for epoch in range(args.num_epochs):
        for batch in tqdm(train_loader):
            train_statistics = train(model, sub_model, batch, model_optim, sub_model_optim, device)
            step += 1
            
            if step % args.train_log_freq == 0:
                for key, val in train_statistics.items():
                    neptune.log_metric(step, f"{key}/train", val)

            if step % args.eval_log_freq == 0:
                vali_statistics = evaluate(model, sub_model, vali_loader, device)
                for key, val in vali_statistics.items():
                    neptune.log_metric(step, f"{key}/vali", val)            

if __name__ == "__main__":
    main()
