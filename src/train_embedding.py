from collections import defaultdict
import argparse

from tqdm import tqdm
import numpy as np

import torch
import torch.nn.functional as F

from model import GNN
from data.dataset import MoleculeDataset
from data.dataloader import SubDataLoader
from data.splitter import random_split
from data.transform import AddRandomWalkSubstruct

import neptune


def compute_bce_with_logits_loss(logits, targets):
    logits = logits.reshape(-1)
    targets = targets.reshape(-1)
    elemwise_loss = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")

    weights = targets / torch.sum(targets) + (1 - targets) / torch.sum(1 - targets)
    weights /= torch.sum(weights)
    loss = torch.sum(weights * elemwise_loss)
    return loss


def compute_bgce_with_logits_loss(logits, targets, q):
    logits = logits.reshape(-1)
    targets = targets.reshape(-1)
    probs = targets * torch.sigmoid(logits) + (1-targets) * torch.sigmoid(-logits)
    elemwise_loss = (1 - probs.pow(q)) / q

    weights = targets / torch.sum(targets) + (1 - targets) / torch.sum(1 - targets)
    weights /= torch.sum(weights)
    loss = torch.sum(weights * elemwise_loss)

    return loss

def compute_binary_statistics(logits, targets):
    logits = logits.reshape(-1)
    targets = targets.reshape(-1)
    tp = torch.sum((torch.logical_and(logits > 0, targets > 0.5).float()))
    fp = torch.sum((torch.logical_and(logits > 0, targets < 0.5).float()))
    fn = torch.sum((torch.logical_and(logits < 0, targets > 0.5).float()))
    tn = torch.sum((torch.logical_and(logits < 0, targets < 0.5).float()))

    acc = (tp + tn) / (tp + fp + fn + tn)
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)

    return {
        "acc": acc,
        "precision": precision,
        "recall": recall,
    }


def train(model, sub_model, batch, loss_func, model_optim, sub_model_optim, device):
    model.train()
    sub_model.train()

    batch = batch.to(device)
    targets = torch.eye(batch.batch_size).to(device)

    graph_reps = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
    sub_graph_reps = sub_model(
        batch.sub_x, batch.sub_edge_index, batch.sub_edge_attr, batch.sub_batch
    )
    logits = torch.matmul(graph_reps, sub_graph_reps.T) / graph_reps.size(0)

    loss = loss_func(logits, targets)

    model_optim.zero_grad()
    sub_model_optim.zero_grad()
    loss.backward()
    model_optim.step()
    sub_model_optim.step()

    loss = loss.detach()
    statistics = {"loss": loss}
    logits = logits.detach()

    binary_statistics = compute_binary_statistics(logits, targets)
    statistics.update(binary_statistics)

    return statistics


def evaluate(model, sub_model, loader, loss_func, device):
    model.eval()
    sub_model.eval()

    cum_statistics = defaultdict(float)
    for batch in tqdm(loader):
        batch = batch.to(device)

        graph_reps = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
        sub_graph_reps = sub_model(
            batch.sub_x, batch.sub_edge_index, batch.sub_edge_attr, batch.sub_batch
        )
        logits = torch.matmul(graph_reps, sub_graph_reps.T) / graph_reps.size(0)

        loss = loss_func(logits, batch.targets)
        binary_statistics = compute_binary_statistics(logits, batch.targets)

        cum_statistics["loss"] += loss
        cum_statistics["cnt"] += batch.batch_size
        for key, val in binary_statistics.items():
            cum_statistics[key] += binary_statistics[key] * batch.batch_size

    cum_cnt = cum_statistics.pop("cnt")
    statistics = {key: val / cum_cnt for key, val in cum_statistics.items()}

    return statistics


def compute_data_statistics(loader):
    cum_cnt = 0
    cum_num_nodes = 0
    cum_num_sub_nodes = 0

    cum_statistics = defaultdict(float)
    for batch in loader:
        unintended_positives = torch.sum(batch.targets.reshape(-1) > 0.5) - batch.batch_size

        cum_statistics["cnt"] += batch.batch_size
        cum_statistics["num_nodes"] += batch.x.size(0)
        cum_statistics["num_sub_nodes"] += batch.sub_x.size(0)
        cum_statistics["unintended_positives"] += unintended_positives

    cum_cnt = cum_statistics.pop("cnt")
    statistics = {key: val / cum_cnt for key, val in cum_statistics.items()}

    return statistics


def main():
    FRAC_TRAIN = 0.95
    FRAC_VALI = 0.05
    FRAC_TEST = 0.0
    TRAIN_BATCH_SIZE = 256
    EVAL_BATCH_SIZE = 32
    NUM_WORKERS = 8
    LR = 1e-3
    TRAIN_LOG_FREQ = 10
    EVAL_LOG_FREQ = 5000
    DATASET_DIR = "../resource/dataset/"
    DATASET = "zinc_standard_agent"
    NUM_EPOCHS = 200
    GCE_Q = 0.7

    # Training settings
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--min_walk_length", type=int, default=10)
    parser.add_argument("--max_walk_length", type=int, default=40)
    parser.add_argument("--loss_scheme", type=str, default="ce")
    parser.add_argument("--gce_q", type=float, default=0.7)
    args = parser.parse_args()

    torch.manual_seed(0)
    np.random.seed(0)
    device = torch.device(0)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(0)

    # set up dataset and transform function.
    dataset = MoleculeDataset(
        DATASET_DIR + DATASET,
        dataset=DATASET,
        transform=AddRandomWalkSubstruct(
            min_walk_length=args.min_walk_length, max_walk_length=args.max_walk_length
        ),
    )
    train_dataset, vali_dataset, test_dataset = random_split(
        dataset,
        null_value=0,
        frac_train=FRAC_TRAIN,
        frac_valid=FRAC_VALI,
        frac_test=FRAC_TEST,
        seed=args.seed,
    )
    train_loader = SubDataLoader(
        train_dataset,
        batch_size=TRAIN_BATCH_SIZE,
        shuffle=True,
        compute_true_target=False,
        num_workers=NUM_WORKERS,
    )
    vali_loader = SubDataLoader(
        vali_dataset,
        batch_size=EVAL_BATCH_SIZE,
        shuffle=True,
        compute_true_target=True,
        num_workers=NUM_WORKERS,
    )

    # set up model
    model = GNN().to(device)
    sub_model = GNN().to(device)

    # set up optimizer
    model_optim = torch.optim.Adam(model.parameters(), lr=LR)
    sub_model_optim = torch.optim.Adam(sub_model.parameters(), lr=LR)

    if args.loss_scheme == "ce":
        loss_func = compute_bce_with_logits_loss
    elif args.loss_scheme == "gce":
        loss_func = lambda logits, targets: compute_bgce_with_logits_loss(
            logits, targets, args.gce_q
            )
    else:
        raise NotImplementedError

    neptune.init(project_qualified_name="sungsahn0215/molfp-learning")
    neptune.create_experiment(name="molfp-embed", params=vars(args))

    """
    data_statistics = compute_data_statistics(vali_loader)
    for key, val in data_statistics.items():
        neptune.log_metric(f"data/{key}", val)
    """

    step = 0
    for epoch in range(NUM_EPOCHS):
        for batch in tqdm(train_loader):
            step += 1

            train_statistics = train(
                model, sub_model, batch, loss_func, model_optim, sub_model_optim, device
                )
            if step % TRAIN_LOG_FREQ == 0:
                for key, val in train_statistics.items():
                    neptune.log_metric(f"train/{key}", step, val)

            if step % EVAL_LOG_FREQ == 0:
                with torch.no_grad():
                    vali_statistics = evaluate(model, sub_model, vali_loader, loss_func, device)

                for key, val in vali_statistics.items():
                    neptune.log_metric(f"vali/{key}", step, val)


if __name__ == "__main__":
    main()
