from collections import defaultdict
import argparse

from tqdm import tqdm
import numpy as np

import torch
import torch.nn.functional as F
from torch_geometric.nn import global_max_pool

from model import GraphEncoder, SubGraphEncoder, SubGraphDecoder
from data.dataset import MoleculeDataset
from data.dataloader import SubDataLoader
from data.splitter import random_split
from data.transform import AddRandomWalkSubStruct

import neptune


def compute_bce_with_logits_loss(logits, targets):
    logits = logits.reshape(-1)
    targets = targets.reshape(-1)
    elemwise_loss = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")

    weights = targets / torch.sum(targets) + (1 - targets) / torch.sum(1 - targets)
    weights /= torch.sum(weights)
    loss = torch.sum(weights * elemwise_loss)
    return loss


def compute_kl_div_loss(mu, log_var):
    kl_div = torch.mean(-0.5 * torch.mean(1 + log_var - mu ** 2 - log_var.exp(), dim=1), dim=0)
    return kl_div


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


def train(
    encoder,
    sub_encoder,
    sub_decoder,
    batch,
    encoder_optim,
    sub_encoder_optim,
    sub_decoder_optim,
    device,
):
    encoder.train()
    sub_encoder.train()
    sub_decoder.train()

    batch = batch.to(device)
    substruct_targets = torch.cat([torch.ones_like(batch.neg_targets), batch.neg_targets], dim=0)

    # Compute KL divergence loss
    sub_graph_mu, sub_graph_logvar = sub_encoder(
        batch.sub_x, batch.sub_edge_index, batch.sub_edge_attr, batch.sub_batch
    )
    sub_graph_reps = sub_encoder.reparameterize(sub_graph_mu, sub_graph_logvar)
    kl_div_loss = compute_kl_div_loss(sub_graph_mu, sub_graph_logvar)

    # Compute recon loss
    recon_logits = sub_decoder(
        sub_graph_reps, batch.x, batch.edge_index, batch.edge_attr, batch.batch_num_nodes
    )
    recon_loss = compute_bce_with_logits_loss(recon_logits, batch.sub_mask)

    # Compute BCE loss
    graph_reps = encoder(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
    neg_graph_reps = graph_reps.index_select(dim=0, index=batch.neg_idxs)
    graph_and_neg_graph_reps = torch.cat([graph_reps, neg_graph_reps], dim=0)
    twice_sub_graph_reps = torch.cat([sub_graph_reps, sub_graph_reps], dim=0)
    substruct_logits = torch.mean(graph_and_neg_graph_reps * twice_sub_graph_reps, dim=1)
    substruct_loss = compute_bce_with_logits_loss(substruct_logits, substruct_targets)

    # Compute total loss
    loss = kl_div_loss + recon_loss + substruct_loss

    encoder_optim.zero_grad()
    sub_encoder_optim.zero_grad()
    sub_decoder_optim.zero_grad()
    loss.backward()
    encoder_optim.step()
    sub_encoder_optim.step()
    sub_decoder_optim.step()

    loss = loss.detach()
    substruct_loss = substruct_loss.detach()
    kl_div_loss = kl_div_loss.detach()
    recon_logits = recon_logits.detach()
    substruct_logits = substruct_logits.detach()
    statistics = {"loss": loss, "substruct_loss": substruct_loss, "kl_div_loss": kl_div_loss}

    recon_binary_statistics = compute_binary_statistics(recon_logits, batch.sub_mask)
    for key, val in recon_binary_statistics.items():
        statistics[f"recon_{key}"] = val

    recon_correct = ((recon_logits > 0) == (batch.sub_mask > 0.5)).float()
    statistics[f"exact_recon_acc"] = -global_max_pool(-recon_correct, batch.batch).mean()

    substruct_binary_statistics = compute_binary_statistics(substruct_logits, substruct_targets)
    for key, val in substruct_binary_statistics.items():
        statistics[f"substruct_{key}"] = val

    return statistics


def evaluate(encoder, sub_encoder, sub_decoder, loader, device):
    encoder.eval()
    sub_encoder.eval()
    sub_decoder.eval()

    cum_statistics = defaultdict(float)
    for batch in tqdm(loader):
        batch = batch.to(device)
        substruct_targets = torch.cat([torch.ones_like(batch.neg_targets), batch.neg_targets], dim=0)

        # Compute KL divergence loss
        sub_graph_mu, sub_graph_logvar = sub_encoder(
            batch.sub_x, batch.sub_edge_index, batch.sub_edge_attr, batch.sub_batch
        )
        sub_graph_reps = sub_encoder.reparameterize(sub_graph_mu, sub_graph_logvar)
        kl_div_loss = compute_kl_div_loss(sub_graph_mu, sub_graph_logvar)

        # Compute recon loss
        recon_logits = sub_decoder(
            sub_graph_reps, batch.x, batch.edge_index, batch.edge_attr, batch.batch_num_nodes
        )
        recon_loss = compute_bce_with_logits_loss(recon_logits, batch.sub_mask)

        # Compute BCE loss
        graph_reps = encoder(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
        neg_graph_reps = graph_reps.index_select(dim=0, index=batch.neg_idxs)
        graph_and_neg_graph_reps = torch.cat([graph_reps, neg_graph_reps], dim=0)
        twice_sub_graph_reps = torch.cat([sub_graph_reps, sub_graph_reps], dim=0)
        substruct_logits = torch.mean(graph_and_neg_graph_reps * twice_sub_graph_reps, dim=1)
        substruct_loss = compute_bce_with_logits_loss(substruct_logits, substruct_targets)

        # Compute total loss
        loss = kl_div_loss + recon_loss + substruct_loss

        # Compute statistics
        statistics = {"loss": loss, "substruct_loss": substruct_loss, "kl_div_loss": kl_div_loss}

        recon_binary_statistics = compute_binary_statistics(recon_logits, batch.sub_mask)
        for key, val in recon_binary_statistics.items():
            statistics[f"recon_{key}"] = val

        recon_correct = ((recon_logits > 0) == (batch.sub_mask > 0.5)).float()
        statistics[f"exact_recon_acc"] = global_mean_pool(recon_correct, batch.batch).mean()

        substruct_binary_statistics = compute_binary_statistics(substruct_logits, substruct_targets)
        for key, val in substruct_binary_statistics.items():
            statistics[f"substruct_{key}"] = val

        for key, val in statistics.items():
            cum_statistics[key] += val * batch.batch_size

        cum_statistics["cnt"] += batch.batch_size

    cum_cnt = cum_statistics.pop("cnt")
    statistics = {key: val / cum_cnt for key, val in cum_statistics.items()}

    return statistics


def compute_data_statistics(loader):
    cum_statistics = defaultdict(float)
    for batch in loader:
        positives = torch.sum(batch.neg_targets.reshape(-1) > 0.5) + batch.batch_size

        cum_statistics["cnt"] += batch.batch_size
        cum_statistics["positive_ratio"] += 0.5 * positives
        cum_statistics["num_nodes"] += batch.x.size(0)
        cum_statistics["num_sub_nodes"] += batch.sub_x.size(0)

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

    # Training settings
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--min_walk_length", type=int, default=2)
    parser.add_argument("--max_walk_length", type=int, default=10)
    parser.add_argument("--loss_scheme", type=str, default="ce")
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
        transform=AddRandomWalkSubStruct(
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

    # set up encoder
    encoder = GraphEncoder().to(device)
    sub_encoder = SubGraphEncoder().to(device)
    sub_decoder = SubGraphDecoder().to(device)

    # set up optimizer
    encoder_optim = torch.optim.Adam(encoder.parameters(), lr=LR)
    sub_encoder_optim = torch.optim.Adam(sub_encoder.parameters(), lr=LR)
    sub_decoder_optim = torch.optim.Adam(sub_decoder.parameters(), lr=LR)

    neptune.init(project_qualified_name="sungsahn0215/molfp-learning")
    neptune.create_experiment(name="molfp-embed", params=vars(args))

    #data_statistics = compute_data_statistics(vali_loader)
    #for key, val in data_statistics.items():
    #    neptune.log_metric(f"data/{key}", val)

    step = 0
    for epoch in range(NUM_EPOCHS):
        for batch in tqdm(train_loader):
            step += 1

            train_statistics = train(
                encoder,
                sub_encoder,
                sub_decoder,
                batch,
                encoder_optim,
                sub_encoder_optim,
                sub_decoder_optim,
                device,
            )
            if step % TRAIN_LOG_FREQ == 0:
                for key, val in train_statistics.items():
                    neptune.log_metric(f"train/{key}", step, val)

            if step % EVAL_LOG_FREQ == 0:
                with torch.no_grad():
                    vali_statistics = evaluate(
                        encoder, sub_encoder, sub_decoder, vali_loader, device
                    )

                for key, val in vali_statistics.items():
                    neptune.log_metric(f"vali/{key}", step, val)


if __name__ == "__main__":
    main()
