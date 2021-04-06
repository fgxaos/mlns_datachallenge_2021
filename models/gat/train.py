### LIBRARIES ###
# Global libraries
import os
from tqdm import tqdm
import wandb

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# Custom libraries
from models.gat.gat import GAT, init_weights
from utils.torch_helper import todevice

### FUNCTION DEFINITIONS ###
def train(train_loader, val_loader, checkpoint_path, input_dim, output_dim, cfg):
    """Function to train the GAT model.

    Args:
        train_loader: DataLoader
            dataset to use for training
        val_loader: DataLoader
            dataset to use for validation
        checkpoint_path: str
            path to which to save the trained model
        input_dim: int
            dimension of the input
        output_dim: int
            dimension of the output
        cfg: Dict
            configuration to use
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = GAT(
        input_dim,
        cfg["gat"]["hidden_dims"],
        output_dim,
        cfg["gat"]["n_heads"],
        cfg["gat"]["dropout"],
        device,
    )
    model.apply(init_weights)
    model.to(device)

    optimizer = optim.Adam(
        model.parameters(), lr=cfg["gat"]["lr"], weight_decay=cfg["gat"]["weight_decay"]
    )
    criterion = nn.BCEWithLogitsLoss()

    best_val_acc = 0

    train_step = 0
    val_step = 0
    for epoch in tqdm(range(cfg["gat"]["n_epochs"]), desc="Training epochs"):
        # Training step
        model.train()
        for i, batch in tqdm(enumerate(train_loader), desc="Training batch"):
            # Get the batch
            edges, features, node_layers, mappings, rows, labels = batch
            features, labels = features.to(device), labels.to(device)
            optimizer.zero_grad()
            out = model(features, node_layers, mappings, rows)
            all_pairs = torch.mm(out, out.t())
            scores = all_pairs[edges.T]
            loss_train = criterion(scores, labels.float())
            acc_train = torch.sum(scores == labels.float()) / len(scores)
            if cfg["use_wandb"]:
                wandb.log(
                    {
                        "train_step": train_step,
                        "loss_train": loss_train,
                        "acc_train": acc_train,
                    }
                )
            loss_train.backward()
            optimizer.step()
            train_step += 1

        # Validation step
        model.eval()
        with torch.no_grad():
            epoch_val_acc = 0
            for i, batch in tqdm(enumerate(val_loader), desc="Validation batch"):
                # Get the batch
                edges, features, node_layers, mappings, rows, labels = batch
                features, labels = features.to(device), labels.to(device)
                out = model(features, node_layers, mappings, rows)
                all_pairs = torch.mm(out, out.t())
                scores = all_pairs[edges.T]
                print("Scores: ", scores)
                print("Labels: ", labels)
                print("Shape scores: ", scores.shape)
                print("Shape labels: ", labels.shape)
                loss_val = criterion(scores, labels.float())
                acc_val = torch.sum(scores == labels.float()) / len(scores)

                if cfg["use_wandb"]:
                    wandb.log(
                        {"val_step": val_step, "loss_val": loss_val, "acc_val": acc_val}
                    )
                val_step += 1
                epoch_val_acc += torch.sum(scores == labels.float())

        # Save the best model (on validation accuracy)
        if epoch_val_acc > best_val_acc:
            best_val_acc = epoch_val_acc
            torch.save(model.state_dict(), checkpoint_path)


def test(test_loader, checkpoint_path, input_dim, output_dim, cfg):
    """Function to test the GAT model.

    Args:
        train_loader: DataLoader
            dataset to use for training
        val_loader: DataLoader
            dataset to use for validation
        checkpoint_path: str
            path to which to save the trained model
        input_dim: int
            dimension of the input
        output_dim: int
            dimension of the output
        cfg: Dict
            configuration to use
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = load_gat_model(checkpoint_path, input_dim, output_dim, device)
    test_preds = torch.zeros(len(test_loader))

    model.eval()
    with torch.no_grad():
        for i, batch in enumerate(test_loader):
            edges, features, node_layers, mappings, rows, labels = batch
            features, labels = features.to(device), labels.to(device)
            out = model(features, node_layers, mappings, rows)
            all_pairs = torch.mm(out, out.t())
            scores = all_pairs[edges.T]


def load_get_model(ckpt_path, input_dim, output_dim, device):
    """Loads a pretrained GAT model.

    Args:
        ckpt_path: str
            path of the pretrained model weights
        input_dim: int
            dimension of the input
        output_dim: int
            dimension of the output
        device: str
            device to use
    Returns:
        model: GAT
            pretrained GAT model
    """
    model = GAT(
        input_dim,
        cfg["gat"]["hidden_dims"],
        output_dim,
        cfg["gat"]["n_heads"],
        cfg["gat"]["dropout"],
        device,
    )
    model.load_state_dict(torch.load(ckpt_path))
    model.eval()

    return model