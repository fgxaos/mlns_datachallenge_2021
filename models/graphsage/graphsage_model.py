### LIBRARIES ###
# Global libraries
import os

import torch
from torch.utils.data import DataLoader

# Custom libraries
from models.graphsage.graphsage import GraphSage
from models.graphsage.train import train, test
from models.graphsage.data_handler import GraphDataset

### FUNCTION DEFINITION ###
def graphsage_predict(training_set, validation_set, testing_set, cfg):
    """Predicts a connection with a GraphSage model.

    Args:
        training_set: List
            list of elements from the training dataset
        validation_set: List
            list of elements from the validation dataset
        testing_set: List
            list of elements from the testing dataset
        cfg: Dict
            configuration to use
    Returns:
        predictions: np.array (values in {0, 1})
            connection predictions for each element of the given dataset.
    """
    # Convert the three sets in three separate nn.Dataset objects
    # The three datasets use the same adjacency matrix from the training set
    # Be careful, don't cheat by adding links that are in the validation set!
    print("Generating datasets...")
    train_dataset = GraphDataset(training_set, cfg, True, "train")
    val_dataset = GraphDataset(validation_set, cfg, True, "val")
    test_dataset = GraphDataset(testing_set, cfg, False, "test")

    # n_features = train_dataset.features.shape[1]
    input_dim, output_dim = train_dataset.get_dims()

    # Create the associate DataLoader objects
    print("Generating dataloaders...")
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg["graphsage"]["batch_size"],
        shuffle=True,
        collate_fn=train_dataset.collate_wrapper,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg["graphsage"]["batch_size"],
        shuffle=True,
        collate_fn=val_dataset.collate_wrapper,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=True,
        collate_fn=test_dataset.collate_wrapper,
    )

    # Load a pretrained GAT model with the given characteristics,
    # or train a new one if there is none
    checkpoint_name = "ft_{}-ep_{}-dout_{}-lr_{}".format(
        input_dim,
        cfg["graphsage"]["n_epochs"],
        cfg["graphsage"]["dropout"],
        cfg["graphsage"]["lr"],
    )
    checkpoint_path = os.path.join(cfg["graphsage"]["checkpoints"], checkpoint_name)

    if not os.path.exists(checkpoint_path):
        train(train_loader, val_loader, checkpoint_path, input_dim, output_dim, cfg)

    graphsage_predictions = test(
        test_loader, checkpoint_path, input_dim, output_dim, cfg
    )

    # Send the validation and test datasets to the GAT model
    if validation_set:
        val_pred = gat_model(val_dataset)
        score = torch.sum(val_pred == val_labels) / len(val_pred)
        return graphsage_predictions.cpu().numpy(), score
    else:
        return graphsage_predictions.cpu().numpy()
