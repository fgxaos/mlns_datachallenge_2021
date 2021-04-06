### LIBRARIES ###
# Global libraries
import csv
import random

# Custom libraries
from utils.types import Node, Edge

### FUNCTION DEFINITIONS ###
def load_test_data(path):
    """Loads the test dataset.

    Args:
        path: str
            path of the test dataset file
    Returns:
        testing_set: List[Edge]
            list of the elements in the test dataset
    """
    with open(path, "r") as f:
        reader = csv.reader(f)
        testing_set = list(reader)

    return [Edge(*element[0].split(" ")) for element in testing_set]


def load_train_data(path, split=None, shuffle=True):
    """Loads the train dataset.

    Args:
        path: str
            path of the train dataset file
        split: float
            percentage of the train dataset to use for validation
            if not specified, does not create a validation dataset
        shuffle: boolean
            whether to shuffle the training (and validation) datasets
    Returns:
        train_set: List[Edge]
            list of the elements in the train dataset
        val_set: List[Edge]
            if specified, list of the elements in the validation dataset
    """
    with open(path, "r") as f:
        reader = csv.reader(f)
        train_data = list(reader)

    if split:
        if shuffle:
            # Shuffle the train set to have different training and validation datasets
            random.shuffle(train_data)

        val_split = int(len(train_data) * split)
        train_set = [Edge(*element[0].split(" ")) for element in train_data[val_split:]]
        val_set = [Edge(*element[0].split(" ")) for element in train_data[:val_split]]

        return train_set, val_set

    else:
        return [Edge(*element[0].split(" ")) for element in train_data]


def load_node_info(path):
    """Loads node information.

    Args:
        path: str
            path of the node information file
    Returns:
        node_info: List[Node]
            list of the node informations
        node_dict: Dict{int: Node}
            dictionary to find a node by ID
    """
    with open(path, "r") as f:
        reader = csv.reader(f)
        node_info = list(reader)
    node_info = [Node(*element) for element in node_info]

    node_dict = {}
    for node in node_info:
        node_dict[node.id] = node

    return node_info, node_dict


def save_predictions(predictions, path):
    """Saves the predictions.

    Args:
        predictions: np.array
            predictions to save
        path: str
            path to use to save the predictions
    """
    results = zip(range(len(predictions)), predictions)
    with open(path, "w") as pred:
        csv_out = csv.writer(pred, fieldnames=["id", "category"])
        csv_out.writeheader()
        for row in results:
            csv_out.writerow(row)
