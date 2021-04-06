### LIBRARIES ###
# Global libraries
import random

import numpy as np

from sklearn.ensemble import GradientBoostingClassifier

# Custom libraries
from utils.file_handler import load_node_info
from models.boosting.features import process_features, tf_idf

### FUNCTION DEFINITION ###


def boosting_predict(training_set, validation_set, testing_set, cfg):
    """Predicts a connection with a Boosting classifier.

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
            predictions for each element of the given dataset
    """
    # Load node information
    node_info, node_dict = load_node_info(cfg["paths"]["node_infos"])

    # Compute the TF-IDF vector of each paper
    abstract_features_tfidf, title_features_tfidf = tf_idf(node_info, cfg["language"])

    # Construct a one-hot embedding for each journal
    journal_names = {}
    journal_idx = 0
    for element in node_info:
        if element.journal not in journal_names:
            journal_names[element.journal] = journal_idx
            journal_idx += 1

    # Compute the number of neighbors of each node
    dict_neighbors = {}
    for edge in training_set:
        if edge.origin not in dict_neighbors:
            dict_neighbors[edge.origin] = set()
        dict_neighbors[edge.origin].add(edge.target)

        if edge.target not in dict_neighbors:
            dict_neighbors[edge.target] = set()
        dict_neighbors[edge.target].add(edge.origin)

    train_features, train_labels = process_features(
        training_set,
        node_dict,
        abstract_features_tfidf,
        title_features_tfidf,
        journal_names,
        dict_neighbors,
        step="train",
    )

    # Initialize basic Boosting classifier
    classifier = GradientBoostingClassifier(
        loss=cfg["boosting"]["loss"],
        learning_rate=cfg["boosting"]["lr"],
        n_estimators=cfg["boosting"]["n_estimators"],
        max_depth=cfg["boosting"]["max_depth"],
        random_state=cfg["boosting"]["random_state"],
    )
    # Train the classifier
    classifier.fit(train_features, train_labels)

    test_features = process_features(
        testing_set,
        node_dict,
        abstract_features_tfidf,
        title_features_tfidf,
        journal_names,
        dict_neighbors,
        training=False,
        step="test",
    )

    predictions_boosting = np.array(classifier.predict(test_features))

    if validation_set:
        val_features, val_labels = process_features(
            validation_set,
            node_dict,
            abstract_features_tfidf,
            title_features_tfidf,
            journal_names,
            dict_neighbors,
            step="val",
        )
        val_pred = np.array(classifier.predict(val_features))
        score = np.sum(val_pred == val_labels) / len(val_pred)
        return predictions_boosting, score
    else:
        return predictions_boosting
