### LIBRARIES ###
# Global libraries
import random

import numpy as np

from sklearn import svm
from sklearn.metrics.pairwise import linear_kernel

# Custom libraries
from utils.file_handler import load_node_info
from models.baseline.features import process_features, tf_idf

### FUNCTION DEFINITION ###


def baseline_predict(training_set, validation_set, testing_set, cfg):
    """Predicts a connection with the baseline method.

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
    features_tfidf = tf_idf(node_info, cfg["language"])

    # to_keep = random.sample(
    #    range(len(training_set)), k=int(round(len(training_set) * 0.05))
    # )
    # training_set_reduced = [training_set[i] for i in range(len(to_keep))]
    # train_features, train_labels = process_features(training_set_reduced, node_dict)
    train_features, train_labels = process_features(
        training_set, node_dict, step="train"
    )

    # Initialize basic SVM
    classifier = svm.LinearSVC(max_iter=10000)
    # Train the SVM
    classifier.fit(train_features, train_labels)

    test_features = process_features(
        testing_set, node_dict, training=False, step="test"
    )

    predictions_SVM = np.array(classifier.predict(test_features))

    if validation_set:
        val_features, val_labels = process_features(
            validation_set, node_dict, step="val"
        )
        val_pred = np.array(classifier.predict(val_features))
        score = np.sum(val_pred == val_labels) / len(val_pred)
        return predictions_SVM, score
    else:
        return predictions_SVM
