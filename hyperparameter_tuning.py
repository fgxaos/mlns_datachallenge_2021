### LIBRARIES ###
# Global libraries
import os
import yaml
from statistics import mean, stdev

import nltk

from ray import tune

# Custom libraries
from models.random.rand_model import random_predict
from models.baseline.baseline_model import baseline_predict
from models.svm.svm_model import svm_predict
from models.rand_forest.rand_forest_model import rand_forest_predict
from models.log_reg.log_reg_model import log_reg_predict
from models.boosting.boosting_model import boosting_predict
from models.gat.gat_model import gat_predict
from models.graphsage.graphsage_model import graphsage_predict

from utils.file_handler import (
    load_train_data,
    load_test_data,
    save_predictions,
)

### LOAD CONFIG ###
with open("cfg.yml", "r") as cfg_file:
    cfg = yaml.safe_load(cfg_file)

# Build non-existing folders
if not os.path.exists(cfg["paths"]["pred_folder"]):
    os.makedirs(cfg["paths"]["pred_folder"])
if not os.path.exists(cfg["gat"]["checkpoints"]):
    os.makedirs(cfg["gat"]["checkpoints"])
# Set up nltk
nltk.download("punkt")
nltk.download("stopwords")

### LOAD DATASETS ###
# Load train data
training_set, validation_set = load_train_data(
    cfg["paths"]["train_set"], split=cfg["val_split"]
)

# Load test data
testing_set = load_test_data(cfg["paths"]["test_set"])


def run_experiment(cfg):
    """Runs a single experiment.

    Args:
        cfg: Dict
            configuration to use
    """
    os.chdir("/home/francois/Documents/Cours CS/MLNS/mlns_kaggle_2021")
    ### BASELINE METHOD ###
    if cfg["method"] == "baseline":
        baseline_predictions, score = baseline_predict(
            training_set, validation_set, testing_set, cfg
        )
        tune.report(val_score=score)

    ### SVM METHOD ###
    elif cfg["method"] == "svm":
        svm_predictions, score = svm_predict(
            training_set, validation_set, testing_set, cfg
        )
        tune.report(val_score=score)

    ### RANDOM FOREST METHOD ###
    elif cfg["method"] == "rand_forest":
        rand_forest_predictions, score = rand_forest_predict(
            training_set, validation_set, testing_set, cfg
        )
        tune.report(val_score=score)

    ### LOGISTIC REGRESSION METHOD ###
    elif cfg["method"] == "log_reg":
        log_reg_predictions, score = log_reg_predict(
            training_set, validation_set, testing_set, cfg
        )
        tune.report(val_score=score)

    ### BOOSTING METHOD ###
    elif cfg["method"] == "boosting":
        boosting_predictions, score = boosting_predict(
            training_set, validation_set, testing_set, cfg
        )
        tune.report(val_score=score)

    ### GRAPHSAGE METHOD ###
    elif cfg["method"] == "graphsage":
        graphsage_predictions, score = graphsage_predict(
            training_set, validation_set, testing_set, cfg
        )
        tune.report(val_score=score)

    ### GAT METHOD ###
    elif cfg["method"] == "gat":
        gat_predictions, score = gat_predict(
            training_set, validation_set, testing_set, cfg
        )
        tune.report(val_score=score)


### CREATE LIST OF EXPERIMENTS ###
analysis = tune.run(
    run_experiment,
    config={
        **cfg,
        "number_tries": 1,
        "C": tune.grid_search([0.001, 0.01, 0.1, 1.0]),
    },
)

### DISPLAY RESULTS ###
print("Best config: ", analysis.get_best_config(metric="val_score", mode="max"))
