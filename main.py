### LIBRARIES ###
# Global libraries
import os
import yaml
from statistics import mean, stdev

import nltk

import wandb

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

# Set up wandb (if activated)
if cfg["use_wandb"]:
    wandb.init(project="mlns_kaggle_2021")
    wandb.config.method = cfg["method"]
    wandb.run.name = f"{cfg['method']}-c_{cfg['svm_C']}"

### MAIN PROGRAM ###
if cfg["validation"]:
    val_scores = []

for i in range(cfg["number_tries"]):
    print(f"=== Try {i+1}/{cfg['number_tries']} ===")
    # Load train data
    if cfg["validation"]:
        training_set, validation_set = load_train_data(
            cfg["paths"]["train_set"], split=cfg["val_split"]
        )
    else:
        training_set = load_train_data(cfg["paths"]["train_set"])
        validation_set = None

    # Load test data
    testing_set = load_test_data(cfg["paths"]["test_set"])

    ### RANDOM METHOD ###
    if cfg["method"] == "random":
        random_pred_path = os.path.join(cfg["paths"]["pred_folder"], "random_test.csv")
        random_predictions = random_predict(len(testing_set))
        save_predictions(random_predictions, random_pred_path)

    ### BASELINE METHOD ###
    elif cfg["method"] == "baseline":
        baseline_pred_path = os.path.join(
            cfg["paths"]["pred_folder"], "baseline_test.csv"
        )
        if cfg["validation"]:
            baseline_predictions, score = baseline_predict(
                training_set, validation_set, testing_set, cfg
            )
            val_scores.append(score)
        else:
            baseline_predictions = baseline_predict(
                training_set, validation_set, testing_set, cfg
            )
        save_predictions(baseline_predictions, baseline_pred_path)

    ### SVM METHOD ###
    elif cfg["method"] == "svm":
        svm_pred_path = os.path.join(cfg["paths"]["pred_folder"], "svm_test.csv")
        if cfg["validation"]:
            svm_predictions, score = svm_predict(
                training_set, validation_set, testing_set, cfg
            )
            val_scores.append(score)
            if cfg["use_wandb"]:
                wandb.config.C = cfg["svm"]["C"]
                wandb.config.max_iter = cfg["svm"]["max_iter"]
                wandb.config.class_weight = cfg["svm"]["class_weight"]
                wandb.log({"val_score": score})
        else:
            svm_predictions = svm_predict(
                training_set, validation_set, testing_set, cfg
            )
        save_predictions(svm_predictions, svm_pred_path)

    ### RANDOM FOREST METHOD ###
    elif cfg["method"] == "rand_forest":
        rand_forest_pred_path = os.path.join(
            cfg["paths"]["pred_folder"], "rand_forest_test.csv"
        )
        if cfg["validation"]:
            rand_forest_predictions, score = rand_forest_predict(
                training_set, validation_set, testing_set, cfg
            )
            val_scores.append(score)
            if cfg["use_wandb"]:
                wandb.config.n_estimators = cfg["rand_forest"]["n_estimators"]
                wandb.config.criterion = cfg["rand_forest"]["criterion"]
                wandb.config.max_depth = cfg["rand_forest"]["max_depth"]
                wandb.config.class_weight = cfg["rand_forest"]["class_weight"]
                wandb.config.random_state = cfg["rand_forest"]["random_state"]
                wandb.log({"val_score": score})
        else:
            rand_forest_predictions = rand_forest_predict(
                training_set, validation_set, testing_set, cfg
            )
        save_predictions(rand_forest_predictions, rand_forest_pred_path)

    ### LOGISTIC REGRESSION METHOD ###
    elif cfg["method"] == "log_reg":
        log_reg_pred_path = os.path.join(
            cfg["paths"]["pred_folder"], "log_reg_test.csv"
        )
        if cfg["validation"]:
            log_reg_predictions, score = log_reg_predict(
                training_set, validation_set, testing_set, cfg
            )
            val_scores.append(score)
            if cfg["use_wandb"]:
                wandb.config.penalty = cfg["log_reg"]["penalty"]
                wandb.config.dual = cfg["log_reg"]["dual"]
                wandb.config.C = cfg["log_reg"]["C"]
                wandb.config.class_weight = cfg["log_reg"]["class_weight"]
                wandb.config.random_state = cfg["log_reg"]["random_state"]
                wandb.log({"val_score": score})
        else:
            log_reg_predictions = rand_forest_predict(
                training_set, validation_set, testing_set, cfg
            )
        save_predictions(log_reg_predictions, log_reg_pred_path)

    ### BOOSTING METHOD ###
    elif cfg["method"] == "boosting":
        boosting_pred_path = os.path.join(
            cfg["paths"]["pred_folder"], "boosting_test.csv"
        )
        if cfg["validation"]:
            boosting_predictions, score = boosting_predict(
                training_set, validation_set, testing_set, cfg
            )
            val_scores.append(score)
            if cfg["use_wandb"]:
                wandb.config.loss = cfg["boosting"]["loss"]
                wandb.config.lr = cfg["boosting"]["lr"]
                wandb.config.n_estimators = cfg["boosting"]["n_estimators"]
                wandb.config.criterion = cfg["boosting"]["criterion"]
                wandb.config.max_depth = cfg["boosting"]["max_depth"]
                wandb.config.random_state = cfg["boosting"]["random_state"]
                wandb.log({"val_score": score})
        else:
            boosting_predictions = boosting_predict(
                training_set, validation_set, testing_set, cfg
            )
        save_predictions(boosting_predictions, boosting_pred_path)

    ### GRAPHSAGE METHOD ###
    elif cfg["method"] == "graphsage":
        graphsage_pred_path = os.path.join(
            cfg["paths"]["pred_folder"], "graphsage_test.csv"
        )
        if cfg["validation"]:
            graphsage_predictions, score = graphsage_predict(
                training_set, validation_set, testing_set, cfg
            )
            val_scores.append(score)
            if cfg["use_wandb"]:
                wandb.log({"val_score": score})
        else:
            graphsage_predictions = graphsage_predict(
                training_set, validation_set, testing_set, cfg
            )
        save_predictions(graphsage_predictions, graphsage_pred_path)

    ### GAT METHOD ###
    elif cfg["method"] == "gat":
        gat_pred_path = os.path.join(cfg["paths"]["pred_folder"], "gat_test.csv")
        if cfg["validation"]:
            gat_predictions, score = gat_predict(
                training_set, validation_set, testing_set, cfg
            )
            val_scores.append(score)
            if cfg["use_wandb"]:
                wandb.log({"val_score": score})
        else:
            gat_predictions = gat_predict(
                training_set, validation_set, testing_set, cfg
            )
        save_predictions(gat_predictions, gat_pred_path)

if cfg["validation"] and cfg["number_tries"] > 1:
    mean_val = mean(val_scores) * 100
    std_val = stdev(val_scores) * 100
    print(f"Validation statistics: {mean_val:.4f} (±{std_val:.4f})%")
    if cfg["use_wandb"]:
        wandb.log({"mean_val": mean_val, "std_val": std_val})
elif cfg["validation"] and cfg["number_tries"] == 1:
    print(f"Validation statistics: {100 * val_scores[0]:.4f}%")