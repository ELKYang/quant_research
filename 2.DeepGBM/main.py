# -*- coding: utf-8 -*-
"""
Created on 2022/11/30 14:03

@author: Yang Fan

脚本说明
"""
import os
import random

import numpy as np
import torch
import wandb
from model_components import TrainGBDT
from train_models import train_GBDT2NN
from utils import EvalTestset, TrainWithLog, eval_metrics, get_data

# wandb offline mode
os.environ["WANDB_API_KEY"] = "f7bb19bc81f5d12f426a67949f434e397dd1dbc0"
os.environ["WANDB_MODE"] = "offline"


def gbdt_offline():
    datas = get_data(wandb.config.data_dir)
    metrics = []
    for train_x, train_y, test_x, test_y, batch_idx in datas:
        if batch_idx == 0:
            gbm, preds = TrainGBDT(train_x, train_y, test_x, test_y)
            metric = eval_metrics(test_y.reshape(-1), preds)
        else:
            preds = gbm.predict(test_x)
            metric = eval_metrics(test_y.reshape(-1), preds)
        wandb.log({"gbdt_offline_mse": metric}, step=batch_idx + 1)
        print(f"GBDT_offline_mse round{batch_idx}: {metric}")
        metrics.append(metric)
    print(f"GBDT Offline Metrics: {metrics}")


def gbdt_online():
    datas = get_data(wandb.config.data_dir)
    metrics = []
    for train_x, train_y, test_x, test_y, batch_idx in datas:
        if batch_idx == 0:
            gbm, preds = TrainGBDT(train_x, train_y, test_x, test_y)
            metric = eval_metrics(test_y, preds)
        else:
            new_gbm = gbm.refit(train_x, train_y.reshape(-1))
            preds = new_gbm.predict(test_x)
            metric = eval_metrics(test_y.reshpe(-1), preds)
        wandb.log({"gbdt_online_mse": metric}, step=batch_idx + 1)
        print(f"GBDT_online_mse round{batch_idx}: {metric}")
        metrics.append(metric)
    print(f"GBDT Online Metrics: {metrics}")


def gbdt2nn_online():
    datas = get_data(wandb.config.data_dir)
    metrics = []
    for train_x, train_y, test_x, test_y, batch_idx in datas:
        if batch_idx == 0:
            print(f"Round{batch_idx}, train_x shape:{train_x.shape}")
            fitted_model, opt, metric, metric_tree = train_GBDT2NN(
                train_x, train_y, test_x, test_y
            )
        else:
            fitted_model, opt, metric = GBDT2NN_Refit(
                train_x,
                train_y,
                test_x,
                test_y,
                fitted_model,
                opt,
                key=f"Online_round{batch_idx}-",
            )
        wandb.log({"gbdt2nn_online_mse": metric}, step=batch_idx + 1)
        print(f"GBDT2NN_online_mse round{batch_idx}: {metric}")
        metrics.append(metric)
    print(f"GBDT2NN Online Metrics: {metrics}")


def GBDT2NN_Refit(train_x, train_y, test_x, test_y, fitted_model, opt, key):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    concat_train_x = np.concatenate(
        [train_x, np.zeros((train_x.shape[0], 1), dtype=np.float32)], axis=-1
    )
    concat_test_x = np.concatenate(
        [test_x, np.zeros((test_x.shape[0], 1), dtype=np.float32)], axis=-1
    )
    if not wandb.config.offline:
        TrainWithLog(
            train_x=concat_train_x,
            train_y=train_y,
            train_y_opt=None,
            test_x=concat_test_x,
            test_y=test_y,
            model=fitted_model,
            opt=opt,
            epoch=wandb.config.online_epochs,
            batch_size=wandb.config.online_batch_size,
            key=key,
            device=device,
        )
    _, pred_y = EvalTestset(
        concat_test_x, test_y, fitted_model, wandb.config.test_batch_size
    )
    metric = eval_metrics(test_y, pred_y)
    return fitted_model, opt, metric


def set_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)


if __name__ == "__main__":
    wandb.init(name="gbdt2nn", project="DeepGBM")
    set_seed(wandb.config.seed)
    if wandb.config.model == "gbdt2nn":
        gbdt2nn_online()
    elif wandb.config.model == "gbdt_online":
        gbdt_online()
    elif wandb.config.model == "gbdt_offline":
        gbdt_offline()
    else:
        raise ValueError(
            "model should be 'gbdt2nn', 'gbdt_online' or 'gbdt_offline'"
        )
    wandb.finish()
