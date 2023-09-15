# -*- coding: utf-8 -*-
"""
Created on 2022/11/25 17:13

@author: Yang Fan

GBDT2NN训练
"""
import gc
import pdb

import numpy as np
import torch
import wandb
from model_components_2 import (
    GBDT2NN,
    EmbeddingModel,
    SubGBDTLeaf_cls,
    TrainGBDT,
)
from utils import (
    EvalTestset,
    GetEmbPred,
    TrainWithLog,
    TrainWithLog_GBDT2NN,
    eval_metrics,
)


def train_GBDT2NN(train_x, train_y, test_x, test_y):
    # 1. Train lightgbm first to get trees.
    # gbm, tree_pred = TrainGBDT(train_x, train_y, test_x, test_y)
    # torch.save(gbm, "light_gbm_tree.pt")
    # torch.save(tree_pred, "tree_pred.pt")
    gbm = torch.load("light_gbm_tree.pt")
    tree_pred = torch.load("tree_pred.pt")
    # 2. Group the trees together and prepare the training data for embedding.
    gbms = SubGBDTLeaf_cls(train_x, test_x, gbm)

    min_len_features = train_x.shape[1]
    used_features = []
    tree_outputs = []
    leaf_preds = []
    test_leaf_preds = []
    n_output = train_y.shape[1]
    max_ntree_per_split = 0
    group_average = []
    for (
        used_feature,
        new_train_y,
        leaf_pred,
        test_leaf_pred,
        avg,
        all_avg,
    ) in gbms:
        used_features.append(used_feature)
        min_len_features = min(min_len_features, len(used_feature))
        tree_outputs.append(new_train_y)
        leaf_preds.append(leaf_pred)
        test_leaf_preds.append(test_leaf_pred)
        group_average.append(avg)
        max_ntree_per_split = max(max_ntree_per_split, leaf_pred.shape[1])
    for i in range(len(used_features)):
        used_features[i] = sorted(used_features[i][:min_len_features])
    n_models = len(used_features)
    group_average = np.asarray(group_average).reshape((n_models, 1, 1))
    for i in range(n_models):
        if leaf_preds[i].shape[1] < max_ntree_per_split:
            leaf_preds[i] = np.concatenate(
                [
                    leaf_preds[i] + 1,
                    np.zeros(
                        [
                            leaf_preds[i].shape[0],
                            max_ntree_per_split - leaf_preds[i].shape[1],
                        ],
                        dtype=np.int32,
                    ),
                ],
                axis=1,
            )
            test_leaf_preds[i] = np.concatenate(
                [
                    test_leaf_preds[i] + 1,
                    np.zeros(
                        [
                            test_leaf_preds[i].shape[0],
                            max_ntree_per_split - test_leaf_preds[i].shape[1],
                        ],
                        dtype=np.int32,
                    ),
                ],
                axis=1,
            )
    leaf_preds = np.concatenate(leaf_preds, axis=1)
    test_leaf_preds = np.concatenate(test_leaf_preds, axis=1)

    # train embedding model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    emb_model = EmbeddingModel(
        n_models,
        max_ntree_per_split,
        wandb.config.embedding_size,
        wandb.config.num_leaves + 1,
        n_output,
        group_average,
    ).to(device)
    tree_layers = wandb.config.tree_layers
    tree_layers.append(wandb.config.embedding_size)
    optimizer = torch.optim.AdamW(
        params=emb_model.parameters(),
        lr=wandb.config.embedding_lr,
        weight_decay=wandb.config.weight_decay,
    )
    tree_outputs = (
        np.asarray(tree_outputs)
        .reshape((n_models, leaf_preds.shape[0]))
        .transpose((1, 0))
    )
    TrainWithLog(
        train_x=leaf_preds,
        train_y=train_y,
        train_y_opt=tree_outputs,
        test_x=test_leaf_preds,
        test_y=test_y,
        model=emb_model,
        opt=optimizer,
        epoch=wandb.config.embedding_epochs,
        batch_size=wandb.config.batch_size,
        key="Embedding-",
        device=device,
    )
    del tree_outputs, test_leaf_preds
    gc.collect()
    # emb_model = torch.load("best_score_embed_model.pt").to(device)
    output_w = (
        emb_model.bout.weight.data.cpu()
        .numpy()
        .reshape(n_models * wandb.config.embedding_size, n_output)
    )
    output_b = np.array(emb_model.bout.bias.data.cpu().numpy().sum())
    concat_train_x = np.concatenate(
        [train_x, np.zeros((train_x.shape[0], 1), dtype=np.float32)], axis=-1
    )
    concat_test_x = np.concatenate(
        [test_x, np.zeros((test_x.shape[0], 1), dtype=np.float32)], axis=-1
    )
    # train gbdt2nn model
    gbdt2nn_model = GBDT2NN(
        concat_train_x.shape[1],
        np.asarray(used_features, dtype=np.int64),
        tree_layers,
        output_w,
        output_b,
        device=device,
    ).to(device)
    opt = torch.optim.AdamW(
        gbdt2nn_model.parameters(),
        lr=wandb.config.lr,
        weight_decay=wandb.config.weight_decay,
        amsgrad=False,
    )

    TrainWithLog_GBDT2NN(
        train_x=concat_train_x,
        train_y=train_y,
        train_y_opt=leaf_preds,
        model_embed=emb_model,
        fun=emb_model.lastlayer,
        test_x=concat_test_x,
        test_y=test_y,
        model=gbdt2nn_model,
        opt=opt,
        epoch=wandb.config.epochs,
        batch_size=wandb.config.batch_size,
        key="GBDT2NN-",
        device=device,
    )
    gbdt2nn_model = torch.load("best_score_gbdt2nn_model.pt")
    _, pred_y = EvalTestset(
        concat_test_x, test_y, gbdt2nn_model, wandb.config.test_batch_size
    )
    metric = eval_metrics(test_y, pred_y)
    metric_tree = eval_metrics(test_y.reshape(-1), tree_pred)
    print(f"GBDT2NN MSE:{metric}, GBDT MSE:{metric_tree}")
    return gbdt2nn_model, opt, metric, metric_tree
