# -*- coding: utf-8 -*-
"""
Created on 2022/11/25 15:22

@author: Yang Fan

模型的组成，包括LightGBM的训练, 多棵树分组, Embedding模型, GBDT2NN模型
"""
import math

import lightgbm as lgb
import numpy as np
import torch
import torch.nn as nn
import wandb
from utils import ModelInterpreter
from wandb.lightgbm import wandb_callback


# LightGBM的训练
def TrainGBDT(train_x, train_y, test_x, test_y):
    params = {
        "task": "train",
        "boosting_type": "gbdt",
        "num_class": 1,
        "objective": "regression",
        "metric": "mse",
        "boost_from_average": True,
        "num_leaves": wandb.config.num_leaves,
        "feature_fraction": wandb.config.feature_fraction,
        "bagging_freq": wandb.config.bagging_freq,
        "bagging_fraction": wandb.config.bagging_fraction,
        "num_threads": wandb.config.num_threads,
        "learning_rate": wandb.config.tree_lr,
        "seed": wandb.config.seed,
    }
    lgb_train = lgb.Dataset(train_x, train_y.reshape(-1), params=params)
    lgb_eval = lgb.Dataset(test_x, test_y.reshape(-1), reference=lgb_train)
    # early_stop_callback = lgb.early_stopping(
    #     stopping_rounds=wandb.config.early_stopping_rounds
    # )
    gbm = lgb.train(
        params,
        lgb_train,
        num_boost_round=wandb.config.num_trees,
        valid_sets=[lgb_eval],
        callbacks=[wandb_callback()],
    )
    preds = gbm.predict(test_x, raw_score=True)
    preds = preds.astype(np.float32)
    return gbm, preds


# 将n棵树分组
def SubGBDTLeaf_cls(train_x, test_x, gbm):
    num_slices = wandb.config.num_slices
    MAX = train_x.shape[1]

    # get leaf prediction index
    leaf_preds = gbm.predict(train_x, pred_leaf=True).reshape(
        train_x.shape[0], -1
    )
    test_leaf_preds = gbm.predict(test_x, pred_leaf=True).reshape(
        test_x.shape[0], -1
    )
    n_trees = leaf_preds.shape[1]

    # get leaf output from each tree
    leaf_output = np.zeros(
        [n_trees, wandb.config.num_leaves], dtype=np.float32
    )
    for tree_id in range(n_trees):
        num_leaf = np.max(leaf_preds[:, tree_id]) + 1
        for leaf_id in range(num_leaf):
            leaf_output[tree_id][leaf_id] = gbm.get_leaf_output(
                tree_id, leaf_id
            )

    modelI = ModelInterpreter(gbm)
    clusterIdx = modelI.EqualGroup(num_slices)
    n_feature = wandb.config.feature_per_group
    treeI = modelI.trees

    for n_idx in range(num_slices):
        tree_indices = np.where(clusterIdx == n_idx)[0]
        trees = {}
        tid = 0
        for jdx in tree_indices:
            trees[str(tid)] = treeI[jdx].raw
            tid += 1

        all_hav = {}
        for jdx, tree in enumerate(tree_indices):
            for kdx, f in enumerate(treeI[tree].feature):
                if f == -2:
                    continue
                if f not in all_hav:
                    all_hav[f] = 0
                all_hav[f] += treeI[tree].gain[kdx]

        all_hav = sorted(all_hav.items(), key=lambda kv: -kv[1])
        used_features = [item[0] for item in all_hav[:n_feature]]

        for kdx in range(max(0, n_feature - len(used_features))):
            used_features.append(MAX)
        cur_leaf_preds = leaf_preds[:, tree_indices]
        cur_test_leaf_preds = test_leaf_preds[:, tree_indices]
        new_train_y = np.zeros(train_x.shape[0])
        for jdx in tree_indices:
            new_train_y += np.take(
                leaf_output[jdx, :].reshape(-1), leaf_preds[:, jdx].reshape(-1)
            )
        new_train_y = new_train_y.reshape(-1, 1).astype(np.float32)
        yield used_features, new_train_y, cur_leaf_preds, cur_test_leaf_preds, np.mean(
            np.take(leaf_output, tree_indices, 0)
        ), np.mean(
            leaf_output
        )


class BatchDense(nn.Module):
    def __init__(self, batch, in_features, out_features, bias_init=None):
        super(BatchDense, self).__init__()
        self.batch = batch
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(
            torch.Tensor(batch, in_features, out_features), requires_grad=True
        )
        self.bias = nn.Parameter(
            torch.Tensor(batch, 1, out_features), requires_grad=True
        )
        self.reset_parameters(bias_init)

    def reset_parameters(self, bias_init=None):
        stdv = math.sqrt(6.0 / (self.in_features + self.out_features))
        self.weight.data.uniform_(-stdv, stdv)
        if bias_init is not None:
            self.bias.data = torch.from_numpy(bias_init)
        else:
            self.bias.data.fill_(0)

    def forward(self, x):
        size = x.size()
        # Todo: avoid the swap axis
        x = x.view(x.size(0), self.batch, -1)
        out = x.transpose(0, 1).contiguous()
        out = torch.baddbmm(self.bias, out, self.weight)
        out = out.transpose(0, 1).contiguous()
        out = out.view(x.size(0), -1)
        return out


class EmbeddingModel(nn.Module):
    def __init__(
        self,
        n_models,
        max_ntree_per_split,
        embsize,
        maxleaf,
        n_output,
        out_bias=None,
        task="regression",
    ):
        super(EmbeddingModel, self).__init__()
        self.task = task
        self.n_models = n_models
        self.maxleaf = maxleaf
        self.fcs = nn.ModuleList()
        self.max_ntree_per_split = max_ntree_per_split

        self.embed_w = nn.Parameter(
            torch.Tensor(n_models, max_ntree_per_split * maxleaf, embsize),
            requires_grad=True,
        )
        # torch.nn.init.xavier_normal(self.embed_w)
        stdv = math.sqrt(1.0 / (max_ntree_per_split))
        self.embed_w.data.normal_(0, stdv)  # .uniform_(-stdv, stdv)

        self.bout = BatchDense(n_models, embsize, 1, out_bias)
        self.bn = nn.BatchNorm1d(embsize * n_models)
        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()
        # self.output_fc = Dense(n_models * embsize, n_output)
        self.dropout = torch.nn.Dropout()
        if task == "regression":
            self.criterion = nn.MSELoss()
        else:
            self.criterion = nn.BCELoss()
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

    def one_hot(self, y, numslot, mask=None):
        y_tensor = (
            y.type(torch.LongTensor).contiguous().view(-1, 1).to(self.device)
        )
        tmp = torch.zeros(
            y_tensor.size()[0],
            numslot,
            device=self.device,
            dtype=torch.float32,
            requires_grad=False,
        ).to(self.device)
        y_one_hot = tmp.scatter_(1, y_tensor.to(self.device), 1)
        if mask is not None:
            y_one_hot = y_one_hot * mask
        y_one_hot = y_one_hot.view(y.shape[0], -1)
        return y_one_hot

    def batchmul(self, x, models, embed_w, length):
        out = self.one_hot(x, length)
        out = out.view(x.size(0), models, -1)
        out = out.transpose(0, 1).contiguous()
        out = torch.bmm(out, embed_w)
        out = out.transpose(0, 1).contiguous()
        out = out.view(x.size(0), -1)
        return out

    def lastlayer(self, x):
        out = self.batchmul(x, self.n_models, self.embed_w, self.maxleaf)
        out = self.bn(out)
        return out

    def forward(self, x):
        out = self.lastlayer(x)
        out = self.dropout(out)
        out = out.view(x.size(0), self.n_models, -1)
        out = self.bout(out)
        # out = self.output_fc(out)
        sum_out = torch.sum(out, -1, True)
        if self.task != "regression":
            return self.sigmoid(sum_out), out
        return sum_out, out

    def joint_loss(self, out, target, out_inner, target_inner, *args):
        return nn.MSELoss()(out_inner, target_inner)

    def true_loss(self, out, target):
        return self.criterion(out, target)


class GBDT2NN(nn.Module):
    def __init__(
        self,
        input_size,
        used_features,
        tree_layers,
        output_w,
        output_b,
        device=None,
    ):
        super(GBDT2NN, self).__init__()
        print("Init GBDT2NN")
        self.n_models = len(used_features)
        self.tree_layers = tree_layers
        n_feature = len(used_features[0])
        used_features = np.asarray(used_features).reshape(-1)
        self.used_features = nn.Parameter(
            torch.from_numpy(used_features).to(device), requires_grad=False
        )
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        assert len(tree_layers) > 0
        self.bdenses = nn.ModuleList()
        self.bns = nn.ModuleList()
        self.bdenses.append(
            BatchDense(self.n_models, n_feature, tree_layers[0])
        )
        for i in range(1, len(tree_layers)):
            self.bdenses.append(
                BatchDense(self.n_models, tree_layers[i - 1], tree_layers[i])
            )
        for i in range(len(tree_layers) - 1):
            self.bns.append(nn.BatchNorm1d(tree_layers[i] * self.n_models))
        self.out_weight = nn.Parameter(
            torch.from_numpy(output_w).to(device), requires_grad=False
        )
        self.out_bias = nn.Parameter(
            torch.from_numpy(output_b).to(device), requires_grad=False
        )
        print("Init GBDT2NN succeed!")
        self.criterion = nn.MSELoss()
        self.device = device

    def batchmul(self, x, f):
        out = x.view(x.size(0), self.n_models, -1)
        out = f(out)
        out = out.view(x.size(0), -1)
        return out

    def lastlayer(self, x):
        out = torch.index_select(
            x.to(self.device), dim=1, index=self.used_features.to(self.device)
        )
        for i in range(len(self.bdenses) - 1):
            out = self.batchmul(out, self.bdenses[i])
            out = self.bns[i](out)
            out = self.relu(out)
        return out

    def forward(self, x):
        out = self.lastlayer(x.float())
        pred = self.batchmul(out, self.bdenses[-1])
        out = torch.addmm(self.out_bias, pred, self.out_weight)
        return out, pred

    def emb_loss(self, emb_pred, emb_target):
        loss_weight = torch.abs(torch.sum(self.out_weight, 1))
        l2_loss = (
            nn.MSELoss(reduction="none")(emb_pred, emb_target) * loss_weight
        )
        return torch.mean(torch.sum(l2_loss, dim=1))

    def joint_loss(self, out, target, emb_pred, emb_target, ratio):
        return (1 - ratio) * self.criterion(
            out, target
        ) + ratio * self.emb_loss(emb_pred, emb_target)

    def true_loss(self, out, target):
        return self.criterion(out.to(self.device), target.to(self.device))
