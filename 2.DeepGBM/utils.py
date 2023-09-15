# -*- coding: utf-8 -*-
"""
Created on 2022/11/25 15:18

@author: Yang Fan

代码运行过程中的工具类辅助函数
"""
import gc
import math
import random

import numpy as np
import pandas as pd
import scipy.sparse
import sklearn
import torch
import wandb
from tqdm import tqdm


def get_data(data_dir):
    assert (
        len(data_dir) == 2
    ), "There should be 2 terms in data_path, data and label respectively."

    data = pd.read_hdf(data_dir[0], key="data")
    label = pd.read_hdf(data_dir[1], key="data")

    data.set_index(keys=["TradingDay", "SecuCode"], inplace=True)
    label.set_index(keys=["TradingDay", "SecuCode"], inplace=True)
    data = data.join(label, how="left").sort_index(axis=0)

    kwargs = {"object": data}
    data_arr = np.array(**kwargs).astype(np.float32)
    del data, label
    gc.collect()
    np.nan_to_num(data_arr, copy=False, nan=0.0)
    data_len = data_arr.shape[0]
    batch_len = data_len // wandb.config.batches
    for batch_idx in range(wandb.config.batches - 1):
        yield data_arr[
            : batch_idx * batch_len + batch_len,
            0:-1,  # train_x
        ], data_arr[
            : batch_idx * batch_len + batch_len,
            [-1],  # train_y
        ], data_arr[
            (batch_idx + 1) * batch_len : (batch_idx + 1) * batch_len  # test_x
            + batch_len,
            0:-1,
        ], data_arr[
            (batch_idx + 1) * batch_len : (batch_idx + 1) * batch_len  # test_y
            + batch_len,
            [-1],
        ], batch_idx

        # yield data_arr[
        #     batch_idx * batch_len : batch_idx * batch_len + batch_len,
        #     0:-1,  # train_x
        # ], data_arr[
        #     batch_idx * batch_len : batch_idx * batch_len + batch_len,
        #     [-1],  # train_y
        # ], data_arr[
        #     (batch_idx + 1) * batch_len : (batch_idx + 1) * batch_len  # test_x
        #     + batch_len,
        #     0:-1,
        # ], data_arr[
        #     (batch_idx + 1) * batch_len : (batch_idx + 1) * batch_len  # test_y
        #     + batch_len,
        #     [-1],
        # ], batch_idx


def getItemByTree(tree, item="split_feature"):
    root = tree.raw["tree_structure"]
    split_nodes = tree.split_nodes
    res = np.zeros(split_nodes + tree.raw["num_leaves"], dtype=np.int32)
    if "value" in item or "threshold" in item or "split_gain" in item:
        res = res.astype(np.float64)

    def getFeature(root, res):
        if "child" in item:
            if "split_index" in root:
                node = root[item]
                if "split_index" in node:
                    res[root["split_index"]] = node["split_index"]
                else:
                    res[root["split_index"]] = (
                        node["leaf_index"] + split_nodes
                    )  # need to check
            else:
                res[root["leaf_index"] + split_nodes] = -1
        elif "value" in item:
            if "split_index" in root:
                res[root["split_index"]] = root["internal_" + item]
            else:
                res[root["leaf_index"] + split_nodes] = root["leaf_" + item]
        else:
            if "split_index" in root:
                res[root["split_index"]] = root[item]
            else:
                res[root["leaf_index"] + split_nodes] = -2
        if "left_child" in root:
            getFeature(root["left_child"], res)
        if "right_child" in root:
            getFeature(root["right_child"], res)

    getFeature(root, res)
    return res


class TreeInterpreter:
    def __init__(self, tree):
        self.raw = tree
        self.split_nodes = self.countSplitNodes(tree)  # 统计分裂节点总数
        self.node_count = self.split_nodes
        self.value = getItemByTree(
            self, item="value"
        )  # 获取节点每次分裂的internal_value以及叶子节点的leaf_value
        self.feature = getItemByTree(self)  # 获取内部分裂节点和叶子节点的序号
        self.gain = getItemByTree(self, "split_gain")  # 获取节点每次分裂的增益

    @staticmethod
    def countSplitNodes(tree):
        root = tree["tree_structure"]

        def counter(root):
            if "split_index" not in root:
                return 0
            return (
                1 + counter(root["left_child"]) + counter(root["right_child"])
            )

        ans = counter(root)
        return ans


class ModelInterpreter:
    def __init__(self, model):
        model = model.dump_model()
        self.n_features_ = model["max_feature_idx"] + 1
        self.trees, self.featurelist, self.threshlist = self.GetTreeSplits(
            model
        )
        self.listcl, self.listcr = self.GetChildren(self.trees)

    @staticmethod
    def GetTreeSplits(model):
        featurelist = []
        threhlist = []
        trees = []
        for idx, tree in enumerate(model["tree_info"]):
            trees.append(TreeInterpreter(tree))
            featurelist.append(trees[-1].feature)
            threhlist.append(getItemByTree(trees[-1], "threshold"))
        return trees, featurelist, threhlist

    @staticmethod
    def GetChildren(trees):
        left_child = []
        right_child = []
        for idx, tree in enumerate(trees):
            left_child.append(getItemByTree(tree, "left_child"))
            right_child.append(getItemByTree(tree, "right_child"))
        return left_child, right_child

    def EqualGroup(self, n_clusters):
        vectors = {}
        for idx, features in enumerate(self.featurelist):
            vectors[idx] = set(features[np.where(features > 0)])
        keys = random.sample(vectors.keys(), len(vectors))
        clusterIdx = np.zeros(len(vectors))  # 对应每个tree所属的group_ID
        trees_per_cluster = len(vectors) // n_clusters
        mod_per_cluster = len(vectors) % n_clusters
        begin = 0
        for idx in range(n_clusters):
            for jdx in range(trees_per_cluster):
                clusterIdx[keys[begin]] = idx
                begin += 1
            if idx < mod_per_cluster:
                clusterIdx[keys[begin]] = idx
                begin += 1
        print(
            f"Tree numbers in each Group: {[np.where(clusterIdx == i)[0].shape for i in range(n_clusters)]}"
        )
        return clusterIdx


def EvalTestset(
    test_x,
    test_y,
    model,
    test_batch_size,
    test_x_opt=None,
    device=None,
):
    test_len = test_x.shape[0]
    test_num_batch = math.ceil(test_len / test_batch_size)
    sum_loss = 0.0
    y_preds = []
    model.eval()
    with torch.no_grad():
        for jdx in range(test_num_batch):
            tst_st = jdx * test_batch_size
            tst_ed = min(test_len, tst_st + test_batch_size)
            inputs = torch.from_numpy(
                test_x[tst_st:tst_ed].astype(np.float32)
            ).to(device)
            if test_x_opt is not None:
                inputs_opt = torch.from_numpy(
                    test_x_opt[tst_st:tst_ed].astype(np.float32)
                ).to(device)
                outputs = model(inputs, inputs_opt)
            else:
                outputs = model(inputs)
            targets = torch.from_numpy(test_y[tst_st:tst_ed]).to(device)
            if isinstance(outputs, tuple):
                outputs = outputs[0]
            y_preds.append(outputs.detach().cpu().numpy())
            loss_tst = model.true_loss(outputs, targets).item()
            sum_loss += (tst_ed - tst_st) * loss_tst
    return sum_loss / test_len, np.concatenate(y_preds, 0)


def TrainWithLog(
    train_x,
    train_y,
    train_y_opt,
    test_x,
    test_y,
    model,
    opt,
    epoch,
    batch_size,
    key="",
    device=None,
    train_x_opt=None,
    test_x_opt=None,
):
    if isinstance(test_x, scipy.sparse.csr_matrix):
        test_x = test_x.todense()
    train_len = train_x.shape[0]
    global_iter = 0
    trn_batch_size = batch_size
    train_num_batch = math.ceil(train_len / trn_batch_size)
    min_loss = float("Inf")

    for epoch in range(epoch):
        shuffled_indices = np.random.permutation(np.arange(train_x.shape[0]))
        Loss_trn_epoch = 0.0
        Loss_trn_log = 0.0
        log_st = 0
        train_tqdm = tqdm(range(train_num_batch))
        for local_iter in train_tqdm:
            trn_st = local_iter * trn_batch_size
            trn_ed = min(train_len, trn_st + trn_batch_size)
            batch_trn_x = train_x[shuffled_indices[trn_st:trn_ed]]
            if isinstance(batch_trn_x, scipy.sparse.csr_matrix):
                batch_trn_x = batch_trn_x.todense()
            inputs = torch.from_numpy(batch_trn_x.astype(np.float32)).to(
                device
            )
            targets = torch.from_numpy(
                train_y[shuffled_indices[trn_st:trn_ed], :]
            ).to(device)
            model.train()
            if train_x_opt is not None:
                inputs_opt = torch.from_numpy(
                    train_x_opt[shuffled_indices[trn_st:trn_ed]].astype(
                        np.float32
                    )
                ).to(device)
                outputs = model(inputs, inputs_opt)
            else:
                outputs = model(inputs)
            opt.zero_grad()
            if isinstance(outputs, tuple) and train_y_opt is not None:
                # targets_inner = torch.from_numpy(s_train_y_opt[trn_st:trn_ed,:]).to(device)
                targets_inner = torch.from_numpy(
                    train_y_opt[shuffled_indices[trn_st:trn_ed], :]
                ).to(device)
                loss_ratio = wandb.config.loss_init * max(
                    0.2,
                    wandb.config.loss_dr ** (epoch // wandb.config.loss_de),
                )  # max(0.5, args.loss_dr ** (epoch // args.loss_de))
                if len(outputs) == 3:
                    loss_val = model.joint_loss(
                        outputs[0],
                        targets,
                        outputs[1],
                        targets_inner,
                        loss_ratio,
                        outputs[2],
                    )
                else:
                    loss_val = model.joint_loss(
                        outputs[0],
                        targets,
                        outputs[1],
                        targets_inner,
                        loss_ratio,
                    )
                loss_val.backward()
                loss_val = model.true_loss(outputs[0], targets)
            elif isinstance(outputs, tuple):
                loss_val = model.true_loss(outputs[0], targets)
                loss_val.backward()
            else:
                loss_val = model.true_loss(outputs, targets)
                loss_val.backward()
            opt.step()
            loss_val = loss_val.item()
            global_iter += 1
            Loss_trn_epoch += (trn_ed - trn_st) * loss_val
            Loss_trn_log += (trn_ed - trn_st) * loss_val

            train_tqdm.set_description(f"Epoch{epoch}")
            train_tqdm.set_postfix({"Loss": Loss_trn_log / (trn_ed - log_st)})

            if global_iter % wandb.config.log_freq == 0:
                wandb.log(
                    {f"{key}train_loss": Loss_trn_log / (trn_ed - log_st)},
                    step=global_iter,
                )
                log_st = trn_ed
                Loss_trn_log = 0.0
            if local_iter == (train_num_batch - 1):
                torch.cuda.empty_cache()
                test_loss, pred_y = EvalTestset(
                    test_x,
                    test_y,
                    model,
                    wandb.config.test_batch_size,
                    test_x_opt,
                    device=device,
                )
                wandb.log({f"{key}test_loss": test_loss}, step=global_iter)
                if test_loss <= min_loss and key == "GBDT2NN-":
                    torch.save(model, "best_score_gbdt2nn_model.pt")
                if test_loss <= min_loss and key == "Embedding-":
                    torch.save(model, "best_score_embed_model.pt")

                min_loss = min(min_loss, test_loss)
        print(
            f"{key}Epoch{epoch}, Test Metric: {test_loss}, Best Metric: {min_loss}, LR: {opt.param_groups[0]['lr']}"
        )
        if key == "GBDT2NN-" and epoch == wandb.config.lr_milestone:
            for params in opt.param_groups:
                params["lr"] = wandb.config.lr * 0.1
        if (
            key == "Embedding-"
            and epoch == wandb.config.embedding_lr_milestone
        ):
            for params in opt.param_groups:
                params["lr"] = params["lr"] * 0.1
        wandb.log(
            {f"{key}epoch": epoch, f"{key}lr": opt.param_groups[0]["lr"]},
            step=epoch,
        )
    print(f"{key}Best Metric: {min_loss}")

    return min_loss


def GetEmbPred(model, fun, X, test_batch_size, device=None):
    model.eval()
    tst_len = X.shape[0]
    test_num_batch = math.ceil(tst_len / test_batch_size)
    print(tst_len, test_num_batch)
    # y_preds = []
    with torch.no_grad():
        for jdx in range(test_num_batch):
            tst_st = jdx * test_batch_size
            tst_ed = min(tst_len, tst_st + test_batch_size)
            inputs = torch.from_numpy(X[tst_st:tst_ed]).to(device)
            if jdx == 0:
                y_preds = fun(inputs).data.cpu().numpy()
            else:
                y_preds = np.concatenate(
                    [y_preds, fun(inputs).data.cpu().numpy()], axis=0
                )
            # t_preds = fun(inputs).data.cpu().numpy()
            # print(jdx, t_preds.shape)
            # y_preds.append(t_preds)
        # y_preds = np.concatenate(y_preds, 0)
    return y_preds


def eval_metrics(true, pred):
    mse_loss = sklearn.metrics.mean_squared_error(true, pred)
    return mse_loss


def TrainWithLog_GBDT2NN(
    train_x,
    train_y,
    train_y_opt,
    test_x,
    test_y,
    model,
    opt,
    epoch,
    batch_size,
    model_embed,
    fun,
    key="",
    device=None,
    train_x_opt=None,
    test_x_opt=None,
):
    if isinstance(test_x, scipy.sparse.csr_matrix):
        test_x = test_x.todense()
    train_len = train_x.shape[0]
    global_iter = 0
    trn_batch_size = batch_size
    train_num_batch = math.ceil(train_len / trn_batch_size)
    min_loss = float("Inf")

    for epoch in range(epoch):
        shuffled_indices = np.random.permutation(np.arange(train_x.shape[0]))
        Loss_trn_epoch = 0.0
        Loss_trn_log = 0.0
        log_st = 0
        train_tqdm = tqdm(range(train_num_batch))
        for local_iter in train_tqdm:
            trn_st = local_iter * trn_batch_size
            trn_ed = min(train_len, trn_st + trn_batch_size)
            batch_trn_x = train_x[shuffled_indices[trn_st:trn_ed]]
            if isinstance(batch_trn_x, scipy.sparse.csr_matrix):
                batch_trn_x = batch_trn_x.todense()
            inputs = torch.from_numpy(batch_trn_x.astype(np.float32)).to(
                device
            )
            targets = torch.from_numpy(
                train_y[shuffled_indices[trn_st:trn_ed], :]
            ).to(device)
            model.train()
            if train_x_opt is not None:
                inputs_opt = torch.from_numpy(
                    train_x_opt[shuffled_indices[trn_st:trn_ed]].astype(
                        np.float32
                    )
                ).to(device)
                outputs = model(inputs, inputs_opt)
            else:
                outputs = model(inputs)
            opt.zero_grad()
            if isinstance(outputs, tuple) and train_y_opt is not None:
                # targets_inner = torch.from_numpy(s_train_y_opt[trn_st:trn_ed,:]).to(device)
                # targets_inner = torch.from_numpy(
                #     train_y_opt[shuffled_indices[trn_st:trn_ed], :]
                # ).to(device)
                model_embed.eval()
                inputs_ = torch.from_numpy(
                    train_y_opt[shuffled_indices[trn_st:trn_ed]]
                ).to(device)
                with torch.no_grad():
                    targets_inner = fun(inputs_)
                loss_ratio = wandb.config.loss_init * max(
                    0.2,
                    wandb.config.loss_dr ** (epoch // wandb.config.loss_de),
                )  # max(0.5, args.loss_dr ** (epoch // args.loss_de))
                if len(outputs) == 3:
                    loss_val = model.joint_loss(
                        outputs[0],
                        targets,
                        outputs[1],
                        targets_inner,
                        loss_ratio,
                        outputs[2],
                    )
                else:
                    loss_val = model.joint_loss(
                        outputs[0],
                        targets,
                        outputs[1],
                        targets_inner,
                        loss_ratio,
                    )
                loss_val.backward()
                loss_val = model.true_loss(outputs[0], targets)
            elif isinstance(outputs, tuple):
                loss_val = model.true_loss(outputs[0], targets)
                loss_val.backward()
            else:
                loss_val = model.true_loss(outputs, targets)
                loss_val.backward()
            opt.step()
            loss_val = loss_val.item()
            global_iter += 1
            Loss_trn_epoch += (trn_ed - trn_st) * loss_val
            Loss_trn_log += (trn_ed - trn_st) * loss_val

            train_tqdm.set_description(f"Epoch{epoch}")
            train_tqdm.set_postfix({"Loss": Loss_trn_log / (trn_ed - log_st)})

            if global_iter % wandb.config.log_freq == 0:
                wandb.log(
                    {f"{key}train_loss": Loss_trn_log / (trn_ed - log_st)},
                    step=global_iter,
                )
                log_st = trn_ed
                Loss_trn_log = 0.0
            if local_iter == (train_num_batch - 1):
                torch.cuda.empty_cache()
                test_loss, pred_y = EvalTestset(
                    test_x,
                    test_y,
                    model,
                    wandb.config.test_batch_size,
                    test_x_opt,
                    device=device,
                )
                pred_y = pred_y.reshape(-1)
                test_y = test_y.reshape(-1)
                n = pred_y.shape[0]
                numerator = (1 / n) * np.einsum(
                    "n,n->",
                    pred_y - np.mean(pred_y, keepdims=True, axis=-1),
                    test_y - np.mean(test_y, keepdims=True, axis=-1),
                )
                denominator = np.std(pred_y, axis=-1) * np.std(test_y, axis=-1)
                wandb.log(
                    {
                        f"{key}test_loss": test_loss,
                        f"{key}test_ic": numerator / denominator,
                    },
                    step=global_iter,
                )
                if test_loss <= min_loss and key == "GBDT2NN-":
                    torch.save(model, "best_score_gbdt2nn_model.pt")
                if test_loss <= min_loss and key == "Embedding-":
                    torch.save(model, "best_score_embed_model.pt")

                min_loss = min(min_loss, test_loss)
        print(
            f"{key}Epoch{epoch}, Test Metric: {test_loss}, Best Metric: {min_loss}, LR: {opt.param_groups[0]['lr']}"
        )
        if key == "GBDT2NN-" and epoch == wandb.config.lr_milestone:
            for params in opt.param_groups:
                params["lr"] = wandb.config.lr * 0.1
        if (
            key == "Embedding-"
            and epoch == wandb.config.embedding_lr_milestone
        ):
            for params in opt.param_groups:
                params["lr"] = params["lr"] * 0.1
        wandb.log(
            {f"{key}epoch": epoch, f"{key}lr": opt.param_groups[0]["lr"]},
            step=epoch,
        )
    print(f"{key}Best Metric: {min_loss}")

    return min_loss
