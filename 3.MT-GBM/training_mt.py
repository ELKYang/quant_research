# -*- coding: utf-8 -*-
"""
Created on 2023/1/5 9:01

@author: Yang Fan

MT-LightGBM和LightGBM训练及对比
"""
import pandas as pd
import numpy as np
import gc
import time
from scipy.special import softmax
import matplotlib.pyplot as plt
import pickle
import lightgbmmt as lgb
from scipy.stats import kurtosis, skew

from numba import jit
from numpy import ndarray
from typing import Callable, Optional, Tuple


# 准备数据
def get_data(data_dir):
    start = time.time()
    train_data = pd.read_hdf(
        data_dir[0],
        key="data",
        where="(TradingDay >= 20190101) & (TradingDay <= 20201231)",
    )
    test_data = pd.read_hdf(
        data_dir[0],
        key="data",
        where="(TradingDay >= 20210101) & (TradingDay <= 20211231)",
    )
    label = pd.read_hdf(data_dir[1], key="data")

    train_data.set_index(keys=["TradingDay", "SecuCode"], inplace=True)
    test_data.set_index(keys=["TradingDay", "SecuCode"], inplace=True)
    label.set_index(keys=["TradingDay", "SecuCode"], inplace=True)

    train_data = train_data.join(label, how="left").sort_index(axis=0)
    test_data = test_data.join(label, how="left").sort_index(axis=0)
    res_label = test_data["Return"].to_frame()
    train_kwargs = {"object": train_data}
    train_data_arr = np.array(**train_kwargs).astype(np.float32)
    test_kwargs = {"object": test_data}
    test_data_arr = np.array(**test_kwargs).astype(np.float32)

    del train_data, test_data, label
    gc.collect()
    end = time.time()
    print(f"Reading data done! Time consuming {end - start} s")
    np.nan_to_num(train_data_arr, copy=False, nan=0.0)
    np.nan_to_num(test_data_arr, copy=False, nan=0.0)

    train_x, train_y, test_x, test_y = (
        train_data_arr[:, 0:-1],
        train_data_arr[:, [-1]],
        test_data_arr[:, 0:-1],
        test_data_arr[:, [-1]],
    )
    del train_data_arr, test_data_arr
    gc.collect()

    print(train_x.shape, train_y.shape, test_x.shape, test_y.shape)
    return train_x, train_y, test_x, test_y, res_label


print("Preparing Data...")
data_path = [
    "/mnt/data1/yangfan/basic_features.hdf5",
    "/mnt/data1/yangfan/labels_1.hdf5",
]
train_x, train_y, test_x, test_y, res_label = get_data(data_path)
print("Done!")

# 多任务
num_labels = 2
params_mt = {
    "seed": 2022,
    # "deterministic": True,
    "num_leaves": 128,
    "boost_from_average": True,
    "learning_rate": 0.015,
    "feature_fraction": 0.5,
    "bagging_freq": 1,
    "bagging_fraction": 0.5,
    # multitask
    "objective": "custom",
    "num_labels": num_labels,
    "tree_learner": "serial2",
    "num_threads": 16,
}


# 有序回归
def get_order_regress_grad_and_hess(
    bins: ndarray, loss_type: str = "Logistic"
) -> Callable[[ndarray, ndarray], Tuple[Optional[ndarray], Optional[ndarray]]]:
    """Calculate the gradient of an ordered regression loss function

    Parameters
    ----------
    bins: ndarray
        Classification threshold for each data.
        shape: [n, d]
    loss_type: str
        The loss function used in ordered regression.
    Returns
    -------
    The gradient and hess of an ordered regression loss function
    """
    n, d = bins.shape
    bins = bins.T

    # Use numba to convert the python function into native code. This operation speeds up the program by nearly two times (Test on one million pieces of data, K=10).
    @jit(nopython=True)
    def grad_and_hess(
        y: ndarray, pred: ndarray, sample_weight: ndarray
    ) -> Tuple[Optional[ndarray], Optional[ndarray]]:
        """

        Parameters
        ----------
        y: ndarray
            True value.
            shape: [n]
        pred: ndarray
            Predict value.
            shape: [n]
        sample_weight: ndarray
            Weights for sample weighting.
            shape: [n]

        Returns
        -------
        grad: ndarray
            shape: [n]
        hess: ndarray
            shape: [n]
        """
        if loss_type == "Logistic":
            temp = np.exp(-np.sign(y - bins) * (pred - bins))
            grad = sample_weight * np.sum(
                (temp * (-np.sign(y - bins))) / (1 + temp), axis=0
            )
            hess = sample_weight * np.sum(temp / np.square(1 + temp), axis=0)
        elif loss_type == "Exponential":
            temp = np.exp(-np.sign(y - bins) * (pred - bins))
            grad = sample_weight * np.sum(temp * (-np.sign(y - bins)), axis=0)
            hess = sample_weight * np.sum(temp, axis=0)
        elif loss_type == "MSE":
            grad = (
                2
                * sample_weight
                * np.sum(((pred - bins) - np.sign(y - bins)), axis=0)
            )
            hess = sample_weight * np.ones(n) * 2 * d
        else:
            raise ValueError(
                "Loss type should be 'Logistic', 'Exponential' or 'MSE'."
            )
        return grad, hess

    return grad_and_hess


def l2_and_ordered_regression(preds, train_data, ep=0):
    labels = train_data.get_label()
    labels = labels.reshape((num_labels, -1)).transpose()
    preds = preds.reshape((num_labels, -1)).transpose()
    label = labels[:, 0]
    pred1, pred2 = preds[:, 0], preds[:, 1]

    # 计算MSE回归的grad和hess
    grad1 = pred1 - label
    hess1 = grad1 * 0.0 + 1

    # 计算有序回归的grad和hess
    n = labels.shape[0]
    bins = np.repeat(
        np.expand_dims(np.percentile(label, [20, 40, 60, 80]), axis=-1),
        n,
        axis=-1,
    ).transpose()
    fn = get_order_regress_grad_and_hess(bins=bins, loss_type="Logistic")
    grad2, hess2 = fn(label, pred2, np.ones(n))

    # 对两个损失进行加权
    # 有序回归的梯度范围[-2, 2], 标准差1.4, MSE梯度范围[-0.1, 0.1], 标准差较小。
    # 两个梯度相关性0.82

    # Equal Weight
    # w1, w2 = 1.0, 1.0

    # RLW
    weight = softmax(np.random.random((1, 2)), axis=-1)
    w1, w2 = weight[0, 0], weight[0, 1]
    grad = w1 * grad1 + w2 * grad2 * 0.1
    hess = w1 * hess1 + w2 * hess2
    grad_concat = (
        np.concatenate(
            [np.expand_dims(grad1, axis=-1), np.expand_dims(grad2, axis=-1)],
            axis=-1,
        )
        .transpose()
        .reshape((-1))
    )
    hess_concat = (
        np.concatenate(
            [np.expand_dims(hess1, axis=-1), np.expand_dims(hess2, axis=-1)],
            axis=-1,
        )
        .transpose()
        .reshape((-1))
    )

    return grad, hess, grad_concat, hess_concat


def cal_ic(y_pred, y_true):
    eps = 1e-8
    Ns = y_true.shape[0]
    numerator = (1 / Ns) * np.einsum(
        "n,n->",
        y_pred - np.mean(y_pred, axis=-1, keepdims=True),
        y_true - np.mean(y_true, axis=-1, keepdims=True),
    )
    denominator = np.std(y_pred, axis=-1) * np.std(y_true, axis=-1)
    return numerator / (denominator + eps)


def metric_mt(preds, train_data):
    labels = train_data.get_label()
    label = labels.reshape((num_labels, -1)).transpose()[:, 0]
    pred = preds.reshape((num_labels, -1)).transpose()[:, 0]
    score = np.mean((label - pred) ** 2)
    ic = cal_ic(pred, label)
    kurt = kurtosis(pred)
    skew_pred = skew(pred)
    return [
        ("MSE", score, False),
        ("IC", ic, True),
        ("Kurt", kurt, False),
        ("Skew", skew_pred, False),
    ]


print("MT-LightGBM Training...")
# 训练
lgb_train_mt = lgb.Dataset(
    train_x, np.repeat(train_y, 2, axis=-1), params=params_mt
)
lgb_eval_mt = lgb.Dataset(
    test_x, np.repeat(test_y, 2, axis=-1), reference=lgb_train_mt
)
eval_result_mt = {}
clf_mt = lgb.train(
    params_mt,
    lgb_train_mt,
    fobj=l2_and_ordered_regression,
    feval=metric_mt,
    num_boost_round=2000,
    valid_sets=[lgb_train_mt, lgb_eval_mt],
    evals_result=eval_result_mt,
)
clf_mt.set_num_labels(2)
print("Done!")

clf_mt.save_model("model.txt")

preds = clf_mt.predict(test_x, raw_score=True, num_iteration=-1)

with open("preds.pkl", "wb") as handle:
    pickle.dump(preds, handle)

res_label["Prediction"] = preds[:, 0]
res_label["Prediction_auxiliary"] = preds[:, 1]
res_label.reset_index(inplace=True)
res_label.to_hdf("multi_task_result.h5", key="data")
with open("alpha.pkl", "wb") as handle:
    pickle.dump(res_label, handle)

with open("mt_rw_mse_1_odr_01.pkl", "wb") as handle:
    pickle.dump(eval_result_mt, handle)
