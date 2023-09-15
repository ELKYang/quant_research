# -*- coding: utf-8 -*-
"""
Created on 2022/10/21 9:33

@author: Yang Fan

有序回归的一阶导函数以及二阶导函数
参考 华泰证券 《量化如何追求模糊的准确：有序回归》
"""
from typing import Callable, Optional, Tuple

import numpy as np
from numba import jit
from numpy import ndarray


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
