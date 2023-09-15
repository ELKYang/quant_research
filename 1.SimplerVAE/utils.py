# -*- coding: utf-8 -*-
"""
Created on 2022/10/2 19:26

@author: Yang Fan

VAE的Loss函数和Metric
"""
from typing import Tuple

import torch
import torch.nn as nn
from torch import Tensor


class VAELoss(nn.Module):
    def __init__(self, gamma: float = 1.0, alpha: float = 0.0) -> None:
        """The objective function of VAE which consists of two parts.

        Parameters
        ----------
        gamma: float
            Gamma is the weight of KLD loss.
        """
        super(VAELoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(
        self,
        y_hat: torch.Tensor,
        y: torch.Tensor,
        z: torch.Tensor,
    ) -> Tuple[float, float, float, float]:
        """

        Parameters
        ----------
        y_hat: torch.Tensor
            Stock prediction.
            shape: [Ns]
        y: torch.Tensor
            Ground Truth.
            shape: [Ns]
        z: torch.Tensor
            Mean of the latent Gaussian.
            shape: [Ns, D]

        Returns
        -------
        Loss
        """
        reconstruction_loss = 10.0 * torch.nn.functional.mse_loss(y_hat, y)

        n = y.shape[0]
        ones = torch.ones(n, 1).cuda()

        pred_matrix = torch.einsum("n, km->nk", y_hat, ones)
        pred_diff = pred_matrix - pred_matrix.T

        y_matrix = torch.einsum("n, km->nk", y, ones)
        y_diff = y_matrix - y_matrix.T
        ranking_aware_loss = self.alpha * torch.sum(
            nn.functional.relu(-pred_diff * y_diff)
        )

        embedding_loss = self.gamma * torch.mean(
            (0.5 * torch.linalg.norm(z, dim=-1) ** 2)
        )

        loss = reconstruction_loss + embedding_loss + ranking_aware_loss
        return loss, reconstruction_loss, embedding_loss, ranking_aware_loss


def rank_ics(y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
    """Calculate the Rank ICs.

    Parameters
    ----------
    y_pred: torch.Tensor
        The predicted ranks of stocks in cross-section on s-th trading day.
        shape: [Ns]
    y_true: torch.Tensor
        The True ranks of stocks in cross-section on s-th trading day.
        shape: [Ns]
    Returns
    -------
    Rank ICs on s-th trading day.
    shape:[1]
    """
    eps = 1e-8
    Ns = y_true.shape[0]
    numerator = (1 / Ns) * torch.einsum(
        "n,n->",
        y_pred - torch.mean(y_pred, dim=-1, keepdim=True),
        y_true - torch.mean(y_true, dim=-1, keepdim=True),
    )
    denominator = torch.std(y_pred, dim=-1) * torch.std(y_true, dim=-1)

    return numerator / (denominator + eps)
