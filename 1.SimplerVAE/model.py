# -*- coding: utf-8 -*-
"""
Created on 2022/9/29 14:34

@author: Yang Fan

Simpler VAE
"""
from typing import Any, Tuple

import torch
import torch.nn as nn
from torch import Tensor


class Encoder(nn.Module):
    def __init__(
        self,
        feature_num: int = 20,
        embedding_dim: int = 128,
        latent_dim: int = 64,
    ) -> None:
        """Feature extractor is used to extract the feature from the raw data.

        Parameters
        ----------
        feature_num: int
            dimension of the input data (named C in paper).
        embedding_dim: int
            dimensions of the feature and hidden variables(gru) (named H in paper).
        """
        super(Encoder, self).__init__()
        self.proj = nn.Sequential(
            nn.Linear(in_features=feature_num, out_features=embedding_dim),
            nn.LeakyReLU(),
        )
        self.gru = nn.GRU(
            input_size=embedding_dim,
            hidden_size=embedding_dim,
            batch_first=True,
        )
        self.z_layer = nn.Linear(embedding_dim, latent_dim)

    def forward(self, x: torch.Tensor) -> tuple[Tensor, Tensor]:
        """FeatureExtractor model

        Parameters
        ----------
        x: torch.Tensor
            The raw stock data.
            shape: [Ns, T, C]

        Returns
        -------
        Mean and log variance of the latent variable.
        shape: tuple([Ns, D], [Ns, D])
        """
        assert (
            len(x.shape) == 3
        ), "Input variable should have 4 dimensions, (Ns, T, C)"
        x = self.proj(x)
        x = torch.squeeze(self.gru(x)[1], dim=0)

        z = self.z_layer(x)

        return x, z


class VAE(nn.Module):
    def __init__(
        self,
        feature_num: int = 20,
        embedding_dim: int = 128,
        latent_dim: int = 64,
        decoder_layers: int = 4,
    ):
        super(VAE, self).__init__()
        self.encoder = Encoder(
            feature_num=feature_num,
            embedding_dim=embedding_dim,
            latent_dim=latent_dim,
        )
        modules = [
            nn.Sequential(
                nn.Linear(latent_dim, latent_dim),
                nn.LeakyReLU(),
                nn.Linear(latent_dim, 1),
            )
        ]
        self.decoder = nn.Sequential(*modules)

    def forward(self, x: torch.Tensor) -> tuple[Tensor, Tensor]:
        """

        Parameters
        ----------
        x: torch.Tensor
            The raw stock data.
            shape: [Ns, T, C]

        Returns
        -------
        y_hat: torch.Tensor
            Stock prediction.
            shape: [Ns]
        mu: torch.Tensor
            Mean of the latent Gaussian.
            shape: [Ns, D]
        logvar: torch.Tensor
            Log variance of the latent variable.
            shape: [Ns, D]
        """
        gru_encode_res, z = self.encoder(x)
        y_hat = torch.squeeze(self.decoder(z), dim=-1)
        return y_hat, z
