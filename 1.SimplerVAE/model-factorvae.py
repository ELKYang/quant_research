# -*- coding: utf-8 -*-
"""
Created on 2022/9/29 14:34

@author: Yang Fan

FactorVAE的模型文件，包含因子特征提取器，因子编码器，因子解码器，因子预测器以及整个模型的整合[注：Batch维度根据Qlib中实现直接使用股票数量Ns]
此版本根据多日的股票数据进行训练，但未向量化batch数据
"""
import math
from typing import Tuple

import torch
import torch.nn as nn
from einops.layers.torch import Rearrange


class FeatureExtractor(nn.Module):
    def __init__(
        self, feature_num: int = 20, embedding_dim: int = 128
    ) -> None:
        """Feature extractor is used to extract the feature from the raw data.

        Parameters
        ----------
        feature_num: int
            dimension of the input data (named C in paper).
        embedding_dim: int
            dimensions of the feature and hidden variables(gru) (named H in paper).
        """
        super(FeatureExtractor, self).__init__()
        self.proj = nn.Linear(
            in_features=feature_num, out_features=embedding_dim
        )
        self.activation = nn.LeakyReLU()
        self.gru = nn.GRU(
            input_size=embedding_dim,
            hidden_size=embedding_dim,
            batch_first=True,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """FeatureExtractor model

        Parameters
        ----------
        x: torch.Tensor
            The raw stock data.
            shape: [Ns, T, C]

        Returns
        -------
        The hidden state of GRU at last time step T as the latent features of stocks, which named 'e' in paper.
        shape: [Ns, H]
        """
        assert (
            len(x.shape) == 3
        ), "Input variable should have 3 dimensions, (Ns, T, C)"

        x = self.proj(x)
        x = self.activation(x)
        x = torch.squeeze(self.gru(x)[1], dim=0)

        return x


class FactorEncoder(nn.Module):
    def __init__(
        self, embedding_dim: int = 128, M: int = 30, K: int = 4
    ) -> None:
        """Factor encoder is used to extract posterior factors z_post from the future stock returns y and the latent features e.

        Parameters
        ----------
        embedding_dim: int
            The dimension of FeatureExtractor result e.
        M: int
            Number of Portfolios.
        K: int
            The dimensions of the mean and variance of the posterior distribution.
        """
        super(FactorEncoder, self).__init__()
        self.portfolio_layer = nn.Sequential(
            nn.Linear(in_features=embedding_dim, out_features=M),
            nn.Softmax(dim=0),
        )
        self.mapping_layer_mu = nn.Linear(in_features=M, out_features=K)
        self.mapping_layer_sigma = nn.Sequential(
            nn.Linear(in_features=M, out_features=K), nn.Softplus()
        )

    def forward(
        self, e: torch.Tensor, y: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """FactorEncoder model

        Parameters
        ----------
        e: torch.Tensor
            The feature e obtained by Ns stocks after passing through the FeatureExtractor
            shape: [Ns, embedding_dim]
        y: torch.Tensor
            Earnings per stock
            shape: [Ns]

        Returns
        -------
        z_post: tuple[torch.Tensor, torch.Tensor]
            Mean and variance of the posterior factors distribution.
            shape: tuple([K], [K])
        """
        a_p = self.portfolio_layer(e)  # 通过每支股票的特征得到M种投资组合中每支股票的权重
        y_p = torch.einsum(
            "n, nm->m", y, a_p
        )  # 将每支股票的收益与对应投资组合中的权重相乘，得到每个投资组合的收益
        # todo: 可视化a_p和y_p, 观察学习出的投资组合每支股票权重最后得到的投资组合收益是否与真实的对应投资组合收益相一致
        """Explanation for this operation in paper:
        Because of the number of individual stocks in cross-section is large and varies with time,
        instead of using stock returns y directly,
        we construct a set of portfolios inspired by (Gu, Kelly, and Xiu 2021),
        these portfolios are dynamically re-weighted on the basis of stock latent features.
        """
        mu_z_post = self.mapping_layer_mu(y_p)
        sigma_z_post = self.mapping_layer_sigma(y_p)
        return mu_z_post, sigma_z_post


class FactorDecoderAlphaLayer(nn.Module):
    def __init__(self, embedding_dim: int = 128) -> None:
        """Alpha layer is used to output idiosyncratic returns Alpha from the latent features e.

        Parameters
        ----------
        embedding_dim: int
            The dimension of the hidden state, setting the same dimension as feature e in the paper.
        """
        super(FactorDecoderAlphaLayer, self).__init__()
        self.proj = nn.Sequential(
            nn.Linear(in_features=embedding_dim, out_features=embedding_dim),
            nn.LeakyReLU(),
        )
        self.mu_alpha_layer = nn.Linear(
            in_features=embedding_dim, out_features=1
        )
        self.sigma_alpha_layer = nn.Sequential(
            nn.Linear(in_features=embedding_dim, out_features=1), nn.Softplus()
        )

    def forward(self, e: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Alpha layer

        Parameters
        ----------
        e: torch.Tensor
            The feature e obtained by Ns stocks after passing through the FeatureExtractor
            shape: [Ns, embedding_dim]

        Returns
        -------
        alpha: tuple[torch.Tensor, torch.Tensor]
            The mean and variance of the idiosyncratic returns Alpha.
            shape: tuple([Ns, 1], [Ns, 1])
        """
        h_alpha = self.proj(e)
        mu_alpha = self.mu_alpha_layer(h_alpha)
        sigma_alpha = self.sigma_alpha_layer(h_alpha)
        return mu_alpha, sigma_alpha


class FactorDecoder(nn.Module):
    def __init__(self, embedding_dim: int = 128, K: int = 4) -> None:
        """The factor decoder is used to calculate stock returns y_hat from factors z and the latent feature e.

        Parameters
        ----------
        embedding_dim: int
            The dimension of FeatureExtractor result e.
        K: int
            The dimensions of the mean and variance of the return distribution.

        """
        super(FactorDecoder, self).__init__()
        self.alpha_layer = FactorDecoderAlphaLayer(embedding_dim)
        self.beta_layer = nn.Linear(in_features=embedding_dim, out_features=K)

    def forward(
        self, z: Tuple[torch.Tensor, torch.Tensor], e: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Factor decoder model

        Parameters
        ----------
        z: tuple[torch.Tensor, torch.Tensor]
            The mean and variance of the posterior or prior factor distribution, getting from FactorEncoder or FactorPredictor
            shape: [K]
        e: torch.Tensor
            The feature e obtained by Ns stocks after passing through the FeatureExtractor
            shape: [Ns, embedding_dim]

        Returns
        -------
        y_pred: tuple[torch.Tensor, torch.Tensor]
            Mean and variance of the stock forecast output distribution.
            shape: tuple([Ns], [Ns])
        """
        mu_alpha, sigma_alpha = self.alpha_layer(e)
        beta = self.beta_layer(e)
        mu_z, sigma_z = torch.unsqueeze(z[0], dim=-1), torch.unsqueeze(
            z[1], dim=-1
        )
        mu_y = torch.squeeze(mu_alpha + torch.matmul(beta, mu_z), dim=-1)
        sigma_y = torch.squeeze(
            torch.sqrt(
                torch.square(sigma_alpha)
                + torch.matmul(torch.square(beta), torch.square(sigma_z))
            ),
            dim=-1,
        )
        return mu_y, sigma_y


# Different from the attention operations in paper which is too simple to fit the posterior factors.
# We use multi head self-attention to fit the posterior factors. However, it is not perfect now.
# class FactorPredictor_private(nn.Module):
#     def __init__(
#         self, embedding_dim: int = 128, dim_head: int = 128, heads_K: int = 4
#     ) -> None:
#         """Factor predictor extracts prior factors z_prior from the stock latent features e.
#
#         Parameters
#         ----------
#         embedding_dim: int
#             The dimension of FeatureExtractor result e.
#         dim_head: int
#             The dimension of the head, same as the dimension if e in paper(All named H).
#         heads_K: int
#             The head number in multi-head self-attention.,
#             also the dimensions of the mean and variance of the prior distribution.
#         """
#         super(FactorPredictor_private, self).__init__()
#         # parameters and layers for multi head attention
#         self.heads = heads_K
#         inner_dim = dim_head * heads_K
#         self.to_qkv = nn.Linear(embedding_dim, inner_dim * 3)
#         self.rearrange = Rearrange("n (h d) -> h n d", h=self.heads)
#         # parameters and pi_prior layers for mu_z_prior and sigma_z_prior
#         self.mapping_layer_mu = nn.Linear(in_features=dim_head, out_features=1)
#         self.mapping_layer_sigma = nn.Sequential(
#             nn.Linear(in_features=dim_head, out_features=1), nn.Softplus()
#         )
#
#     def forward(self, e: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
#         """Factor predictor model
#
#         Parameters
#         ----------
#         e: torch.Tensor
#             The feature e obtained by Ns stocks after passing through the FeatureExtractor
#             shape: [Ns, embedding_dim]
#
#         Returns
#         -------
#         z_prior: tuple[torch.Tensor, torch.Tensor]
#             Mean and variance of the prior factors distribution.
#             shape: tuple([K], [K])
#         """
#         # multi-head self-attention
#         # Different from the attention operations in paper which is too simple to fit the posterior factors.
#         query, key, value = self.to_qkv(e).chunk(3, dim=-1)
#         query, key, value = (
#             self.rearrange(query),
#             self.rearrange(key),
#             self.rearrange(value),
#         )
#         scores = torch.nn.functional.softmax(
#             torch.matmul(query, key.transpose(-1, -2)) / math.sqrt(query.shape[-1]),
#             dim=-1,
#         )
#         attention = torch.matmul(scores, value)  # [heads_K Ns H]
#         h_multi = attention.sum(dim=1)
#         # pi_prior layer
#         mu_z_prior = torch.squeeze(self.mapping_layer_mu(h_multi), dim=-1)
#         sigma_z_prior = torch.squeeze(self.mapping_layer_sigma(h_multi), dim=-1)
#         return mu_z_prior, sigma_z_prior


class FactorPredictor(nn.Module):
    def __init__(self, embedding_dim: int = 128, K: int = 4) -> None:
        """Factor predictor extracts prior factors z_prior from the stock latent features e.

        Parameters
        ----------
        embedding_dim: int
            The dimension of FeatureExtractor result e.
        K: int
            The head number in multi-head self-attention.,
            also the dimensions of the mean and variance of the prior distribution.
        """
        super(FactorPredictor, self).__init__()
        # parameters and layers for multi head attention
        self.w_key = torch.nn.Parameter(
            torch.randn([K, 1]), requires_grad=True
        )
        self.w_value = torch.nn.Parameter(
            torch.randn([K, 1]), requires_grad=True
        )
        self.q = torch.nn.Parameter(
            torch.randn(embedding_dim), requires_grad=True
        )
        # parameters and pi_prior layers for mu_z_prior and sigma_z_prior
        self.mapping_layer_mu = nn.Linear(
            in_features=embedding_dim, out_features=1
        )
        self.mapping_layer_sigma = nn.Sequential(
            nn.Linear(in_features=embedding_dim, out_features=1), nn.Softplus()
        )

    def forward(self, e: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Factor predictor model

        Parameters
        ----------
        e: torch.Tensor
            The feature e obtained by Ns stocks after passing through the FeatureExtractor
            shape: [Ns, embedding_dim]


        Returns
        -------
        z_prior: tuple[torch.Tensor, torch.Tensor]
            Mean and variance of the prior factors distribution.
            shape: tuple([K], [K])
        """
        e = torch.unsqueeze(e, dim=0)
        k = torch.einsum("kd,dnh->knh", self.w_key, e)
        v = torch.einsum("kd,dnh->knh", self.w_value, e)
        a_att = torch.einsum("h,knh->kn", self.q, k) / (
            torch.norm(self.q, p=2) * torch.norm(k, p=2, dim=-1, keepdim=False)
        )
        a_att = torch.maximum(a_att, torch.as_tensor(0))
        a_att = a_att / torch.maximum(
            torch.sum(a_att, dim=-1, keepdim=True), torch.as_tensor(1e-6)
        )

        h_multi = torch.einsum("kn,knh->kh", a_att, v)
        # pi_prior layer
        mu_z_prior = torch.squeeze(self.mapping_layer_mu(h_multi), dim=-1)
        sigma_z_prior = torch.squeeze(
            self.mapping_layer_sigma(h_multi), dim=-1
        )
        return mu_z_prior, sigma_z_prior


class FactorVAE(nn.Module):
    def __init__(self, C=20, H=128, M=30, K=4):
        """The whole architecture of FactorVAE
        @article{duan2022factorvae,
        title={FactorVAE: A Probabilistic Dynamic Factor Model Based on Variational Autoencoder for Predicting Cross-sectional Stock Returns},
        author={Duan, Yitong and Wang, Lei and Zhang, Qizhong and Li, Jian},
        year={2022}
        }

        Parameters
        ----------
        C: int
            Dimension of the input data (named C in paper).
        H: int
            Dimensions of the feature and other hidden variables (named H in paper).
        M: int
            Number of Portfolios.
        K: int
            The dimensions of the mean and variance of the posterior and prior distribution.
        """
        super(FactorVAE, self).__init__()
        self.feature_extractor = FeatureExtractor(
            feature_num=C, embedding_dim=H
        )
        self.factor_encoder = FactorEncoder(embedding_dim=H, M=M, K=K)
        self.factor_decoder = FactorDecoder(embedding_dim=H, K=K)
        self.factor_predictor = FactorPredictor(embedding_dim=H, K=K)

    def forward(
        self, x: torch.Tensor, y: torch.Tensor = None, mode: str = "Train"
    ) -> tuple:
        """FactorVAE
        A prior-posterior learning method based on VAE: train a factor predictor only given the historical observation data, which predicts factors to approximate the optimal posterior factors above, called prior factors.
        Then we use the factor decoder to calculate the stock returns by the prior factors without any future information leakage, as the predicted returns of model.

        Parameters
        ----------
        x: torch.Tensor
            The raw stock data.
            shape: [Ns, T, C]
        y: torch.Tensor
            Earnings per stock.
            shape: [Ns]
        mode: str
            To distinguish training and testing, and implement a prior-posterior training strategy.

        Returns
        -------
        Train only:
            z_post: tuple[torch.Tensor, torch.Tensor]
                Mean and variance of the posterior factors distribution.
                shape: tuple([K], [K])
            z_prior: tuple[torch.Tensor, torch.Tensor]
                Mean and variance of the prior factors distribution.
                shape: tuple([K], [K])
            y_rec: tuple[torch.Tensor, torch.Tensor]
                Mean and variance of the posterior stock forecast output distribution.
                shape: tuple([Ns], [Ns])
        y_pred: tuple[torch.Tensor, torch.Tensor]
            Mean and variance of the prior stock forecast output distribution.
            shape: tuple([Ns], [Ns])
        """
        if mode == "Train":
            if y is None:
                raise ValueError(
                    "In train mode, we need the stocks return to calculate the posterior distribution."
                )
            e = self.feature_extractor(x)
            z_post = self.factor_encoder(e, y)
            z_prior = self.factor_predictor(e)
            y_rec = self.factor_decoder(z_post, e)
            y_pred = self.factor_decoder(z_prior, e)
            return z_post, z_prior, y_rec, y_pred
        elif mode == "Val" or mode == "Test":
            e = self.feature_extractor(x)
            z_prior = self.factor_predictor(e)
            y_pred = self.factor_decoder(z_prior, e)
            return y_pred
        else:
            raise ValueError("mode should be 'Train', 'Val' or 'Test'.")
