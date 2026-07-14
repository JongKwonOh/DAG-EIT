# #----------------------------------------------------------------------------
# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# This work is licensed under a Creative Commons
# Attribution-NonCommercial-ShareAlike 4.0 International License.
# You should have received a copy of the license along with this
# work. If not, see http://creativecommons.org/licenses/by-nc-sa/4.0/

"""Loss functions used in the paper
"Elucidating the Design Space of Diffusion-Based Generative Models"."""

import torch
from torch_utils import persistence
from torch_utils import training_stats

#----------------------------------------------------------------------------
# Shared logging helper (VP/VE/EDM 공통)
#----------------------------------------------------------------------------

def _report_loss_channels(
    loss: torch.Tensor,
    report_channels=(0, 1),
    report_channel_mean: bool = True,
    report_allch_mean: bool = True,
    channel_mean_name: str = "Loss/loss_ch01_mean",
    all_mean_name: str = "Loss/loss_allch_mean",
):
    """
    loss: (B,C,H,W) (권장) 또는 (B,C,...) 형태를 가정.
    - Loss/loss_ch{ch}: 채널별 텐서 report
    - Loss/loss_ch01_mean: report_channels 평균 스칼라 report
    - Loss/loss_allch_mean: 전체 평균 스칼라 report
    """
    if (not torch.is_tensor(loss)) or loss.ndim < 2:
        return

    # 채널 차원은 dim=1 로 가정
    C = loss.shape[1]
    if C <= 0:
        return

    # 1) 채널별 report
    ch_tuple = tuple(report_channels) if report_channels is not None else tuple()
    if len(ch_tuple) > 0:
        for ch in ch_tuple:
            ch = int(ch)
            if 0 <= ch < C:
                # (B,1,...)로 남기기
                training_stats.report(f"Loss/loss_ch{ch}", loss[:, ch:ch+1].detach())

    # 2) report_channels 평균 (스칼라)
    if report_channel_mean and len(ch_tuple) > 0:
        ch_list = [int(ch) for ch in ch_tuple if 0 <= int(ch) < C]
        if len(ch_list) > 0:
            ch_mean = loss[:, ch_list, ...].detach().mean()
            training_stats.report(channel_mean_name, ch_mean)

    # 3) 전체 채널 평균 (스칼라)
    if report_allch_mean:
        all_mean = loss.detach().mean()
        training_stats.report(all_mean_name, all_mean)


#----------------------------------------------------------------------------
# Loss function corresponding to the variance preserving (VP) formulation
#----------------------------------------------------------------------------

@persistence.persistent_class
class VPLoss:
    def __init__(
        self,
        beta_d=19.9,
        beta_min=0.1,
        epsilon_t=1e-5,
        report_channels=(0, 1),
        report_channel_mean=True,   # (0,1) 같은 지정 채널 평균도 로그로 남길지
        report_allch_mean=True,     # 전체 채널 평균도 로그로 남길지
    ):
        self.beta_d = beta_d
        self.beta_min = beta_min
        self.epsilon_t = epsilon_t
        self.report_channels = tuple(report_channels) if report_channels is not None else tuple()
        self.report_channel_mean = bool(report_channel_mean)
        self.report_allch_mean = bool(report_allch_mean)

    def __call__(self, net, images, labels, augment_pipe=None):
        rnd_uniform = torch.rand([images.shape[0], 1, 1, 1], device=images.device)
        sigma = self.sigma(1 + rnd_uniform * (self.epsilon_t - 1))
        weight = 1 / sigma ** 2

        y, augment_labels = augment_pipe(images) if augment_pipe is not None else (images, None)
        n = torch.randn_like(y) * sigma
        D_yn = net(y + n, sigma, labels, augment_labels=augment_labels)

        loss = weight * ((D_yn - y) ** 2)   # (B,C,H,W)

        # Logging only (does NOT affect optimization)
        _report_loss_channels(
            loss,
            report_channels=self.report_channels,
            report_channel_mean=self.report_channel_mean,
            report_allch_mean=self.report_allch_mean,
            channel_mean_name="Loss/loss_ch01_mean",
            all_mean_name="Loss/loss_allch_mean",
        )

        return loss

    def sigma(self, t):
        t = torch.as_tensor(t)
        return ((0.5 * self.beta_d * (t ** 2) + self.beta_min * t).exp() - 1).sqrt()


#----------------------------------------------------------------------------
# Loss function corresponding to the variance exploding (VE) formulation
#----------------------------------------------------------------------------

@persistence.persistent_class
class VELoss:
    def __init__(
        self,
        sigma_min=0.02,
        sigma_max=100,
        report_channels=(0, 1),
        report_channel_mean=True,
        report_allch_mean=True,
    ):
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.report_channels = tuple(report_channels) if report_channels is not None else tuple()
        self.report_channel_mean = bool(report_channel_mean)
        self.report_allch_mean = bool(report_allch_mean)

    def __call__(self, net, images, labels, augment_pipe=None):
        rnd_uniform = torch.rand([images.shape[0], 1, 1, 1], device=images.device)
        sigma = self.sigma_min * ((self.sigma_max / self.sigma_min) ** rnd_uniform)
        weight = 1 / sigma ** 2

        y, augment_labels = augment_pipe(images) if augment_pipe is not None else (images, None)
        n = torch.randn_like(y) * sigma
        D_yn = net(y + n, sigma, labels, augment_labels=augment_labels)

        loss = weight * ((D_yn - y) ** 2)   # (B,C,H,W)

        # VP와 동일하게 로그 남기기
        _report_loss_channels(
            loss,
            report_channels=self.report_channels,
            report_channel_mean=self.report_channel_mean,
            report_allch_mean=self.report_allch_mean,
            channel_mean_name="Loss/loss_ch01_mean",
            all_mean_name="Loss/loss_allch_mean",
        )

        return loss


#----------------------------------------------------------------------------
# Improved loss function proposed in "Elucidating the Design Space..." (EDM)
#----------------------------------------------------------------------------

@persistence.persistent_class
class EDMLoss:
    def __init__(
        self,
        P_mean=-1.2,
        P_std=1.2,
        sigma_data=0.5,
        report_channels=(0, 1),
        report_channel_mean=True,
        report_allch_mean=True,
    ):
        self.P_mean = P_mean
        self.P_std = P_std
        self.sigma_data = sigma_data
        self.report_channels = tuple(report_channels) if report_channels is not None else tuple()
        self.report_channel_mean = bool(report_channel_mean)
        self.report_allch_mean = bool(report_allch_mean)

    def __call__(self, net, images, labels=None, augment_pipe=None):
        rnd_normal = torch.randn([images.shape[0], 1, 1, 1], device=images.device)
        sigma = (rnd_normal * self.P_std + self.P_mean).exp()
        weight = (sigma ** 2 + self.sigma_data ** 2) / (sigma * self.sigma_data) ** 2

        y, augment_labels = augment_pipe(images) if augment_pipe is not None else (images, None)
        n = torch.randn_like(y) * sigma
        D_yn = net(y + n, sigma, labels, augment_labels=augment_labels)

        loss = weight * ((D_yn - y) ** 2)   # (B,C,H,W)

        # VP와 동일하게 로그 남기기
        _report_loss_channels(
            loss,
            report_channels=self.report_channels,
            report_channel_mean=self.report_channel_mean,
            report_allch_mean=self.report_allch_mean,
            channel_mean_name="Loss/loss_ch01_mean",
            all_mean_name="Loss/loss_allch_mean",
        )

        return loss
