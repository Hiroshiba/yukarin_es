"""モデルのモジュール。ネットワークの出力から損失を計算する。"""

from dataclasses import dataclass
from typing import Self

import torch
from torch import Tensor, nn
from torch.nn.functional import mse_loss

from hiho_pytorch_base.batch import BatchOutput
from hiho_pytorch_base.config import ModelConfig
from hiho_pytorch_base.network.predictor import Predictor
from hiho_pytorch_base.utility.pytorch_utility import detach_cpu
from hiho_pytorch_base.utility.train_utility import DataNumProtocol


@dataclass
class ModelOutput(DataNumProtocol):
    """学習時のモデルの出力。損失と、イテレーション毎に計算したい値を含む"""

    loss: Tensor
    """逆伝播させる損失"""

    duration_loss: Tensor
    """音素継続時間損失"""

    def detach_cpu(self) -> Self:
        """全てのTensorをdetachしてCPUに移動"""
        self.loss = detach_cpu(self.loss)
        self.duration_loss = detach_cpu(self.duration_loss)
        return self


class Model(nn.Module):
    """学習モデルクラス"""

    def __init__(self, model_config: ModelConfig, predictor: Predictor):
        super().__init__()
        self.model_config = model_config
        self.predictor = predictor

    def forward(self, batch: BatchOutput) -> ModelOutput:
        """データをネットワークに入力して損失などを計算する"""
        duration_output_list = self.predictor(
            phoneme_id_list=batch.phoneme_id_list,
            speaker_id=batch.speaker_id,
        )  # [(L,)]

        # 一括で損失計算
        pred_duration_all = torch.cat(duration_output_list, dim=0)  # (sum(L),)
        target_duration_all = torch.cat(batch.phoneme_duration_list, dim=0)  # (sum(L),)

        # 音素継続時間損失
        duration_loss = mse_loss(pred_duration_all, target_duration_all)

        return ModelOutput(
            loss=duration_loss,
            duration_loss=duration_loss,
            data_num=batch.data_num,
        )
