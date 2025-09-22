"""評価値計算モジュール"""

from dataclasses import dataclass
from typing import Self

import torch
from torch import Tensor, nn
from torch.nn.functional import mse_loss

from hiho_pytorch_base.batch import BatchOutput
from hiho_pytorch_base.generator import Generator, GeneratorOutput
from hiho_pytorch_base.utility.pytorch_utility import detach_cpu
from hiho_pytorch_base.utility.train_utility import DataNumProtocol


@dataclass
class EvaluatorOutput(DataNumProtocol):
    """評価値"""

    loss: Tensor

    def detach_cpu(self) -> Self:
        """全てのTensorをdetachしてCPUに移動"""
        self.loss = detach_cpu(self.loss)
        return self


def calculate_value(output: EvaluatorOutput) -> Tensor:
    """評価値の良し悪しを計算する関数。高いほど良い。"""
    return -1 * output.loss


class Evaluator(nn.Module):
    """評価値を計算するクラス"""

    def __init__(self, generator: Generator):
        super().__init__()
        self.generator = generator

    @torch.no_grad()
    def forward(self, batch: BatchOutput) -> EvaluatorOutput:
        """データをネットワークに入力して評価値を計算する"""
        output_result: GeneratorOutput = self.generator(
            phoneme_id_list=batch.phoneme_id_list,
            speaker_id=batch.speaker_id,
        )

        # 予測結果とターゲットを結合して一括計算
        pred_duration_all = torch.cat(output_result.duration, dim=0)  # (sum(L),)
        target_duration_all = torch.cat(batch.phoneme_duration_list, dim=0)  # (sum(L),)

        # 音素継続時間損失
        loss = mse_loss(pred_duration_all, target_duration_all)

        return EvaluatorOutput(
            loss=loss,
            data_num=batch.data_num,
        )
