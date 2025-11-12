"""バッチ処理モジュール"""

from dataclasses import dataclass
from typing import Self

import torch
from torch import Tensor

from .data.data import OutputData
from .utility.pytorch_utility import to_device


@dataclass
class BatchOutput:
    """バッチ処理後のデータ構造"""

    phoneme_id_list: list[Tensor]  # [(L,)]
    phoneme_duration_list: list[Tensor]  # [(L,)]
    speaker_id: Tensor  # (B,)

    @property
    def data_num(self) -> int:
        """バッチサイズを返す"""
        return self.speaker_id.shape[0]

    def to_device(self, device: str, non_blocking: bool) -> Self:
        """データを指定されたデバイスに移動"""
        self.phoneme_id_list = to_device(
            self.phoneme_id_list, device, non_blocking=non_blocking
        )
        self.phoneme_duration_list = to_device(
            self.phoneme_duration_list, device, non_blocking=non_blocking
        )
        self.speaker_id = to_device(self.speaker_id, device, non_blocking=non_blocking)
        return self


def collate_stack(values: list[Tensor]) -> Tensor:
    """Tensorのリストをスタックする"""
    return torch.stack(values)


def collate_dataset_output(data_list: list[OutputData]) -> BatchOutput:
    """DatasetOutputのリストをBatchOutputに変換"""
    if len(data_list) == 0:
        raise ValueError("batch is empty")

    return BatchOutput(
        phoneme_id_list=[d.phoneme_id for d in data_list],
        phoneme_duration_list=[d.phoneme_duration for d in data_list],
        speaker_id=collate_stack([d.speaker_id for d in data_list]),
    )
