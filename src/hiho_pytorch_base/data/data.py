"""データ処理モジュール"""

from dataclasses import dataclass

import numpy
import torch
from torch import Tensor

from hiho_pytorch_base.data.phoneme import ArpaPhoneme


@dataclass
class InputData:
    """データ処理前のデータ構造"""

    phonemes: list[ArpaPhoneme]
    speaker_id: int


@dataclass
class OutputData:
    """データ処理後のデータ構造"""

    phoneme_id: Tensor  # (L,) 音素ID
    phoneme_duration: Tensor  # (L,) 音素継続時間
    speaker_id: Tensor


def preprocess(d: InputData, is_eval: bool) -> OutputData:
    """全ての変換・検証・配列化処理を統合"""
    # 音素情報の抽出
    phoneme_ids = numpy.array(
        [ArpaPhoneme.phoneme_list.index(p.phoneme) for p in d.phonemes],
        dtype=numpy.int32,
    )
    phoneme_durations = numpy.array(
        [p.duration for p in d.phonemes], dtype=numpy.float32
    )

    # Tensor変換
    return OutputData(
        phoneme_id=torch.from_numpy(phoneme_ids).long(),
        phoneme_duration=torch.from_numpy(phoneme_durations).float(),
        speaker_id=torch.tensor(d.speaker_id).long(),
    )
