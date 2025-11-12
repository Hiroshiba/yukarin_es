"""メインのネットワークモジュール"""

import torch
from torch import Tensor, nn
from torch.nn.utils.rnn import pad_sequence

from ..config import NetworkConfig
from .conformer.encoder import Encoder
from .transformer.utility import make_non_pad_mask


class Predictor(nn.Module):
    """メインのネットワーク"""

    def __init__(
        self,
        phoneme_size: int,
        phoneme_embedding_size: int,
        hidden_size: int,
        speaker_size: int,
        speaker_embedding_size: int,
        encoder: Encoder,
    ):
        super().__init__()

        self.hidden_size = hidden_size

        # TODO: 推論時は行列演算を焼き込める。精度的にdoubleにする必要があるかも
        self.phoneme_embedder = nn.Sequential(
            nn.Embedding(phoneme_size, phoneme_embedding_size),
            nn.Linear(phoneme_embedding_size, phoneme_embedding_size),
            nn.Linear(phoneme_embedding_size, phoneme_embedding_size),
            nn.Linear(phoneme_embedding_size, phoneme_embedding_size),
            nn.Linear(phoneme_embedding_size, phoneme_embedding_size),
        )

        # TODO: 推論時は行列演算を焼き込める。精度的にdoubleにする必要があるかも
        self.speaker_embedder = nn.Sequential(
            nn.Embedding(speaker_size, speaker_embedding_size),
            nn.Linear(speaker_embedding_size, speaker_embedding_size),
            nn.Linear(speaker_embedding_size, speaker_embedding_size),
            nn.Linear(speaker_embedding_size, speaker_embedding_size),
            nn.Linear(speaker_embedding_size, speaker_embedding_size),
        )

        # Conformer前の写像
        embedding_size = phoneme_embedding_size
        self.pre_conformer = nn.Linear(
            embedding_size + speaker_embedding_size, hidden_size
        )

        self.encoder = encoder

        # 出力ヘッド - 音素長予測用
        self.duration_head = nn.Linear(hidden_size, 1)

    def forward(  # noqa: D102
        self,
        *,
        phoneme_id_list: list[Tensor],  # [(L,)]
        speaker_id: Tensor,  # (B,)
    ) -> list[Tensor]:  # duration_list [(L,)]
        device = speaker_id.device
        batch_size = len(phoneme_id_list)

        # シーケンスをパディング
        phoneme_lengths = torch.tensor(
            [seq.shape[0] for seq in phoneme_id_list], device=device
        )
        padded_phoneme_ids = pad_sequence(phoneme_id_list, batch_first=True)  # (B, L)

        # 埋め込み
        phoneme_embed = self.phoneme_embedder(padded_phoneme_ids)  # (B, L, ?)

        # 話者埋め込み
        speaker_embed = self.speaker_embedder(speaker_id)  # (B, ?)
        max_length = padded_phoneme_ids.size(1)
        speaker_embed = speaker_embed.unsqueeze(1).expand(
            batch_size, max_length, -1
        )  # (B, L, ?)

        # 埋め込みを結合
        h = torch.cat([phoneme_embed, speaker_embed], dim=2)  # (B, L, ?)

        # Conformer前の投影
        h = self.pre_conformer(h)  # (B, L, ?)

        # マスキング
        mask = make_non_pad_mask(phoneme_lengths).unsqueeze(-2).to(device)  # (B, 1, L)

        # Conformerエンコーダ
        h, _ = self.encoder(x=h, cond=None, mask=mask)  # (B, L, ?)

        # 出力ヘッド - 全音素に対して継続時間を予測
        duration = self.duration_head(h).squeeze(-1)  # (B, L)

        # 音素長をリストで返す
        return [duration[i, :length] for i, length in enumerate(phoneme_lengths)]


def create_predictor(config: NetworkConfig) -> Predictor:
    """設定からPredictorを作成"""
    encoder = Encoder(
        hidden_size=config.hidden_size,
        condition_size=0,
        block_num=config.conformer_block_num,
        dropout_rate=config.conformer_dropout_rate,
        positional_dropout_rate=config.conformer_dropout_rate,
        attention_head_size=8,
        attention_dropout_rate=config.conformer_dropout_rate,
        use_macaron_style=True,
        use_conv_glu_module=True,
        conv_glu_module_kernel_size=31,
        feed_forward_hidden_size=config.hidden_size * 4,
        feed_forward_kernel_size=3,
    )
    return Predictor(
        phoneme_size=config.phoneme_size,
        phoneme_embedding_size=config.phoneme_embedding_size,
        hidden_size=config.hidden_size,
        speaker_size=config.speaker_size,
        speaker_embedding_size=config.speaker_embedding_size,
        encoder=encoder,
    )
