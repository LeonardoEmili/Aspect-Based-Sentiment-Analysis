from typing import *
from stud.utils import HParams
from stud.constants import *
import torch
import torch.nn as nn

class AttentionLSTM(nn.Module):
    def __init__(
        input_dim: int,
        hparams: HParams,
        axial_shape: Tuple[int, int] = AXIAL_SHAPE_DEFAULT
    ):
        super().__init__()
        self.positional_encoder = AxialPositionalEmbedding(input_dim, axial_shape=axial_shape)
        self.att = nn.MultiheadAttention(input_dim, hparams.num_heads, dropout=hparams.dropout)
        self.lstm = nn.LSTM(input_dim, hparams.hidden_dim, batch_first=True, bidirectional=True)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        add_positional_encoding: bool = True
    ) -> torch.Tensor:
        lstm_x, _ = self.lstm(x)

        # Add positional encodings
        if add_positional_encoding:
            x = self.positional_encoder(x)

        # Apply self-attention
        x, _ = self.att(x, x, x, key_padding_mask=mask)
        x = torch.cat([x, lstm_x], dim=-1)
        return x
