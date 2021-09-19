from typing import *
from stud.layers.multihead_attention import MultiheadSelfAttention
from stud.layers.positional_encoder import PositionalEncoding
from stud.utils import HParams
import torch
import torch.nn as nn
import math

class TransformerEncoder(nn.Module):
    '''
    Torch implementation of Transformer encoder.
    Adapted from: https://uvadlc-notebooks.readthedocs.io/en/latest/index.html
    '''
    def __init__(
        self,
        input_dim: int,
        hparams: HParams,
        activation_fn: Optional[Callable] = nn.ReLU(inplace=True)
    ):                                                    
        super().__init__()
        self.self_att = MultiheadSelfAttention(input_dim, input_dim, hparams.num_heads)
        self.positional_encoder = PositionalEncoding(input_dim)
        self.dropout = nn.Dropout(hparams.dropout)
        self.fc = nn.Sequential(
            nn.Linear(input_dim, hparams.hidden_dim),
            nn.Dropout(hparams.dropout),
            activation_fn,
            nn.Linear(hparams.hidden_dim, input_dim)
        )
        self.norm1 = nn.LayerNorm(input_dim)
        self.norm2 = nn.LayerNorm(input_dim)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        add_positional_encoding: bool = True
    ) -> torch.Tensor:
        # Add positional encodings
        if add_positional_encoding:
            x = self.positional_encoder(x)
        
        # Apply attention
        att_out = self.self_att(x, mask=mask)
        X = self.dropout(att_out)
        x = self.norm1(x)

        # Simple feed-forward network
        fc_out = self.fc(x)
        x = self.dropout(fc_out)
        x = self.norm2(x)
        return x
