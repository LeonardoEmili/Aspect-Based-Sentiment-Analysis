from typing import *
from torch.nn.utils.rnn import pad_sequence
import math
import torch
import torch.nn as nn

class PositionalEncoding(nn.Module):
    '''
    Torch implementation of positional encoder.
    Source: https://pytorch.org/tutorials/beginner/transformer_tutorial.html
    '''
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        x = x.transpose(0, 1)
        x = x + self.pe[:x.size(0)]
        x = self.dropout(x)
        x = x.transpose(0, 1)
        return x