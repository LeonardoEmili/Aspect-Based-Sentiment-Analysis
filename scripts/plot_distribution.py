#!/usr/bin/env python3
# -*- coding: utf-8 -*-

'''
Plots the distribution of preprocessed input data.
python -m scripts.plot_distribution [PREPROCESSED_PATH] [PLOT_TYPE]
Author: Leonardo Emili
'''

import plotly.graph_objects as go
from torchtext.vocab import Vocab
from typing import *
import fire
import torch
import sys

sys.path.append('hw2')
from hw2.stud.dataset import ABSADataset

POS_TAGS = ['NOUN', 'VERB', 'ADP', '.', 'DET', 'ADJ', 'ADV', 'CONJ', 'PRON', 'PRT', 'NUM', 'X']
POLARITY_TAGS = ['positive', 'negative', 'neutral', 'conflict']

def get_aspect_indices(sample: Dict[str, Any], ner_vocab: Vocab) -> List[int]:
    ''' Returns the indices of tokens labeled as aspect terms. '''
    return [i for i,x in enumerate(sample['ner']) if x.item() != ner_vocab['O']]

def pos_distribution(preprocessed_path: str, width: int, height: int):
    ds: ABSADataset = torch.load(preprocessed_path)

    def extract_values(x: Dict[str, torch.Tensor], y: Dict[str, torch.Tensor], idxs: List[int]) -> Tuple[List, List]:
        pos_tags = ds.pos_vocab.lookup_tokens(x['pos_indices'][idxs].tolist())
        polarities = ds.polarity_vocab.lookup_tokens(y['token_polarity'][idxs].tolist())
        return [f'{pos}_{polarity}' for pos, polarity in zip(pos_tags, polarities)]

    # Get the indices of tokens tagged as aspect terms
    indices = [get_aspect_indices(sample, ds.ner_ext_vocab) for _, sample in ds.train_ds]
    annotated_by_pos_sentiment = [v for sample, idxs in zip(ds.train_ds, indices) for v in extract_values(*sample, idxs)]

    # Create n arrays of values (POLARITY_TAGS) for each POS tag (POS_TAGS)
    distribution_counter = Counter(annotated_by_pos_sentiment)
    plot_data = [[distribution_counter[f'{pos}_{polarity}'] for pos in POS_TAGS] for polarity in POLARITY_TAGS]

    # Rename the class '.' as PUNCT (i.e. abbr. for punctuation class)
    POS_TAGS[POS_TAGS.index('.')] = 'PUNCT'

    # Plot the figure
    legend = dict(yanchor='top', xanchor='right')
    fig = go.Figure(data=[go.Bar(name=polarity, x=POS_TAGS, y=data) for data, polarity in zip(plot_data, POLARITY_TAGS)])
    fig.update_layout(barmode='group', width=width, height=height, legend=legend)

    fig.show()

def main(
    preprocessed_path: Optional[str] = 'data/preprocessing/SemEval-2014.pth',
    plot_type: Optional[str] = 'pos',
    width: Optional[int] = 1000,
    height: Optional[int] = 500
    ):
    if plot_type == 'pos':
        pos_distribution(preprocessed_path, width, height)
    else:
        raise ValueError(f'Plot type {plot_type} not available.')


if __name__ == '__main__':
    fire.Fire(main)
