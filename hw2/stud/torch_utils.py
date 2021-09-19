from typing import *
from torch.nn.utils.rnn import pad_sequence
import torch.nn as nn
import torch

from stud.constants import PADDING_INDEX

def compute_scatter_mask(
    input_ids: torch.Tensor,
    add_special_tokens: bool,
    offset: int = 0
) -> torch.Tensor:
    '''
    Efficient computation of the scatter mask of `input_ids`.

    :param input_ids: 2D tensor containing for each row (identifying a token), the number of wordpieces it has been split into.
    :param add_special_tokens: flag indicating whether to add leading/trailing symbols (e.g. as required by BERT-like models)
    :param offset: the index to start with (e.g. may be useful if using padding)
    :return 1D torch.Tensor with values indicating to which token they belong to (e.g. [1,2,2,3,4,5,5,6])
    '''
    l_tokens, r_tokens = (1,1) if add_special_tokens else (0,0)
    length = len(input_ids) + l_tokens + r_tokens
    repeats = torch.ones(length, dtype=torch.int64)
    repeats[l_tokens:length-r_tokens] = torch.count_nonzero(input_ids, dim=-1)
    mask = torch.repeat_interleave(torch.arange(offset, len(repeats) + offset), repeats)
    return mask

def batch_scatter_mean(
    src: torch.Tensor, 
    labels: torch.Tensor,
    remove_special_tokens: bool = True,
    return_tensors: bool = True
) -> List[torch.Tensor]:
    ''' Batch-version of scatter_mean function. '''
    if not remove_special_tokens:
        batch_out = [scatter_mean(seq, mask) for seq, mask in zip(src, labels)]
    else:
        batch_out = [scatter_mean(seq, mask)[1:-1] for seq, mask in zip(src, labels)]
    if return_tensors:
        batch_out = pad_sequence(batch_out, batch_first=True, padding_value=PADDING_INDEX)
    return batch_out

def scatter_mean(src: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    '''
    Scatter mean implementation in PyTorch taken from:
    https://discuss.pytorch.org/t/groupby-aggregate-mean-in-pytorch/45335/2
    '''
    labels = labels.view(labels.size(0), 1).expand(-1, src.size(1))
    unique_labels, labels_count = labels.unique(dim=0, return_counts=True)
    res = torch.zeros_like(unique_labels, dtype=torch.float).scatter_add_(0, labels, src)
    res = res / labels_count.float().unsqueeze(1)
    return res

def gpus(device: str = 'cpu', use_cuda: bool = False) -> int:
    ''' Utility to determine wheter to use GPU/CPU. '''
    use_cuda = use_cuda or device in ['cuda', 'gpu']
    return 1 if use_cuda and torch.cuda.is_available() else 0

def get_device(model: nn.Module, gpu_device: str = 'cuda:0') -> str:
    ''' Utility function used to determine the device the input model is running on. '''
    return gpu_device if next(model.parameters()).is_cuda else 'cpu'

def arg_where(x: torch.Tensor, mask: torch.Tensor, return_tensors: bool = False) -> Union[torch.Tensor, List]:
    ''' Simple implementation of missing arg_where in Pytorch. '''
    indices, *_ = mask.nonzero(as_tuple=True)
    if not return_tensors:
        indices = indices.tolist()
    return indices

def arg_where_equals(x: torch.Tensor, v: Any) -> Union[torch.Tensor, List]:
    ''' Useful overload of arg_where function. '''
    return arg_where(x, x==v)