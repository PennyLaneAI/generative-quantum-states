import copy

import torch
import torch.nn as nn

PAD_TOKEN_V0 = 0
START_TOKEN_V0 = 1
END_TOKEN_V0 = 2
TOKEN_SHIFT_V0 = 3

PAD_TOKEN_V1 = -1
START_TOKEN_V1 = 0
END_TOKEN_V1 = 1
TOKEN_SHIFT_V1 = 2

def clones(module: nn.Module, n_clones: int):
    """ helper function which produces n_clones copies of a layer """
    return nn.ModuleList([copy.deepcopy(module) for _ in range(n_clones)])


def subsequent_mask(size):
    """ mask out subsequent positions """
    attn_shape = (1, size, size)
    mask = torch.triu(torch.ones(attn_shape), diagonal=1).type(torch.uint8)
    return mask == 0


def make_std_mask(tgt, pad):
    """ Create a mask to hide padding and future words. """
    tgt_mask = (tgt != pad).unsqueeze(-2)  # noqa
    tgt_mask = tgt_mask & subsequent_mask(tgt.size(-1)).type_as(tgt_mask.data)
    return tgt_mask
