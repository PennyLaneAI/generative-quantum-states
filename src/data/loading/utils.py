import numpy as np
import os

from src.models.transformer.utils import make_std_mask
from src.utils import measurement2multihot


def load_data(path, fname, sample_size=-1, seed=0, return_meas=True):
    all_data = np.load(os.path.join(path, fname))
    if not return_meas:
        all_data = measurement2multihot(all_data, )
    if sample_size > 0:
        np.random.seed(seed)
        idxes = np.random.choice(len(all_data), sample_size, replace=False) if sample_size > 0 else np.arange(
            len(all_data))
        sampled_data = all_data[idxes]
        return sampled_data
    else:
        return all_data


class TgtBatch:
    """ object for holding a batch of data with mask """
    def __init__(self, tgt, pad):
        self.tgt = tgt[:, :-1]
        self.tgt_y = tgt[:, 1:]
        self.tgt_mask = make_std_mask(self.tgt, pad)
        self.ntokens = (self.tgt_y != pad).data.sum()  # noqa