from abc import ABC, abstractmethod

import numpy as np
import os
import torch
from torch.utils.tensorboard import SummaryWriter
from typing import Dict, Any, Union

from src.data.loading import DatasetGCTransformer


class BaseTrainer(ABC):
    use_cuda = torch.cuda.is_available()
    FloatTensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor  # noqa
    LongTensor = torch.cuda.LongTensor if use_cuda else torch.LongTensor  # noqa

    def __init__(
            self,
            train_dataset: Union[DatasetGCTransformer],
            val_dataset: Union[None, DatasetGCTransformer],
            save_dir: str,
            iterations: int,
            lr: float,
            batch_size: int,
            rng: np.random.Generator,
            tensorboard_dir: str,
    ):
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.save_dir = save_dir
        self.iterations = iterations
        self.lr = lr
        self.rng = rng
        self.batch_size = batch_size
        self.tensorboard_dir = tensorboard_dir
        self.checkpoint_dir = os.path.join(save_dir, 'checkpoints/')

        # tensorboard
        tensorboard_dir = tensorboard_dir if tensorboard_dir is not None else 'tensorboard'
        self.summary_writer = SummaryWriter(os.path.join(save_dir, tensorboard_dir))

    def _save_model(self, ckpt: Dict[str, Any], ckpt_fn: str = None) -> str:
        """ basic method to save checkpoints stored as a dict in the ckpt variable """
        if ckpt_fn is None:
            ckpt_fn = 'checkpoint.pth.tar'

        save_as = os.path.join(self.checkpoint_dir, ckpt_fn)
        torch.save(ckpt, save_as)
        return save_as

    @abstractmethod
    def _train_loop_epochs(self, start_epoch, end_epoch):
        """ implement training loop in epoch mode """
        pass

    @abstractmethod
    def train(self, *args, **kwargs):
        """ implement this to train the model(s) by calling _train_loop """
        pass

    @abstractmethod
    def _initialize_optimizers(self): pass

    @abstractmethod
    def _initialize_meters(self): pass

    @abstractmethod
    def _reset_meters(self): pass
