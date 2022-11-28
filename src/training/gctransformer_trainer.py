import copy
import numpy as np
import time
import torch
import torch.backends.cudnn as cudnn
from tqdm import tqdm
from typing import Union, List

from constants import *
from src.data.loading import DatasetGCTransformer
from src.data.loading.dataloading import batch_generator_epoch_mixed, batch_generator_iterations_mixed
from src.data.loading.dataloading import batch_generator_epoch, batch_generator_iterations
from src.models.gctransformer import GCTransformer, KLDivLoss
from src.training import BaseTrainer
from src.training.utils import AverageMeter, warmup_sqrt_decay_lr_scheduler, warm_up_cosine_lr_scheduler

__all__ = ['GCTransformerTrainer']


class GCTransformerTrainer(BaseTrainer):

    def __init__(
            self,
            model: GCTransformer,
            train_dataset: Union[DatasetGCTransformer, List[DatasetGCTransformer]],
            val_dataset: Union[DatasetGCTransformer, List[DatasetGCTransformer], None],
            test_dataset: Union[DatasetGCTransformer, List[DatasetGCTransformer]],
            save_dir: str,
            iterations: int,
            lr: float,
            final_lr: float,
            lr_scheduler: str,
            warmup_frac: float,
            weight_decay: float,
            batch_size: int,
            rng: np.random.Generator,
            device,
            tensorboard_dir: str = None,
            eval_every: int = 1,
            epoch_mode=True,
    ):
        super(GCTransformerTrainer, self).__init__(
            train_dataset, val_dataset, save_dir, iterations, lr, batch_size, rng, tensorboard_dir
        )

        self.model = model
        self.eval_every = eval_every
        self.final_lr = final_lr
        self.lr_scheduler_name = lr_scheduler
        self.warmup_frac = warmup_frac
        self.weight_decay = weight_decay
        self.epoch_mode = epoch_mode
        self.device = device

        self.test_dataset = test_dataset

        if isinstance(self.train_dataset, List):
            self._batch_generator_epoch = batch_generator_epoch_mixed
            self._batch_generator_iters = batch_generator_iterations_mixed

            num_train_samples = sum([len(ds) for ds in train_dataset])
        else:
            self._batch_generator_epoch = batch_generator_epoch
            self._batch_generator_iters = batch_generator_iterations
            num_train_samples = len(train_dataset)

        self.total_steps = int(
            num_train_samples // batch_size + int(num_train_samples % batch_size > 0)
        ) * iterations

        # loss function
        self.criterion = KLDivLoss(padding_idx=model.pad_token)

        # init optimizers and meters
        self._initialize_optimizers()
        self._initialize_meters()

    def train(self):
        cudnn.benchmark = True if self.use_cuda else cudnn.benchmark

        if self.use_cuda:
            self.model = self.model.cuda()

        # run training
        if self.epoch_mode:
            return self._train_loop_epochs(1, self.iterations + 1)

        raise NotImplementedError

    def save_model(self, epoch, is_best=False):
        model_fp = self._save_model(
            ckpt={
                'epoch': epoch,
                'model_state_dict': copy.deepcopy(self.model.state_dict()),
                'optimizer': copy.deepcopy(self.optimizer.state_dict()),
            },
            ckpt_fn=f'checkpoint_{epoch}.pth.tar' if not is_best else 'model_best.pth.tar'
        )
        return model_fp

    def _train_loop_iterations(self):
        best_val_loss = np.inf
        self.model.train()

        # progress bar
        pbar = tqdm(
            enumerate(self._batch_generator_iters(
                self.train_dataset, batch_size=self.batch_size, pad_token=self.model.pad_token, rng=self.rng,
                iterations=self.iterations
            )), total=self.iterations, desc='Step'
        )

        # iterate through data
        for step, (tgt_batch, graphs_batch, *_) in pbar:
            # train step
            t0 = time.time()
            self._forward_step(
                tgt=tgt_batch.tgt,
                tgt_y=tgt_batch.tgt_y,
                tgt_mask=tgt_batch.tgt_mask,
                graphs_batch=graphs_batch,
                norm=tgt_batch.ntokens.item(),
                is_train=True,
                update_meter=True
            )
            step_time = time.time() - t0

            # decay learning rate after every step
            self.scheduler.step()

            # eval over dataset
            if step % self.eval_every == 0:
                train_loss = self._eval_epoch(dataset=self.train_dataset)
                val_loss = self._eval_epoch(self.val_dataset)
                test_loss = self._eval_epoch(self.test_dataset)

                if not np.isnan(val_loss) and val_loss < best_val_loss:
                    best_val_loss = val_loss
                    self.save_model(step, is_best=True)

                pstr = self.get_progress_str(train_loss, test_loss, val_loss, step=step)
                pbar.set_postfix_str(pstr)

                # add summaries to tensorboard
                self._collect_summaries(train_loss, test_loss, val_loss, step_time=step_time, step=step)

                self.model.train()

        val_loss = self._eval_epoch(self.val_dataset)
        train_loss = self._eval_epoch(self.train_dataset)
        test_loss = self._eval_epoch(self.test_dataset)

        return train_loss, test_loss, val_loss

    def _train_loop_epochs(self, start_epoch, end_epoch):
        best_val_loss = np.inf

        pbar = tqdm(range(start_epoch, end_epoch), desc='Epoch')

        for epoch in pbar:
            self.model.train()
            self._reset_meters()

            # train for one epoch
            t0 = time.time()
            for tgt_batch, graphs_batch, *_ in self._batch_generator_epoch(
                    self.train_dataset, batch_size=self.batch_size, pad_token=self.model.pad_token, rng=self.rng,
                    shuffle=True, device=self.device
            ):
                _ = self._forward_step(
                    tgt=tgt_batch.tgt,
                    tgt_y=tgt_batch.tgt_y,
                    tgt_mask=tgt_batch.tgt_mask,
                    graphs_batch=graphs_batch,
                    norm=tgt_batch.ntokens.item(),
                    is_train=True,
                    update_meter=True
                )

                # decay learning rate after every step
                self.scheduler.step()

            train_total_loss = self.total_loss_meter.average()

            # eval model
            if epoch % self.eval_every == 0:
                val_total_loss = self._eval_epoch(self.val_dataset)
                test_total_loss = self._eval_epoch(self.test_dataset)

                if not np.isnan(val_total_loss) and val_total_loss < best_val_loss:
                    best_val_loss = val_total_loss
                    self.save_model(epoch, is_best=True)

            else:
                test_total_loss, val_total_loss = np.nan, np.nan

            epoch_time = time.time() - t0

            pstr = self.get_progress_str(train_total_loss, test_total_loss, val_total_loss, step=epoch)
            pbar.set_postfix_str(pstr)

            # add summaries to tensorboard
            self._collect_summaries(train_total_loss, test_total_loss, val_total_loss, step=epoch, step_time=epoch_time)

        train_loss = self._eval_epoch(self.train_dataset)
        val_loss = self._eval_epoch(self.val_dataset)
        test_loss = self._eval_epoch(self.test_dataset)

        return train_loss, val_loss, test_loss

    def _eval_epoch(self, dataset):
        if dataset is None:
            return np.nan

        self.model.eval()
        self._reset_meters()

        for tgt_batch, graphs_batch, *_ in self._batch_generator_epoch(
                dataset, batch_size=self.batch_size, pad_token=self.model.pad_token, shuffle=False, rng=self.rng,
                device=self.device
        ):
            self._forward_step(
                tgt=tgt_batch.tgt,
                tgt_y=tgt_batch.tgt_y,
                tgt_mask=tgt_batch.tgt_mask,
                graphs_batch=graphs_batch,
                norm=tgt_batch.ntokens.item(),
                is_train=False,
                update_meter=True
            )

        # collect stats
        total_loss = self.total_loss_meter.average()

        return total_loss

    def _forward_step(self, tgt, tgt_y, tgt_mask, norm, graphs_batch, is_train, update_meter=True):
        graph_embed, out = self.model.forward(tgt, tgt_mask, coupling_graph=graphs_batch)
        log_probs = self.model.generator(out)

        if is_train:
            total_loss = self.criterion(
                log_probs.contiguous().view(-1, log_probs.size(-1)), tgt_y.contiguous().view(-1)
            ) / norm

            self.optimizer.zero_grad()
            total_loss.backward()
            self.optimizer.step()
        else:
            with torch.no_grad():
                total_loss = self.criterion(log_probs.contiguous().view(-1, log_probs.size(-1)),
                                            tgt_y.contiguous().view(-1)) / norm

        # update meters
        total_loss = total_loss.item()

        if update_meter:
            self.total_loss_meter.update(val=total_loss, n=tgt.shape[0])

        return total_loss

    def get_progress_str(self, train_total_loss, test_total_loss, val_total_loss, step):
        pstr = f'step {step} | lr: {self.scheduler.get_last_lr()[0]:.8f}, train total-loss: {train_total_loss:.4f}'

        if not np.isnan(test_total_loss):
            pstr = pstr + f', test total-loss: {test_total_loss:.4f}'
        if not np.isnan(val_total_loss):
            pstr = pstr + f'val total-loss: {val_total_loss:.4f}'

        return pstr

    def _collect_summaries(self, train_total_loss, test_total_loss, val_total_loss, step, step_time):
        self.summary_writer.add_scalars('total_loss/',
                                        {'train': train_total_loss, 'test': test_total_loss, 'val': val_total_loss},
                                        global_step=step)
        self.summary_writer.add_scalar('step_time/', step_time, global_step=step)
        self.summary_writer.add_scalar('learning_rate/', self.scheduler.get_last_lr()[0], global_step=step)
        self.summary_writer.flush()

    def _initialize_optimizers(self):
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=self.lr, betas=(0.9, 0.98), eps=1e-9, weight_decay=self.weight_decay)

        if self.lr_scheduler_name == COSINE_ANNEALING_WARM_RESTARTS_SCHEDULER:
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                self.optimizer, T_0=self.iterations // 5, T_mult=1
            )
        elif self.lr_scheduler_name == COSINE_ANNEALING_SCHEDULER:
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=self.total_steps, eta_min=self.final_lr
            )
        elif self.lr_scheduler_name == WARMUP_SQRT_SCHEDULER:
            warmup_steps = int(self.warmup_frac * self.total_steps)
            self.scheduler = warmup_sqrt_decay_lr_scheduler(
                optimizer=self.optimizer, model_size=self.model.model_size, factor=1, warmup=warmup_steps
            )
        elif self.lr_scheduler_name == WARMUP_COSINE_SCHEDULER:
            warmup_steps = int(self.warmup_frac * self.total_steps)
            self.scheduler = warm_up_cosine_lr_scheduler(
                optimizer=self.optimizer, epochs=self.total_steps, warm_up_epochs=warmup_steps, eta_min=self.final_lr
            )
        else:
            raise ValueError(f'unknown lr_scheduler {self.lr_scheduler_name}')

    def _initialize_meters(self):
        self.total_loss_meter = AverageMeter()

    def _reset_meters(self):
        self.total_loss_meter.reset()
