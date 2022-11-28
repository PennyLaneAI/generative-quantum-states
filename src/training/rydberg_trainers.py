import copy
import numpy as np
import time
import torch
import torch.backends.cudnn as cudnn
from tqdm.auto import tqdm

from constants import *
from src.models.transformer import ConditionalTransformer, LabelSmoothing
from src.training.utils import AverageMeter, warmup_sqrt_decay_lr_scheduler, warm_up_cosine_lr_scheduler
from src.data.loading.utils import TgtBatch

__all__ = ['RydbergConditionalTransformerTrainer']


def batch_generator_from_cond_datasets(datasets: dict, batch_size: int, num_iterations: int, token_shift: int,
                                       pad_token: int, start_token: int,
                                       device: torch.device, ):
    """ function iterates over dataset, each time sampling a random batch of size batch_size """
    keys = list(datasets.keys())

    for step in range(1, num_iterations + 1):
        key = keys[np.random.choice(len(keys))]
        dataset = datasets[key]
        if batch_size > 0:
            indices = np.random.choice(len(dataset), batch_size, replace=True)
            batch_data = dataset[indices]
        else:
            batch_data = dataset
        batch_size = len(batch_data)
        batch_data = np.asarray(batch_data, dtype=np.int)
        batch_data += token_shift
        batch_data = np.concatenate(
            [np.ones((batch_size, 1)) * start_token, batch_data, np.ones((batch_size, 1)) * pad_token], axis=1)
        batch_data = torch.from_numpy(batch_data).long().to(device)
        tgt_batch = TgtBatch(batch_data, pad=pad_token)
        condition = torch.from_numpy(np.array([list(key)])).float().to(device)
        yield tgt_batch, condition


class RydbergConditionalTransformerTrainer():

    def __init__(
            self,
            model: ConditionalTransformer,
            train_dataset: dict,
            iterations: int,
            lr: float,
            final_lr: float,
            lr_scheduler: str,
            warmup_frac: float,
            weight_decay: float,
            batch_size: int,
            rng: np.random.Generator,
            # tensorboard_dir: str = None,
            smoothing: float = 0.0,
            eval_every: int = 1000,
            epoch_mode: bool = False,
            transfomer_version: int = 0,
            test_dataset: dict = None,
            # save_dir: str = None,
            device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
    ):
        self.train_dataset = train_dataset
        self.rng = rng

        self.lr = lr
        self.warmup_frac = warmup_frac
        self.model = model.to(device)
        self.iterations = iterations
        self.eval_every = eval_every
        self.num_train_data = len(self.train_dataset)
        self.final_lr = final_lr
        self.lr_scheduler_name = lr_scheduler
        self.weight_decay = weight_decay
        self.epoch_mode = epoch_mode
        self.transformer_version = transfomer_version
        self.batch_size = batch_size
        self.test_dataset = test_dataset
        self.device = device
        if epoch_mode:
            self.total_steps = int(1 + len(train_dataset) / batch_size) * iterations
        else:
            self.total_steps = iterations

        # loss function
        self.criterion = LabelSmoothing(
            size=self.model.n_outcomes + model.token_shift, padding_idx=0, smoothing=smoothing
        )

        # init optimizers and meters
        self._initialize_optimizers()
        self._initialize_meters()

    def set_device(self, device: torch.device):
        self.device = device
        self.model.to(device)

    def train(self):
        if self.device.type[:4] == 'cuda':
            cudnn.benchmark = True
        # run training
        if self.epoch_mode:
            return self._train_loop_epochs(1, self.iterations + 1)
        else:
            return self._train_loop_iterations()

    def _train_loop_epochs(self, start_epoch, end_epoch):
        raise NotImplementedError

    def save_model(self, epoch, is_best=False):
        model_fp = self._save_model(
            ckpt={
                'epoch': epoch,
                'model_state_dict': copy.deepcopy(self.model.state_dict()),
                'optimizer': copy.deepcopy(self.optimizer.state_dict()),
            },
            ckpt_fn='checkpoint.pth.tar' if not is_best else 'model_best_pth.tar'
        )
        return model_fp

    def _train_loop_iterations(self):

        # progress bar
        pbar = tqdm(
            batch_generator_from_cond_datasets(self.train_dataset, batch_size=self.batch_size,
                                               num_iterations=self.iterations,
                                               token_shift=self.model.token_shift,
                                               start_token=self.model.start_token,
                                               pad_token=self.model.pad_token, device=self.device),
            total=self.iterations, desc='Step'
        )

        # iterate through data
        for step, (tgt_batch, condition) in enumerate(pbar):
            self.model.train()
            # train step
            t0 = time.time()
            self._forward_step(
                tgt=tgt_batch.tgt,
                tgt_y=tgt_batch.tgt_y,
                tgt_mask=tgt_batch.tgt_mask,
                norm=tgt_batch.ntokens.item(),
                cond_var=condition,
                is_train=True,
                update_meter=True
            )
            step_time = time.time() - t0

            # decay learning rate after every step
            self.scheduler.step()
            if self.eval_every > 0 and step % self.eval_every == 0:
                avg_loss = self.total_loss_meter.average()
                self._reset_meters()
                pbar.set_postfix(loss=avg_loss)

        return self.model

    def _forward_step(self, tgt, tgt_y, tgt_mask, norm, cond_var, is_train, update_meter=True):
        out = self.model.forward(tgt, tgt_mask, cond_var=cond_var)
        log_probs = self.model.generator(out)

        if is_train:
            loss = self.criterion(log_probs.contiguous().view(-1, log_probs.size(-1)),
                                  tgt_y.contiguous().view(-1)) / norm
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        else:
            with torch.no_grad():
                loss = self.criterion(log_probs.contiguous().view(-1, log_probs.size(-1)),
                                      tgt_y.contiguous().view(-1)) / norm

        # update meters
        loss = loss.item()

        if update_meter:
            self.total_loss_meter.update(val=loss, n=tgt.shape[0])

        return loss

    def get_progress_str(self, train_loss, test_loss, step):
        pstr = f'step {step} | lr: {round(self.scheduler.get_last_lr()[0], 8)}, train loss: {train_loss:.4f}'
        pstr += f', test loss: {test_loss:.4f}'
        return pstr

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
        self.total_variation_dist_meter = AverageMeter()

    def _reset_meters(self):
        self.total_loss_meter.reset()
        self.total_variation_dist_meter.reset()