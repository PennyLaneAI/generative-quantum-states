from copy import deepcopy

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from src.models.transformer.layers import DecoderLayer
from src.models.transformer.modules import MultiHeadAttention, PositionwiseFeedForward, PositionalEncoding, Embeddings
from src.models.transformer.models import Decoder, Generator
from src.models.transformer.utils import subsequent_mask, make_std_mask
from src.models.transformer.utils import TOKEN_SHIFT_V0, TOKEN_SHIFT_V1, PAD_TOKEN_V0, PAD_TOKEN_V1, START_TOKEN_V0, \
    START_TOKEN_V1
from torch import LongTensor, FloatTensor


def init_transformer(n_qubits, n_outcomes, n_layers=6, d_model=512, d_ff=2048, n_heads=8, dropout=0.1, version=0):
    """ helper function which constructs a model from hyperparameters"""
    attn = MultiHeadAttention(n_heads, d_model, dropout)
    ff = PositionwiseFeedForward(d_model, d_ff, dropout)
    position = PositionalEncoding(d_model, dropout)

    if version == 0:
        token_shift = TOKEN_SHIFT_V0
    elif version == 1:
        token_shift = TOKEN_SHIFT_V1
    else:
        raise ValueError

    vocab_size = n_outcomes + token_shift

    model = Transformer(
        decoder=Decoder(DecoderLayer(d_model, deepcopy(attn), deepcopy(ff), dropout), n_layers=n_layers),
        tgt_embed=nn.Sequential(Embeddings(d_model, vocab_size), deepcopy(position)),
        generator=Generator(d_model, vocab_size),
        n_qubits=n_qubits,
        n_outcomes=n_outcomes,
        version=version
    )

    # init weights
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)

    return model


class Transformer(nn.Module):
    def __init__(self, decoder, tgt_embed, generator, n_qubits, n_outcomes, version=0, device=torch.device('cpu')):
        super(Transformer, self).__init__()
        self.decoder = decoder
        self.tgt_embed = tgt_embed
        self.generator = generator
        self.n_qubits = n_qubits
        self.n_outcomes = n_outcomes

        if version == 0:
            self.pad_token = PAD_TOKEN_V0
            self.start_token = START_TOKEN_V0
            self.token_shift = TOKEN_SHIFT_V0
        elif version == 1:
            self.pad_token = PAD_TOKEN_V1
            self.start_token = START_TOKEN_V1
            self.token_shift = TOKEN_SHIFT_V1
        else:
            raise ValueError
        self.device = device

    def to(self, device):
        self.device = device
        return super().to(device)

    def forward(self, tgt, tgt_mask):
        """ process masked target sequences """
        return self.decoder(self.tgt_embed(tgt), tgt_mask)

    def probs(self, tgt):
        tgt0 = tgt[:, :-1]
        tgt1 = tgt[:, 1:]

        tgt_mask = make_std_mask(tgt0, self.pad_token)
        out = self.forward(tgt0, tgt_mask)
        log_probs = self.generator(out)
        conditional_probs = torch.exp(log_probs).cpu()
        tgt_one_hot = F.one_hot(tgt1)

        probs = (conditional_probs * tgt_one_hot).sum(dim=-1)
        probs = torch.prod(probs, dim=-1)
        probs = probs / probs.sum()

        return probs

    def sample_batch(self, batch_size):
        num_valid_samples = 0
        valid_samples = None

        while num_valid_samples < batch_size:
            tgt = torch.ones(batch_size - num_valid_samples, 1).type(LongTensor) * self.start_token

            for i in range(self.n_qubits):
                mask = make_std_mask(tgt, pad=self.pad_token)
                out = self.forward(tgt, mask)
                log_probs = self.generator(out[:, -1])
                probs = torch.exp(log_probs).detach()
                probs = probs / probs.sum(axis=-1).reshape(batch_size - num_valid_samples, 1)

                # draw samples from model p(a_n | a_{n-1}, ..., a_{0})
                next_outcomes = torch.multinomial(probs, 1, replacement=True)

                tgt = torch.cat([tgt, next_outcomes], dim=1)

            # remove begin token and map back to sample space
            tgt = tgt.cpu().numpy()
            tgt = tgt[:, 1:] - self.token_shift

            # discard invalid samples
            tgt = tgt[np.all(tgt >= 0, axis=1), :]

            if valid_samples is None:
                valid_samples = tgt
            else:
                valid_samples = np.concatenate([valid_samples, tgt], axis=0)

            num_valid_samples = valid_samples.shape[0]

        return valid_samples

    def sample(self, num_samples, batch_size=500) -> np.ndarray:
        batch_sizes = [batch_size] * int(num_samples // batch_size)
        batch_sizes = batch_sizes + [] if num_samples % batch_size == 0 else [num_samples % batch_size]
        batch_samples = [self.sample_batch(bs) for bs in tqdm(batch_sizes, postfix='generating samples from model...')]
        samples = np.concatenate(batch_samples, axis=0)
        return samples

    # def probs2(self, tgt):
    #     """ this function computes the model probabilities for outcomes tgt which are assumed to be in the shape
    #     N x (nq + 3) with values [[<start>, q_1 + 3, ..., q_n + 3, <end>], ...]
    #     """
    #     num_samples = tgt.shape[0]
    #     tgt_mask = make_std_mask(tgt, self.pad_token)
    #     out = self.forward(tgt, tgt_mask)
    #     log_probs = self.generator(out)
    #     conditional_probs = torch.exp(log_probs).cpu()
    #     probs = torch.ones(tgt.shape[0]).type(torch.FloatTensor)
    #
    #     for nq in range(self.n_qubits):
    #         idx = torch.arange(0, num_samples).type(torch.LongTensor)
    #         cp = conditional_probs[idx, nq, tgt[:, nq + 1]]
    #         probs = probs * cp
    #
    #     probs = probs / torch.sum(probs)
    #
    #     return probs

#
#
# def inference_test():
#     test_model = make_model(4, 9, 1)
#     test_model.eval()
#     tgt = torch.ones(2, 1).type(torch.LongTensor)
#
#     for i in range(4):
#         mask = subsequent_mask(tgt.size(1)).type_as(tgt.data)
#         out = test_model(tgt, mask)
#         log_probs = test_model.generator(out[:, -1])
#         _, next_word = torch.max(log_probs, dim=1)
#         next_word = next_word.data[0]
#         tgt = torch.cat([tgt, torch.empty(2, 1).type_as(tgt.data).fill_(next_word)], dim=1)
#
#     print("Example untrained model prediction:", tgt)
