from copy import deepcopy

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F  # noqa
from tqdm import tqdm

from src.models.gctransformer.layers import DecoderLayer
from src.models.gctransformer.models import Decoder, Generator
from src.models.gctransformer.utils import make_std_mask
from src.models.gctransformer.modules import (
    MultiHeadAttention,
    PositionwiseFeedForward,
    PositionalEncoding,
    Embeddings
)

LongTensor = torch.cuda.LongTensor if torch.cuda.is_available() else torch.LongTensor
FloatTensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor


class GCTransformer(nn.Module):
    def __init__(
            self, encoder, decoder, tgt_embed, generator, n_outcomes, d_model,
            pad_token, start_token, end_token, token_shift
    ):
        super(GCTransformer, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.tgt_embed = tgt_embed
        self.generator = generator
        self.n_outcomes = n_outcomes
        self.d_model = d_model

        self.pad_token = pad_token
        self.start_token = start_token
        self.end_token = end_token
        self.token_shift = token_shift

    def forward(self, tgt, tgt_mask, coupling_graph):
        """ process masked target sequences """
        tgt_embed = self.tgt_embed(tgt)

        # map coupling graph to model dimension and add to embedded tgt sequence
        graph_embed = self.encoder(coupling_graph)
        graph_embed = graph_embed[:, :tgt_embed.shape[1], :]
        tgt_embed = tgt_embed + graph_embed

        return graph_embed, self.decoder(tgt_embed, tgt_mask)

    def sample_batch(self, batch_size, coupling_graph, qubits):
        num_valid_samples = 0
        valid_samples = None

        num_restarts = 0

        while num_valid_samples < batch_size:
            tgt = torch.ones(batch_size - num_valid_samples, 1).type(
                LongTensor) * self.start_token

            for i in range(qubits):
                mask = make_std_mask(tgt, pad=self.pad_token)

                _, out = self.forward(tgt, mask, coupling_graph)
                log_probs = self.generator(out[:, -1])
                probs = torch.exp(log_probs).detach()
                probs = probs / probs.sum(axis=-1).reshape(
                    batch_size - num_valid_samples, 1)

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

            num_restarts += 1
            if num_restarts > 10:
                print(
                    f'sampling timed out: got {num_valid_samples} instead of {batch_size}')
                break

        return valid_samples

    def sample(self, samples, coupling_graph, qubits, batch_size=1000,
               print_progress=True) -> np.ndarray:
        batch_sizes = [batch_size] * int(samples // batch_size)
        batch_sizes = batch_sizes + (
            [] if samples % batch_size == 0 else [samples % batch_size])

        if print_progress:
            pbar = tqdm(batch_sizes,
                        postfix=f'generating {samples} samples from model...')
        else:
            pbar = batch_sizes

        batch_samples = [self.sample_batch(bs, coupling_graph, qubits) for bs in pbar]
        samples = np.concatenate(batch_samples, axis=0)
        return samples


def init_gctransformer(n_outcomes, encoder, n_layers, d_model, d_ff, n_heads, dropout,
                       pad_token, start_token,
                       end_token, token_shift) -> GCTransformer:
    """ helper function which constructs a model from hyperparameters"""
    attn = MultiHeadAttention(n_heads, d_model, dropout)
    ff = PositionwiseFeedForward(d_model, d_ff, dropout)
    position = PositionalEncoding(d_model, dropout)

    vocab_size = n_outcomes + token_shift

    model = GCTransformer(
        encoder=encoder,
        decoder=Decoder(DecoderLayer(d_model, deepcopy(attn), deepcopy(ff), dropout),
                        n_layers=n_layers),
        tgt_embed=nn.Sequential(Embeddings(d_model, vocab_size), deepcopy(position)),
        generator=Generator(d_model, vocab_size),
        n_outcomes=n_outcomes,
        d_model=d_model,
        pad_token=pad_token,
        start_token=start_token,
        end_token=end_token,
        token_shift=token_shift
    )

    # init weights
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)

    return model
