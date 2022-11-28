from copy import deepcopy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F  # noqa
from tqdm.auto import tqdm, trange

from src.models.transformer.layers import DecoderLayer
from src.models.transformer.modules import MultiHeadAttention, PositionwiseFeedForward, PositionalEncoding, Embeddings
from src.models.transformer.models import Decoder, Generator
from src.models.transformer.utils import make_std_mask
from src.models.transformer.utils import TOKEN_SHIFT_V0, TOKEN_SHIFT_V1, PAD_TOKEN_V0, PAD_TOKEN_V1, START_TOKEN_V0, \
    START_TOKEN_V1
from torch import LongTensor, FloatTensor


def init_conditional_transformer(n_outcomes, encoder, n_layers=6, d_model=512, d_ff=2048,
                                 n_heads=8, dropout=0.1, version=0, use_prompt=False):
    """ helper function which constructs a model from hyperparameters"""
    # Prompt: replace the first token with an embedding provided by the conditional variable
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
    pos_enc = deepcopy(position)
    model = ConditionalTransformer(
        encoder=encoder,
        decoder=Decoder(DecoderLayer(d_model, deepcopy(attn), deepcopy(ff), dropout), n_layers=n_layers),
        tgt_embed=nn.Sequential(Embeddings(d_model, vocab_size), pos_enc),
        generator=Generator(d_model, vocab_size),
        n_outcomes=n_outcomes,
        version=version,
        use_prompt=use_prompt,
        pos_enc=pos_enc,
    )

    # init weights
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)

    return model


class ConditionalTransformer(nn.Module):
    def __init__(self, encoder, decoder, tgt_embed, generator, n_outcomes, version=0,
                 use_prompt=True, pos_enc=None, device=torch.device('cpu')):
        super(ConditionalTransformer, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.tgt_embed = tgt_embed
        self.generator = generator
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
        # Prompt: replace the first token with an embedding provided by the conditional variable
        self.use_prompt = use_prompt
        self.pos_enc = pos_enc
        if use_prompt: assert pos_enc is not None
        self.device = device

    def to(self, device):
        self.device = device
        return super().to(device)

    def forward(self, tgt, tgt_mask, cond_var):
        """ process masked target sequences """
        tgt_embed = self.tgt_embed(tgt)

        # map conditional variable to model dimension
        cond_embed = self.encoder(cond_var).unsqueeze(1)
        if self.use_prompt:
            # we assume the first token is the token_start,
            # and we replace its embedding with the conditional variable embedding
            x = tgt_embed
            x[:, 0, :] = self.pos_enc(cond_embed).squeeze(1)
        else:
            # add to embedded tgt sequence
            x = tgt_embed + cond_embed

        return self.decoder(x, tgt_mask)

    def probs(self, tgt, cond_var):
        tgt0 = tgt[:, :-1]
        tgt1 = tgt[:, 1:]

        tgt_mask = make_std_mask(tgt0, self.pad_token)
        out = self.forward(tgt0, tgt_mask, cond_var)
        log_probs = self.generator(out)
        conditional_probs = torch.exp(log_probs).cpu()
        tgt_one_hot = F.one_hot(tgt1)

        probs = (conditional_probs * tgt_one_hot).sum(dim=-1)
        probs = torch.prod(probs, dim=-1)
        probs = probs / probs.sum()

        return probs

    @torch.no_grad()
    def sample_batch(self, batch_size, cond_var, num_qubits):
        self.eval()
        cond_var = cond_var.to(self.device)
        num_valid_samples = 0
        valid_samples = None

        while num_valid_samples < batch_size:
            tgt = torch.ones(batch_size - num_valid_samples, 1).type(LongTensor) * self.start_token
            tgt = tgt.to(self.device)

            for i in range(num_qubits):
                mask = make_std_mask(tgt, pad=self.pad_token)

                out = self.forward(tgt, mask, cond_var)
                log_probs = self.generator(out[:, -1])
                probs = torch.exp(log_probs).detach()
                probs = probs / probs.sum(axis=-1).reshape(batch_size - num_valid_samples, 1)

                # draw samples from model p(a_n | a_{n-1}, ..., a_{0})
                next_outcomes = torch.multinomial(probs, 1, replacement=True)
                # next_outcomes = dist.Categorical(probs).sample()
                tgt = torch.cat([tgt, next_outcomes], dim=1)

            # remove begin token and map back to sample space
            tgt = tgt.cpu().numpy()
            tgt = tgt[:, 1:] - self.token_shift

            # discard invalid samples (i.e., the ones predicting start/end/pad tokens at wrong positions)
            tgt = tgt[np.all(tgt >= 0, axis=1), :]

            if valid_samples is None:
                valid_samples = tgt
            else:
                valid_samples = np.concatenate([valid_samples, tgt], axis=0)

            num_valid_samples = valid_samples.shape[0]

        return valid_samples

    def sample(self, cond_var, num_samples, num_qubits, batch_size=1000, print_progress=True) -> np.ndarray:
        batch_sizes = [batch_size] * int(num_samples // batch_size)
        batch_sizes = batch_sizes + [] if num_samples % batch_size == 0 else [num_samples % batch_size]

        if print_progress:
            pbar = tqdm(batch_sizes, postfix=f'generating {num_samples} samples from model...')
        else:
            pbar = batch_sizes

        batch_samples = [self.sample_batch(bs, cond_var, num_qubits) for bs in pbar]
        samples = np.concatenate(batch_samples, axis=0)
        return samples
