import torch.nn as nn

from src.models.transformer.utils import clones
from src.models.transformer.modules import SublayerConnection


class DecoderLayer(nn.Module):
    def __init__(self, size, self_attn, feed_forward, dropout):
        super(DecoderLayer, self).__init__()
        self.size = size
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 3)

    def forward(self, x, tgt_mask):
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, tgt_mask))
        x = self.sublayer[1](x, lambda x: self.self_attn(x, x, x, tgt_mask))
        return self.sublayer[2](x, self.feed_forward)
