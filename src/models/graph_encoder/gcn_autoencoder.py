from typing import Tuple

import torch
import torch.nn as nn
import torch_geometric.nn as gnn
from torch_geometric.utils import negative_sampling

__all__ = ['GCNAutoencoder']

EPS = 1e-15


class GCNEncoder(nn.Module):
    def __init__(self, in_node_dim, out_node_dim, n_layers):
        super(GCNEncoder, self).__init__()
        assert n_layers > 0
        channel_nums = [in_node_dim]
        channel_nums = channel_nums + [2 ** (n_layers - 1 - i) * out_node_dim for i in range(n_layers - 1)]
        channel_nums = channel_nums + [out_node_dim]

        self._out_channels = out_node_dim

        # graph convolutional layers
        layers = []

        for c_in, c_out in zip(channel_nums[:-1], channel_nums[1:]):
            layers.append(gnn.GCNConv(in_channels=c_in, out_channels=c_out))

        self.layers = nn.Sequential(*layers)

    def forward(self, data):
        x, edge_index, edge_weight = data.x, data.edge_index, data.edge_weight

        for layer in self.layers[:-1]:
            x = layer(x, edge_index, edge_weight).relu()

        return self.layers[-1](x, edge_index, edge_weight)


class GCNAutoencoder(nn.Module):
    def __init__(self, in_node_dim, out_node_dim, n_layers):
        super(GCNAutoencoder, self).__init__()
        self.gae = gnn.GAE(
            encoder=GCNEncoder(in_node_dim, out_node_dim, n_layers)
        )

    def forward(self, data) -> Tuple[torch.Tensor, torch.Tensor]:
        # encode graph
        z = self.gae.encoder(data)
        return z

    def recon_loss(self, z, pos_edge_index, true_weights):
        pred_weights_pos = self.gae.decoder(z, pos_edge_index, sigmoid=True)
        loss_vec = -(torch.log(pred_weights_pos + EPS) * true_weights + (1 - true_weights) * torch.log(
            1 - pred_weights_pos + EPS))
        pos_loss = torch.mean(loss_vec)

        neg_edge_index = negative_sampling(pos_edge_index, z.size(0))
        neg_loss = -torch.log(1 - self.gae.decoder(z, neg_edge_index, sigmoid=True) + EPS).mean()

        return pos_loss + neg_loss

    # def l2_loss(self, z, pos_edge_index, true_weights):
    #     pred_weights_pos = self.gae.decoder(z, pos_edge_index, sigmoid=True)
    #     pos_loss_vec = torch.mean((pred_weights_pos - true_weights) ** 2)
    #
    #     neg_edge_index = negative_sampling(pos_edge_index, z.size(0))
    #     pred_weights_neg = self.gae.decoder(z, neg_edge_index, sigmoid=True)
    #     neg_loss_vec = torch.mean(pred_weights_neg ** 2)
    #
    #     return 0.5 * (neg_loss_vec + pos_loss_vec)
