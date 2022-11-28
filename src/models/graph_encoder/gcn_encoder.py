import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv  # noqa

__all__ = ['GCNEncoder']


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
            layers.append(GCNConv(in_channels=c_in, out_channels=c_out))

        self.layers = nn.Sequential(*layers)

    def forward(self, data):
        x, edge_index, edge_weight = data.x, data.edge_index, data.edge_weight

        for layer in self.layers[:-1]:
            x = layer(x, edge_index, edge_weight).relu()

        graph_embed = self.layers[-1](x, edge_index, edge_weight)

        # reshape into B x N x d_model
        graph_embed = graph_embed.view(data.num_graphs, -1, self._out_channels)

        graph_embed = torch.concat([
            graph_embed, torch.zeros(size=(graph_embed.shape[0], 1, graph_embed.shape[-1]), device=graph_embed.device)
        ], dim=1)

        return graph_embed
