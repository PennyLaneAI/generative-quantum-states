import torch.nn as nn
from torch_geometric.nn import GCNConv  # noqa

__all__ = ['GCNProjEncoder']


class GCNProjEncoder(nn.Module):
    def __init__(self, n_nodes, in_node_dim, out_node_dim, n_layers, transformer_dim):
        super(GCNProjEncoder, self).__init__()
        assert n_layers > 0
        channel_nums = [in_node_dim]
        channel_nums = channel_nums + [2 ** (n_layers - 1 - i) * out_node_dim for i in range(n_layers - 1)]
        channel_nums = channel_nums + [out_node_dim]

        self._n_nodes = n_nodes
        self._out_channels = out_node_dim
        self._transformer_dim = transformer_dim

        # graph convolutional layers
        self.gcn_layers = []

        for c_in, c_out in zip(channel_nums[:-1], channel_nums[1:]):
            self.gcn_layers.append(GCNConv(in_channels=c_in, out_channels=c_out))

        # projection
        self.out_proj = nn.Linear(in_features=n_nodes * out_node_dim, out_features=transformer_dim)

        self.layers = nn.Sequential(*self.gcn_layers, self.out_proj)

    def forward(self, data):
        x, edge_index, edge_weight = data.x, data.edge_index, data.edge_weight

        for layer in self.layers[:-1]:
            x = layer(x, edge_index, edge_weight).relu()

        graph_embed = x.view(-1, self._out_channels * self._n_nodes)
        graph_proj = self.layers[-1](graph_embed)

        # reshape into B x N x d_model
        graph_proj = graph_proj.view(data.num_graphs, -1, self._transformer_dim)

        return graph_proj
