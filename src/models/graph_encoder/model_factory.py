from src.models.graph_encoder import GCNProjEncoder, GCNEncoder, GCNAutoencoder


def get_graph_encoder(arch: str, in_node_dim, gcn_dim, gcn_layers, d_model, qubits=None):
    if arch.startswith('gcn_proj'):
        # gcn wihout global aggregation
        return GCNProjEncoder(
            qubits, in_node_dim=in_node_dim, out_node_dim=gcn_dim, n_layers=gcn_layers, transformer_dim=d_model
        )

    if arch.startswith('gcn_encoder'):
        assert gcn_dim == d_model
        # gcn using cnn for aggregation followed by global average pooling
        return GCNEncoder(
            in_node_dim=in_node_dim, out_node_dim=gcn_dim, n_layers=gcn_layers
        )

    if arch.startswith('gcn_autoencoder'):
        assert gcn_dim == d_model
        # gcn using cnn for aggregation followed by global average pooling
        return GCNAutoencoder(
            in_node_dim=in_node_dim, out_node_dim=gcn_dim, n_layers=gcn_layers
        )

    raise ValueError(f'unknown graph encoder {arch}')
