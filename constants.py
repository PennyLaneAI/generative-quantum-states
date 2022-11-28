import os as _os
import pennylane as _qml

ROOT_DIR = _os.path.dirname(_os.path.abspath(__file__))
DATA_DIR = _os.path.join(ROOT_DIR, 'data/')

# classical shadows
PAULI_ENSEMBLE = [_qml.PauliX, _qml.PauliY, _qml.PauliZ]
NUM_MMT_OUTCOMES = 6

# transformer architectures
TF_ARCHS = {
    'transformer_l4_d128_h4': dict(d_model=128, n_layers=4, n_head=4),
    'transformer_l8_d32_h8': dict(d_model=32, n_layers=8, n_head=8),
}

# gcn architectures
GCN_ARCHS = {
    'gcn_proj_3_16': dict(gcn_layers=3, gcn_dim=16, cnn_channels=0),
    'gcn_proj_3_128': dict(gcn_layers=3, gcn_dim=128, cnn_channels=0),
    'gcn_encoder_2_128': dict(gcn_layers=2, gcn_dim=128, cnn_channels=0),
    'gcn_encoder_3_32': dict(gcn_layers=3, gcn_dim=32, cnn_channels=0),
    'gcn_encoder_3_64': dict(gcn_layers=3, gcn_dim=64, cnn_channels=0),
    'gcn_encoder_3_128': dict(gcn_layers=3, gcn_dim=128, cnn_channels=0),
    'gcn_autoencoder_2_128': dict(gcn_layers=2, gcn_dim=128, cnn_channels=0),
}

# gcn features
ONE_HOT_FEATURE = 'one_hot'
WDEGREE_FEATURE = 'weighted_degree'

# learning rate schedulers
COSINE_ANNEALING_WARM_RESTARTS_SCHEDULER = 'cosine_annealing_warm_restarts'
COSINE_ANNEALING_SCHEDULER = 'cosine_annealing'
WARMUP_SQRT_SCHEDULER = 'warmup_sqrt'
WARMUP_COSINE_SCHEDULER = 'warmup_cosine'
