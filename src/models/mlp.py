from torch import nn
import torch


class MLP(nn.Module):
    def __init__(self, input_size, output_size, n_layers, hidden_size,
                 activation='ELU', input_layer_norm=False, output_batch_size=None, device: torch.device = None,
                 output_factor: float = 1.):
        super(MLP, self).__init__()
        self.n_layers = n_layers
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.output_size = output_size
        activation_fn = getattr(nn, activation)
        layers = []
        if n_layers <= 1:
            layers.append(nn.Linear(input_size, output_size))
        else:
            layers.append(nn.Linear(input_size, hidden_size))
            layers.append(activation_fn())
            for _ in range(n_layers - 2):
                layers.append(nn.Linear(hidden_size, hidden_size))
                layers.append(activation_fn())
            layers.append(nn.Linear(hidden_size, output_size))
        self.layers = nn.Sequential(*layers)
        self.input_layer_norm = input_layer_norm
        if self.input_layer_norm:
            self.layernorm = nn.LayerNorm(input_size)
        self.output_batch_size = output_batch_size
        self.device = device
        if device is not None:
            self.to(device)
        self.output_factor = output_factor

    def forward(self, x):
        if self.input_layer_norm:
            x = self.layernorm(x)
        out = self.layers(x)
        if len(out) == 1 and self.output_batch_size is not None:
            out = out.repeat(self.output_batch_size, 1)
        return out * self.output_factor
