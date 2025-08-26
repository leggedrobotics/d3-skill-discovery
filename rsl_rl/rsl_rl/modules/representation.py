from __future__ import annotations

import torch
import torch.nn as nn

from rsl_rl.utils import extract_batch_shape, flatten_batch, unflatten_batch

##
# Sub Nets
##


class StateMLP(nn.Module):
    """Defines a simple multi-layer perceptron (MLP) network."""

    is_recurrent = False

    def __init__(
        self,
        state: dict[str, torch.Tensor] | torch.Tensor,
        hidden_layers: list[int],
        latent_dim: int,
        activation,
        layer_norm: bool = False,
    ):
        super().__init__()

        # state dim
        if isinstance(state, dict):
            in_dim = 0
            self.state_dim = {}
            for key, tensor in state.items():
                in_dim += tensor.flatten(1).shape[-1]
                self.state_dim[key] = tensor.flatten(1).shape[1:]

        else:
            in_dim = state.flatten(1).shape[-1]
            self.state_dim = state.flatten(1).shape[1:]

        layers = []
        for hidden_dim in hidden_layers:
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(activation)
            if layer_norm:
                layers.append(nn.LayerNorm(hidden_dim))
            in_dim = hidden_dim
        layers.append(nn.Linear(in_dim, latent_dim))
        self.mlp = nn.Sequential(*layers)

    def forward(self, state: dict[str, torch.Tensor] | torch.Tensor):
        batch_dim = extract_batch_shape(state, self.state_dim)

        state = flatten_batch(state, self.state_dim)

        if isinstance(state, dict):

            state = torch.cat([tensor.flatten(1) for _, tensor in state.items()], dim=-1)
        latent = self.mlp(state)

        return unflatten_batch(latent, batch_dim)


class StateRepresentation(nn.Module):
    """the state representation network phi(s): s -> z"""

    def __init__(
        self,
        obs: torch.Tensor | dict[str, torch.Tensor],
        hidden_layers: list[int],
        latent_dim: int,
        activation: str,
        layer_norm: bool = False,
    ):
        super().__init__()

        self.state_mlp = StateMLP(obs, hidden_layers, latent_dim, get_activation(activation), layer_norm=layer_norm)

    def forward(self, state):
        return self.state_mlp(state)


def get_activation(act_name):
    if act_name == "elu":
        return nn.ELU()
    elif act_name == "selu":
        return nn.SELU()
    elif act_name == "relu":
        return nn.ReLU()
    elif act_name == "crelu":
        return nn.CReLU()
    elif act_name == "lrelu":
        return nn.LeakyReLU()
    elif act_name == "tanh":
        return nn.Tanh()
    elif act_name == "sigmoid":
        return nn.Sigmoid()
    else:
        print("invalid activation function!")
        return None
