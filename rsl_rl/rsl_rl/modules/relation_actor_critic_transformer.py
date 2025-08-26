#  Copyright 2021 ETH Zurich, NVIDIA CORPORATION
#  SPDX-License-Identifier: BSD-3-Clause


from __future__ import annotations

import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
from torch.func import functional_call, stack_module_state, vmap
from typing import Literal

from rsl_rl.utils.timer import TIMER_CUMULATIVE

from .body_transformer import ActionDetokenizer, BodyTransformer, ValueDetokenizer

# from .body_transformer_pytorch import ActionDetokenizer, BodyTransformer, ValueDetokenizer
from .simba import SimBa
from .transformer import PoolingModule, TransformerLayer

##
# - Transformer
##


class RelationalTransformer(nn.Module):
    is_recurrent = False

    def __init__(
        self,
        obs_dict: dict[str, torch.Tensor],
        out_dim: int,
        activation,
        embedding_dim=128,
        transformer_hidden_dim=128,
        out_mlp_dims: list[int] = [256],
        num_heads=4,
        num_transformer_layers=2,
    ):
        super().__init__()

        self.activation = activation

        # Extract shapes from observation dictionary
        my_pos_shape = obs_dict["my_velocity"].shape[-1]
        lidar_scan_shape = obs_dict["lidar_scan"].shape[-1]
        lidar_scan_top_shape = obs_dict["lidar_scan_top"].shape[-1]
        boxes_poses_shape = obs_dict["boxes_poses"].shape[-1]
        _batch_size = obs_dict["my_velocity"].shape[0]

        # Entity count
        num_boxes = obs_dict["boxes_poses"].shape[1]
        num_entities = num_boxes + 1  # +1 for self
        print(f"[INFO] Number of entities: {num_entities}")

        ##
        # - Embeddings
        ##

        # Self embedding: lidar scans and position
        assert lidar_scan_shape == lidar_scan_top_shape, "lidar_scan and lidar_scan_top must have the same shape"
        # Circular convolution for lidar scans
        self.circular_conv = nn.Conv2d(
            in_channels=1, out_channels=4, kernel_size=(2, 4), stride=1, padding_mode="circular"
        )

        # Compute the flattened output dimension after convolution
        dummy_lidar_im = torch.stack([obs_dict["lidar_scan"], obs_dict["lidar_scan_top"]], dim=1).unsqueeze(1).cpu()
        dummy_out = self.circular_conv(dummy_lidar_im)
        flattened_out_dim = dummy_out.view(dummy_out.size(0), -1).shape[-1]

        # Define the self embedder
        self.self_embedder = nn.Sequential(
            nn.Linear(my_pos_shape + flattened_out_dim, embedding_dim),
            activation,
        )
        # Box embedding
        self.box_embedder = nn.Sequential(
            nn.Linear(boxes_poses_shape, embedding_dim),
            activation,
        )

        # Print number of parameters in the embeddings
        num_embedding_params = sum(p.numel() for p in self.self_embedder.parameters()) + sum(
            p.numel() for p in self.box_embedder.parameters()
        )
        print(f"[INFO] number of parameters in self embedder: {num_embedding_params:_}")

        ##
        # - Transformer
        ##

        # Transformer parameters
        self.embedding_dim = embedding_dim  # Dimension of embeddings
        self.num_heads = num_heads  # Number of attention heads
        self.num_layers = num_transformer_layers  # Number of transformer layers

        # Transformer encoder layer
        # Stack multiple transformer layers
        self.transformer_layers = nn.ModuleList(
            [
                TransformerLayer(
                    embedding_dim=self.embedding_dim,
                    num_heads=self.num_heads,
                    dim_feedforward=transformer_hidden_dim,
                    dropout=0.0,
                )
                for _ in range(self.num_layers)
            ]
        )

        # This will be updated with new observations
        # attentin mask, values are 0 for allowed positions and -inf for masked positions
        # has to be updated with new observations
        self.attention_mask = torch.zeros(_batch_size, num_entities, num_entities)
        # TODO attention mask

        num_transformer_params = sum(p.numel() for p in self.transformer_layers.parameters())
        print(f"[INFO] number of parameters in transformer layers: {num_transformer_params:_}")

        ##
        # - Pooling module
        ##
        num_seeds = 1  # Number of seed vectors for PMA
        self.pooling = PoolingModule(
            embedding_dim=self.embedding_dim,
            num_heads=self.num_heads,
            pooling_type="pma",
            num_seeds=num_seeds,
        )

        # check pooling permutation invariance
        dummy_pooling_input = torch.randn(32, 124, self.embedding_dim)
        shuffled_dummy_pooling_input = dummy_pooling_input[:, torch.randperm(124), :]
        dummy_pooling_output = self.pooling(dummy_pooling_input)
        shuffled_pooing_output = self.pooling(shuffled_dummy_pooling_input)
        assert torch.allclose(
            dummy_pooling_output, shuffled_pooing_output, atol=1e-6
        ), "Pooling is not permutation invariant"

        ##
        # - Output layer
        ##
        # Final output layer (e.g., action logits)
        out_mlp_in_dim = self.embedding_dim * num_seeds
        out_layers = []
        for hidden_dim in out_mlp_dims:
            out_layers.append(nn.Linear(out_mlp_in_dim, hidden_dim))
            out_layers.append(activation)
            out_mlp_in_dim = hidden_dim
        out_layers.append(nn.Linear(out_mlp_in_dim, out_dim))
        self.output_layer = nn.Sequential(*out_layers)

        num_output_params = sum(p.numel() for p in self.output_layer.parameters())
        print(f"[INFO] number of parameters in output layer: {num_output_params:_}")

    def forward(self, obs_dict):
        # Extract observations
        my_velocity = obs_dict["my_velocity"]
        lidar_scan = obs_dict["lidar_scan"]
        lidar_scan_top = obs_dict["lidar_scan_top"]
        boxes_poses = obs_dict["boxes_poses"]

        batch_size = my_velocity.size(0)

        # Process lidar scans
        lidar_im = torch.stack([lidar_scan, lidar_scan_top], dim=1).unsqueeze(1)
        lidar_features = self.circular_conv(lidar_im)
        lidar_features = lidar_features.view(batch_size, -1)

        # Self embedding
        self_features = torch.cat([my_velocity, lidar_features], dim=-1)
        self_embedding = self.self_embedder(self_features)  # Shape: [batch_size, embedding_dim]

        # Box embeddings
        box_embeddings = self.box_embedder(boxes_poses)  # Shape: [batch_size, num_boxes, embedding_dim]

        # Combine embeddings to a sequence (set of entities)
        # Shape: [batch_size, num_entities, embedding_dim]
        entity_embeddings = torch.cat([self_embedding.unsqueeze(1), box_embeddings], dim=1)  # self has to be first

        # update attention mask
        self.attention_mask = None  # self._attention_mask(  # TODO: update with new observations

        # Apply transformer layers
        for layer in self.transformer_layers:
            entity_embeddings = layer(entity_embeddings, attn_mask=self.attention_mask)

        # Pooling
        pooled_output = self.pooling(entity_embeddings, attn_mask=self.attention_mask)

        if self.pooling.pooling_type == "pma" and self.pooling.num_seeds > 1:
            pooled_output = pooled_output.view(batch_size, -1)  # Shape: [batch_size, embedding_dim * num_seeds]

        # Compute output
        output = self.output_layer(pooled_output)  # Shape: [batch_size, out_dim]

        return output

    def _attention_mask(self, visible_entities: torch.Tensor) -> torch.Tensor:
        """
        Compute the attention mask for the transformer.
        The mask is a binary matrix where 0 indicates allowed positions and -inf indicates masked positions.
        The mask is symmetric and has shape [batch_size, num_entities, num_entities].

        the input is a binary mask of visible entities wrt to the agent (self), i.e, the first entry
        input shape: [batch_size, num_entities -1]. The agent is always visible to itself
        if visible_entities is None, all entities are visible
        """

        # Add the agent to the visible entities
        visible_entities = torch.cat(
            [torch.ones(visible_entities.size(0), 1).to(visible_entities.device), visible_entities], dim=1
        )
        # Compute the mask
        mask = torch.matmul(visible_entities.unsqueeze(-1), visible_entities.unsqueeze(1))
        mask = mask * -1e9  # Set masked positions to -inf
        return mask


##
# - MLP
##


class DictMLP(nn.Module):
    is_recurrent = False

    def __init__(
        self,
        obs_dict: dict[str, torch.Tensor],
        hidden_layers: list[int],
        out_dim: int,
        activation,
    ):
        super().__init__()

        in_dim = 0
        for _, tensor in obs_dict.items():
            in_dim += tensor.flatten(1).shape[-1]

        layers = []
        for hidden_dim in hidden_layers:
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(activation)
            in_dim = hidden_dim
        layers.append(nn.Linear(in_dim, out_dim))
        self.mlp = nn.Sequential(*layers)

    @torch.jit.unused
    def forward_dict(self, obs_dict: dict[str, torch.Tensor]) -> torch.Tensor:
        """Python-only forward that handles dict input."""
        x = torch.cat([obs_dict[key].flatten(1) for key in sorted(obs_dict.keys())], dim=-1)
        return self.mlp(x)

    def forward_tensor(self, obs: torch.Tensor) -> torch.Tensor:
        """TorchScript forward used for exporting."""
        return self.mlp(obs)

    def forward(self, obs):
        if torch.jit.is_scripting():
            return self.forward_tensor(obs)
        else:
            # We are in normal Python mode --> expect dict
            return self.forward_dict(obs)


class MLP(nn.Module):
    is_recurrent = False

    def __init__(self, obs: torch.Tensor, hidden_layers: list[int], out_dim: int, activation):
        super().__init__()
        in_dim = obs.shape[-1]
        layers = []
        for hidden_dim in hidden_layers:
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(activation)
            in_dim = hidden_dim
        layers.append(nn.Linear(in_dim, out_dim))
        self.mlp = nn.Sequential(*layers)
        self.activation = activation

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.mlp(x)
        return x


class EfficientMLPEnsemble(nn.Module):
    """
    A more efficient ensemble implementation that processes all models in one batch.
    This avoids the overhead of vmap and functional_call.
    """

    def __init__(self, input_dim, hidden_layers, output_dim, num_models, activation=nn.ReLU()):
        super().__init__()
        self.num_models = num_models
        self.output_dim = output_dim

        # Create all weights directly as batched parameters
        # For each layer, we'll have weights of shape [num_models, out_features, in_features]

        # First layer (input to first hidden)
        self.w1 = nn.Parameter(torch.empty(num_models, hidden_layers[0], input_dim))
        self.b1 = nn.Parameter(torch.empty(num_models, hidden_layers[0]))

        # Build middle layers
        self.middle_weights = nn.ParameterList()
        self.middle_biases = nn.ParameterList()

        for i in range(1, len(hidden_layers)):
            self.middle_weights.append(nn.Parameter(torch.empty(num_models, hidden_layers[i], hidden_layers[i - 1])))
            self.middle_biases.append(nn.Parameter(torch.empty(num_models, hidden_layers[i])))

        # Output layer
        self.w_out = nn.Parameter(torch.empty(num_models, output_dim, hidden_layers[-1]))
        self.b_out = nn.Parameter(torch.empty(num_models, output_dim))

        self.activation = activation
        self.reset_parameters()

    def reset_parameters(self):
        # Initialize all parameters using standard initialization
        nn.init.kaiming_uniform_(self.w1)
        nn.init.zeros_(self.b1)

        for w, b in zip(self.middle_weights, self.middle_biases):
            nn.init.kaiming_uniform_(w)
            nn.init.zeros_(b)

        nn.init.kaiming_uniform_(self.w_out)
        nn.init.zeros_(self.b_out)

    def forward(self, x):
        # x shape: [batch_size, input_dim]
        batch_size = x.shape[0]

        # First layer
        # [batch_size, input_dim] -> [batch_size, 1, input_dim] -> [batch_size, num_models, input_dim]
        x_expanded = x.unsqueeze(1).expand(-1, self.num_models, -1)

        # Perform batched matmul for first layer
        # [batch_size, num_models, input_dim] @ [num_models, hidden_layers[0], input_dim].transpose(-1, -2)
        # -> [batch_size, num_models, hidden_layers[0]]
        h = torch.bmm(
            x_expanded.view(-1, 1, x.shape[-1]), self.w1.view(self.num_models, self.w1.shape[1], -1).transpose(-1, -2)
        ).view(batch_size, self.num_models, -1)

        # Add bias and apply activation
        h = h + self.b1.unsqueeze(0)
        h = self.activation(h)

        # Middle layers
        for w, b in zip(self.middle_weights, self.middle_biases):
            # [batch_size, num_models, prev_dim] @ [num_models, curr_dim, prev_dim].transpose(-1, -2)
            # -> [batch_size, num_models, curr_dim]
            h = torch.bmm(h.view(-1, 1, h.shape[-1]), w.view(self.num_models, w.shape[1], -1).transpose(-1, -2)).view(
                batch_size, self.num_models, -1
            )
            h = h + b.unsqueeze(0)
            h = self.activation(h)

        # Output layer
        # [batch_size, num_models, hidden_dim] @ [num_models, output_dim, hidden_dim].transpose(-1, -2)
        # -> [batch_size, num_models, output_dim]
        out = torch.bmm(
            h.view(-1, 1, h.shape[-1]), self.w_out.view(self.num_models, self.w_out.shape[1], -1).transpose(-1, -2)
        ).view(batch_size, self.num_models, -1)
        out = out + self.b_out.unsqueeze(0)

        if self.output_dim == 1:
            out = out.squeeze(-1)  # [batch_size, num_models]

        return out


class ParallelCritics(nn.Module):
    """
    A batched/parallel MLP ensemble where each critic has its own parameters,
    but all are evaluated in a single pass.

    - We flatten the dict-based observations once.
    - For each layer, we do a batched matmul across the [num_critics] dimension.
    """

    def __init__(
        self,
        obs_dict_example: dict[str, torch.Tensor],
        hidden_dims: list[int],
        num_critics: int,
        activation=nn.ReLU(),
        out_dim: int = 1,
    ):
        super().__init__()
        self.num_critics = num_critics
        self.activation = activation

        # 1) Figure out the total input dimension (flattening all dict entries except batch).
        in_dim = 0
        for _, tensor in obs_dict_example.items():
            # 'tensor.shape[1:]' is everything but the batch dimension
            in_dim += torch.prod(torch.tensor(tensor.shape[1:])).item()

        # 2) Build up the full layer structure: [in_dim -> ...hidden_dims... -> out_dim]
        layer_sizes = [in_dim] + hidden_dims + [out_dim]

        # 3) We'll create parameters for each layer in shape [num_critics, out_features, in_features].
        #    That means each critic has its own slice along the 0th dimension.
        self.weights = nn.ParameterList()
        self.biases = nn.ParameterList()

        for i in range(len(layer_sizes) - 1):
            in_features = layer_sizes[i]
            out_features = layer_sizes[i + 1]

            w = nn.Parameter(torch.empty(num_critics, out_features, in_features))
            b = nn.Parameter(torch.empty(num_critics, out_features))

            # Initialize
            nn.init.xavier_uniform_(w, gain=1.0)
            nn.init.zeros_(b)

            self.weights.append(w)
            self.biases.append(b)

    def _flatten_obs_dict(self, obs_dict: dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Flatten and concatenate all observations in sorted(key) order.
        Result shape: [batch_size, total_in_dim].
        """
        # Sort keys so the dimension ordering is consistent
        return torch.cat(
            [obs_dict[k].flatten(1) for k in sorted(obs_dict.keys())],
            dim=-1,
        )

    def forward(self, obs_dict: dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Evaluates ALL critics in parallel:
          - Input: dict of Tensors, each shape [batch_size, ...].
          - Output: Tensor of shape [batch_size, num_critics]
            (or [batch_size, num_critics, out_dim] if out_dim > 1).
        """
        # 1) Flatten dict once => [batch_size, in_dim]
        x = self._flatten_obs_dict(obs_dict)

        # 2) Expand x to [num_critics, batch_size, in_dim]
        #    so we can do a batched matmul across the critic dimension.
        #    shape => [num_critics, batch_size, current_layer_dim]
        x = x.unsqueeze(0).expand(self.num_critics, -1, -1)

        # 3) Go through each layer in a batched fashion
        for layer_idx, (W, b) in enumerate(zip(self.weights, self.biases)):
            # W: [num_critics, out_features, in_features]
            # b: [num_critics, out_features]
            # x: [num_critics, batch_size, in_features]
            # out => [num_critics, batch_size, out_features]

            # matmul: out = x @ W^T + b, done in a batched way
            # torch.baddbmm lets us do: out = b + x * W
            #   b must be [num_critics, 1, out_features], so we unsqueeze(1).
            x = torch.baddbmm(
                b.unsqueeze(1),  # shape [num_critics, 1, out_features]
                x,  # [num_critics, batch_size, in_features]
                W.transpose(1, 2),  # [num_critics, in_features, out_features]
            )
            # Apply activation if not the last layer
            is_last = layer_idx == len(self.weights) - 1
            if not is_last:
                x = self.activation(x)

        # 4) Now x => [num_critics, batch_size, out_dim].
        #    We typically want [batch_size, num_critics, out_dim].
        x = x.transpose(0, 1)  # => [batch_size, num_critics, out_dim]

        # 5) If out_dim == 1, squeeze down to [batch_size, num_critics]
        if x.shape[-1] == 1:
            x = x.squeeze(-1)

        return x


import copy
import torch
import torch.nn as nn
from torch.func import functional_call, stack_module_state, vmap


class ModelEnsemble(nn.Module):
    def __init__(self, critic_cls, critic_kwargs, num_critics):
        super().__init__()
        self.num_critics = num_critics

        # 1) Build 'num_critics' real models
        models = [critic_cls(**critic_kwargs) for _ in range(num_critics)]

        # 2) Stack their parameters/buffers
        self.params, self.stacked_buffers = stack_module_state(models)

        # 3) Create a single “base” model on meta
        base = copy.deepcopy(models[0])
        base.to("meta")

        # 4) Attach it *without* registering it as a submodule
        object.__setattr__(self, "base_model", base)
        # at this point, there's no "base_model" in self._modules

    def forward(self, x: torch.Tensor):
        def single_critic_forward(p, b):
            return functional_call(self.base_model, (p, b), (x,))

        outs = vmap(single_critic_forward, in_dims=(0, 0))(self.params, self.stacked_buffers)
        outs = outs.transpose(0, 1)  # => [batch_size, num_critics, out_dim]
        if outs.shape[-1] == 1:
            outs = outs.squeeze(-1)
        return outs

    def _apply(self, fn):
        # Let PyTorch recurse on real submodules, but we have no "base_model" submodule
        super()._apply(fn)

        # Now apply fn to stacked params/buffers
        for k, v in self.params.items():
            self.params[k] = fn(v)
        for k, b in self.stacked_buffers.items():
            self.stacked_buffers[k] = fn(b)

        return self

    def to(self, device, *args, **kwargs):
        """
        Only move real ensemble params/buffers. The meta base_model remains untouched.
        """
        for k, v in self.params.items():
            self.params[k] = v.to(device, *args, **kwargs)
        for k, b in self.stacked_buffers.items():
            self.stacked_buffers[k] = b.to(device, *args, **kwargs)
        return self


##
# - ActorCritic
##
class RelationalActorCriticTransformer(nn.Module):
    is_recurrent = False

    def __init__(
        self,
        actor_obs_dict: dict,
        critic_obs_dict: dict,
        num_actions,
        activation="elu",
        init_noise_std=1.0,
        actor_layers: list[int] = [512, 256, 256],
        critic_layers: list[int] = [512, 256, 256],
        architecture: Literal["MLP", "Transformer", "SimBa", "BodyTransformer"] = "MLP",
        log_std_range: tuple[float, float] = (-5, 2),
        squashed_output=False,
        num_critics: int = 1,
        **kwargs,
    ):
        if kwargs:
            print(
                "ActorCritic.__init__ got unexpected arguments, which will be ignored: "
                + str([key for key in kwargs.keys()])
            )
        super().__init__()
        activation = get_activation(activation)
        self.num_actions = num_actions
        self.architecture = architecture
        # Policy and Value networks
        if architecture == "Transformer":
            self.actor = RelationalTransformer(actor_obs_dict, num_actions * 2, activation)
            self.critics = torch.nn.ModuleList(
                [RelationalTransformer(critic_obs_dict, 1, activation) for _ in range(num_critics)]
            )

        elif architecture == "MLP":
            self.actor = MLP(self._flatten_dict(actor_obs_dict), actor_layers, num_actions * 2, activation)
            crit_layers = critic_layers
            self.critics = torch.nn.ModuleList(
                [MLP(self._flatten_dict(critic_obs_dict), crit_layers, 1, activation) for _ in range(num_critics)]
            )

            # self.critic_ensemble = ModelEnsemble(
            #     critic_cls=MLP,
            #     critic_kwargs={
            #         "obs": self._flatten_dict(critic_obs_dict),
            #         "hidden_layers": crit_layers,
            #         "out_dim": 1,
            #         "activation": activation,
            #     },
            #     num_critics=num_critics,
            # )

        elif architecture == "SimBa":
            self.actor = SimBa(
                self._flatten_dict(actor_obs_dict), actor_layers, num_actions * 2, activation, expansion=1
            )
            self.critics = torch.nn.ModuleList(
                [
                    SimBa(self._flatten_dict(critic_obs_dict), critic_layers, 1, activation, expansion=1)
                    for _ in range(num_critics)
                ]
            )

            # v-mappable
            # self.critic_ensemble = ModelEnsemble(
            #     critic_cls=SimBa,
            #     critic_kwargs={
            #         "obs": self._flatten_dict(critic_obs_dict),
            #         "hidden_layers": [128, 128, 128],
            #         "out_dim": 1,
            #         "activation": activation,
            #     },
            #     num_critics=num_critics,
            # )

        elif architecture == "BodyTransformer":

            # adjacency matrix 13x13
            # links:
            # 1. Torso (1 node)
            # 2. Left Front Leg (3 nodes)
            # 3. Left Hind Leg (3 nodes)
            # 4. Right Front Leg (3 nodes)
            # 5. Right Hind Leg (3 nodes)

            adjacency_matrix = torch.tensor(
                [  # 0, 1, 2, 3, 4, 5, 6, 7, 8, 9,10,11,12          link     | joint
                    [0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],  # 0:  Torso    | NO JOINT
                    [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],  # 1:  LF Hip   | LF Hip Abduction/Adduction
                    [1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],  # 2:  LH Hip   | LH Hip Abduction/Adduction
                    [1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],  # 3:  RF Hip   | RF Hip Abduction/Adduction
                    [1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],  # 4:  RH Hip   | RH Hip Abduction/Adduction
                    [0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],  # 5:  LF Thigh | LF Hip Flexion/Extension
                    [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],  # 6:  LH Thigh | LH Hip Flexion/Extension
                    [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0],  # 7:  RF Thigh | RF Hip Flexion/Extension
                    [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1],  # 8:  RH Thigh | RH Hip Flexion/Extension
                    [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],  # 9:  LF Shank | LF Knee Flexion/Extension
                    [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],  # 10: LH Shank | LH Knee Flexion/Extension
                    [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],  # 11: RF Shank | RF Knee Flexion/Extension
                    [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],  # 12: RH Shank | RH Knee Flexion/Extension
                ]
            )
            # obs node dict specifies which observation belongs to which node
            obs_node_dict: dict[str, list[int] | tuple[list[int], torch.Tensor]] = {}
            # dictionary that maps observation names to the node id they belong to.
            # If the observation belongs to multiple nodes, an indexing matrix is used of shape (num_nodes, num_obs) which tells which node gets which observation
            # If no indexing matrix is used, the observation is assumed to belong to all nodes which is the same as if the indexing matrix is full of ones

            # - Observation index to node maps:
            # i-th row of this matrix is a boolean mask for observations, which are passed
            # to the i-th node in the provided node list
            joint_index_to_node_map = torch.eye(12).bool()
            skill_to_node_map = torch.zeros((5, 26))
            skill_to_node_map[0, 0:2] = 1  # postion to torso
            skill_to_node_map[0, 14:] = 1  # base_velocity to torso
            skill_to_node_map[1:, 2:14] = 1  # feet to shanks

            obs_node_dict["origin"] = [0]
            obs_node_dict["height_scan"] = [0]
            obs_node_dict["base_lin_vel"] = [0]
            obs_node_dict["base_ang_vel"] = [0]
            obs_node_dict["projected_gravity"] = [0]
            obs_node_dict["actions"] = ([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12], joint_index_to_node_map)
            obs_node_dict["joint_pos"] = ([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12], joint_index_to_node_map)
            obs_node_dict["joint_vel"] = ([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12], joint_index_to_node_map)
            obs_node_dict["skill"] = [9, 10, 11, 12]  # hardcoded: skills only for shanks
            obs_node_dict["skill"] = ([0, 9, 10, 11, 12], skill_to_node_map.bool())
            # hardcoded: pos 2d, feet 12d, base_vel 8d

            if True:
                ##
                # - testing non-skill disocvery for debugging
                ##
                obs_node_dict: dict[str, list[int] | tuple[list[int], torch.Tensor]] = {}
                obs_node_dict["base_lin_vel"] = [0]
                obs_node_dict["base_ang_vel"] = [0]
                obs_node_dict["projected_gravity"] = [0]
                obs_node_dict["velocity_commands"] = [0]
                obs_node_dict["actions"] = ([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12], joint_index_to_node_map)
                obs_node_dict["joint_pos"] = ([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12], joint_index_to_node_map)
                obs_node_dict["joint_vel"] = ([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12], joint_index_to_node_map)

                if False:
                    # even simpler:
                    adjacency_matrix = torch.ones_like(adjacency_matrix)
                    all_nodes_list = [i for i in range(13)]
                    obs_node_dict: dict[str, list[int] | tuple[list[int], torch.Tensor]] = {}
                    obs_node_dict["base_lin_vel"] = all_nodes_list
                    obs_node_dict["base_ang_vel"] = all_nodes_list
                    obs_node_dict["projected_gravity"] = all_nodes_list
                    obs_node_dict["velocity_commands"] = all_nodes_list
                    obs_node_dict["actions"] = all_nodes_list
                    obs_node_dict["joint_pos"] = all_nodes_list
                    obs_node_dict["joint_vel"] = all_nodes_list

            assert sorted(obs_node_dict.keys()) == sorted(
                actor_obs_dict.keys()
            ), "obs_node_dict keys do not match actor_obs_dict keys"
            # actor
            actor_BoT = BodyTransformer(
                obs_dict=actor_obs_dict,
                obs_node_dict=obs_node_dict,
                embedding_dim=64,
                num_heads=2,
                num_layers=10,
                adjacency_mat=adjacency_matrix,
                feedforward_dim=128,
                activation=activation,
                bot_variant="mix",
                pre_layer_norm=True,
            )

            actor_detokenizer = ActionDetokenizer(
                embedding_dim=actor_BoT.node_embedders[0].out_dim,
                node_to_joint=[-1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
                dim_per_out_node=2,
            )
            self.actor = nn.Sequential(actor_BoT, actor_detokenizer)

            # critic
            # We may not want to use the same architecture for the critic as for the actor as it is quite big
            # Also, we may have many critics, so it should be small
            critics_as_BoT = False
            if critics_as_BoT:
                critics = []
                for i in range(num_critics):
                    critic_BoT = BodyTransformer(
                        obs_dict=critic_obs_dict,
                        obs_node_dict=obs_node_dict,
                        embedding_dim=64,
                        num_heads=2,
                        num_layers=7,
                        adjacency_mat=adjacency_matrix,
                        feedforward_dim=128,
                        activation=activation,
                        pre_layer_norm=True,
                        bot_variant="hard",
                    )

                    critic_detokenizer = ValueDetokenizer(
                        embedding_dim=critic_BoT.node_embedders[0].out_dim,
                        node_to_value=[i for i in range(13)],
                    )

                    critics.append(nn.Sequential(critic_BoT, critic_detokenizer))
                self.critics = nn.ModuleList(critics)
            else:
                self.critics = torch.nn.ModuleList(
                    [MLP(critic_obs_dict, [256, 256, 256], 1, activation) for _ in range(num_critics)]
                )
        else:
            raise ValueError(f"Unknown architecture: {architecture}")

        print(f"[INFO] Architecture: {architecture}")
        print(f"[INFO] Actor number of parameters: {sum(p.numel() for p in self.actor.parameters()):_}")
        print(f"[INFO] Critic number of parameters: {sum(p.numel() for p in self.critics[0].parameters()):_}")
        print(
            f"[INFO] Number of critics: {num_critics} ==> Total critic number of parameters: {sum(p.numel() for p in self.critics.parameters()):_}"
        )
        print(f"[INFO] Actor Transformer: \n{self.actor}\n")
        print(f"[INFO] Critic Transformer: \n{self.critics[0]}\n")

        # Action noise
        # self.std = nn.Parameter(init_noise_std * torch.ones(num_actions))
        self.distribution = None
        # disable args validation for speedup
        Normal.set_default_validate_args = False
        self.log_std_range = log_std_range
        self.squashed_output = squashed_output

        # init log std
        a = log_std_range[0]
        b = log_std_range[1]
        alpha = (torch.log(torch.tensor(init_noise_std)) - a) / (b - a)
        argument = 2.0 * alpha - 1.0
        self.init_std_offset = torch.atanh(argument)

        # seems that we get better performance without init
        # self.init_memory_weights(self.memory_a, 0.001, 0.)
        # self.init_memory_weights(self.memory_c, 0.001, 0.)

    @staticmethod
    # not used at the moment
    def init_weights(sequential, scales):
        [
            torch.nn.init.orthogonal_(module.weight, gain=scales[idx])
            for idx, module in enumerate(mod for mod in sequential if isinstance(mod, nn.Linear))
        ]

    def reset(self, dones=None):
        pass

    def forward(self):
        raise NotImplementedError

    @property
    def std(self):
        return self.distribution.stddev

    @property
    def action_mean(self):
        return self.distribution.mean

    @property
    def action_std(self):
        return self.distribution.stddev

    @property
    def entropy(self):
        return self.distribution.entropy().sum(dim=-1)

    def update_distribution(self, observations):
        if self.architecture == "BodyTransformer":
            logits = self.actor(observations)
            mean = logits[..., 0]
            std_logits = logits[..., 1]

        else:
            logits = self.actor(self._flatten_dict(observations))
            mean, std_logits = logits.chunk(2, dim=-1)
        log_std = (torch.tanh(std_logits + self.init_std_offset) * 0.5 + 0.5) * (
            self.log_std_range[1] - self.log_std_range[0]
        ) + self.log_std_range[0]

        self.distribution = Normal(mean, torch.exp(log_std))

    def act(self, observations, return_log_probs=False, return_pre_tanh_actions=False, **kwargs):
        TIMER_CUMULATIVE.start("acting")
        self.update_distribution(observations)
        if self.squashed_output:
            action_sample = self.distribution.sample()
            scaled_action = torch.tanh(action_sample)
        else:
            action_sample = scaled_action = self.distribution.sample()

        if return_log_probs:
            if self.squashed_output:
                log_probs = self.distribution.log_prob(action_sample)
                log_det_jacobian = torch.log(1 - scaled_action.pow(2) + 1e-9)
                log_prob_scaled = log_probs - log_det_jacobian
                return scaled_action, log_prob_scaled.sum(dim=-1)
            else:
                return action_sample, self.distribution.log_prob(scaled_action).sum(dim=-1)

        if return_pre_tanh_actions:
            if self.squashed_output:
                return scaled_action, action_sample
            else:
                return action_sample, action_sample

        TIMER_CUMULATIVE.stop("acting")
        return scaled_action

    def get_actions_log_prob(self, actions, is_pre_tanh_action: bool = False):

        if self.squashed_output:
            if not is_pre_tanh_action:
                eps = 1e-6
                # Clip actions to prevent numerical issues
                clipped_actions = actions.clamp(-1 + eps, 1 - eps)
                # Invert the tanh transformation
                pre_tanh_action = 0.5 * (torch.log(1 + clipped_actions) - torch.log(1 - clipped_actions))
                scaled_action = clipped_actions
            else:
                eps = 1e-9
                pre_tanh_action = actions
                scaled_action = torch.tanh(pre_tanh_action)
            # Compute log probability under the normal distribution
            log_probs = self.distribution.log_prob(pre_tanh_action)
            # Compute the log determinant of the Jacobian of the tanh transformation
            log_det_jacobian = torch.log(1 - scaled_action.pow(2) + eps)
            # Adjust log probabilities
            log_prob_scaled = log_probs - log_det_jacobian
            # Sum over action dimensions if necessary
            return log_prob_scaled.sum(dim=-1)
        else:
            # actions is = scaled_action since we are not squashing the output
            return self.distribution.log_prob(actions).sum(dim=-1)

    def squash_action(self, actions):
        if self.squashed_output:
            return torch.tanh(actions)
        return actions

    def act_inference(self, observations):
        actions_mean = self.actor(self._flatten_dict(observations)).chunk(2, dim=-1)[0]
        if self.squashed_output:
            return torch.tanh(actions_mean)
        return actions_mean

    def evaluate(self, critic_observations, **kwargs):

        # flatten before
        flat_obs = self._flatten_dict(critic_observations)

        TIMER_CUMULATIVE.start("critic_evaluation")
        values_1 = self.naive_ensemble_forward(flat_obs)
        TIMER_CUMULATIVE.stop("critic_evaluation")
        # TIMER_CUMULATIVE.start("vmap_ensemble_evaluation")
        # values_2 = self.vmap_ensemble_forward(flat_obs)
        # TIMER_CUMULATIVE.stop("vmap_ensemble_evaluation")

        return values_1

    def naive_ensemble_forward(self, ciritc_obs: torch.Tensor) -> torch.Tensor:
        return torch.stack([critic(ciritc_obs) for critic in self.critics], dim=-1).squeeze(2)

    def vmap_ensemble_forward(self, critic_obs: torch.Tensor) -> torch.Tensor:
        return self.critic_ensemble(critic_obs)

    def _flatten_dict(self, obs_dict: dict[str, torch.Tensor]) -> torch.Tensor:
        return torch.cat([obs_dict[k].flatten(1) for k in sorted(obs_dict.keys())], dim=-1)

    def prepare_export(self):
        """Modify the actor to only predict means (remove std)."""
        if hasattr(self.actor, "output_linear"):
            old_last_layer = self.actor.output_linear  # Get the last layer
            in_dim = old_last_layer.in_features
            out_dim = old_last_layer.out_features // 2
            # Create a new last layer with reduced output size
            new_last_layer = nn.Linear(in_dim, out_dim)
            # Copy the relevant part of the weights and biases
            new_last_layer.weight.data = old_last_layer.weight.data[:out_dim].clone()
            new_last_layer.bias.data = old_last_layer.bias.data[:out_dim].clone()
            # Clone the actor and modify the last layer
            new_actor = copy.deepcopy(self.actor)
            new_actor.output_linear = new_last_layer

        elif hasattr(self.actor, "mlp"):
            # Assuming the actor is an MLP
            # Modify the last layer to output only means
            old_last_layer = self.actor.mlp[-1]  # Get the last layer
            in_dim = old_last_layer.in_features
            out_dim = old_last_layer.out_features // 2  # Keep only the first half

            # Create a new last layer with reduced output size
            new_last_layer = nn.Linear(in_dim, out_dim)

            # Copy the relevant part of the weights and biases
            new_last_layer.weight.data = old_last_layer.weight.data[:out_dim].clone()
            new_last_layer.bias.data = old_last_layer.bias.data[:out_dim].clone()

            # Clone the actor and modify the last layer
            new_actor = copy.deepcopy(self.actor)
            new_actor.mlp[-1] = new_last_layer  # Replace last layer

        # Replace the actor with the modified one
        self.actor = new_actor


def get_activation(act_name):
    if act_name == "elu":
        return nn.ELU()
    elif act_name == "selu":
        return nn.SELU()
    elif act_name == "silu":
        return nn.SiLU()
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
