# Copyright (c) 2025, Robotic Systems Lab - Legged Robotics at ETH Zürich
#
# SPDX-License-Identifier: BSD-3-Clause

#  Copyright 2021 ETH Zurich, NVIDIA CORPORATION
#  SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
import torch.nn as nn
from torch.distributions import Normal
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from rsl_rl.modules.actor_critic import get_activation
from rsl_rl.utils import unpad_trajectories

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
        num_heads=4,
        num_transformer_layers=2,
    ):
        super().__init__()

        self.first_key = list(obs_dict.keys())[0]
        self.num_input_dims = len(obs_dict[self.first_key].shape[1:])

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
        num_embedding_params = (
            sum(p.numel() for p in self.self_embedder.parameters())
            + sum(p.numel() for p in self.box_embedder.parameters())
            + sum(p.numel() for p in self.circular_conv.parameters())
        )
        print(f"[INFO] Number of parameters in self embedder: {num_embedding_params:_}")

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

        num_transformer_params = sum(p.numel() for p in self.transformer_layers.parameters())
        print(f"[INFO] Number of parameters in transformer layers: {num_transformer_params:_}")

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
        print(f"[INFO] Number of parameters in pooling module: {sum(p.numel() for p in self.pooling.parameters()):_}")

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
        pooling_output_dim = self.embedding_dim * num_seeds if num_seeds > 1 else self.embedding_dim
        self.output_layer = nn.Sequential(
            nn.Linear(pooling_output_dim, pooling_output_dim),
            activation,
            nn.Linear(pooling_output_dim, out_dim),
        )

        num_output_params = sum(p.numel() for p in self.output_layer.parameters())
        print(f"[INFO] Number of parameters in output layer: {num_output_params:_}")

    def flatten_batch(
        self, tensor: torch.Tensor, batch_shape: torch.Size, flattened_batch_size: torch.Tensor
    ) -> torch.Tensor:
        """
        Flatten the batch dimensions of a tensor to a single dimension.
        """
        return tensor.contiguous().view(flattened_batch_size, *tensor.shape[len(batch_shape) :])

    def unflatten_batch(self, tensor: torch.Tensor, batch_shape: torch.Size) -> torch.Tensor:
        """
        Unflatten the batch dimensions of a tensor to the original shape.
        """
        return tensor.view(*batch_shape, *tensor.shape[1:])

    def forward(self, obs_dict):
        # Extract observations
        my_velocity = obs_dict["my_velocity"]
        lidar_scan = obs_dict["lidar_scan"]
        lidar_scan_top = obs_dict["lidar_scan_top"]
        boxes_poses = obs_dict["boxes_poses"]

        # flatten batch dimensions (not necessary for inference, only for training)
        batch_size = obs_dict[self.first_key].shape[: -self.num_input_dims]
        flattened_batch_size = torch.tensor(batch_size).prod().item()

        my_velocity = self.flatten_batch(my_velocity, batch_size, flattened_batch_size)
        lidar_scan = self.flatten_batch(lidar_scan, batch_size, flattened_batch_size)
        lidar_scan_top = self.flatten_batch(lidar_scan_top, batch_size, flattened_batch_size)
        boxes_poses = self.flatten_batch(boxes_poses, batch_size, flattened_batch_size)

        # # Process lidar scans
        lidar_im = torch.stack([lidar_scan, lidar_scan_top], dim=-2).unsqueeze(-3)
        lidar_features = self.circular_conv(lidar_im)
        lidar_features = lidar_features.view(flattened_batch_size, -1)

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
            pooled_output = pooled_output.view(
                flattened_batch_size, -1
            )  # Shape: [batch_size, embedding_dim * num_seeds]

        # Compute output
        output = self.output_layer(pooled_output)  # Shape: [batch_size, out_dim]

        # unflatten batch dimensions
        output = self.unflatten_batch(output, batch_size)

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


class MLP_rec(nn.Module):
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
        self.first_key = list(obs_dict.keys())[0]
        self.num_input_dims = len(obs_dict[self.first_key].shape[1:])

    def forward(self, obs_dict):
        batch_dim = len(obs_dict[self.first_key].shape) - self.num_input_dims

        x = torch.cat([tensor.flatten(batch_dim) for _, tensor in obs_dict.items()], dim=-1)

        return self.mlp(x)


##
# - Actor-Critic
##
class RelationalActorCriticRecurrent(nn.Module):
    is_recurrent = True

    def __init__(
        self,
        actor_obs_dict: dict,
        critic_obs_dict: dict,
        num_actions,
        activation="elu",
        rnn_type="lstm",
        rnn_hidden_size=128,
        rnn_num_layers=1,
        init_noise_std=1.0,
        **kwargs,
    ):
        if kwargs:
            print(
                "ActorCriticRecurrent.__init__ got unexpected arguments, which will be ignored: " + str(kwargs.keys()),
            )

        super().__init__()

        activation = get_activation(activation)

        use_mlp = False
        rnn_input_dim = 128
        # MLPs
        if use_mlp:
            self.actor_main = MLP_rec(
                obs_dict=actor_obs_dict, hidden_layers=[128, 128], out_dim=rnn_input_dim, activation=activation
            )
            self.critic_main = MLP_rec(
                obs_dict=critic_obs_dict, hidden_layers=[128, 128], out_dim=rnn_input_dim, activation=activation
            )
        else:
            self.actor_main = RelationalTransformer(actor_obs_dict, rnn_input_dim, activation)
            self.critic_main = RelationalTransformer(critic_obs_dict, rnn_input_dim, activation)

        # RNNs
        self.memory_a = Memory(rnn_input_dim, type=rnn_type, num_layers=rnn_num_layers, hidden_size=rnn_hidden_size)
        self.memory_c = Memory(rnn_input_dim, type=rnn_type, num_layers=rnn_num_layers, hidden_size=rnn_hidden_size)

        # Output layers
        self.actor_out = nn.Linear(rnn_hidden_size, num_actions)
        self.critic_out = nn.Linear(rnn_hidden_size, 1)

        # Action noise
        self.std = nn.Parameter(init_noise_std * torch.ones(num_actions))
        self.distribution = None
        # disable args validation for speedup
        Normal.set_default_validate_args = False

        print(f"\n[INFO] Actor main num params: {sum(p.numel() for p in self.actor_main.parameters()):_}")
        print(f"[INFO] Actor RNN num params: {sum(p.numel() for p in self.memory_a.parameters()):_}")
        print(f"[INFO] Acror out num params: {sum(p.numel() for p in self.actor_out.parameters()):_}")
        print(f"[INFO] Critic main num params: {sum(p.numel() for p in self.critic_main.parameters()):_}")
        print(f"[INFO] Critic RNN num params: {sum(p.numel() for p in self.memory_c.parameters()):_}")
        print(f"[INFO] Critic out num params: {sum(p.numel() for p in self.critic_out.parameters()):_}")
        print(f"[INFO] TOTAL NUM PARAMS: {sum(p.numel() for p in self.parameters()):_}")

        print(f"\n[INFO] Actor main:\n{self.actor_main}")
        print(f"\n[INFO] Actor RNN:\n{self.memory_a}")
        print(f"\n[INFO] Acror out:\n{self.actor_out}")
        print(f"\n[INFO] Critic main:\n{self.critic_main}")
        print(f"\n[INFO] Critic RNN:\n{self.memory_c}")
        print(f"\n[INFO] Critic out:\n{self.critic_out}")
        print(f"\n[INFO] Critic RNN:\n{self.memory_c}")

    @staticmethod
    # not used at the moment
    def init_weights(sequential, scales):
        [
            torch.nn.init.orthogonal_(module.weight, gain=scales[idx])
            for idx, module in enumerate(mod for mod in sequential if isinstance(mod, nn.Linear))
        ]

    def reset(self, dones=None):
        self.memory_a.reset(dones)
        self.memory_c.reset(dones)

    def forward(self):
        raise NotImplementedError

    @property
    def action_mean(self):
        return self.distribution.mean

    @property
    def action_std(self):
        return self.distribution.stddev

    @property
    def entropy(self):
        return self.distribution.entropy().sum(dim=-1)

    def update_distribution(self, observations, masks=None, hidden_states=None):
        mlp_out = self.actor_main(observations)
        rnn_out = self.memory_a(mlp_out, masks, hidden_states)
        mean = self.actor_out(rnn_out.squeeze(0))
        self.distribution = Normal(mean, mean * 0.0 + self.std)

    def act(self, observations, **kwargs):
        self.update_distribution(observations, **kwargs)
        return self.distribution.sample()

    def get_actions_log_prob(self, actions):
        return self.distribution.log_prob(actions).sum(dim=-1)

    def act_inference(self, observations):
        mlp_out = self.actor_main(observations)
        rnn_out = self.memory_a(mlp_out)
        mean = self.actor_out(rnn_out.squeeze(0))
        return mean

    def evaluate(self, critic_observations, masks=None, hidden_states=None):
        mlp_out_c = self.critic_main(critic_observations)
        rnn_out_c = self.memory_c(mlp_out_c, masks, hidden_states)
        value = self.critic_out(rnn_out_c.squeeze(0))
        return value

    def get_hidden_states(self):
        return self.memory_a.hidden_states, self.memory_c.hidden_states


##
# - Memory
##


class Memory(torch.nn.Module):
    def __init__(self, input_size, type="lstm", num_layers=1, hidden_size=256):
        super().__init__()
        # RNN
        rnn_cls = nn.GRU if type.lower() == "gru" else nn.LSTM
        self.rnn = rnn_cls(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers)
        self.hidden_states = None

    def forward(self, input, masks=None, hidden_states=None):
        batch_mode = masks is not None
        if batch_mode:
            # batch mode (policy update): need saved hidden states
            if hidden_states is None:
                raise ValueError("Hidden states not passed to memory module during policy update")

            lengths = masks.sum(dim=0).cpu()  # get the lengths of the sequences
            packed_input = pack_padded_sequence(input, lengths, enforce_sorted=False)  # remove padding
            packed_output, _ = self.rnn(packed_input, hidden_states)  # forward pass with packed input
            out, _ = pad_packed_sequence(packed_output)  # add padding back
            # out, _ = self.rnn(input, hidden_states)
            out = unpad_trajectories(out, masks)  # remove padding
        else:
            # inference mode (collection): use hidden states of last step
            out, self.hidden_states = self.rnn(input.unsqueeze(0), self.hidden_states)
        return out

    def reset(self, dones=None):
        # When the RNN is an LSTM, self.hidden_states_a is a list with hidden_state and cell_state
        for hidden_state in self.hidden_states:
            hidden_state[..., dones == 1, :] = 0.0
