# Copyright (c) 2025, Robotic Systems Lab - Legged Robotics at ETH Zürich
#
# SPDX-License-Identifier: BSD-3-Clause

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Literal, Tuple, Union

from rsl_rl.utils import TIMER_CUMULATIVE

from .transformer import TransformerLayer

#############################################################################
# Optional small embedding MLP
#############################################################################


class SmallEmbedMLP(nn.Module):
    """
    A small MLP for embedding each node's local features,
    rather than a single Linear. This can be helpful if
    you want a bit more expressive 'tokenizer'.
    """

    def __init__(
        self, in_dim: int, out_dim: int, hidden_size: int = 0, activation=nn.ReLU()  # 0 => skip, use single linear
    ):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.hidden_size = hidden_size
        self.activation = activation

        if hidden_size <= 0:
            # single linear
            self.model = nn.Linear(in_dim, out_dim)
        else:
            # MLP: in_dim -> hidden_size -> out_dim
            self.model = nn.Sequential(nn.Linear(in_dim, hidden_size), activation, nn.Linear(hidden_size, out_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: shape [B, in_dim]
        return self.model(x)


#############################################################################
# BodyTransformer
#############################################################################


class BodyTransformer(nn.Module):
    """
    Body Transformer (BoT) [1]_ with two variants:
      - BoT-Hard: every layer uses adjacency-based masking
      - BoT-Mix: interleaves masked and unmasked layers

    This version uses PyTorch's built-in nn.TransformerEncoder and
    nn.TransformerEncoderLayer for the core transformations.
    Additionally:
      - We can optionally insert a small MLP 'tokenizer' per node
      - We do standard pre-LN or post-LN in the Transformer
      - We add node_pos_embedding to differentiate nodes

    References:
        .. [1] Sferrazza et al. "Body Transformer: Leveraging Robot Embodiment for Policy Learning" arXiv preprint https://arxiv.org/abs/2408.06316 (2024).
    """

    def __init__(
        self,
        # The "mapping-based" inputs
        obs_dict: Dict[str, torch.Tensor],
        obs_node_dict: Dict[str, Union[List[int], Tuple[List[int], torch.Tensor]]],
        adjacency_mat: torch.Tensor,  # shape [num_nodes, num_nodes], in {0,1}
        # Transformer config
        num_layers: int,
        embedding_dim: int,
        num_heads: int,
        feedforward_dim: int = 256,
        dropout: float = 0.0,
        # Bot variant
        bot_variant: Literal["hard", "mix"] = "hard",
        # Additional config
        activation=nn.ReLU(),
        tokenizer_hidden_size: int = 0,  # if >0, embed each node with MLP
        pre_layer_norm: bool = True,  # whether to use pre-LN or post-LN
    ):
        """
        Args:
            obs_dict: Example dict {key: shape([B, obs_dim])} to infer shapes.
            obs_node_dict: either
              (A) obs_node_dict[key] = [node_ids]
              (B) obs_node_dict[key] = ( [node_ids], indexing_mat ), shape [len(node_ids), obs_dim]
            adjacency_mat: shape [num_nodes, num_nodes], 1=>connected, 0=>not
            num_layers: # of Transformer layers
            embedding_dim: dimension for each node embedding
            num_heads: multi-head count
            feedforward_dim: dimension in the Transformer feed-forward
            dropout: dropout
            bot_variant: "hard"=>all masked, "mix"=>interleave masked/unmasked
            activation: feed-forward activation
            tokenizer_hidden_size: if >0, we embed each node input with an MLP
            pre_layer_norm: True => "pre-LN" style in Transformer
        """
        super().__init__()

        self.bot_variant = bot_variant.lower()
        assert self.bot_variant in ["hard", "mix"], "Must be 'hard' or 'mix'."

        # 1) number of nodes
        self.num_nodes = adjacency_mat.shape[0]

        # 2) adjacency + self-loops => boolean buffer
        adjacency_mat = adjacency_mat.clone()
        diag_idx = torch.arange(self.num_nodes)
        adjacency_mat[diag_idx, diag_idx] = 1
        float_mask = torch.zeros_like(adjacency_mat, dtype=torch.float32)
        float_mask[~adjacency_mat.bool()] = float("-inf")
        self.register_buffer("adjacency_additive_mask", float_mask)

        # 3) figure out input dimension per node
        node_input_dims = [0] * self.num_nodes
        self.indexing_info = {}
        indexing_buffer_id = 0

        for obs_key, example_val in obs_dict.items():
            obs_dim = example_val.shape[1]
            entry = obs_node_dict[obs_key]

            if isinstance(entry, list):
                # each node in entry sees full obs
                for nid in entry:
                    node_input_dims[nid] += obs_dim
            else:
                # partial indexing
                node_ids, index_mat = entry
                buf_name = f"_indexbuf_{indexing_buffer_id}"
                indexing_buffer_id += 1

                index_mat = index_mat.clone()
                self.register_buffer(buf_name, index_mat)

                # store the node_ids and buffer name per obs_key
                self.indexing_info[obs_key] = (node_ids, buf_name)

                for i, nid in enumerate(node_ids):
                    row = index_mat[i]
                    node_input_dims[nid] += row.sum().item()

        # 4) build node tokenizers
        self.node_embedders = nn.ModuleList()
        for nid in range(self.num_nodes):
            in_dim = int(node_input_dims[nid])
            embedder = SmallEmbedMLP(
                in_dim=in_dim, out_dim=embedding_dim, hidden_size=tokenizer_hidden_size, activation=activation
            )
            self.node_embedders.append(embedder)

        # 5) node positional encoding
        self.node_pos_embedding = nn.Embedding(self.num_nodes, embedding_dim)

        # 6) mask usage mix or hard
        self.mask_usage = []
        if self.bot_variant == "hard":
            self.mask_usage = [True] * num_layers
        else:
            # e.g. alternate masked/unmasked
            self.mask_usage = [(i % 2 == 0) for i in range(num_layers)]

        # 7) obs that are not splitted across nodes
        self.obs_node_dict_list = {}
        for obs_key, entry in obs_node_dict.items():
            if isinstance(entry, list):
                self.obs_node_dict_list[obs_key] = entry

        # 8) Build the Transformer layers using PyTorch
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=num_heads,
            dim_feedforward=feedforward_dim,
            dropout=dropout,
            activation=activation,
            batch_first=True,
            norm_first=pre_layer_norm,
        )
        self.num_layers = num_layers
        self.encoder_layers = nn.ModuleList([encoder_layer for _ in range(num_layers)])

    def forward_dict(self, obs_dict: Dict[str, torch.Tensor]) -> torch.Tensor:
        # print(f"Memory before forward: {torch.cuda.memory_allocated() / 1e9:.2f} GB")

        batch_size = next(iter(obs_dict.values())).size(0)
        node_features = [[] for _ in range(self.num_nodes)]

        # 1) gather input features for each node
        for obs_key, val in obs_dict.items():
            val = val.reshape(batch_size, -1)
            if obs_key in self.obs_node_dict_list:
                # replicate entire obs
                node_ids = self.obs_node_dict_list[obs_key]
                for nid in node_ids:
                    node_features[nid].append(val)
            else:
                # partial indexing
                node_ids, buf_name = self.indexing_info[obs_key]
                index_mat = getattr(self, buf_name)
                for i, nid in enumerate(node_ids):
                    row = index_mat[i]
                    partial_val = val[:, row]
                    node_features[nid].append(partial_val)

        # 2) embed each node
        embedded_nodes = []
        for nid in range(self.num_nodes):
            if len(node_features[nid]) == 0:
                feats_cat = torch.zeros(batch_size, 0, device=val.device, dtype=val.dtype)
            else:
                feats_cat = torch.cat(node_features[nid], dim=1)
            emb = self.node_embedders[nid](feats_cat)
            embedded_nodes.append(emb)

        # stack => [B, num_nodes, embedding_dim]
        x = torch.stack(embedded_nodes, dim=1)

        # 3) add node pos embedding
        node_indices = torch.arange(self.num_nodes, device=x.device).unsqueeze(0)
        pos_emb = self.node_pos_embedding(node_indices)  # shape [1, num_nodes, embedding_dim]
        x = x + pos_emb

        # 4) pass through the layers
        for i, layer in enumerate(self.encoder_layers):
            if self.mask_usage[i]:
                x = layer(x, src_mask=self.adjacency_additive_mask)
            else:
                x = layer(x)
        # print(f"Memory after forward: {torch.cuda.memory_allocated() / 1e9:.2f} GB")

        return x  # [B, num_nodes, embedding_dim]

    def forward(self, obs) -> torch.Tensor:
        # we skip TorchScript fallback
        return self.forward_dict(obs)


#############################################################################
# Detokenizers
#############################################################################


class ActionDetokenizer(nn.Module):
    """
    Convert [B, num_nodes, embedding_dim] => [B, num_joints, out_dim]
    where node_to_joint[i] = -1 => no action, else index of joint.
    If multiple outputs needed (like mean, log_std), set dim_per_out_node >= 2
    """

    def __init__(
        self,
        embedding_dim: int,
        node_to_joint: List[int],
        dim_per_out_node: int,
    ):
        super().__init__()
        self.node_to_joint = node_to_joint
        self.dim_per_out_node = dim_per_out_node

        self.num_nodes = len(node_to_joint)
        valid_joints = [j for j in node_to_joint if j >= 0]
        if not valid_joints:
            raise ValueError("No valid joints in node_to_joint.")
        self.num_joints = max(valid_joints) + 1

        # each joint has a linear from embedding_dim->dim_per_out_node
        self.node_heads = nn.ModuleList([nn.Linear(embedding_dim, dim_per_out_node) for _ in range(self.num_joints)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, num_nodes, embedding_dim]
        return => [B, num_joints, dim_per_out_node]
        in ascending joint index.
        """
        B, N, D = x.shape
        if N != self.num_nodes:
            raise ValueError(f"Mismatch: x.shape[1]={N}, expected {self.num_nodes}.")

        out = x.new_zeros(B, self.num_joints, self.dim_per_out_node)
        for node_idx, joint_idx in enumerate(self.node_to_joint):
            if joint_idx < 0:
                continue
            local_out = self.node_heads[joint_idx](x[:, node_idx, :])
            out[:, joint_idx, :] = local_out
        return out


class ValueDetokenizer(nn.Module):
    """
    Convert [B, num_nodes, embedding_dim] => single scalar or average across nodes.
    If you want a single global node to produce the value, pass in node_to_value
    or do an average across all nodes.
    """

    def __init__(
        self,
        embedding_dim: int,
        node_to_value: List[int],  # e.g. which nodes produce a value? or all
        average_values: bool = True,
    ):
        """
        node_to_value[i] = -1 => skip
        else produce a value from node i
        If average_values=True, we average them, else sum or do separate
        """
        super().__init__()
        self.node_to_value = node_to_value
        self.average_values = average_values

        self.num_nodes = len(node_to_value)
        self.value_heads = nn.ModuleList()
        used_nodes = 0
        for node_idx in node_to_value:
            if node_idx >= 0:
                used_nodes += 1
        # We'll define a linear for each node that is >=0
        # Actually let's define a separate linear for each i in [0..num_nodes) that is not -1
        # but store them in order. We'll keep an internal mapping from "node i" to "value_head index"
        self._node2head_idx = {}
        head_count = 0
        for i, flag in enumerate(node_to_value):
            if flag >= 0:
                self._node2head_idx[i] = head_count
                self.value_heads.append(nn.Linear(embedding_dim, 1))
                head_count += 1

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, num_nodes, embedding_dim]
        returns => [B, 1], single scalar if we average or sum
        or we could do multiple. We'll do a single scalar for simplicity
        """
        B, N, D = x.shape
        if N != self.num_nodes:
            raise ValueError(f"Mismatch in node count. x.shape[1]={N}, expected {self.num_nodes}.")

        all_values = []
        # gather each node's predicted value
        for i in range(self.num_nodes):
            if i in self._node2head_idx:
                idx = self._node2head_idx[i]
                node_val = self.value_heads[idx](x[:, i, :])  # shape [B,1]
                all_values.append(node_val)
        if len(all_values) == 0:
            # no node is used
            return x.new_zeros(B, 1)
        all_values = torch.cat(all_values, dim=1)  # [B, #used_nodes]

        if self.average_values:
            # average
            val = all_values.mean(dim=1, keepdim=True)  # [B,1]
        else:
            val = all_values.sum(dim=1, keepdim=True)
        return val
