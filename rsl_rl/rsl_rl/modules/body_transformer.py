# Copyright (c) 2025, ETH Zurich, Rafael Cathomen
#
# SPDX-License-Identifier: BSD-3-Clause

import torch
import torch.nn as nn
from typing import Literal

from .transformer import TransformerLayer

##
# Optional small embedding MLP
##


class SmallEmbedMLP(nn.Module):
    """
    Optional single linear or MLP for embedding.
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        tokenizer_layers: list[int] = [],
        activation=nn.ReLU(),
    ):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim

        if len(tokenizer_layers) == 0:
            # single linear
            self.model = nn.Linear(in_dim, out_dim)
        else:
            # MLP: in_dim -> tokenizer_layers -> out_dim
            layers = []
            prev_dim = in_dim
            for layer_dim in tokenizer_layers:
                layers.append(nn.Linear(prev_dim, layer_dim))
                layers.append(activation)
                prev_dim = layer_dim
            layers.append(nn.Linear(prev_dim, out_dim))
            self.model = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


##
# Body Transformer
##
class BodyTransformer(nn.Module):
    """
    Body Transformer (BoT) [1]_ with two variants:
      - BoT-Hard: every layer uses adjacency-based masking
      - BoT-Mix: interleaves masked and unmasked layers

    Also supports a custom obs_node_dict that can either:
      - map an obs key to [node_ids] (replicate entire feature to those nodes)
      - map an obs key to ( [node_ids], indexing_mat ) which shapes how each node sees that feature.

    References:
        .. [1] Sferrazza et al. "Body Transformer: Leveraging Robot Embodiment for Policy Learning" arXiv preprint https://arxiv.org/abs/2408.06316 (2024).
    """

    def __init__(
        self,
        obs_dict: dict[str, torch.Tensor],
        obs_node_dict: dict[str, list[int] | tuple[list[int], torch.Tensor]],
        adjacency_mat: torch.Tensor,  # shape [num_nodes, num_nodes], values in {0,1}
        num_layers: int,
        embedding_dim: int,
        num_heads: int,
        feedforward_dim: int = 256,
        dropout: float = 0.0,
        activation=nn.ReLU(),
        bot_variant: Literal["hard", "mix"] = "hard",
        tokenizer_layers: list[int] = [],
        pre_layer_norm: bool = True,
    ):
        """
        Args:
            obs_dict: Example dictionary to infer shapes.
            obs_node_dict:
               Maps each obs_key -> either:
                 (A) [node_ids]
                 (B) ( [node_ids], indexing_mat ) with shape [len(node_ids), obs_dim]
                     controlling partial dimension assignment to each node.
            adjacency_mat: shape [num_nodes, num_nodes] with 1 for edges, 0 for no edge
            num_layers: how many TransformerLayer blocks
            embedding_dim: dimension for each node embedding
            num_heads: multi-head count
            feedforward_dim: dimension of feedforward in each TransformerLayer
            dropout: dropout rate
            activation: activation function in feedforward sub-layer
            bot_variant: "hard" => every layer masked, "mix" => interleave masked/unmasked
            tokenizer_layers: if list is not empty, use those layers as tokenizers
            pre_layer_norm: if True, use pre-LN style, else post-LN style
        """
        super().__init__()

        # - Hard vs Mix
        self.bot_variant = bot_variant.lower()
        assert self.bot_variant in ["hard", "mix"], "bot_variant must be 'hard' or 'mix'"

        # - Number of nodes, adjacency as a buffer
        self.num_nodes = adjacency_mat.shape[0]
        adjacency_mat = adjacency_mat.clone()
        diag_idx = torch.arange(self.num_nodes)
        adjacency_mat[diag_idx, diag_idx] = 1.0  # self-loops
        self.register_buffer("adjacency_mat", adjacency_mat.bool())

        # - adjacency-based mask for masked layers
        adjacency_mat = adjacency_mat.clone()
        diag_idx = torch.arange(self.num_nodes)
        adjacency_mat[diag_idx, diag_idx] = 1
        float_mask = torch.zeros_like(adjacency_mat, dtype=torch.float32)
        float_mask[~adjacency_mat.bool()] = float("-inf")
        self.register_buffer("adjacency_additive_mask", float_mask)

        # - Determine the input dimension per node
        node_input_dims = [0] * self.num_nodes
        self.indexing_info = {}  # obs_key -> (node_ids, buffer_name)
        indexing_buffer_id = 0

        for obs_key, example_val in obs_dict.items():
            obs_dim = example_val.size(1)
            # get node ids for this obs_key
            entry = obs_node_dict[obs_key]

            if isinstance(entry, list):
                # each id in entry gets the entire feature
                for nid in entry:
                    node_input_dims[nid] += obs_dim
            else:
                # the feature needs to be split across nodes
                node_ids, index_mat = entry
                # index_mat: [len(node_ids), obs_dim]
                # i-th row of index_mat is a boolean mask for observations for the i-th node
                buf_name = f"_indexbuf_{indexing_buffer_id}"
                indexing_buffer_id += 1

                index_mat = index_mat.clone()  # ensure no side effects
                self.register_buffer(buf_name, index_mat)

                # We'll record it in indexing_info
                self.indexing_info[obs_key] = (node_ids, buf_name)

                for i, nid in enumerate(node_ids):
                    # dimensionally, we still consider obs_dim added
                    assert obs_dim == len(
                        index_mat[i]
                    ), f"obs_dim={obs_dim} vs index_mat[{i}]={len(index_mat[i])}, should be equal"
                    node_input_dims[nid] += index_mat[i].sum().item()

        # - Embedding layers
        self.node_embedders = nn.ModuleList()
        for nid in range(self.num_nodes):
            in_dim = node_input_dims[nid]
            print(f"BoT INFO: Node {nid:>2}: input_dim= {in_dim:>3}")
            embedder = SmallEmbedMLP(
                in_dim=in_dim, out_dim=embedding_dim, tokenizer_layers=tokenizer_layers, activation=activation
            )
            self.node_embedders.append(embedder)

        # - Positional encoding
        self.node_pos_embedding = nn.Embedding(self.num_nodes, embedding_dim)

        # - Transformer layers
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            layer = TransformerLayer(
                embedding_dim=embedding_dim,
                num_heads=num_heads,
                dim_feedforward=feedforward_dim,
                dropout=dropout,
                activation=activation,
                pre_layer_norm=pre_layer_norm,
            )
            self.layers.append(layer)

        # - For BoT-Hard vs BoT-Mix, define mask usage
        self.mask_usage = []
        if self.bot_variant == "hard":
            # all masked
            self.mask_usage = [True] * num_layers
        else:
            # e.g. alternate
            for i in range(num_layers):
                self.mask_usage.append(i % 2 == 0)

        # 7) We'll store obs_node_dict as-is, for the "list" keys
        #    but we remove the indexing_mat from it, because we already saved them as buffers
        self.obs_node_dict_list = {}
        for obs_key, entry in obs_node_dict.items():
            if isinstance(entry, list):
                self.obs_node_dict_list[obs_key] = entry
            else:
                # we've stored indexing matrix in indexing_info
                pass

    @torch.jit.unused
    def forward_dict(self, obs_dict: dict[str, torch.Tensor]) -> torch.Tensor:
        batch_size = next(iter(obs_dict.values())).size(0)
        node_features = [[] for _ in range(self.num_nodes)]
        # - assign features to nodes
        for obs_key, val in obs_dict.items():
            val = val.reshape(batch_size, -1)  # ensure [B, obs_dim]
            if obs_key in self.obs_node_dict_list:
                # means the observation goes to all nodes in the list (no splitting)
                node_ids = self.obs_node_dict_list[obs_key]
                for nid in node_ids:
                    node_features[nid].append(val)
            else:
                # means we stored it in indexing_info and we need to split it
                node_ids, buf_name = self.indexing_info[obs_key]
                indexing_mat = getattr(self, buf_name)  # retrieve buffer

                # indexing_mat has shape [len(node_ids), obs_dim]
                for i, nid in enumerate(node_ids):
                    obs_mask = indexing_mat[i]  # shape [obs_dim]
                    # partial => val * row
                    partial_val = val[:, obs_mask]
                    node_features[nid].append(partial_val)

        # - embedding
        embedded_per_node = []
        for nid in range(self.num_nodes):
            if len(node_features[nid]) == 0:
                feats_cat = torch.zeros(batch_size, 0, device=val.device, dtype=val.dtype)
            else:
                feats_cat = torch.cat(node_features[nid], dim=1)  # [B, sum_of_dims]
            embedded = self.node_embedders[nid](feats_cat)  # => [B, embedding_dim]
            embedded_per_node.append(embedded)

        x = torch.stack(embedded_per_node, dim=1)  # [B, num_nodes, embedding_dim]

        # - add positional encoding
        node_indices = torch.arange(self.num_nodes, device=x.device).unsqueeze(0)
        pos_embed = self.node_pos_embedding(node_indices)
        x = x + pos_embed  # [B, num_nodes, embedding_dim]

        # - forward through transformer layers
        for i, layer in enumerate(self.layers):
            if self.mask_usage[i]:
                x = layer(x, attn_mask=self.adjacency_additive_mask)
            else:
                x = layer(x, attn_mask=None)

        return x  # shape [B, num_nodes, embedding_dim]

    def forward_tensor(self, obs: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("forward_tensor is not implemented. Use forward_dict instead. ")

    def forward(self, obs) -> torch.Tensor:
        if torch.jit.is_scripting():
            return self.forward_tensor(obs)
        else:
            return self.forward_dict(obs)


##
# Detokenizers
##


class ActionDetokenizer(nn.Module):
    """
    Converts per-node embeddings [B, num_nodes, embedding_dim]
    into a concatenated action vector of shape [B, num_joints * dim_per_out_node].

    node_to_joint: a list of length = num_nodes
      - node_to_joint[i] = -1 => node i does not produce actions
      - node_to_joint[i] = j => node i corresponds to joint j
        (j is in [0..num_joints-1], i.e., [-1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11] = ignore torso)

    dim_per_out_node: how many outputs each actuated node produces
      e.g. 2 for (mean, log_std).

    The final ordering of the action vector is in ascending joint index,
    i.e. joints 0, 1, 2... up to num_joints-1.
    """

    def __init__(
        self,
        embedding_dim: int,
        node_to_joint: list[int],
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

        valid_ids = [i for i in node_to_joint if i >= 0]
        assert len(set(valid_ids)) == len(valid_ids), "node_to_joint must have unique joint indices."

        # each joint has a linear from embedding_dim->dim_per_out_node
        self.node_heads = nn.ModuleList([nn.Linear(embedding_dim, dim_per_out_node) for _ in range(self.num_joints)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, num_nodes, embedding_dim]
        return => [B, num_joints, dim_per_out_node]
        in ascending joint index.
        """
        B, N, D = x.shape
        # if N != self.num_nodes:
        #     raise ValueError(f"Mismatch: x.shape[1]={N}, expected {self.num_nodes}.")

        out = x.new_zeros(B, self.num_joints, self.dim_per_out_node)
        for node_idx, joint_idx in enumerate(self.node_to_joint):
            if joint_idx < 0:
                continue
            out[:, joint_idx, :] = self.node_heads[joint_idx](x[:, node_idx, :])
        return out


class ValueDetokenizer(nn.Module):
    """
    Convert [B, num_nodes, embedding_dim] => [B, 1] by projecting and averaging or summing
    """

    def __init__(
        self,
        embedding_dim: int,
        node_to_value: list[int],  # e.g. which nodes produce a value? or all
        average_values: bool = True,
    ):
        """
        node_to_value[i] = -1 => skip node i
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
        # Linear detokenizer for each node
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
