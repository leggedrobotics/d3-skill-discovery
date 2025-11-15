# Copyright (c) 2025, Robotic Systems Lab - Legged Robotics at ETH Zürich
#
# SPDX-License-Identifier: BSD-3-Clause

#  Copyright 2021 ETH Zurich, NVIDIA CORPORATION
#  SPDX-License-Identifier: BSD-3-Clause


from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from rsl_rl.utils.timer import TIMER_CUMULATIVE


class MaskedMultiHeadAttention(nn.Module):
    """
    Implements multi-head attention with an optional mask.
    """

    def __init__(self, embedding_dim: int, num_heads: int, dropout=0.0):
        """Initializes the multi-head attention module.

        Args:
            embedding_dim (int): Dimension of the embedding.
            num_heads (int): Number of attention heads.
            dropout (float, optional): Dropout rate. Defaults to 0.0.
        """
        super().__init__()
        assert embedding_dim % num_heads == 0, "Embedding dim must be divisible by num_heads."

        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.head_dim = embedding_dim // num_heads

        # Combine Q, K, V projections into a single linear layer for efficiency
        self.in_proj = nn.Linear(embedding_dim, 3 * embedding_dim, bias=False)

        # Output projection
        self.out_proj = nn.Linear(embedding_dim, embedding_dim, bias=False)

        # Dropout
        self.dropout = nn.Dropout(dropout)

        # Scaling factor for attention scores
        self.scale = self.head_dim**-0.5

    def forward(self, query, key, value, attn_mask=None):
        """
        query: [batch_size, seq_len_q, embedding_dim]
        key:   [batch_size, seq_len_k, embedding_dim]
        value: [batch_size, seq_len_k, embedding_dim]
        attn_mask: optional mask, broadcastable to [batch_size, num_heads, seq_len_q, seq_len_k]
        """

        batch_size, seq_len_q, _ = query.size()
        _, seq_len_k, _ = key.size()

        if (query is key) and (key is value):
            # Single projection if all three are identical (common in self-attention))
            x = query
            combined = self.in_proj(x)  # [batch_size, seq_len_q, 3 * embedding_dim]
            Q, K, V = combined.split(self.embedding_dim, dim=-1)
        else:
            # project each one separately.
            Q = self.in_proj(query)[..., : self.embedding_dim]
            K = self.in_proj(key)[..., self.embedding_dim : 2 * self.embedding_dim]
            V = self.in_proj(value)[..., 2 * self.embedding_dim :]

        # Split in num_heads by reshaping batch_size, seq_len, embedding_dim] to
        # [batch_size, num_heads, seq_len, head_dim]
        Q = Q.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)

        # Scaled dot-product attention
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale  # [B, H, seq_q, seq_k]

        # Apply mask if provided
        if attn_mask is not None:
            # Expand mask to [B, H, seq_q, seq_k] if needed
            if attn_mask.dim() == 2:
                attn_mask = attn_mask.unsqueeze(0).unsqueeze(0)
            elif attn_mask.dim() == 3:
                attn_mask = attn_mask.unsqueeze(1)
            attn_scores = attn_scores + attn_mask

        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Multiply by V
        attn_output = torch.matmul(attn_weights, V)

        # Recombine heads
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len_q, self.embedding_dim)

        # Final output projection
        output = self.out_proj(attn_output)

        return output


class TransformerLayer(nn.Module):
    """Implements a Transformer layer with configurable pre- or post-layer normalization."""

    def __init__(
        self,
        embedding_dim,
        num_heads,
        dim_feedforward=256,
        dropout=0.0,
        activation=nn.ReLU(),
        pre_layer_norm=True,
    ):
        super().__init__()

        self.self_attn = MaskedMultiHeadAttention(embedding_dim, num_heads, dropout=dropout)

        # Feedforward network
        self.linear1 = nn.Linear(embedding_dim, dim_feedforward)
        self.activation = activation
        self.linear2 = nn.Linear(dim_feedforward, embedding_dim)

        # Layer normalization
        self.norm1 = nn.LayerNorm(embedding_dim)
        self.norm2 = nn.LayerNorm(embedding_dim)

        # Dropout layers
        self.dropout = nn.Dropout(dropout)
        self.dropout_ffn = nn.Dropout(dropout)

        self.pre_layer_norm = pre_layer_norm

        # Pre or Post LayerNorm
        self.forward_fn = self._forward_pre_norm if pre_layer_norm else self._forward_post_norm

    def forward(self, src, attn_mask=None):
        return self.forward_fn(src, attn_mask)

    def _forward_pre_norm(self, src, attn_mask):
        """Pre-LayerNorm Transformer forward pass."""
        src2 = self.self_attn(self.norm1(src), self.norm1(src), self.norm1(src), attn_mask=attn_mask)
        src = src + self.dropout(src2)

        src2 = self.linear2(self.dropout_ffn(self.activation(self.linear1(self.norm2(src)))))
        src = src + self.dropout(src2)

        return src

    def _forward_post_norm(self, src, attn_mask):
        """Post-LayerNorm Transformer forward pass (Standard Transformer)."""
        src2 = self.self_attn(src, src, src, attn_mask=attn_mask)
        src = src + self.dropout(src2)
        src = self.norm1(src)

        src2 = self.linear2(self.dropout_ffn(self.activation(self.linear1(src))))
        src = src + self.dropout(src2)
        src = self.norm2(src)

        return src


class PoolingModule(nn.Module):
    """This module implements different pooling methods to ensure permutation invariance (Transformer is permutation equivariant)."""

    def __init__(self, embedding_dim, num_heads, pooling_type="mean", num_seeds=1):
        super(PoolingModule, self).__init__()
        self.pooling_type = pooling_type.lower()
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.num_seeds = num_seeds  # Number of seed vectors for PMA

        if self.pooling_type == "pma":
            # Initialize seed vectors as learnable parameters
            self.seed_vectors = nn.Parameter(torch.randn(self.num_seeds, embedding_dim))
            # Multihead attention layer for PMA
            self.attention = MaskedMultiHeadAttention(embedding_dim, num_heads)
        elif self.pooling_type not in ["max", "mean", "sum"]:
            raise ValueError("Invalid pooling type. Must be 'max', 'mean', 'sum', or 'pma'.")

    def forward(self, x, attn_mask=None):
        """
        x: Transformer outputs of shape [batch_size, num_entities, embedding_dim]
        attn_mask: Optional attention mask for PMA
        Returns:
            Pooled output of shape [batch_size, embedding_dim] (for num_seeds=1) or [batch_size, num_seeds, embedding_dim]
        """
        if self.pooling_type == "max":
            # Max pooling over entities
            pooled_output, _ = torch.max(x, dim=1)  # Shape: [batch_size, embedding_dim]
        elif self.pooling_type == "mean":
            # Mean pooling over entities
            pooled_output = torch.mean(x, dim=1)  # Shape: [batch_size, embedding_dim]
        elif self.pooling_type == "sum":
            # Sum pooling over entities
            pooled_output = torch.sum(x, dim=1)  # Shape: [batch_size, embedding_dim]
        elif self.pooling_type == "pma":
            # Pooling by Multihead Attention
            # Repeat seed vectors for batch size
            batch_size = x.size(0)
            seed_vectors = self.seed_vectors.unsqueeze(0).repeat(
                batch_size, 1, 1
            )  # Shape: [batch_size, num_seeds, embedding_dim]

            # Apply attention: Query=seed_vectors, Key=Value=x (entities)
            pooled_output = self.attention(
                query=seed_vectors,  # [batch_size, num_seeds, embedding_dim]
                key=x,  # [batch_size, num_entities, embedding_dim]
                value=x,  # [batch_size, num_entities, embedding_dim]
                attn_mask=attn_mask,  # Mask between seeds and entities if needed
            )  # Output shape: [batch_size, num_seeds, embedding_dim]

            if self.num_seeds == 1:
                pooled_output = pooled_output.squeeze(1)  # Shape: [batch_size, embedding_dim]
        else:
            raise ValueError("Invalid pooling type.")

        return pooled_output
