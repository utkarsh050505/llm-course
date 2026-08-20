#!/usr/bin/env python3
from __future__ import annotations

import os
import regex as re
from typing import IO, BinaryIO, Iterable, Optional, Type
from collections import defaultdict
import math
import numpy.typing as npt
import torch
import torch.nn as nn
import heapq
from . import helper_functions, helper_classes


def run_positionwise_feedforward(
    d_model: int,
    d_ff: int,
    weights: dict[str, torch.FloatTensor],
    in_features: torch.FloatTensor,
) -> torch.FloatTensor:
    """Given the weights of a position-wise feedforward network, return
    the output of your implementation with these weights.

    Args:
        d_model: int
            Dimensionality of the feedforward input and output.
        d_ff: int
            Dimensionality of the feedforward network's inner layer.
        weights: dict[str, torch.FloatTensor]
            State dict of our reference implementation.
            Keys: 'w1.weight' shape (d_ff, d_model),
                  'w2.weight' shape (d_model, d_ff).
        in_features: torch.FloatTensor
            Tensor to run your implementation on.

    Returns:
        torch.FloatTensor with the same leading dims as in_features,
        last dim = d_model.
    """
    raise NotImplementedError


def run_scaled_dot_product_attention(
    K: torch.FloatTensor,
    Q: torch.FloatTensor,
    V: torch.FloatTensor,
    mask: Optional[torch.BoolTensor] = None,
    pdrop: Optional[float] = None,
) -> torch.FloatTensor:
    """Given key (K), query (Q), and value (V) tensors, return
    the output of your scaled dot product attention implementation.

    Args:
        K: torch.FloatTensor
            Shape (batch_size, ..., seq_len, key_dimension).
        Q: torch.FloatTensor
            Shape (batch_size, ..., seq_len, key_dimension).
        V: torch.FloatTensor
            Shape (batch_size, ..., seq_len, value_dimension).
        mask: Optional[torch.BoolTensor]
            Shape (seq_len, seq_len). Positions where mask=True are
            filled with -inf before softmax (they are ignored).
        pdrop: Optional[float]
            If given, apply dropout to the post-softmax attention weights.

    Returns:
        torch.FloatTensor of shape (batch_size, ..., seq_len, value_dimension).
    """
    raise NotImplementedError


def run_multihead_self_attention(
    d_model: int,
    num_heads: int,
    attn_pdrop: float,
    weights: dict[str, torch.FloatTensor],
    in_features: torch.FloatTensor,
) -> torch.FloatTensor:
    """Given the key, query, and value projection weights of a naive unbatched
    implementation of multi-head attention, return the output of an optimized batched
    implementation. This implementation should handle the key, query, and value projections
    for all heads in a single matrix multiply.

    Args:
        d_model: int
            Dimensionality of the feedforward input and output.
        num_heads: int
            Number of heads to use in multi-headed attention.
        attn_pdrop: float
            Drop-out the attention probabilities with this rate.
        weights: dict[str, torch.FloatTensor]
            Keys:
            - 'q_heads.{N}.weight', 'k_heads.{N}.weight', 'v_heads.{N}.weight'
              for N in 0..num_heads-1, each shape (d_key, d_model).
            - 'output_proj.weight' shape (d_model, d_value * num_heads).
        in_features: torch.FloatTensor
            Shape (batch_size, seq_len, d_model).

    Returns:
        torch.FloatTensor shape (batch_size, seq_len, d_model).
    """
    raise NotImplementedError


def run_transformer_block(
    d_model: int,
    num_heads: int,
    d_ff: int,
    attn_pdrop: float,
    residual_pdrop: float,
    weights: dict[str, torch.FloatTensor],
    in_features: torch.FloatTensor,
) -> torch.FloatTensor:
    """Given the weights of a pre-norm Transformer block and input features,
    return the output of running the Transformer block on the input features.

    Pre-norm block:
        z = x + dropout(MultiHeadSelfAttention(RMSNorm(x)))
        y = z + dropout(FFN(RMSNorm(z)))

    Args:
        d_model: int
        num_heads: int
        d_ff: int
        attn_pdrop: float  -- attention dropout rate
        residual_pdrop: float  -- residual/sublayer dropout rate
        weights: dict[str, torch.FloatTensor]
            Keys:
            - 'attn.q_proj.weight', 'attn.k_proj.weight', 'attn.v_proj.weight'
              each shape (num_heads * d_k, d_model)
            - 'attn.output_proj.weight' shape (d_model, d_model)
            - 'ln1.weight', 'ln2.weight' each shape (d_model,)
            - 'ffn.w1.weight' shape (d_ff, d_model)
            - 'ffn.w2.weight' shape (d_model, d_ff)
        in_features: torch.FloatTensor
            Shape (batch_size, sequence_length, d_model).

    Returns:
        FloatTensor of shape (batch_size, sequence_length, d_model).
    """
    raise NotImplementedError


def run_transformer_lm(
    vocab_size: int,
    context_length: int,
    d_model: int,
    num_layers: int,
    num_heads: int,
    d_ff: int,
    attn_pdrop: float,
    residual_pdrop: float,
    weights: dict[str, torch.FloatTensor],
    in_indices: torch.LongTensor,
) -> torch.FloatTensor:
    """Given the weights of a Transformer language model and input indices,
    return the output of running a forward pass on the input indices.

    Architecture:
        1. token_embeddings + position_embeddings -> dropout
        2. num_layers x Transformer block (pre-norm)
        3. Final RMSNorm
        4. Linear projection to vocab_size (weight-tied with token_embeddings)
        -> Returns raw logits (no softmax)

    Args:
        vocab_size, context_length, d_model, num_layers, num_heads, d_ff: int
        attn_pdrop, residual_pdrop: float
        weights: dict[str, torch.FloatTensor]
            Keys:
            - 'token_embeddings.weight'   shape (vocab_size, d_model)
            - 'position_embeddings.weight' shape (context_length, d_model)
            - 'layers.{i}.attn.q_proj.weight', 'layers.{i}.attn.k_proj.weight',
              'layers.{i}.attn.v_proj.weight', 'layers.{i}.attn.output_proj.weight'
            - 'layers.{i}.ln1.weight', 'layers.{i}.ln2.weight'
            - 'layers.{i}.ffn.w1.weight', 'layers.{i}.ffn.w2.weight'
            - 'ln_final.weight'
            - 'lm_head.weight'  (tied to token_embeddings.weight)
        in_indices: torch.LongTensor
            Shape (batch_size, sequence_length), sequence_length <= context_length.

    Returns:
        FloatTensor of shape (batch_size, sequence_length, vocab_size).
    """
    raise NotImplementedError


def run_rmsnorm(
    d_model: int,
    eps: float,
    weights: dict[str, torch.FloatTensor],
    in_features: torch.FloatTensor,
) -> torch.FloatTensor:
    """Given the weights of a RMSNorm affine transform,
    return the output of running RMSNorm on the input features.

    RMSNorm(a_i) = (a_i / RMS(a)) * g_i
    where RMS(a) = sqrt(mean(a^2) + eps)

    Args:
        d_model: int
        eps: float  -- small value for numerical stability (typically 1e-5)
        weights: dict with key 'weight', shape (d_model,)
        in_features: torch.FloatTensor  -- shape (*, d_model)

    Returns:
        FloatTensor of same shape as in_features.
    """
    raise NotImplementedError


def run_gelu(in_features: torch.FloatTensor) -> torch.FloatTensor:
    """Given a tensor of inputs, return the output of applying GELU
    to each element.

    GELU(x) = x * 0.5 * (1 + erf(x / sqrt(2)))

    Args:
        in_features: torch.FloatTensor -- arbitrary shape

    Returns:
        FloatTensor of same shape as in_features.
    """
    raise NotImplementedError


def run_get_batch(
    dataset: npt.NDArray, batch_size: int, context_length: int, device: str
) -> tuple[torch.Tensor, torch.Tensor]:
    """Given a dataset (a 1D numpy array of integers) and a desired batch size and
    context length, sample language modeling input sequences and their corresponding
    labels from the dataset.

    Args:
        dataset: np.array
            1D numpy array of integer token IDs in the dataset.
        batch_size: int
            Desired batch size to sample.
        context_length: int
            Desired context length of each sampled example.
        device: str
            PyTorch device string (e.g., 'cpu' or 'cuda:0').

    Returns:
        Tuple of torch.LongTensors of shape (batch_size, context_length).
        First item: sampled input sequences.
        Second item: corresponding language modeling labels (inputs shifted right by 1).
    """
    raise NotImplementedError


def run_softmax(in_features: torch.FloatTensor, dim: int) -> torch.FloatTensor:
    """Given a tensor of inputs, return the output of softmaxing the given `dim`
    of the input.

    Must be numerically stable: subtract max before computing exp.

    Args:
        in_features: torch.FloatTensor -- arbitrary shape
        dim: int -- dimension to apply softmax over

    Returns:
        FloatTensor of same shape as in_features.
    """
    raise NotImplementedError


def run_cross_entropy(inputs: torch.FloatTensor, targets: torch.LongTensor):
    """Given a tensor of inputs and targets, compute the average cross-entropy
    loss across examples.

    Must be numerically stable: subtract max from logits before computing exp.

    Args:
        inputs: torch.FloatTensor
            Shape (batch_size, num_classes) -- raw unnormalized logits.
        targets: torch.LongTensor
            Shape (batch_size,) -- index of the correct class for each example.

    Returns:
        Scalar tensor: mean cross-entropy loss across the batch.
    """
    raise NotImplementedError


def run_gradient_clipping(parameters: Iterable[torch.nn.Parameter], max_l2_norm: float):
    """Given a set of parameters, clip their combined gradients to have l2 norm
    at most max_l2_norm.

    Compute the global L2 norm across ALL parameters' gradients combined.
    If it exceeds max_l2_norm, scale all gradients down by the same factor.
    Use eps=1e-6 for numerical stability.

    Args:
        parameters: collection of trainable parameters.
        max_l2_norm: a positive value -- the maximum allowed L2 norm.

    Modifies gradients in-place. Returns None.
    """
    raise NotImplementedError


def get_adamw_cls() -> Type[torch.optim.Optimizer]:
    """Return a torch.optim.Optimizer subclass that implements AdamW.

    Your AdamW class __init__ should accept:
        params, lr, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0

    AdamW update rule (t starts at 1):
        m = beta1*m + (1-beta1)*g
        v = beta2*v + (1-beta2)*g^2
        alpha_t = lr * sqrt(1-beta2^t) / (1-beta1^t)
        theta -= alpha_t * m / (sqrt(v) + eps)   # gradient update
        theta -= lr * weight_decay * theta         # weight decay (decoupled)

    Store per-parameter state (m, v, step) in self.state[p].
    """
    raise NotImplementedError


def run_get_lr_cosine_schedule(
    it: int,
    max_learning_rate: float,
    min_learning_rate: float,
    warmup_iters: int,
    cosine_cycle_iters: int,
):
    """Given the parameters of a cosine learning rate decay schedule (with linear
    warmup) and an iteration number, return the learning rate at that iteration.

    Schedule:
        if it < warmup_iters:
            lr = (it / warmup_iters) * max_learning_rate
        elif it <= cosine_cycle_iters:
            lr = min_learning_rate + 0.5*(1 + cos(pi*(it-warmup_iters)/(cosine_cycle_iters-warmup_iters)))
                 * (max_learning_rate - min_learning_rate)
        else:
            lr = min_learning_rate

    Args:
        it: int -- current iteration number
        max_learning_rate: float -- peak learning rate
        min_learning_rate: float -- minimum / final learning rate
        warmup_iters: int -- number of linear warmup steps
        cosine_cycle_iters: int -- total number of cosine annealing steps

    Returns:
        float: learning rate at iteration `it`.
    """
    raise NotImplementedError


def run_save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    iteration: int,
    out: str | os.PathLike | BinaryIO | IO[bytes],
):
    """Serialize model weights, optimizer state, and iteration number to disk.

    Use torch.save() with a dict containing all three. Example structure:
        {'model': model.state_dict(),
         'optimizer': optimizer.state_dict(),
         'iteration': iteration}

    Args:
        model: torch.nn.Module
        optimizer: torch.optim.Optimizer
        iteration: int -- number of training iterations completed so far
        out: file path or file-like object to write to
    """
    raise NotImplementedError


def run_load_checkpoint(
    src: str | os.PathLike | BinaryIO | IO[bytes],
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
):
    """Restore model and optimizer state from a checkpoint file.

    Args:
        src: file path or file-like object to read from
        model: torch.nn.Module -- restore state into this model
        optimizer: torch.optim.Optimizer -- restore state into this optimizer

    Returns:
        int -- the iteration number that was saved in the checkpoint.
    """
    raise NotImplementedError


def get_tokenizer(
    vocab: dict[int, bytes],
    merges: list[tuple[bytes, bytes]],
    special_tokens: Optional[list[str]] = None,
):
    """Given a vocabulary, a list of merges, and a list of special tokens,
    return a BPE tokenizer that uses the provided vocab, merges, and special tokens.

    Args:
        vocab: dict[int, bytes] -- token ID -> bytes
        merges: list[tuple[bytes, bytes]] -- BPE merges in creation order
        special_tokens: Optional[list[str]] -- special tokens (never split by BPE)

    Returns:
        A Tokenizer instance (your implementation in helper_classes.py).
    """
    return helper_classes.Tokenizer(vocab=vocab, merges=merges, special_tokens=special_tokens)


def run_train_bpe(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str],
):
    """Train a byte-level BPE tokenizer on the text file at input_path.

    Algorithm:
        1. Initialize vocab with all 256 byte values (ids 0-255) + special tokens.
        2. Pre-tokenize the corpus using the GPT-2 regex pattern.
        3. Count the frequency of every adjacent byte pair across all pre-tokens.
        4. Repeatedly:
            a. Find the most frequent pair. Break ties by taking the
               lexicographically GREATER pair.
            b. Merge all occurrences of that pair into a new token.
            c. Add the new token to the vocab.
            d. Record the merge.
           Until vocab reaches vocab_size or no pairs remain.

    Efficiency requirement:
        After each merge, only update counts for pairs that OVERLAP with
        the merged pair. Use a heap to pick the best pair in O(log n).

    Args:
        input_path: str | os.PathLike -- path to training text file (UTF-8)
        vocab_size: int -- target vocabulary size
        special_tokens: list[str] -- e.g. ['<|endoftext|>']
                        Added to vocab but never merged into or from.

    Returns:
        vocab:  dict[int, bytes] -- token_id -> token bytes
        merges: list[tuple[bytes, bytes]] -- merge operations in order
    """
    raise NotImplementedError