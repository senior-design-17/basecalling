"""
Forward and Backward Score Computation for CRF Decoding

This module implements the forward and backward pass computation for CRF scores,
matching Dorado's implementation in CPUDecoder.cpp.
"""

import torch
import numpy as np
from typing import Tuple


def scan(
    Ms: torch.Tensor,
    fixed_stay_score: float,
    idx: torch.Tensor,
    v0: torch.Tensor
) -> torch.Tensor:
    """
    Scan operation for forward/backward pass computation.
    Operates in TNC (Time, Batch, States) format.
    
    Args:
        Ms: Transition scores of shape [T, N, num_states, n_base]
        fixed_stay_score: Fixed score for stay transitions
        idx: Indices tensor for state transitions
        v0: Initial values of shape [N, num_states]
    
    Returns:
        Alpha tensor of shape [T + 1, N, num_states]
    """
    T = Ms.shape[0]
    N = Ms.shape[1]
    C = Ms.shape[2]  # num_states
    
    alpha = torch.full((T + 1, N, C), -1e38, dtype=Ms.dtype, device=Ms.device)
    alpha[0] = v0
    
    for t in range(T):
        # Scored steps: add transition scores to previous states
        scored_steps = alpha[t, :, idx] + Ms[t]
        
        # Scored stay: add fixed stay score to all previous states
        scored_stay = (alpha[t] + fixed_stay_score).unsqueeze(-1)
        
        # Concatenate stay and steps
        scored_transitions = torch.cat([scored_stay, scored_steps], dim=-1)
        
        # Log-sum-exp over all transitions
        alpha[t + 1] = torch.logsumexp(scored_transitions, dim=-1)
    
    return alpha


def forward_scores(scores_TNC: torch.Tensor, fixed_stay_score: float) -> torch.Tensor:
    """
    Compute forward log probabilities.
    
    Args:
        scores_TNC: CRF scores of shape [T, N, C] where C = 4^(state_len + 1)
        fixed_stay_score: Fixed score for stay transitions
    
    Returns:
        Forward scores of shape [T + 1, N, num_states]
    """
    T = scores_TNC.shape[0]  # Signal length
    N = scores_TNC.shape[1]   # Batch size
    C = scores_TNC.shape[2]   # 4^state_len * 4 = 4^(state_len + 1)
    
    n_base = 4
    state_len = int(np.log(C) / np.log(n_base) - 1)
    
    # Reshape to [T, N, num_states, n_base]
    Ms = scores_TNC.reshape(T, N, -1, n_base)
    
    # Number of states per timestep
    num_states = int(n_base ** state_len)
    
    # Initial values (zeros)
    v0 = torch.zeros((N, num_states), dtype=scores_TNC.dtype, device=scores_TNC.device)
    
    # Indices: for each state, the indices of the 4 states that could precede it via a step
    idx = torch.arange(num_states, device=scores_TNC.device)
    idx = idx.repeat_interleave(n_base).reshape(n_base, -1).t().contiguous()
    
    return scan(Ms, fixed_stay_score, idx, v0)


def backward_scores(scores_TNC: torch.Tensor, fixed_stay_score: float) -> torch.Tensor:
    """
    Compute backward log probabilities.
    
    Args:
        scores_TNC: CRF scores of shape [T, N, C] where C = 4^(state_len + 1)
        fixed_stay_score: Fixed score for stay transitions
    
    Returns:
        Backward scores of shape [T + 1, N, num_states]
    """
    N = scores_TNC.shape[1]   # Batch size
    C = scores_TNC.shape[2]   # 4^state_len * 4 = 4^(state_len + 1)
    
    n_base = 4
    state_len = int(np.log(C) / np.log(n_base) - 1)
    
    # Number of states per timestep
    num_states = int(n_base ** state_len)
    
    # Final values (zeros)
    vT = torch.zeros((N, num_states), dtype=scores_TNC.dtype, device=scores_TNC.device)
    
    # Indices for successor states
    idx = torch.arange(num_states, device=scores_TNC.device)
    idx = idx.repeat_interleave(n_base).reshape(n_base, -1).t().contiguous()
    idx_T_flat = idx.flatten().argsort()
    
    # Reorder scores using advanced indexing
    # scores_TNC is [T, N, C], idx_T_flat is [C] - reorder last dimension
    T = scores_TNC.shape[0]
    N = scores_TNC.shape[1]
    Ms_T = scores_TNC[:, :, idx_T_flat]  # [T, N, C] with reordered last dim
    
    # Reshape to [T, N, num_states, n_base] for scan
    Ms_T = Ms_T.reshape(T, N, num_states, n_base)
    
    # For each state, indices of states that could succeed it
    idx_T_successor = idx >> 2  # Right shift by 2 (equivalent to dividing by 4)
    
    # Run scan in reverse, then flip back
    bwd = scan(Ms_T.flip(0), fixed_stay_score, idx_T_successor.long(), vT).flip(0)
    
    return bwd


def compute_posterior_probabilities(
    forward_scores: torch.Tensor,
    backward_scores: torch.Tensor
) -> torch.Tensor:
    """
    Compute posterior probabilities from forward and backward scores.
    
    Args:
        forward_scores: Forward scores of shape [T + 1, N, num_states]
        backward_scores: Backward scores of shape [T + 1, N, num_states]
    
    Returns:
        Posterior probabilities of shape [T + 1, N, num_states]
    """
    # Posterior = softmax(forward + backward)
    posts = torch.softmax(forward_scores + backward_scores, dim=-1)
    return posts


def compute_forward_backward_posterior(
    scores_TNC: torch.Tensor,
    fixed_stay_score: float = 2.0
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute forward scores, backward scores, and posterior probabilities.
    
    Args:
        scores_TNC: CRF scores of shape [T, N, C]
        fixed_stay_score: Fixed stay score (default: 2.0)
    
    Returns:
        Tuple of (forward_scores, backward_scores, posterior_probs)
        All have shape [T + 1, N, num_states]
    """
    fwd = forward_scores(scores_TNC, fixed_stay_score)
    bwd = backward_scores(scores_TNC, fixed_stay_score)
    posts = compute_posterior_probabilities(fwd, bwd)
    
    return fwd, bwd, posts

