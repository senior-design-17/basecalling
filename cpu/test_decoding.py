"""
Test script for beam search decoding implementation.

This script tests the decoding pipeline:
1. Generate synthetic CRF scores
2. Compute forward/backward scores and posteriors
3. Perform beam search decoding
4. Verify output format and basic properties
"""

import torch
import numpy as np
from pathlib import Path
import sys

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent))

from forward_backward import compute_forward_backward_posterior
from beam_search_decode import beam_search_decode


def generate_synthetic_crf_scores(
    num_blocks: int = 100,
    num_states: int = 256,  # 4^4 states
    batch_size: int = 1,
    seed: int = 42
) -> torch.Tensor:
    """
    Generate synthetic CRF scores for testing.
    
    Args:
        num_blocks: Number of time steps
        num_states: Number of states (should be 4^state_len)
        batch_size: Batch size
        seed: Random seed
    
    Returns:
        CRF scores tensor of shape [T, N, C] where C = num_states * 4
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    # C = num_states * 4 = 4^(state_len + 1)
    num_transitions = num_states * 4
    
    # Generate scores in range [-2, 2] to keep them reasonable
    scores = torch.randn(num_blocks, batch_size, num_transitions) * 0.5
    
    # Clamp to [-5, 5] range (matching model output range)
    scores = torch.clamp(scores, -5.0, 5.0)
    
    return scores


def test_forward_backward_computation():
    """Test forward and backward score computation."""
    print("=" * 60)
    print("Test 1: Forward and Backward Score Computation")
    print("=" * 60)
    
    num_blocks = 50
    num_states = 256  # 4^4
    batch_size = 2
    
    scores = generate_synthetic_crf_scores(num_blocks, num_states, batch_size)
    
    print(f"Input scores shape: {scores.shape}")
    print(f"Expected: [T={num_blocks}, N={batch_size}, C={num_states * 4}]")
    
    fixed_stay_score = 2.0
    fwd, bwd, posts = compute_forward_backward_posterior(scores, fixed_stay_score)
    
    print(f"\nForward scores shape: {fwd.shape}")
    print(f"Expected: [T+1={num_blocks + 1}, N={batch_size}, num_states={num_states}]")
    assert fwd.shape == (num_blocks + 1, batch_size, num_states), \
        f"Forward scores shape mismatch: {fwd.shape}"
    
    print(f"Backward scores shape: {bwd.shape}")
    assert bwd.shape == (num_blocks + 1, batch_size, num_states), \
        f"Backward scores shape mismatch: {bwd.shape}"
    
    print(f"Posterior probabilities shape: {posts.shape}")
    assert posts.shape == (num_blocks + 1, batch_size, num_states), \
        f"Posterior probabilities shape mismatch: {posts.shape}"
    
    # Check that posteriors sum to 1
    post_sums = posts.sum(dim=-1)
    print(f"\nPosterior probability sums (should be ~1.0):")
    print(f"  Min: {post_sums.min().item():.6f}")
    print(f"  Max: {post_sums.max().item():.6f}")
    print(f"  Mean: {post_sums.mean().item():.6f}")
    
    assert torch.allclose(post_sums, torch.ones_like(post_sums), atol=1e-5), \
        "Posterior probabilities do not sum to 1"
    
    # Check for NaN/Inf
    assert not torch.isnan(fwd).any(), "Forward scores contain NaN"
    assert not torch.isnan(bwd).any(), "Backward scores contain NaN"
    assert not torch.isnan(posts).any(), "Posterior probabilities contain NaN"
    assert not torch.isinf(fwd).any(), "Forward scores contain Inf"
    assert not torch.isinf(bwd).any(), "Backward scores contain Inf"
    
    print("✓ Forward and backward computation test passed!")
    return fwd, bwd, posts, scores


def test_beam_search_decode():
    """Test beam search decoding."""
    print("\n" + "=" * 60)
    print("Test 2: Beam Search Decoding")
    print("=" * 60)
    
    num_blocks = 100
    num_states = 256  # 4^4
    batch_size = 1
    
    # Generate scores
    scores = generate_synthetic_crf_scores(num_blocks, num_states, batch_size)
    
    # Compute forward/backward/posterior
    fixed_stay_score = 2.0
    fwd, bwd, posts = compute_forward_backward_posterior(scores, fixed_stay_score)
    
    # Prepare inputs for beam search
    # Scores need to be [num_blocks, num_transitions]
    scores_2d = scores[:, 0, :]  # Take first batch, keep time dimension
    
    # Backward guides: [num_blocks + 1, num_states]
    back_guides = bwd[:, 0, :]  # Take first batch, keep time dimension
    
    # Posts: [num_blocks + 1, num_states]
    posts_2d = posts[:, 0, :]  # Take first batch, keep time dimension
    
    print(f"Input shapes:")
    print(f"  Scores: {scores_2d.shape}")
    print(f"  Back guides: {back_guides.shape}")
    print(f"  Posts: {posts_2d.shape}")
    
    # Run beam search
    sequence, qstring, moves = beam_search_decode(
        scores_2d,
        back_guides,
        posts_2d,
        max_beam_width=32,
        beam_cut=100.0,
        fixed_stay_score=fixed_stay_score,
        q_shift=0.0,
        q_scale=1.0
    )
    
    print(f"\nDecoding results:")
    print(f"  Sequence length: {len(sequence)}")
    print(f"  Quality string length: {len(qstring)}")
    print(f"  Moves length: {len(moves)}")
    
    assert len(sequence) == len(qstring), \
        "Sequence and quality string lengths don't match"
    assert len(sequence) == sum(moves), \
        "Sequence length doesn't match sum of moves"
    assert len(moves) == num_blocks, \
        f"Moves length mismatch: got {len(moves)}, expected {num_blocks}"
    
    # Check sequence contains only valid bases
    valid_bases = set('ACGT')
    assert all(c in valid_bases for c in sequence), \
        f"Sequence contains invalid bases: {set(sequence) - valid_bases}"
    
    # Check quality string is valid ASCII
    assert all(33 <= ord(c) <= 126 for c in qstring), \
        "Quality string contains invalid ASCII characters"
    
    # Moves should be 0 or 1
    assert all(m == 0 or m == 1 for m in moves), \
        f"Moves contains invalid values: {set(moves)}"
    
    # First move should always be 1
    assert moves[0] == 1, "First move should always be 1"
    
    print(f"\n  Sequence preview (first 50 bases): {sequence[:50]}")
    print(f"  Quality preview (first 50): {qstring[:50]}")
    print(f"  Moves summary: {sum(moves)} emits, {len(moves) - sum(moves)} stays")
    
    print("✓ Beam search decoding test passed!")
    return sequence, qstring, moves


def test_decoding_consistency():
    """Test that decoding is consistent across runs with same inputs."""
    print("\n" + "=" * 60)
    print("Test 3: Decoding Consistency")
    print("=" * 60)
    
    num_blocks = 50
    num_states = 256
    batch_size = 1
    
    scores = generate_synthetic_crf_scores(num_blocks, num_states, batch_size, seed=123)
    fixed_stay_score = 2.0
    fwd, bwd, posts = compute_forward_backward_posterior(scores, fixed_stay_score)
    
    scores_2d = scores[:, 0, :]  # Take first batch, keep time dimension
    back_guides = bwd[:, 0, :]  # Take first batch, keep time dimension
    posts_2d = posts[:, 0, :]  # Take first batch, keep time dimension
    
    # Run decoding twice
    seq1, qstr1, moves1 = beam_search_decode(
        scores_2d, back_guides, posts_2d,
        max_beam_width=32, beam_cut=100.0, fixed_stay_score=fixed_stay_score
    )
    
    seq2, qstr2, moves2 = beam_search_decode(
        scores_2d, back_guides, posts_2d,
        max_beam_width=32, beam_cut=100.0, fixed_stay_score=fixed_stay_score
    )
    
    # Results should be identical
    assert seq1 == seq2, "Sequences differ between runs"
    assert qstr1 == qstr2, "Quality strings differ between runs"
    assert moves1 == moves2, "Moves differ between runs"
    
    print("✓ Decoding is consistent across runs")
    print(f"  Sequence length: {len(seq1)}")
    print(f"  Unique bases: {set(seq1)}")


def test_beam_width_effect():
    """Test that different beam widths produce different (but valid) results."""
    print("\n" + "=" * 60)
    print("Test 4: Beam Width Effect")
    print("=" * 60)
    
    num_blocks = 50
    num_states = 256
    batch_size = 1
    
    scores = generate_synthetic_crf_scores(num_blocks, num_states, batch_size, seed=456)
    fixed_stay_score = 2.0
    fwd, bwd, posts = compute_forward_backward_posterior(scores, fixed_stay_score)
    
    scores_2d = scores[:, 0, :]  # Take first batch, keep time dimension
    back_guides = bwd[:, 0, :]  # Take first batch, keep time dimension
    posts_2d = posts[:, 0, :]  # Take first batch, keep time dimension
    
    # Test different beam widths
    beam_widths = [8, 16, 32]
    results = {}
    
    for beam_width in beam_widths:
        seq, qstr, moves = beam_search_decode(
            scores_2d, back_guides, posts_2d,
            max_beam_width=beam_width, beam_cut=100.0, fixed_stay_score=fixed_stay_score
        )
        results[beam_width] = {
            'sequence': seq,
            'length': len(seq),
            'moves': moves
        }
        print(f"  Beam width {beam_width}: sequence length {len(seq)}")
    
    # All should be valid
    for beam_width, result in results.items():
        assert result['length'] > 0, f"Zero-length sequence for beam width {beam_width}"
        assert len(set(result['sequence']) - set('ACGT')) == 0, \
            f"Invalid bases for beam width {beam_width}"
    
    print("✓ All beam widths produce valid results")


def test_edge_cases():
    """Test edge cases."""
    print("\n" + "=" * 60)
    print("Test 5: Edge Cases")
    print("=" * 60)
    
    # Test with very few blocks
    print("  Testing with minimal blocks (5)...")
    scores = generate_synthetic_crf_scores(num_blocks=5, num_states=256, batch_size=1, seed=789)
    fwd, bwd, posts = compute_forward_backward_posterior(scores, 2.0)
    seq, qstr, moves = beam_search_decode(
        scores[:, 0, :], bwd[:, 0, :], posts[:, 0, :],
        max_beam_width=8, beam_cut=100.0, fixed_stay_score=2.0
    )
    assert len(seq) > 0, "Should produce sequence even with few blocks"
    print("    ✓ Minimal blocks test passed")
    
    # Test with very small beam width
    print("  Testing with small beam width (4)...")
    seq2, qstr2, moves2 = beam_search_decode(
        scores[:, 0, :], bwd[:, 0, :], posts[:, 0, :],
        max_beam_width=4, beam_cut=100.0, fixed_stay_score=2.0
    )
    assert len(seq2) > 0, "Should produce sequence even with small beam"
    print("    ✓ Small beam width test passed")
    
    print("✓ Edge cases handled correctly")


def main():
    """Run all tests."""
    print("=" * 60)
    print("Beam Search Decoding Test Suite")
    print("=" * 60)
    print()
    
    try:
        # Test 1: Forward/backward computation
        fwd, bwd, posts, scores = test_forward_backward_computation()
        
        # Test 2: Beam search decoding
        seq, qstr, moves = test_beam_search_decode()
        
        # Test 3: Consistency
        test_decoding_consistency()
        
        # Test 4: Beam width effect
        test_beam_width_effect()
        
        # Test 5: Edge cases
        test_edge_cases()
        
        print("\n" + "=" * 60)
        print("All tests passed! ✓")
        print("=" * 60)
        
    except AssertionError as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

