"""
Integration script for CRF layer and decoder.

This script demonstrates the complete pipeline:
1. Generate synthetic LSTM features (mimicking FPGA output)
2. Process through CRF layer to get emission scores
3. Compute forward/backward scores and posterior probabilities
4. Perform beam search decoding
5. Output ACTG sequence and quality scores
"""

import torch
import numpy as np
from pathlib import Path
import sys

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent))

from crf_layer_implementation import CRFEncoder, load_crf_encoder_with_weights
from forward_backward import compute_forward_backward_posterior
from beam_search_decode import beam_search_decode


def generate_synthetic_lstm_features(
    num_time_steps: int = 1667,
    feature_dim: int = 384,
    batch_size: int = 1,
    seed: int = 42
) -> torch.Tensor:
    """
    Generate synthetic LSTM features that mimic FPGA output.
    
    Args:
        num_time_steps: Number of time steps (default: 1667, typical for 10k signal samples)
        feature_dim: Feature dimension (default: 384, LSTM output size)
        batch_size: Batch size (default: 1)
        seed: Random seed for reproducibility
    
    Returns:
        Synthetic LSTM features tensor of shape [batch_size, num_time_steps, feature_dim]
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    # Generate features that roughly match LSTM output distribution
    # LSTM outputs are typically bounded and have some structure
    features = torch.randn(batch_size, num_time_steps, feature_dim) * 0.5
    
    # Apply tanh-like saturation to mimic LSTM activations
    features = torch.tanh(features)
    
    return features


def run_crf_decoding_pipeline(
    lstm_features: torch.Tensor,
    crf_encoder: CRFEncoder,
    fixed_stay_score: float = 2.0,
    max_beam_width: int = 32,
    beam_cut: float = 100.0,
    q_shift: float = 0.0,
    q_scale: float = 1.0
) -> tuple:
    """
    Run the complete CRF decoding pipeline.
    
    Args:
        lstm_features: LSTM features of shape [N, T, 384]
        crf_encoder: CRF encoder module
        fixed_stay_score: Fixed score for stay transitions (default: 2.0)
        max_beam_width: Maximum beam width for decoding (default: 32)
        beam_cut: Beam cutoff parameter (default: 100.0)
        q_shift: Quality score shift (default: 0.0)
        q_scale: Quality score scale (default: 1.0)
    
    Returns:
        Tuple of (sequence, quality_string, moves)
    """
    batch_size, num_time_steps, feature_dim = lstm_features.shape
    
    print(f"\n{'='*60}")
    print(f"Step 1: CRF Layer Forward Pass")
    print(f"{'='*60}")
    print(f"Input shape: {lstm_features.shape}")
    
    # Step 1: Forward pass through CRF layer
    crf_encoder.eval()
    with torch.no_grad():
        crf_scores = crf_encoder(lstm_features)  # [N, T, 1024]
    
    print(f"CRF scores shape: {crf_scores.shape}")
    print(f"CRF scores range: [{crf_scores.min().item():.4f}, {crf_scores.max().item():.4f}]")
    
    # Step 2: Reshape for forward/backward computation
    # Forward/backward expects [T, N, C] format
    crf_scores_TNC = crf_scores.permute(1, 0, 2)  # [T, N, 1024]
    
    print(f"\n{'='*60}")
    print(f"Step 2: Forward/Backward Computation")
    print(f"{'='*60}")
    print(f"Reshaped scores shape: {crf_scores_TNC.shape}")
    
    # Step 3: Compute forward, backward, and posterior probabilities
    forward_scores, backward_scores, posterior_probs = compute_forward_backward_posterior(
        crf_scores_TNC, fixed_stay_score
    )
    
    print(f"Forward scores shape: {forward_scores.shape}")
    print(f"Backward scores shape: {backward_scores.shape}")
    print(f"Posterior probabilities shape: {posterior_probs.shape}")
    
    # Verify posteriors sum to 1
    post_sums = posterior_probs.sum(dim=-1)
    print(f"Posterior probability sums: min={post_sums.min().item():.6f}, "
          f"max={post_sums.max().item():.6f}, mean={post_sums.mean().item():.6f}")
    
    # Step 4: Prepare inputs for beam search decoding
    # Decoder expects 2D tensors (single batch)
    # For batch_size > 1, we'll process each batch element separately
    results = []
    
    print(f"\n{'='*60}")
    print(f"Step 3: Beam Search Decoding")
    print(f"{'='*60}")
    
    for batch_idx in range(batch_size):
        # Extract single batch element
        scores_2d = crf_scores_TNC[:, batch_idx, :]  # [T, 1024]
        back_guides_2d = backward_scores[:, batch_idx, :]  # [T+1, num_states]
        posts_2d = posterior_probs[:, batch_idx, :]  # [T+1, num_states]
        
        print(f"\nDecoding batch element {batch_idx + 1}/{batch_size}...")
        print(f"  Scores shape: {scores_2d.shape}")
        print(f"  Back guides shape: {back_guides_2d.shape}")
        print(f"  Posts shape: {posts_2d.shape}")
        
        # Step 5: Perform beam search decoding
        sequence, quality_string, moves = beam_search_decode(
            scores_2d,
            back_guides_2d,
            posts_2d,
            max_beam_width=max_beam_width,
            beam_cut=beam_cut,
            fixed_stay_score=fixed_stay_score,
            q_shift=q_shift,
            q_scale=q_scale
        )
        
        results.append((sequence, quality_string, moves))
        
        print(f"\n  Decoding complete!")
        print(f"  Sequence length: {len(sequence)}")
        print(f"  Quality string length: {len(quality_string)}")
        print(f"  Number of moves: {len(moves)}")
        print(f"  Emits: {sum(moves)}, Stays: {len(moves) - sum(moves)}")
    
    return results


def format_output(
    sequence: str,
    quality_string: str,
    read_id: str = "synthetic_read",
    max_display_length: int = 80
) -> str:
    """
    Format output as FASTQ-like string for display.
    
    Args:
        sequence: DNA sequence string (ACGT)
        quality_string: Quality score string
        read_id: Read identifier
        max_display_length: Maximum sequence length to display in preview
    
    Returns:
        Formatted string
    """
    lines = [
        f"Read ID: {read_id}",
        f"Sequence length: {len(sequence)}",
        f"Sequence (first {min(max_display_length, len(sequence))} bases):",
        f"  {sequence[:max_display_length]}{'...' if len(sequence) > max_display_length else ''}",
        f"Quality string (first {min(max_display_length, len(quality_string))} chars):",
        f"  {quality_string[:max_display_length]}{'...' if len(quality_string) > max_display_length else ''}",
        "",
        "FASTQ format:",
        f"@{read_id}",
        sequence,
        "+",
        quality_string
    ]
    return "\n".join(lines)


def main():
    """Main integration test function."""
    print("="*60)
    print("CRF Layer and Decoder Integration Test")
    print("="*60)
    
    # Configuration
    num_time_steps = 1667  # Typical for 10k signal samples (reduced by stride=6)
    feature_dim = 384      # LSTM output dimension
    batch_size = 1         # Number of sequences to process
    seed = 42              # Random seed
    
    # Decoding parameters
    fixed_stay_score = 2.0
    max_beam_width = 32
    beam_cut = 100.0
    q_shift = 0.0
    q_scale = 1.0
    
    # Step 0: Generate synthetic LSTM features
    print(f"\n{'='*60}")
    print(f"Step 0: Generate Synthetic LSTM Features")
    print(f"{'='*60}")
    print(f"Time steps: {num_time_steps}")
    print(f"Feature dimension: {feature_dim}")
    print(f"Batch size: {batch_size}")
    
    lstm_features = generate_synthetic_lstm_features(
        num_time_steps=num_time_steps,
        feature_dim=feature_dim,
        batch_size=batch_size,
        seed=seed
    )
    print(f"Generated LSTM features shape: {lstm_features.shape}")
    print(f"LSTM features range: [{lstm_features.min().item():.4f}, {lstm_features.max().item():.4f}]")
    
    # Step 1: Create CRF encoder
    print(f"\n{'='*60}")
    print(f"Initializing CRF Encoder")
    print(f"{'='*60}")
    
    # Try to load from model directory, otherwise use random weights
    model_dir = Path(__file__).parent.parent / "model" / "dna_r10.4.1_e8.2_400bps_hac@v5.2.0"
    
    if model_dir.exists():
        print(f"Loading CRF encoder from model directory: {model_dir}")
        try:
            crf_encoder = load_crf_encoder_with_weights(str(model_dir))
            print("✓ Successfully loaded CRF encoder with weights")
        except Exception as e:
            print(f"Warning: Could not load weights ({e})")
            print("  Using randomly initialized weights instead")
            crf_encoder = CRFEncoder(
                insize=384,
                n_base=4,
                state_len=4,
                bias=False,
                use_decomposition=False,
                clamp_active=True
            )
    else:
        print(f"Model directory not found: {model_dir}")
        print("  Using randomly initialized weights")
        crf_encoder = CRFEncoder(
            insize=384,
            n_base=4,
            state_len=4,
            bias=False,
            use_decomposition=False,
            clamp_active=True
        )
    
    # Step 2: Run decoding pipeline
    results = run_crf_decoding_pipeline(
        lstm_features,
        crf_encoder,
        fixed_stay_score=fixed_stay_score,
        max_beam_width=max_beam_width,
        beam_cut=beam_cut,
        q_shift=q_shift,
        q_scale=q_scale
    )
    
    # Step 3: Display results
    print(f"\n{'='*60}")
    print(f"Final Output: ACTG Sequences and Quality Scores")
    print(f"{'='*60}")
    
    for batch_idx, (sequence, quality_string, moves) in enumerate(results):
        print(f"\n{format_output(sequence, quality_string, f'synthetic_read_{batch_idx + 1}')}")
        
        # Additional statistics
        print(f"\nStatistics:")
        print(f"  Sequence length: {len(sequence)}")
        print(f"  Base composition: A={sequence.count('A')}, C={sequence.count('C')}, "
              f"G={sequence.count('G')}, T={sequence.count('T')}")
        
        # Quality score statistics
        quality_scores = [ord(q) - 33 for q in quality_string]  # Convert to Phred scores
        print(f"  Quality scores: min={min(quality_scores)}, max={max(quality_scores)}, "
              f"mean={sum(quality_scores)/len(quality_scores):.2f}")
        
        print(f"  Moves: {len(moves)} total, {sum(moves)} emits, {len(moves) - sum(moves)} stays")
    
    print(f"\n{'='*60}")
    print(f"Integration test complete! ✓")
    print(f"{'='*60}")
    
    return results


if __name__ == "__main__":
    main()

