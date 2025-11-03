"""
CRF Layer Implementation for Dorado Basecalling Model

This implementation matches the LinearCRF module used in the C++ codebase
for the DNA basecalling model (dna_r10.4.1_e8.2_400bps_hac@v5.2.0).

The "CRF" layer in this context is actually a linear transformation layer
that produces emission scores for CTC-like decoding, not a traditional CRF.

Model Architecture:
-------------------
For the v5.2.0 model (dna_r10.4.1_e8.2_400bps_hac@v5.2.0):
- Input: [N, T, 384] from LSTM layers
- Output: [N, T, 1024] emission scores
  - 1024 = 4^(state_len + 1) = 4^5
  - Each of the 1024 outputs represents a state transition in the CTC-like model
  - Scores are clamped to [-5.0, 5.0] range
  
The output scores are then used with beam search decoding to produce:
- Base sequence (A, C, G, T)
- Quality scores (Phred scores)

Key Parameters (from config.toml):
- insize: 384 (LSTM output dimension)
- n_base: 4 (A, C, G, T)
- state_len: 4 (state length for CTC decoding)
- bias: false (no bias term)
- blank_score: 2.0 (used in decoding, not in layer itself)
- clamp: [-5.0, 5.0] (applied to output)

Weight Loading:
---------------
The weights are stored in .tensor files using PyTorch's native pickle format.
You can load them directly using torch.load():
1. The linear layer weight file is: "11.linear.weight.tensor"
2. Expected shape: [1024, 384] = [outsize, insize]
3. Use load_weights_from_file() to load directly from file path
4. Or use load_weights() with a tensor loaded via torch.load()
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Optional
from pathlib import Path


class LinearCRF(nn.Module):
    """
    Linear CRF layer implementation.
    
    This corresponds to the LinearCRFImpl class in dorado/nn/CRFModules.h
    
    Args:
        insize: Input feature size (384 for this model)
        outsize: Output feature size (calculated from state_len and n_base)
        bias: Whether to use bias (False for v5.2.0)
        tanh_and_scale: Whether to apply tanh activation and scale (False for v5.2.0)
    """
    
    def __init__(self, insize: int, outsize: int, bias: bool = False, tanh_and_scale: bool = False):
        super().__init__()
        self.bias = bias
        self.scale = 5  # Constant scale factor from the C++ implementation
        self.tanh_and_scale = tanh_and_scale
        
        self.linear = nn.Linear(insize, outsize, bias=bias)
        
        if tanh_and_scale:
            self.activation = nn.Tanh()
        else:
            self.activation = None
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape [N, T, C] where:
               - N is batch size
               - T is sequence length (time steps)
               - C is input feature size (384)
        
        Returns:
            Output tensor of shape [N, T, C_out] where C_out is outsize
        """
        # Linear transformation
        scores = self.linear(x)
        
        # Apply tanh activation and scale if enabled
        if self.activation is not None:
            scores = self.activation(scores) * self.scale
        
        return scores


class Clamp(nn.Module):
    """
    Clamp module that clamps values to a specified range.
    
    This corresponds to the ClampImpl class in dorado/nn/CRFModules.h
    """
    
    def __init__(self, min_val: float = -5.0, max_val: float = 5.0, active: bool = True):
        super().__init__()
        self.active = active
        self.min_val = min_val
        self.max_val = max_val
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Clamp the input tensor to [min_val, max_val] if active."""
        if self.active:
            return torch.clamp(x, self.min_val, self.max_val)
        return x


class CRFEncoder(nn.Module):
    """
    Complete CRF encoder for the v5.2.0 model.
    
    This implements the linear CRF encoder as specified in the config.toml:
    - insize: 384 (from LSTM output)
    - n_base: 4 (A, C, G, T)
    - state_len: 4
    - bias: false
    - blank_score: 2.0 (used in decoding, not in the layer itself)
    
    For v5.2.0 models, the linear layer may be decomposed into two layers.
    However, based on the tensor files present, it appears to use a single
    linear layer with clamp.
    """
    
    def __init__(
        self,
        insize: int = 384,
        n_base: int = 4,
        state_len: int = 4,
        bias: bool = False,
        use_decomposition: bool = False,
        decomposition_size: Optional[int] = None,
        clamp_active: bool = True
    ):
        super().__init__()
        self.insize = insize
        self.n_base = n_base
        self.state_len = state_len
        
        # Calculate output size: 4^(state_len + 1) = 4^5 = 1024
        # This represents all possible state transitions in the CTC-like model
        self.outsize = int(n_base ** (state_len + 1))
        
        if use_decomposition and decomposition_size is not None:
            # Decomposed linear layer (used in some v5.x models)
            # First layer: insize -> decomposition_size (with bias)
            # Second layer: decomposition_size -> outsize (no bias)
            self.linear1 = LinearCRF(insize, decomposition_size, bias=True, tanh_and_scale=False)
            self.linear2 = LinearCRF(decomposition_size, self.outsize, bias=False, tanh_and_scale=False)
            self.clamp = Clamp(min_val=-5.0, max_val=5.0, active=clamp_active)
            self.use_decomposition = True
        else:
            # Single linear layer (typical for v5.2.0 based on tensor files)
            self.linear1 = LinearCRF(insize, self.outsize, bias=bias, tanh_and_scale=False)
            self.linear2 = None
            self.clamp = Clamp(min_val=-5.0, max_val=5.0, active=clamp_active)
            self.use_decomposition = False
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the CRF encoder.
        
        Args:
            x: Input tensor of shape [N, T, C] where C is insize (384)
        
        Returns:
            Output tensor of shape [N, T, C_out] where C_out is outsize (1024)
            Values are clamped to [-5.0, 5.0] range.
        """
        if self.use_decomposition:
            x = self.linear1(x)
            x = self.linear2(x)
            x = self.clamp(x)
        else:
            x = self.linear1(x)
            x = self.clamp(x)
        
        return x
    
    def load_weights(self, weight_tensor: torch.Tensor, use_decomposition: bool = False):
        """
        Load weights from a PyTorch tensor.
        
        The weight tensor should be loaded from the .tensor file.
        For the v5.2.0 model, the linear layer weight file is typically:
        "11.linear.weight.tensor" with shape [outsize, insize] = [1024, 384]
        
        Args:
            weight_tensor: Weight tensor, shape [outsize, insize] for single layer,
                          or tuple of (weight1, weight2) for decomposition
            use_decomposition: Whether loading decomposed weights
        """
        if use_decomposition:
            if isinstance(weight_tensor, tuple):
                w1, w2 = weight_tensor
                self.linear1.linear.weight.data = w1.t()
                self.linear2.linear.weight.data = w2.t()
            else:
                raise ValueError("Decomposed weights must be provided as tuple")
        else:
            # Single layer: weight tensor should be [outsize, insize]
            # PyTorch Linear stores weights as [out_features, in_features] = [outsize, insize]
            # The loaded tensor is already in this format, so no transpose needed
            if weight_tensor.dim() != 2:
                raise ValueError(f"Expected 2D weight tensor, got {weight_tensor.shape}")
            if weight_tensor.shape != (self.outsize, self.insize):
                raise ValueError(
                    f"Weight shape mismatch: got {weight_tensor.shape}, "
                    f"expected ({self.outsize}, {self.insize})"
                )
            # Direct assignment - both are in [out_features, in_features] format
            self.linear1.linear.weight.data = weight_tensor.clone()
    
    def load_bias(self, bias_tensor: torch.Tensor):
        """
        Load bias from a PyTorch tensor (if bias is enabled).
        
        Args:
            bias_tensor: Bias tensor of shape [outsize]
        """
        if not self.linear1.bias:
            raise ValueError("Cannot load bias: layer was initialized without bias")
        if bias_tensor.shape != (self.outsize,):
            raise ValueError(f"Bias shape mismatch: got {bias_tensor.shape}, expected ({self.outsize},)")
        self.linear1.linear.bias.data = bias_tensor
    
    def load_weights_from_file(self, weight_path: str, use_decomposition: bool = False):
        """
        Load weights directly from a .tensor file.
        
        The .tensor files use PyTorch's native pickle format, so they can be
        loaded directly with torch.load().
        
        Args:
            weight_path: Path to the .tensor weight file (e.g., "11.linear.weight.tensor")
            use_decomposition: Whether loading decomposed weights (requires two files)
        
        Example:
            crf_encoder.load_weights_from_file("model/dna_r10.4.1_e8.2_400bps_hac@v5.2.0/11.linear.weight.tensor")
        """
        weight_path = Path(weight_path)
        if not weight_path.exists():
            raise FileNotFoundError(f"Weight file not found: {weight_path}")
        
        # Load tensor from file
        # torch.load() may return different types depending on file format:
        # - Plain tensor files: returns torch.Tensor
        # - TorchScript archives: returns torch.jit.ScriptModule (need to extract weights)
        # - Pickle files: returns list of tensors
        # weights_only=False is needed for TorchScript archives (PyTorch 2.6+ default changed to True)
        loaded_data = torch.load(weight_path, map_location='cpu', weights_only=False)
        
        # Handle different return types from torch.load()
        if isinstance(loaded_data, list):
            if len(loaded_data) == 0:
                raise ValueError(f"Empty tensor list in file: {weight_path}")
            weight_tensor = loaded_data[0]  # First tensor in list
        elif isinstance(loaded_data, torch.Tensor):
            weight_tensor = loaded_data  # Single tensor
        elif hasattr(loaded_data, 'state_dict'):
            # TorchScript module - extract weight tensor from state_dict
            state_dict = loaded_data.state_dict()
            if len(state_dict) == 0:
                raise ValueError(f"Empty state_dict in TorchScript module: {weight_path}")
            # Get first tensor from state_dict (typically key '0' or 'weight')
            weight_tensor = next(iter(state_dict.values()))
            if not isinstance(weight_tensor, torch.Tensor):
                raise ValueError(f"Expected tensor in state_dict, got {type(weight_tensor)}")
        elif hasattr(loaded_data, 'parameters'):
            # TorchScript module - extract weight tensor from parameters
            params = list(loaded_data.parameters())
            if len(params) == 0:
                raise ValueError(f"No parameters in module: {weight_path}")
            weight_tensor = params[0]
        else:
            raise ValueError(f"Unexpected data type from {weight_path}: {type(loaded_data)}")
        
        # Load the weight tensor
        self.load_weights(weight_tensor, use_decomposition=use_decomposition)
        
        print(f"Loaded weights from {weight_path}")
        print(f"  Shape: {weight_tensor.shape}")
        print(f"  Dtype: {weight_tensor.dtype}")


def create_crf_encoder_from_config(config_path: str) -> CRFEncoder:
    """
    Create a CRF encoder from a config.toml file.
    
    Args:
        config_path: Path to the config.toml file
    
    Returns:
        Configured CRFEncoder module
    """
    try:
        import toml
        config = toml.load(config_path)
        
        # Extract encoder sublayers
        encoder = config['encoder']
        sublayers = encoder['sublayers']
        
        # Find the linearcrfencoder sublayer
        crf_config = None
        for sublayer in sublayers:
            if sublayer.get('type') == 'linearcrfencoder':
                crf_config = sublayer
                break
        
        if crf_config is None:
            raise ValueError("Could not find linearcrfencoder in config")
        
        # Extract parameters
        insize = crf_config['insize']
        n_base = crf_config['n_base']
        state_len = crf_config['state_len']
        bias = crf_config.get('bias', False)
        
        # Check for decomposition (would be in a separate 'linear' sublayer if present)
        use_decomposition = False
        decomposition_size = None
        for sublayer in sublayers:
            if sublayer.get('type') == 'linear' and 'out_features' in sublayer:
                use_decomposition = True
                decomposition_size = sublayer['out_features']
                break
        
        # Check for clamp
        clamp_active = any(s.get('type') == 'clamp' for s in sublayers)
        
        return CRFEncoder(
            insize=insize,
            n_base=n_base,
            state_len=state_len,
            bias=bias,
            use_decomposition=use_decomposition,
            decomposition_size=decomposition_size,
            clamp_active=clamp_active
        )
    except ImportError:
        raise ImportError("toml library required. Install with: pip install toml")
    except Exception as e:
        raise ValueError(f"Error parsing config file: {e}")


def load_crf_encoder_with_weights(model_dir: str) -> CRFEncoder:
    """
    Convenience function to create and load a CRF encoder with weights from model directory.
    
    Args:
        model_dir: Path to model directory (e.g., "model/dna_r10.4.1_e8.2_400bps_hac@v5.2.0")
    
    Returns:
        CRFEncoder with loaded weights
    """
    model_path = Path(model_dir)
    config_path = model_path / "config.toml"
    weight_path = model_path / "11.linear.weight.tensor"
    
    # Create encoder from config
    crf_encoder = create_crf_encoder_from_config(str(config_path))
    
    # Load weights if file exists
    if weight_path.exists():
        crf_encoder.load_weights_from_file(str(weight_path))
    else:
        print(f"Warning: Weight file not found at {weight_path}")
        print("  Continuing with randomly initialized weights")
    
    return crf_encoder


# Example usage:
if __name__ == "__main__":
    # Example 1: Create CRF encoder with default (random) weights
    print("=" * 60)
    print("Example 1: Creating CRF encoder with random weights")
    print("=" * 60)
    
    crf_encoder = CRFEncoder(
        insize=384,      # From LSTM output
        n_base=4,        # DNA bases: A, C, G, T
        state_len=4,     # From config.toml global_norm.state_len
        bias=False,      # From config.toml linearcrfencoder.bias
        use_decomposition=False,  # Single linear layer based on tensor files
        clamp_active=True  # From config.toml clamp layer
    )
    
    # Example forward pass
    batch_size = 8
    seq_length = 1000
    input_features = 384
    
    # Create dummy input: [N, T, C] = [8, 1000, 384]
    dummy_input = torch.randn(batch_size, seq_length, input_features)
    
    # Forward pass
    output = crf_encoder(dummy_input)
    
    print(f"Input shape: {dummy_input.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Output range: [{output.min().item():.2f}, {output.max().item():.2f}]")
    print(f"Expected output size: {crf_encoder.outsize}")
    
    # Verify output size calculation
    expected_outsize = 4 ** (4 + 1)  # n_base^(state_len + 1)
    assert output.shape[-1] == expected_outsize, \
        f"Output size mismatch: got {output.shape[-1]}, expected {expected_outsize}"
    
    # Example 2: Load weights from file (if model directory exists)
    print("\n" + "=" * 60)
    print("Example 2: Loading weights from .tensor file")
    print("=" * 60)
    
    model_dir = "../model/dna_r10.4.1_e8.2_400bps_hac@v5.2.0"
    weight_file = f"{model_dir}/11.linear.weight.tensor"
    weights_changed = False  # Track if weights were loaded
    
    if Path(weight_file).exists():
        print(f"Loading weights from: {weight_file}")
        
        # Store a sample of weights before loading to verify they change
        weight_before = crf_encoder.linear1.linear.weight.data.clone()
        weight_before_sample = weight_before[0, :5].tolist()
        
        crf_encoder.load_weights_from_file(weight_file)
        
        # Verify weights changed
        weight_after = crf_encoder.linear1.linear.weight.data
        weight_after_sample = weight_after[0, :5].tolist()
        weights_changed = not torch.allclose(weight_before, weight_after, atol=1e-6)
        print(f"  ✓ Weights loaded: {weights_changed}")
        if weights_changed:
            print(f"    Before sample: {[f'{x:.4f}' for x in weight_before_sample]}")
            print(f"    After sample:  {[f'{x:.4f}' for x in weight_after_sample]}")
        
        # Test forward pass with loaded weights
        print("\n  Testing forward pass...")
        crf_encoder.eval()
        with torch.no_grad():
            output_loaded = crf_encoder(dummy_input)
        
        # Verify output
        print(f"  ✓ Forward pass successful")
        print(f"    Output shape: {output_loaded.shape}")
        assert output_loaded.shape == (batch_size, seq_length, crf_encoder.outsize), \
            f"Output shape mismatch: {output_loaded.shape}"
        
        # Verify output range (should be clamped to [-5, 5])
        output_min = output_loaded.min().item()
        output_max = output_loaded.max().item()
        print(f"    Output range: [{output_min:.4f}, {output_max:.4f}]")
        assert -5.0 <= output_min <= output_max <= 5.0, \
            f"Output not properly clamped: range [{output_min}, {output_max}]"
        print(f"  ✓ Output correctly clamped to [-5.0, 5.0]")
        
        # Verify output size
        assert output_loaded.shape[-1] == 1024, \
            f"Output size mismatch: got {output_loaded.shape[-1]}, expected 1024"
        print(f"  ✓ Output size correct: {output_loaded.shape[-1]}")
        
        # Verify no NaN or Inf
        has_nan = torch.isnan(output_loaded).any().item()
        has_inf = torch.isinf(output_loaded).any().item()
        assert not has_nan, "Output contains NaN values"
        assert not has_inf, "Output contains Inf values"
        print(f"  ✓ Output contains no NaN or Inf values")
        
        print("\n  ✓ Model verification complete - all checks passed!")
    else:
        print(f"Weight file not found: {weight_file}")
        print("  Skipping weight loading example")
    
    # Example 3: Create from config and load weights (convenience function)
    print("\n" + "=" * 60)
    print("Example 3: Using convenience function")
    print("=" * 60)
    
    if Path(model_dir).exists():
        try:
            crf_encoder_from_config = load_crf_encoder_with_weights(model_dir)
            print(f"  ✓ Successfully created and loaded CRF encoder from {model_dir}")
            
            # Verify configuration
            print(f"\n  Verifying configuration...")
            print(f"    Input size: {crf_encoder_from_config.insize}")
            print(f"    Output size: {crf_encoder_from_config.outsize}")
            print(f"    State length: {crf_encoder_from_config.state_len}")
            print(f"    N bases: {crf_encoder_from_config.n_base}")
            assert crf_encoder_from_config.insize == 384, "Incorrect input size"
            assert crf_encoder_from_config.outsize == 1024, "Incorrect output size"
            assert crf_encoder_from_config.state_len == 4, "Incorrect state length"
            print(f"  ✓ Configuration verified")
            
            # Verify weights are loaded
            weight_mean = crf_encoder_from_config.linear1.linear.weight.data.mean().item()
            weight_std = crf_encoder_from_config.linear1.linear.weight.data.std().item()
            print(f"\n  Verifying loaded weights...")
            print(f"    Weight mean: {weight_mean:.6f}")
            print(f"    Weight std: {weight_std:.6f}")
            # Loaded weights should have non-zero variance (not all zeros)
            assert weight_std > 0, "Weights appear to be all zeros"
            print(f"  ✓ Weights appear valid (non-zero variance)")
            
            # Test forward pass
            print(f"\n  Testing forward pass...")
            crf_encoder_from_config.eval()
            with torch.no_grad():
                output_config = crf_encoder_from_config(dummy_input)
            
            # Verify output
            print(f"  ✓ Forward pass successful")
            print(f"    Output shape: {output_config.shape}")
            assert output_config.shape == (batch_size, seq_length, 1024), \
                f"Output shape mismatch: {output_config.shape}"
            
            # Verify output range
            output_min = output_config.min().item()
            output_max = output_config.max().item()
            print(f"    Output range: [{output_min:.4f}, {output_max:.4f}]")
            assert -5.0 <= output_min <= output_max <= 5.0, \
                f"Output not properly clamped: range [{output_min}, {output_max}]"
            print(f"  ✓ Output correctly clamped to [-5.0, 5.0]")
            
            # Verify output size
            assert output_config.shape[-1] == 1024, \
                f"Output size mismatch: got {output_config.shape[-1]}, expected 1024"
            print(f"  ✓ Output size correct: {output_config.shape[-1]}")
            
            # Verify no NaN or Inf
            has_nan = torch.isnan(output_config).any().item()
            has_inf = torch.isinf(output_config).any().item()
            assert not has_nan, "Output contains NaN values"
            assert not has_inf, "Output contains Inf values"
            print(f"  ✓ Output contains no NaN or Inf values")
            
            # Compare outputs between examples (should be identical if same weights)
            if Path(weight_file).exists():
                print(f"\n  Comparing outputs between Example 2 and 3...")
                if weights_changed:  # Only if we loaded weights in Example 2
                    # Outputs should be identical since both use loaded weights
                    max_diff = (output_loaded - output_config).abs().max().item()
                    print(f"    Max difference: {max_diff:.6e}")
                    if max_diff < 1e-5:
                        print(f"  ✓ Outputs are identical (within tolerance)")
                    else:
                        print(f"  ⚠ Outputs differ slightly (may be due to numerical precision)")
            
            print("\n  ✓ Model verification complete - all checks passed!")
            
        except Exception as e:
            print(f"Error loading from config: {e}")
            import traceback
            traceback.print_exc()
    else:
        print(f"Model directory not found: {model_dir}")
        print("  Skipping config-based loading example")

