"""
Quick test script to verify .tensor file loading works correctly.
"""

import torch
from pathlib import Path
from crf_layer_implementation import CRFEncoder, load_crf_encoder_with_weights

# Test loading a tensor file directly
model_dir = Path("../model/dna_r10.4.1_e8.2_400bps_hac@v5.2.0")
weight_file = model_dir / "11.linear.weight.tensor"

if weight_file.exists():
    print("Testing tensor file loading...")
    print(f"File: {weight_file}")
    print(f"File size: {weight_file.stat().st_size / 1024 / 1024:.2f} MB")
    
    # Try loading with torch.load()
    try:
        loaded_data = torch.load(weight_file, map_location='cpu', weights_only=False)
        print(f"\nLoaded data type: {type(loaded_data)}")
        
        # Extract tensor for inspection
        if isinstance(loaded_data, list):
            print(f"  List length: {len(loaded_data)}")
            if len(loaded_data) > 0:
                tensor = loaded_data[0]
                print(f"  First tensor shape: {tensor.shape}")
                print(f"  First tensor dtype: {tensor.dtype}")
        elif isinstance(loaded_data, torch.Tensor):
            tensor = loaded_data
            print(f"  Tensor shape: {tensor.shape}")
            print(f"  Tensor dtype: {tensor.dtype}")
            print(f"  Tensor min: {tensor.min().item():.4f}")
            print(f"  Tensor max: {tensor.max().item():.4f}")
            print(f"  Tensor mean: {tensor.mean().item():.4f}")
        elif hasattr(loaded_data, 'state_dict'):
            # TorchScript module
            state_dict = loaded_data.state_dict()
            print(f"  TorchScript module with {len(state_dict)} state_dict entries")
            if len(state_dict) > 0:
                tensor = next(iter(state_dict.values()))
                print(f"  Extracted tensor shape: {tensor.shape}")
                print(f"  Extracted tensor dtype: {tensor.dtype}")
                print(f"  Extracted tensor min: {tensor.min().item():.4f}")
                print(f"  Extracted tensor max: {tensor.max().item():.4f}")
                print(f"  Extracted tensor mean: {tensor.mean().item():.4f}")
        elif hasattr(loaded_data, 'parameters'):
            # Module with parameters
            params = list(loaded_data.parameters())
            print(f"  Module with {len(params)} parameters")
            if len(params) > 0:
                tensor = params[0]
                print(f"  First parameter shape: {tensor.shape}")
                print(f"  First parameter dtype: {tensor.dtype}")
                print(f"  First parameter min: {tensor.min().item():.4f}")
                print(f"  First parameter max: {tensor.max().item():.4f}")
                print(f"  First parameter mean: {tensor.mean().item():.4f}")
        
        # Test loading into CRF encoder
        print("\n" + "=" * 60)
        print("Testing CRF encoder weight loading...")
        crf_encoder = CRFEncoder(
            insize=384,
            n_base=4,
            state_len=4,
            bias=False,
            clamp_active=True
        )
        
        # Store initial weights for comparison
        initial_weight_mean = crf_encoder.linear1.linear.weight.data.mean().item()
        
        # Load weights
        crf_encoder.load_weights_from_file(str(weight_file))
        
        # Check if weights changed
        loaded_weight_mean = crf_encoder.linear1.linear.weight.data.mean().item()
        print(f"Initial weight mean: {initial_weight_mean:.6f}")
        print(f"Loaded weight mean: {loaded_weight_mean:.6f}")
        print(f"Weights changed: {abs(initial_weight_mean - loaded_weight_mean) > 1e-6}")
        
        print("\n✓ Weight loading successful!")
        
    except Exception as e:
        print(f"\n✗ Error loading tensor: {e}")
        import traceback
        traceback.print_exc()
else:
    print(f"Tensor file not found: {weight_file}")
    print("Make sure you're running from the project root directory")

