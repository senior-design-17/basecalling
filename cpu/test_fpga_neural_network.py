"""
Test script for FPGA Neural Network inference.

This script tests the FPGA neural network implementation by:
1. Loading the model from the model directory
2. Creating synthetic input data (normalized signal)
3. Running inference
4. Verifying output shape and properties
"""

import torch
import numpy as np
from pathlib import Path
import sys

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent))

from fpga_neural_network import FPGANeuralNetwork, create_fpga_network


def test_weight_loading():
    """Test that weights can be loaded correctly."""
    print("=" * 60)
    print("Test 1: Weight Loading")
    print("=" * 60)
    
    model_dir = Path(__file__).parent.parent / "model" / "dna_r10.4.1_e8.2_400bps_hac@v5.2.0"
    
    if not model_dir.exists():
        print(f"❌ Model directory not found: {model_dir}")
        return False
    
    try:
        network = create_fpga_network(str(model_dir))
        print("✓ Weights loaded successfully")
        return True
    except Exception as e:
        print(f"❌ Error loading weights: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_input_shape():
    """Test that the network accepts correct input shapes."""
    print("\n" + "=" * 60)
    print("Test 2: Input Shape Handling")
    print("=" * 60)
    
    model_dir = Path(__file__).parent.parent / "model" / "dna_r10.4.1_e8.2_400bps_hac@v5.2.0"
    network = create_fpga_network(str(model_dir))
    
    # Test with shape [batch, time]
    x1 = torch.randn(1, 10000, dtype=torch.float32)
    try:
        out1 = network(x1)
        print(f"✓ Input shape [1, 10000]: output shape {out1.shape}")
        assert out1.shape == (1, 384, 1667), f"Expected (1, 384, 1667), got {out1.shape}"
    except Exception as e:
        print(f"❌ Error with shape [1, 10000]: {e}")
        return False
    
    # Test with shape [batch, 1, time]
    x2 = torch.randn(1, 1, 10000, dtype=torch.float32)
    try:
        out2 = network(x2)
        print(f"✓ Input shape [1, 1, 10000]: output shape {out2.shape}")
        assert out2.shape == (1, 384, 1667), f"Expected (1, 384, 1667), got {out2.shape}"
    except Exception as e:
        print(f"❌ Error with shape [1, 1, 10000]: {e}")
        return False
    
    # Test with batch size > 1
    x3 = torch.randn(4, 10000, dtype=torch.float32)
    try:
        out3 = network(x3)
        print(f"✓ Input shape [4, 10000]: output shape {out3.shape}")
        assert out3.shape == (4, 384, 1667), f"Expected (4, 384, 1667), got {out3.shape}"
    except Exception as e:
        print(f"❌ Error with shape [4, 10000]: {e}")
        return False
    
    return True


def test_output_properties():
    """Test that output has expected properties."""
    print("\n" + "=" * 60)
    print("Test 3: Output Properties")
    print("=" * 60)
    
    model_dir = Path(__file__).parent.parent / "model" / "dna_r10.4.1_e8.2_400bps_hac@v5.2.0"
    network = create_fpga_network(str(model_dir))
    
    # Create normalized signal input (simulating 16-bit normalized values)
    # In practice, these would be normalized signal values from POD5
    x = torch.randn(1, 10000, dtype=torch.float32) * 10 + 94.0  # Mean ~94, std ~10
    
    with torch.no_grad():
        output = network(x)
    
    print(f"Output shape: {output.shape}")
    print(f"Expected shape: (1, 384, 1667)")
    assert output.shape == (1, 384, 1667), f"Expected (1, 384, 1667), got {output.shape}"
    
    print(f"Output dtype: {output.dtype}")
    assert output.dtype == torch.float32, f"Expected float32, got {output.dtype}"
    
    print(f"Output min: {output.min().item():.6f}")
    print(f"Output max: {output.max().item():.6f}")
    print(f"Output mean: {output.mean().item():.6f}")
    print(f"Output std: {output.std().item():.6f}")
    
    # Check for NaN or Inf
    has_nan = torch.isnan(output).any().item()
    has_inf = torch.isinf(output).any().item()
    
    if has_nan:
        print("❌ Output contains NaN values!")
        return False
    if has_inf:
        print("❌ Output contains Inf values!")
        return False
    
    print("✓ Output properties are valid")
    return True


def test_standardization():
    """Test that standardization is applied correctly."""
    print("\n" + "=" * 60)
    print("Test 4: Standardization")
    print("=" * 60)
    
    model_dir = Path(__file__).parent.parent / "model" / "dna_r10.4.1_e8.2_400bps_hac@v5.2.0"
    network = create_fpga_network(str(model_dir))
    
    # Create input with known mean and std
    mean = 94.0
    std = 24.0
    x = torch.ones(1, 10000, dtype=torch.float32) * mean
    
    # After standardization, should be approximately zero
    standardized = network.standardize(x)
    standardized_mean = standardized.mean().item()
    standardized_std = standardized.std().item()
    
    print(f"Input mean: {mean}")
    print(f"Standardized mean: {standardized_mean:.6f} (expected ~0.0)")
    print(f"Standardized std: {standardized_std:.6f} (expected ~0.0 for constant input)")
    
    # For constant input, standardized should be ~0
    if abs(standardized_mean) > 0.01:
        print(f"❌ Standardization mean is not close to 0: {standardized_mean}")
        return False
    
    print("✓ Standardization works correctly")
    return True


def test_conv_stack():
    """Test that ConvStack produces correct intermediate shapes."""
    print("\n" + "=" * 60)
    print("Test 5: ConvStack Intermediate Shapes")
    print("=" * 60)
    
    model_dir = Path(__file__).parent.parent / "model" / "dna_r10.4.1_e8.2_400bps_hac@v5.2.0"
    network = create_fpga_network(str(model_dir))
    
    x = torch.randn(1, 1, 10000, dtype=torch.float32)
    x = network.standardize(x)
    
    # Test each conv layer
    x1 = network.conv1(x)
    print(f"After Conv1: {x1.shape} (expected [1, 16, 10000])")
    assert x1.shape == (1, 16, 10000), f"Expected (1, 16, 10000), got {x1.shape}"
    
    x2 = network.conv2(x1)
    print(f"After Conv2: {x2.shape} (expected [1, 16, 10000])")
    assert x2.shape == (1, 16, 10000), f"Expected (1, 16, 10000), got {x2.shape}"
    
    x3 = network.conv3(x2)
    print(f"After Conv3: {x3.shape} (expected [1, 384, 1667])")
    assert x3.shape == (1, 384, 1667), f"Expected (1, 384, 1667), got {x3.shape}"
    
    print("✓ ConvStack produces correct shapes")
    return True


def test_inference_speed():
    """Test inference speed (basic benchmark)."""
    print("\n" + "=" * 60)
    print("Test 6: Inference Speed")
    print("=" * 60)
    
    model_dir = Path(__file__).parent.parent / "model" / "dna_r10.4.1_e8.2_400bps_hac@v5.2.0"
    network = create_fpga_network(str(model_dir))
    
    x = torch.randn(1, 10000, dtype=torch.float32)
    
    # Warmup
    with torch.no_grad():
        _ = network(x)
    
    # Benchmark
    import time
    num_runs = 10
    times = []
    
    with torch.no_grad():
        for _ in range(num_runs):
            start = time.time()
            _ = network(x)
            times.append(time.time() - start)
    
    avg_time = np.mean(times)
    std_time = np.std(times)
    
    print(f"Average inference time: {avg_time*1000:.2f} ms ± {std_time*1000:.2f} ms")
    print(f"Throughput: {1.0/avg_time:.2f} chunks/second")
    
    print("✓ Inference speed test completed")
    return True


def test_deterministic():
    """Test that inference is deterministic (same input -> same output)."""
    print("\n" + "=" * 60)
    print("Test 7: Deterministic Inference")
    print("=" * 60)
    
    model_dir = Path(__file__).parent.parent / "model" / "dna_r10.4.1_e8.2_400bps_hac@v5.2.0"
    network = create_fpga_network(str(model_dir))
    network.eval()
    
    # Set random seed for reproducibility
    torch.manual_seed(42)
    x = torch.randn(1, 10000, dtype=torch.float32)
    
    with torch.no_grad():
        output1 = network(x)
        output2 = network(x)
    
    # Check if outputs are identical
    max_diff = (output1 - output2).abs().max().item()
    print(f"Maximum difference between two forward passes: {max_diff:.2e}")
    
    if max_diff > 1e-5:
        print(f"❌ Outputs are not identical (max diff: {max_diff})")
        return False
    
    print("✓ Inference is deterministic")
    return True


def main():
    """Run all tests."""
    print("\n" + "=" * 60)
    print("FPGA Neural Network Test Suite")
    print("=" * 60)
    
    tests = [
        ("Weight Loading", test_weight_loading),
        ("Input Shape", test_input_shape),
        ("Output Properties", test_output_properties),
        ("Standardization", test_standardization),
        ("ConvStack Shapes", test_conv_stack),
        ("Inference Speed", test_inference_speed),
        ("Deterministic", test_deterministic),
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"\n❌ Test '{test_name}' failed with exception: {e}")
            import traceback
            traceback.print_exc()
            results.append((test_name, False))
    
    # Summary
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✓ PASS" if result else "❌ FAIL"
        print(f"{status}: {test_name}")
    
    print(f"\nTotal: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All tests passed!")
        return 0
    else:
        print(f"\n⚠️  {total - passed} test(s) failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())

