# CRF Layer Implementation for Dorado Basecalling Model

This Python implementation provides the CRF (Conditional Random Field) layer used in the Dorado basecalling model `dna_r10.4.1_e8.2_400bps_hac@v5.2.0`.

## Overview

The "CRF" layer in this context is actually a **linear transformation layer** that produces emission scores for CTC-like decoding, not a traditional CRF. It transforms LSTM outputs into emission scores that are then decoded using beam search to produce DNA base sequences.

## Model Configuration

Based on the `config.toml` file, the CRF encoder has the following parameters:

- **insize**: 384 (from LSTM output)
- **n_base**: 4 (DNA bases: A, C, G, T)
- **state_len**: 4
- **bias**: false
- **blank_score**: 2.0 (used in decoding, not in the layer)
- **clamp**: [-5.0, 5.0] (applied to output)

The output size is calculated as: `4^(state_len + 1) = 4^5 = 1024`

## Architecture

```
Input [N, T, 384] (from LSTM)
    ↓
Linear Layer (384 → 1024, no bias)
    ↓
Clamp [-5.0, 5.0]
    ↓
Output [N, T, 1024] (emission scores)
```

Where:

- **N** = batch size
- **T** = sequence length (time steps)
- **384** = LSTM feature dimension
- **1024** = number of possible state transitions (4^5)

## Usage

### Basic Usage

```python
import torch
from crf_layer_implementation import CRFEncoder

# Create CRF encoder matching v5.2.0 model configuration
crf_encoder = CRFEncoder(
    insize=384,      # From LSTM output
    n_base=4,        # DNA bases: A, C, G, T
    state_len=4,     # From config.toml global_norm.state_len
    bias=False,      # From config.toml linearcrfencoder.bias
    use_decomposition=False,  # Single linear layer
    clamp_active=True  # From config.toml clamp layer
)

# Forward pass
batch_size = 8
seq_length = 1000
input_features = 384

# Input: [N, T, C] = [8, 1000, 384]
dummy_input = torch.randn(batch_size, seq_length, input_features)

# Forward pass
output = crf_encoder(dummy_input)  # Output: [8, 1000, 1024]
```

### Creating from Config File

```python
from crf_layer_implementation import create_crf_encoder_from_config

# Load from config.toml
crf_encoder = create_crf_encoder_from_config(
    "model/dna_r10.4.1_e8.2_400bps_hac@v5.2.0/config.toml"
)
```

### Loading Weights

The `.tensor` files use PyTorch's native pickle format and can be loaded directly:

```python
# Method 1: Load weights directly from file
crf_encoder.load_weights_from_file("model/dna_r10.4.1_e8.2_400bps_hac@v5.2.0/11.linear.weight.tensor")

# Method 2: Load manually with torch.load() and then load into encoder
weight_tensor = torch.load("11.linear.weight.tensor", map_location='cpu')
if isinstance(weight_tensor, list):
    weight_tensor = weight_tensor[0]  # torch.load() returns a list
crf_encoder.load_weights(weight_tensor)

# Method 3: Use convenience function to load from model directory
from crf_layer_implementation import load_crf_encoder_with_weights
crf_encoder = load_crf_encoder_with_weights("model/dna_r10.4.1_e8.2_400bps_hac@v5.2.0")
```

## Components

### 1. LinearCRF

The core linear layer that performs the transformation:

```python
linear_crf = LinearCRF(
    insize=384,
    outsize=1024,
    bias=False,
    tanh_and_scale=False  # For v5.2.0, no tanh activation
)
```

### 2. Clamp

Clamps output values to the range [-5.0, 5.0]:

```python
clamp = Clamp(min_val=-5.0, max_val=5.0, active=True)
```

### 3. CRFEncoder

Complete encoder combining linear layer and clamp:

```python
crf_encoder = CRFEncoder(
    insize=384,
    n_base=4,
    state_len=4,
    bias=False,
    clamp_active=True
)
```

## Output Interpretation

The output tensor has shape `[N, T, 1024]` where:

- Each of the 1024 dimensions represents a possible state transition
- Scores are logits/emission scores for the CTC-like decoder
- Values are clamped to [-5.0, 5.0]
- These scores are then used with beam search decoding to produce:
  - DNA base sequence (A, C, G, T)
  - Quality scores (Phred scores)

## Differences from Traditional CRF

This is **not** a traditional CRF that models pairwise dependencies. Instead:

1. It's a simple linear transformation layer
2. The "CRF" name refers to the decoding process (CTC-like with state transitions)
3. It produces emission scores, not transition scores
4. The 1024 outputs represent all possible state transitions in a compact encoding

## Model Version Notes

- **v5.2.0**: Single linear layer (384 → 1024) with clamp
- **v4.x**: May use decomposed linear layers or different activation
- **Pre-v4**: Uses tanh activation with scaling

## Dependencies

- PyTorch (tested with 1.9+)
- numpy
- toml (for config file parsing)

## Files

- `crf_layer_implementation.py`: Main implementation
- `config.toml`: Model configuration (in model directory)

## References

The implementation is based on:

- `dorado/nn/CRFModules.h` and `dorado/nn/CRFModules.cpp`
- `dorado/basecall/model/CRFModel.cpp`
- `model/dna_r10.4.1_e8.2_400bps_hac@v5.2.0/config.toml`
