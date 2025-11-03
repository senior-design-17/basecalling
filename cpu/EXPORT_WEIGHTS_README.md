# Weight Export Tool for FPGA

This tool exports model weights from PyTorch `.tensor` files into FPGA-friendly formats suitable for hardware inference.

## Overview

`export_weights_for_fpga.py` converts PyTorch weight tensors into multiple output formats:

- **Binary files**: Raw byte data for direct memory loading
- **Text files**: Human-readable CSV format for verification
- **Header files**: C/C++ compatible arrays with metadata
- **Metadata files**: Statistical information about the weights

The tool also supports optional fixed-point conversion for quantization-aware export.

## Requirements

- Python 3.7+
- PyTorch
- NumPy

All dependencies are listed in `requirements.txt`. Install with:

```bash
pip install -r requirements.txt
```

## Basic Usage

The script can process either a single tensor file or all `.tensor` files in a directory:

```bash
# Single file mode
python export_weights_for_fpga.py <tensor_file> [options]

# Directory mode (processes all .tensor files in the directory)
python export_weights_for_fpga.py <directory_path> [options]
```

### Examples

**Single File Mode:**

```bash
# Export all formats for a weight tensor
python export_weights_for_fpga.py model/11.linear.weight.tensor

# Export to specific directory
python export_weights_for_fpga.py model/11.linear.weight.tensor --output-dir weights_export

# Export only binary format
python export_weights_for_fpga.py model/11.linear.weight.tensor --binary
```

**Directory Mode:**

```bash
# Export all tensor files in a directory
python export_weights_for_fpga.py model/

# Export with reduced verbosity (useful for many files)
python export_weights_for_fpga.py model/ --quiet

# Export only binary format for all files
python export_weights_for_fpga.py model/ --binary
```

## Command-Line Options

### Required Arguments

- `input_path`: Path to a `.tensor` file or a directory containing `.tensor` files
  - If a file: processes that single file
  - If a directory: processes all `.tensor` files found in the directory

### Output Options

- `--output-dir DIR`: Output directory for exported files (default: `weights_export`)
- `--output-name NAME`: Base name for output files (default: inferred from tensor filename). Only used in single file mode. In directory mode, filenames are always inferred from the source files.
- `--quiet`: Reduce output verbosity (useful when processing multiple files in directory mode)
- `--all`: Export all formats (binary, text, header, metadata) - **This is the default if no format is specified**
- `--binary`: Export binary file (`.bin`)
- `--text`: Export text file (`.txt`, CSV format)
- `--header`: Export C/C++ header file (`.h`)
- `--metadata`: Export metadata file (`_metadata.txt`)

### Binary Format Options

- `--binary-dtype TYPE`: Data type for binary output
  - Choices: `float32`, `float16`, `int32`, `int16`, `int8`
  - Default: `float32`

### Fixed-Point Conversion

- `--fixed-point`: Enable fixed-point conversion (quantization)
- `--integer-bits N`: Number of bits for integer part (default: 8)
- `--fractional-bits N`: Number of bits for fractional part (default: 8)

## Usage Examples

### Directory Mode - Process All Tensor Files

```bash
# Export all tensor files in a model directory
python export_weights_for_fpga.py model/

# With quiet mode (less verbose output)
python export_weights_for_fpga.py model/ --quiet

# Export only binary files for all tensors
python export_weights_for_fpga.py model/ --binary

# Export with fixed-point conversion for all files
python export_weights_for_fpga.py model/ --fixed-point --integer-bits 8 --fractional-bits 8
```

When processing a directory, the script will:

- Automatically find all `.tensor` files in the specified directory
- Process each file sequentially
- Show progress with `[n/total]` indicators
- Display a summary at the end with success/failure counts
- Preserve original filenames in the output directory

### Export All Formats (Default)

```bash
python export_weights_for_fpga.py model/11.linear.weight.tensor
```

This creates:

- `11.linear.weight.bin` - Binary file
- `11.linear.weight.txt` - CSV text file
- `11.linear.weight.h` - C/C++ header
- `11.linear.weight_metadata.txt` - Metadata

### Export Specific Formats

```bash
# Only binary and header
python export_weights_for_fpga.py model/11.linear.weight.tensor --binary --header

# Only text for verification
python export_weights_for_fpga.py model/11.linear.weight.tensor --text
```

### Custom Output Directory

```bash
python export_weights_for_fpga.py model/11.linear.weight.tensor --output-dir fpga_weights
```

### Fixed-Point Quantization

```bash
# Convert to 8.8 fixed-point (8 integer bits, 8 fractional bits)
python export_weights_for_fpga.py model/11.linear.weight.tensor \
    --fixed-point \
    --integer-bits 8 \
    --fractional-bits 8

# Convert to 4.12 fixed-point (4 integer bits, 12 fractional bits)
python export_weights_for_fpga.py model/11.linear.weight.tensor \
    --fixed-point \
    --integer-bits 4 \
    --fractional-bits 12
```

### Integer Binary Export

```bash
# Export as 16-bit integers
python export_weights_for_fpga.py model/11.linear.weight.tensor \
    --binary \
    --binary-dtype int16

# Export as 8-bit integers
python export_weights_for_fpga.py model/11.linear.weight.tensor \
    --binary \
    --binary-dtype int8
```

## Output Formats

### Binary File (`.bin`)

Raw binary data in the specified dtype. Data is stored in C-order (row-major) as a flattened array.

**Use case**: Direct memory loading into FPGA buffers

**Example**:

```bash
python export_weights_for_fpga.py weights.tensor --binary --binary-dtype float32
```

### Text File (`.txt`)

CSV format with one value per line, suitable for verification and debugging.

**Format**: Scientific notation (e.g., `1.234567890e-01`)

**Use case**: Human verification, debugging, comparison with reference values

### Header File (`.h`)

C/C++ compatible header file containing:

- Array declaration with weights
- Shape macros
- Size constants
- Statistical metadata in comments

**Example output**:

```c
// Auto-generated weight file
// Shape: [128, 256]
// Total elements: 32768
// Min: -1.234567890e-01, Max: 1.234567890e-01
// Mean: 0.000000000e+00, Std: 0.045678901e+00

#ifndef WEIGHTS_H
#define WEIGHTS_H

#include <stdint.h>

// Weight array shape: [128, 256]
#define WEIGHTS_SHAPE {128, 256}
#define WEIGHTS_SIZE 32768

static const float weights[] = {
  1.234567890e-01, -2.345678901e-02, 3.456789012e-03, ...
};

#endif // WEIGHTS_H
```

**Use case**: C/C++ FPGA inference code integration

### Metadata File (`_metadata.txt`)

Human-readable metadata including:

- Shape
- Data type
- Element count
- Min/max values
- Mean and standard deviation
- Memory size

**Example output**:

```
# Weight Metadata
# ================

shape: [128, 256]
dtype: torch.float32
numel: 32768
min: -1.234567890
max: 1.234567890
mean: 0.000000000
std: 0.045678901
memory_size_bytes: 131072
```

**Use case**: Documentation, verification, understanding weight distributions

## Fixed-Point Conversion

When `--fixed-point` is enabled, the tool converts floating-point weights to fixed-point representation:

1. Scales values by `2^fractional_bits`
2. Clamps to representable range
3. Rounds to nearest integer
4. Stores metadata about the conversion

**Note**: The fixed-point values are converted back to float for storage (divided by scale). The metadata file contains the scale factor needed to reconstruct the integer representation.

**Use case**: Quantization-aware export for FPGA implementations that use fixed-point arithmetic

## Tensor File Format

The tool expects PyTorch `.tensor` files created with `torch.save()`. It handles various formats:

- Direct tensor objects
- Lists containing tensors
- TorchScript modules with `state_dict()`
- PyTorch modules with `parameters()`

## Troubleshooting

### File Not Found

```
FileNotFoundError: Tensor file not found: model/weights.tensor
```

**Solution**: Check the path to your tensor file is correct.

### Empty Tensor

```
ValueError: Empty tensor list in file: model/weights.tensor
```

**Solution**: Ensure the tensor file contains valid weight data.

### Invalid Data Type

```
ValueError: Unexpected data type from model/weights.tensor: <class 'dict'>
```

**Solution**: The tool expects a tensor, list of tensors, or module. If you have a state dict, extract the tensor first.

## Integration Example

### Loading Binary Weights in C/C++

```c
#include "weights.h"

// Access weights
float value = weights[0];  // First weight
float value2 = weights[WEIGHTS_SIZE - 1];  // Last weight

// Iterate with shape information
int idx = 0;
for (int i = 0; i < WEIGHTS_SHAPE[0]; i++) {
    for (int j = 0; j < WEIGHTS_SHAPE[1]; j++) {
        float w = weights[idx++];
        // Use weight...
    }
}
```

### Loading Binary Weights in Python (Verification)

```python
import numpy as np

# Load binary weights
weights = np.fromfile('weights.bin', dtype=np.float32)
weights = weights.reshape([128, 256])  # Reshape to original shape
```

## Notes

- All tensors are flattened in C-order (row-major) for consistency
- Binary files use little-endian byte order (standard for most systems)
- Header files use valid C identifiers (numbers prefixed with `w_` if needed)
- Fixed-point conversion metadata is included in the metadata file
