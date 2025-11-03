"""
Export model weights to FPGA-friendly formats.

This script loads weights from .tensor files and exports them in formats
suitable for FPGA inference:
- Binary file (raw float32 bytes)
- Text file (CSV format for verification)
- Header file with metadata (C/C++ compatible)
- Optional fixed-point conversion

The script can process either a single tensor file or all .tensor files
in a directory. When processing a directory, it automatically finds all
.tensor files and processes them in batch.
"""

import torch
import numpy as np
import argparse
from pathlib import Path
from typing import Optional, Tuple


def load_tensor_file(tensor_path: str) -> torch.Tensor:
    """
    Load a tensor from a .tensor file.
    
    Args:
        tensor_path: Path to the .tensor file
        
    Returns:
        Extracted weight tensor
    """
    tensor_path = Path(tensor_path)
    if not tensor_path.exists():
        raise FileNotFoundError(f"Tensor file not found: {tensor_path}")
    
    # Load tensor from file
    loaded_data = torch.load(tensor_path, map_location='cpu', weights_only=False)
    
    # Handle different return types from torch.load()
    if isinstance(loaded_data, list):
        if len(loaded_data) == 0:
            raise ValueError(f"Empty tensor list in file: {tensor_path}")
        weight_tensor = loaded_data[0]
    elif isinstance(loaded_data, torch.Tensor):
        weight_tensor = loaded_data
    elif hasattr(loaded_data, 'state_dict'):
        state_dict = loaded_data.state_dict()
        if len(state_dict) == 0:
            raise ValueError(f"Empty state_dict in TorchScript module: {tensor_path}")
        weight_tensor = next(iter(state_dict.values()))
        if not isinstance(weight_tensor, torch.Tensor):
            raise ValueError(f"Expected tensor in state_dict, got {type(weight_tensor)}")
    elif hasattr(loaded_data, 'parameters'):
        params = list(loaded_data.parameters())
        if len(params) == 0:
            raise ValueError(f"No parameters in module: {tensor_path}")
        weight_tensor = params[0]
    else:
        raise ValueError(f"Unexpected data type from {tensor_path}: {type(loaded_data)}")
    
    return weight_tensor


def convert_to_fixed_point(
    tensor: torch.Tensor,
    integer_bits: int = 8,
    fractional_bits: int = 8,
    signed: bool = True
) -> Tuple[torch.Tensor, dict]:
    """
    Convert floating-point tensor to fixed-point representation.
    
    Args:
        tensor: Input float tensor
        integer_bits: Number of bits for integer part
        fractional_bits: Number of bits for fractional part
        signed: Whether the value is signed
        
    Returns:
        Tuple of (fixed_point_tensor, metadata_dict)
    """
    # Calculate scale factor
    scale = 2.0 ** fractional_bits
    
    # Clamp values to representable range
    if signed:
        max_val = (2.0 ** (integer_bits + fractional_bits - 1)) - 1
        min_val = -(2.0 ** (integer_bits + fractional_bits - 1))
    else:
        max_val = (2.0 ** (integer_bits + fractional_bits)) - 1
        min_val = 0
    
    # Scale and convert to integer
    scaled = tensor * scale
    clamped = torch.clamp(scaled, min_val, max_val)
    fixed_point = clamped.round().long()
    
    metadata = {
        'integer_bits': integer_bits,
        'fractional_bits': fractional_bits,
        'signed': signed,
        'scale': scale,
        'min_val': min_val / scale,
        'max_val': max_val / scale,
        'original_min': tensor.min().item(),
        'original_max': tensor.max().item(),
        'original_mean': tensor.mean().item(),
        'original_std': tensor.std().item(),
    }
    
    return fixed_point, metadata


def export_binary(tensor: torch.Tensor, output_path: str, dtype: str = 'float32'):
    """
    Export tensor as raw binary file.
    
    Args:
        tensor: Input tensor
        output_path: Output file path
        dtype: Output data type ('float32', 'float16', 'int32', 'int16', 'int8')
    """
    output_path = Path(output_path)
    
    # Convert to numpy and desired dtype
    np_array = tensor.detach().cpu().numpy()
    
    if dtype == 'float32':
        np_array = np_array.astype(np.float32)
    elif dtype == 'float16':
        np_array = np_array.astype(np.float16)
    elif dtype == 'int32':
        np_array = np_array.astype(np.int32)
    elif dtype == 'int16':
        np_array = np_array.astype(np.int16)
    elif dtype == 'int8':
        np_array = np_array.astype(np.int8)
    else:
        raise ValueError(f"Unsupported dtype: {dtype}")
    
    # Flatten and write as binary
    np_array = np_array.flatten('C')  # C-order (row-major)
    np_array.tofile(output_path)
    
    print(f"  ✓ Binary file: {output_path}")
    print(f"    Size: {output_path.stat().st_size / 1024:.2f} KB")
    print(f"    Dtype: {dtype}")
    print(f"    Elements: {len(np_array)}")


def export_text(tensor: torch.Tensor, output_path: str, format: str = 'csv'):
    """
    Export tensor as text file.
    
    Args:
        tensor: Input tensor
        output_path: Output file path
        format: Output format ('csv', 'space', 'hex')
    """
    output_path = Path(output_path)
    
    # Convert to numpy and flatten
    np_array = tensor.detach().cpu().numpy()
    np_array = np_array.flatten('C')  # C-order (row-major)
    
    if format == 'csv':
        # CSV format: one value per line
        np.savetxt(output_path, np_array, fmt='%.9e', delimiter=',')
    elif format == 'space':
        # Space-separated values
        np.savetxt(output_path, np_array, fmt='%.9e', delimiter=' ')
    elif format == 'hex':
        # Hexadecimal format
        with open(output_path, 'w') as f:
            for val in np_array:
                f.write(f"{val.hex()}\n")
    else:
        raise ValueError(f"Unsupported format: {format}")
    
    print(f"  ✓ Text file: {output_path}")
    print(f"    Size: {output_path.stat().st_size / 1024:.2f} KB")
    print(f"    Format: {format}")


def export_header(
    tensor: torch.Tensor,
    output_path: str,
    var_name: str = "weights",
    dtype: str = "float"
):
    """
    Export tensor as C/C++ header file.
    
    Args:
        tensor: Input tensor
        output_path: Output file path
        var_name: Variable name in header
        dtype: C data type ('float', 'double', 'int32_t', 'int16_t', 'int8_t')
    """
    output_path = Path(output_path)
    
    # Convert to numpy and flatten
    np_array = tensor.detach().cpu().numpy()
    np_array = np_array.flatten('C')  # C-order (row-major)
    
    # Determine C type
    if dtype == 'float':
        c_type = 'float'
        fmt = '%.9ef'
    elif dtype == 'double':
        c_type = 'double'
        fmt = '%.18e'
    elif dtype == 'int32_t':
        c_type = 'int32_t'
        fmt = '%d'
    elif dtype == 'int16_t':
        c_type = 'int16_t'
        fmt = '%d'
    elif dtype == 'int8_t':
        c_type = 'int8_t'
        fmt = '%d'
    else:
        raise ValueError(f"Unsupported dtype: {dtype}")
    
    # Write header file
    with open(output_path, 'w') as f:
        f.write(f"// Auto-generated weight file\n")
        f.write(f"// Shape: {list(tensor.shape)}\n")
        f.write(f"// Total elements: {len(np_array)}\n")
        f.write(f"// Min: {np_array.min():.9e}, Max: {np_array.max():.9e}\n")
        f.write(f"// Mean: {np_array.mean():.9e}, Std: {np_array.std():.9e}\n")
        f.write(f"\n")
        f.write(f"#ifndef {var_name.upper()}_H\n")
        f.write(f"#define {var_name.upper()}_H\n")
        f.write(f"\n")
        f.write(f"#include <stdint.h>\n")
        f.write(f"\n")
        f.write(f"// Weight array shape: {list(tensor.shape)}\n")
        f.write(f"#define {var_name.upper()}_SHAPE {{{', '.join(map(str, tensor.shape))}}}\n")
        f.write(f"#define {var_name.upper()}_SIZE {len(np_array)}\n")
        f.write(f"\n")
        f.write(f"static const {c_type} {var_name}[] = {{\n")
        
        # Write values in rows of 8
        for i in range(0, len(np_array), 8):
            row_vals = [fmt % val for val in np_array[i:i+8]]
            if i + 8 < len(np_array):
                f.write(f"  {', '.join(row_vals)},\n")
            else:
                # Last row, remove trailing comma
                f.write(f"  {', '.join(row_vals)}\n")
        
        f.write(f"}};\n")
        f.write(f"\n")
        f.write(f"#endif // {var_name.upper()}_H\n")
    
    print(f"  ✓ Header file: {output_path}")
    print(f"    Size: {output_path.stat().st_size / 1024:.2f} KB")
    print(f"    Variable: {var_name}")


def export_metadata(tensor: torch.Tensor, output_path: str):
    """
    Export metadata about the tensor.
    
    Args:
        tensor: Input tensor
        output_path: Output file path
    """
    output_path = Path(output_path)
    
    np_array = tensor.detach().cpu().numpy()
    
    metadata = {
        'shape': list(tensor.shape),
        'dtype': str(tensor.dtype),
        'numel': int(tensor.numel()),
        'min': float(np_array.min()),
        'max': float(np_array.max()),
        'mean': float(np_array.mean()),
        'std': float(np_array.std()),
        'memory_size_bytes': int(tensor.numel() * tensor.element_size()),
    }
    
    with open(output_path, 'w') as f:
        f.write("# Weight Metadata\n")
        f.write("# ================\n\n")
        for key, value in metadata.items():
            f.write(f"{key}: {value}\n")
    
    print(f"  ✓ Metadata file: {output_path}")


def process_single_tensor(
    tensor_path: str,
    output_dir: Path,
    output_name: Optional[str],
    export_all: bool,
    export_binary_flag: bool,
    export_text_flag: bool,
    export_header_flag: bool,
    export_metadata_flag: bool,
    binary_dtype: str,
    fixed_point: bool,
    integer_bits: int,
    fractional_bits: int,
    verbose: bool = True
):
    """
    Process a single tensor file and export it in the requested formats.
    
    Args:
        tensor_path: Path to the tensor file
        output_dir: Output directory for exported files
        output_name: Base name for output files (None to infer from tensor_path)
        export_all: Export all formats
        export_binary_flag: Export binary file
        export_text_flag: Export text file
        export_header_flag: Export header file
        export_metadata_flag: Export metadata file
        binary_dtype: Binary output dtype
        fixed_point: Enable fixed-point conversion
        integer_bits: Fixed-point integer bits
        fractional_bits: Fixed-point fractional bits
        verbose: Print detailed information
    """
    if verbose:
        print("=" * 60)
        print(f"Loading weights from tensor file: {tensor_path}")
        print("=" * 60)
    
    weight_tensor = load_tensor_file(tensor_path)
    
    if verbose:
        print(f"\nTensor Information:")
        print(f"  Shape: {weight_tensor.shape}")
        print(f"  Dtype: {weight_tensor.dtype}")
        print(f"  Elements: {weight_tensor.numel():,}")
        print(f"  Memory: {weight_tensor.numel() * weight_tensor.element_size() / 1024:.2f} KB")
        print(f"  Min: {weight_tensor.min().item():.6f}")
        print(f"  Max: {weight_tensor.max().item():.6f}")
        print(f"  Mean: {weight_tensor.mean().item():.6f}")
        print(f"  Std: {weight_tensor.std().item():.6f}")
    
    # Handle fixed-point conversion
    if fixed_point:
        if verbose:
            print(f"\nConverting to fixed-point...")
            print(f"  Integer bits: {integer_bits}")
            print(f"  Fractional bits: {fractional_bits}")
        weight_tensor, fp_metadata = convert_to_fixed_point(
            weight_tensor,
            integer_bits=integer_bits,
            fractional_bits=fractional_bits
        )
        if verbose:
            print(f"  Scale: {fp_metadata['scale']}")
            print(f"  Range: [{fp_metadata['min_val']:.6f}, {fp_metadata['max_val']:.6f}]")
        # Convert to float for storage
        weight_tensor = weight_tensor.float() / fp_metadata['scale']
    
    # Determine output name
    if output_name:
        base_name = output_name
    else:
        tensor_path_obj = Path(tensor_path)
        base_name = tensor_path_obj.stem  # Remove extension
    
    if verbose:
        print(f"\n" + "=" * 60)
        print(f"Exporting weights...")
        print("=" * 60)
    
    # Export formats
    if export_all or export_binary_flag:
        binary_path = output_dir / f"{base_name}.bin"
        export_binary(weight_tensor, binary_path, dtype=binary_dtype)
    
    if export_all or export_text_flag:
        text_path = output_dir / f"{base_name}.txt"
        export_text(weight_tensor, text_path, format='csv')
    
    if export_all or export_header_flag:
        header_path = output_dir / f"{base_name}.h"
        var_name = base_name.replace('.', '_').replace('-', '_')
        # Ensure variable name starts with a letter or underscore (valid C identifier)
        if var_name and var_name[0].isdigit():
            var_name = 'w_' + var_name  # Prefix with 'w_' if starts with digit
        if not var_name:
            var_name = 'weights'  # Fallback name
        export_header(weight_tensor, header_path, var_name=var_name)
    
    if export_all or export_metadata_flag:
        metadata_path = output_dir / f"{base_name}_metadata.txt"
        export_metadata(weight_tensor, metadata_path)


def main():
    parser = argparse.ArgumentParser(
        description="Export model weights to FPGA-friendly formats"
    )
    parser.add_argument(
        'input_path',
        type=str,
        help='Path to .tensor file or directory containing .tensor files'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='weights_export',
        help='Output directory (default: weights_export)'
    )
    parser.add_argument(
        '--output-name',
        type=str,
        default=None,
        help='Base name for output files (default: inferred from tensor_file). Only used for single file mode.'
    )
    parser.add_argument(
        '--binary',
        action='store_true',
        help='Export binary file'
    )
    parser.add_argument(
        '--text',
        action='store_true',
        help='Export text file'
    )
    parser.add_argument(
        '--header',
        action='store_true',
        help='Export C/C++ header file'
    )
    parser.add_argument(
        '--metadata',
        action='store_true',
        help='Export metadata file'
    )
    parser.add_argument(
        '--all',
        action='store_true',
        help='Export all formats'
    )
    parser.add_argument(
        '--binary-dtype',
        type=str,
        default='float32',
        choices=['float32', 'float16', 'int32', 'int16', 'int8'],
        help='Binary output dtype (default: float32)'
    )
    parser.add_argument(
        '--fixed-point',
        action='store_true',
        help='Convert to fixed-point representation'
    )
    parser.add_argument(
        '--integer-bits',
        type=int,
        default=8,
        help='Fixed-point integer bits (default: 8)'
    )
    parser.add_argument(
        '--fractional-bits',
        type=int,
        default=8,
        help='Fixed-point fractional bits (default: 8)'
    )
    parser.add_argument(
        '--quiet',
        action='store_true',
        help='Reduce output verbosity (useful when processing multiple files)'
    )
    
    args = parser.parse_args()
    
    # Determine output formats
    export_all = args.all or (not args.binary and not args.text and not args.header and not args.metadata)
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Check if input is a directory or file
    input_path = Path(args.input_path)
    if not input_path.exists():
        raise FileNotFoundError(f"Input path not found: {input_path}")
    
    if input_path.is_dir():
        # Directory mode: find all .tensor files
        tensor_files = sorted(input_path.glob("*.tensor"))
        if len(tensor_files) == 0:
            raise ValueError(f"No .tensor files found in directory: {input_path}")
        
        print("=" * 60)
        print(f"Found {len(tensor_files)} tensor file(s) in directory: {input_path}")
        print("=" * 60)
        
        if not args.quiet:
            print(f"\nFiles to process:")
            for i, tf in enumerate(tensor_files, 1):
                print(f"  {i}. {tf.name}")
            print()
        
        # Process each tensor file
        successful = 0
        failed = 0
        
        for i, tensor_file in enumerate(tensor_files, 1):
            try:
                if not args.quiet:
                    print(f"\n[{i}/{len(tensor_files)}] Processing: {tensor_file.name}")
                    print("-" * 60)
                
                process_single_tensor(
                    tensor_path=str(tensor_file),
                    output_dir=output_dir,
                    output_name=None,  # Always infer from filename in directory mode
                    export_all=export_all,
                    export_binary_flag=args.binary,
                    export_text_flag=args.text,
                    export_header_flag=args.header,
                    export_metadata_flag=args.metadata,
                    binary_dtype=args.binary_dtype,
                    fixed_point=args.fixed_point,
                    integer_bits=args.integer_bits,
                    fractional_bits=args.fractional_bits,
                    verbose=not args.quiet
                )
                
                successful += 1
                
            except Exception as e:
                failed += 1
                print(f"\n❌ Error processing {tensor_file.name}: {e}")
                if not args.quiet:
                    import traceback
                    traceback.print_exc()
        
        # Summary
        print(f"\n" + "=" * 60)
        print(f"Batch export complete!")
        print(f"  Successful: {successful}")
        print(f"  Failed: {failed}")
        print(f"  Output directory: {output_dir}")
        print("=" * 60)
        
    else:
        # Single file mode
        process_single_tensor(
            tensor_path=str(input_path),
            output_dir=output_dir,
            output_name=args.output_name,
            export_all=export_all,
            export_binary_flag=args.binary,
            export_text_flag=args.text,
            export_header_flag=args.header,
            export_metadata_flag=args.metadata,
            binary_dtype=args.binary_dtype,
            fixed_point=args.fixed_point,
            integer_bits=args.integer_bits,
            fractional_bits=args.fractional_bits,
            verbose=True
        )
        
        print(f"\n" + "=" * 60)
        print(f"Export complete!")
        print(f"Output directory: {output_dir}")
        print("=" * 60)


if __name__ == "__main__":
    main()

