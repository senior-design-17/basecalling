# CPU Basecalling Components

This directory contains Python implementations for CPU-based basecalling components, including POD5 parsing, signal normalization, and CRF layer processing.

## Setup

### 1. Create a virtual environment

```bash
python3 -m venv .
# On macOS/Linux:
source bin/activate
# On Windows:
bin\activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

## Adding New Python Packages

When you need to add a new Python package to the project:

1. **Activate the virtual environment** (if not already activated):

   ```bash
   source bin/activate  # macOS/Linux
   # or
   bin\activate  # Windows
   ```

2. **Install the new package**:

   ```bash
   pip install <package-name>
   # Or with a specific version:
   pip install <package-name>==<version>
   ```

3. **Update `requirements.txt`**:

   ```bash
   pip freeze > requirements.txt
   ```

   This will update the file with all currently installed packages and their versions, including the new one.

4. **Commit the changes**:
   ```bash
   git add requirements.txt
   git commit -m "Add <package-name> dependency"
   ```

**Note**: When you run `pip freeze`, it includes all packages in your environment. If you want to keep `requirements.txt` minimal (only direct dependencies), you can use `pip freeze` to identify the new package, then manually add it to the file and remove any unrelated packages that were added.

## Files

- `stream_basecaller.py`: Main script for streaming basecalling from POD5 files
- `crf_layer_implementation.py`: CRF layer implementation for transforming LSTM features
- `test_weight_loading.py`: Test script for loading model weights
- `CRF_LAYER_README.md`: Detailed documentation for the CRF layer implementation

## Usage

See the main project README.md for usage instructions.
