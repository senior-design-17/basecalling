<!-- a79d8b46-c29b-49f1-b99f-38e7b102b724 7367baaa-70fe-4a34-911c-b08cb1672dac -->

# Basecalling Workflow Outline

## Overview

Basecalling is the process of converting raw electrical signals from a nanopore sequencer to an ACTG sequence and quality score string.

This plan lays out a CPU and FPGA real-time basecalling architecture that aims to be fast and power-efficient. The high-level workflow is based on the [Dorado](https://github.com/nanoporetech/dorado) basecaller (also available in the `./dorado/` directory), which converts `.pod5` data to a `.fastq` file.

## 1. POD5 parsing, signal normalization, and chunking on CPU

A CPU shall receive POD5 data from the sequencer. The raw signal data and read_id shall be extracted using Python's POD5 library. The signal data shall be normalized similar to Dorado's implementation, chunked, and sent to the FPGA via PCIe.

The functionality for this shall be implemented in `./cpu/stream_basecaller.py`.

**Relevant Resources:**

- [Python POD5 Package](https://pypi.org/project/pod5/)
- [Dorado's Normalization Implementation](https://github.com/nanoporetech/dorado/blob/f9443bb8695f075dadc60bf4d1d92d8fd4361668/dorado/read_pipeline/nodes/ScalerNode.cpp#L271)

## 2. Neural Network Inference on FPGA

The FPGA shall receive data for each chunk from the CPU via PCIe using a memory-mapped DMA interface.

**PCIe Memory-Mapped Interface:**

```SystemVerilog
-- FPGA PCIe input interface (memory-mapped)
typedef struct packed {
    logic [127:0] read_id;
    logic [31:0] chunk_id;
    logic [15:0] actual_length;     // Actual signal length (≤ 10000)
    logic [31:0] data_length;      // Total data length in bytes (actual_length * 2)
    logic is_last;                  // Last chunk for this read
    logic data_ready;               // Data ready flag (set by CPU when data written)
} chunk_input_header;

-- Signal data buffer (20 KB per chunk)
-- Located at PCIe BAR address: BASE_ADDR + (chunk_id * BUFFER_SIZE)
logic [15:0] signal_buffer[0:9999]; // 10000 × 16-bit values = 20 KB
```

**PCIe Transfer Process:**

1. CPU writes normalized signal data to `signal_buffer` (up to `actual_length` samples)
2. CPU writes header metadata to control register at `BASE_ADDR`
3. CPU sets `data_ready` flag in control register
4. FPGA polls control register or receives interrupt when `data_ready` is set
5. FPGA reads header from control register
6. FPGA performs DMA read of signal data from `signal_buffer` via PCIe
7. FPGA clears `data_ready` flag after reading

**Input Specifications:**

- **Transfer Protocol**: PCIe DMA (memory-mapped I/O)
- **Signal Data**: Normalized 16-bit signal values (up to 10,000 samples per chunk)
- **Format**: 16-bit integer values representing normalized signal
- **Maximum Size**: 10,000 × 2 bytes = 20 KB per chunk
- **Memory Layout**:
  - Control register: `BASE_ADDR` (contains header)
  - Data buffer: `BASE_ADDR + 0x1000 + (chunk_id * 0x6000)` (20 KB aligned)
- **Data Ordering**: Signal samples stored sequentially: sample[0], sample[1], ..., sample[actual_length-1]

For each chunk, the FPGA shall input the signal data into a neural network and output the LSTM features (before the CRF layer).

The functionality for this shall be implemented in `./fpga/`.

### 2.1 Neural Network Architecture

**Relevant Resources:**

- [LSTM on FPGA](https://vast.cs.ucla.edu/sites/default/files/publications/ASP-DAC2017-1352-11.pdf)
- [Dorado DNA model](./model/dna_r10.4.1_e8.2_400bps_hac@v5.2.0/)

The FPGA shall implement the feature extraction portion of a CRF (Conditional Random Field) neural network with the following architecture:

**Input Processing:**

- Input: Raw signal data (10,000 samples, 16-bit floats)
- Standardization: Apply z-score normalization with mean=94.0, stdev=24.0
- Input shape: [1, 1, 10000] (batch=1, channels=1, time=10000)

**Convolutional Stack (ConvStack):**

1. **Conv1**: 1→16 channels, kernel=5, stride=1, padding=2, activation=swish, batch_norm
2. **Conv2**: 16→16 channels, kernel=5, stride=1, padding=2, activation=swish, batch_norm
3. **Conv3**: 16→384 channels, kernel=19, stride=6, padding=9, activation=tanh, batch_norm
   - Output shape: [1, 384, 1667] (time dimension reduced by stride=6)

**LSTM Stack (LSTMStack):**

- 6 bidirectional LSTM layers, each with 384 hidden units
- Alternating forward/reverse directions: [reverse, forward, reverse, forward, reverse, forward]
- Each LSTM processes the full sequence with hidden state propagation
- Output shape: [1, 384, 1667]

### 2.2 FPGA Neural Network Output

The FPGA produces a tensor of shape [1, 384, 1667] containing:

- **LSTM Features**: 384-dimensional feature vectors at each time step (1667 time steps)
- **Format**: Floating point values representing the LSTM hidden states
- **Note**: The Linear CRF layer and decoding are performed on CPU (see Section 3)

### 2.3 FPGA Output Interface

The FPGA shall output data for each processed chunk via PCIe using a simple memory-mapped DMA interface.

**PCIe Memory-Mapped Interface:**

```SystemVerilog
-- FPGA PCIe output interface (memory-mapped)
typedef struct packed {
    logic [127:0] read_id;
    logic [31:0] chunk_id;
    logic [15:0] actual_length;     // Original signal length
    logic [15:0] time_steps;        // Number of time steps (typically 1667)
    logic [15:0] feature_dim;       // Feature dimension (384)
    logic [31:0] data_length;      // Total data length in bytes (time_steps * feature_dim * 4)
    logic is_last;                  // Last chunk for this read
    logic data_ready;               // Data ready flag (set when features written to buffer)
} chunk_output_header;

-- Feature data buffer (2.56 MB per chunk)
-- Located at PCIe BAR address: BASE_ADDR + (chunk_id * BUFFER_SIZE)
logic [31:0] feature_buffer[0:640127]; // 640128 × 32-bit floats = [1667, 384]
```

**PCIe Transfer Process:**

1. FPGA computes LSTM features and writes to `feature_buffer` in row-major order
2. FPGA writes header metadata to status register at `BASE_ADDR`
3. FPGA sets `data_ready` flag in status register
4. CPU polls status register or receives interrupt when `data_ready` is set
5. CPU reads header from status register
6. CPU performs DMA read of feature data from `feature_buffer` via PCIe
7. CPU clears `data_ready` flag after reading

**Output Specifications:**

- **Transfer Protocol**: PCIe DMA (memory-mapped I/O)
- **Features**: LSTM output features of shape [1667, 384] (time_steps × feature_dim)
- **Format**: 32-bit floating point values (IEEE 754 single precision)
- **Total Elements**: 1667 × 384 = 640,128 float32 values
- **Total Size**: 640,128 × 4 bytes = ~2.56 MB per chunk
- **Memory Layout**:
  - Status register: `BASE_ADDR` (contains header)
  - Data buffer: `BASE_ADDR + 0x1000 + (chunk_id * 0x280000)` (2.56 MB aligned)
- **Data Ordering**: Features stored in row-major order: [t=0, f=0], [t=0, f=1], ..., [t=0, f=383], [t=1, f=0], ...
- **Timing**: Output available within 100ms of input completion

## 3. CRF Layer, Decoding, Chunk Stitching, and FASTQ Writing on CPU

The CPU shall receive the LSTM feature data from the FPGA for each chunk. The CPU shall perform the following operations:

**Relevant Resources:**

- [Dorado's Beam Search Implementation](https://github.com/nanoporetech/dorado/blob/main/dorado/basecall/decode/beam_search.cpp)
- [Dorado's CPU Decoder](https://github.com/nanoporetech/dorado/blob/main/dorado/basecall/decode/CPUDecoder.cpp)

### 3.1 Linear CRF Layer

- **Input**: LSTM features of shape [1667, 384]
- **Linear Layer**: 384 → 1024 features
  - 1024 classes (4^4 × 4 = 4^5 = 1024, representing 4-base k-mers with 4 possible transitions)
  - Activation: Clamp to [-5.0, 5.0] range
- **Output**: CRF scores of shape [1667, 1024]

### 3.2 Beam Search Decoding Process

The CPU shall implement beam search decoding (matching Dorado's implementation) to convert CRF scores into base sequences. This maintains multiple candidate paths simultaneously, providing better accuracy than greedy decoding.

**Beam Search Parameters:**

- **beam_width**: Maximum number of candidate paths to maintain (default: 32, max: 256)
- **beam_cut**: Log-space threshold for pruning low-scoring paths (default: 100.0)
- **blank_score**: Fixed score for "stay" transitions (default: 2.0)

**Step 1: Forward and Backward Score Computation**

- **Forward Pass**: Compute forward log probabilities using a scan operation
  - For each time step t, accumulate scores considering all possible transitions (4 bases) and stay operations
  - Use log-sum-exp for numerical stability
- **Backward Pass**: Compute backward log probabilities by reversing the sequence
- **Posterior Calculation**: Compute softmax(forward + backward) to get posterior probabilities for each state at each time step

**Step 2: Beam Search Decoding**

For each chunk, maintain a beam of candidate paths:

1. **Initialization**: Select top `beam_width` initial states based on backward guide scores
2. **Iterative Expansion**: For each time step (block):
   - Expand each beam element by considering:
     - All 4 possible base transitions (A, C, G, T)
     - One "stay" operation (no base emitted)
   - Score each candidate: `prev_score + transition_score + backward_guide_score`
   - Handle stay/step merging: When a stay path and step path produce equivalent sequences (same sequence hash), merge using log-sum-exp
   - Apply beam pruning:
     - Calculate beam cutoff: `max_score - log(beam_cut)`
     - Keep paths with scores ≥ cutoff, up to `beam_width` elements
     - Use binary search to maintain 80-100% of target beam width when needed
3. **Path Selection**: At the final time step, select the highest-scoring path
4. **Backward Pass (Path Reconstruction)**:
   - Follow backpointers from the best final element to reconstruct the optimal path
   - Extract move sequence (0=stay, 1=emit base) and state sequence

**Step 3: Sequence Generation**

- Convert state sequence to base sequence using the following mapping:
  - State % 4 = 0 → 'A', 1 → 'C', 2 → 'G', 3 → 'T'
- Apply move sequence to determine when bases are emitted
- Generate final base sequence of variable length (typically 200-400 bases)

**Step 4: Quality Score Calculation**

- For each emitted base, compute quality score from posterior probabilities:
  - Compute probability for the called base and alternative bases
  - Use posterior probabilities from forward/backward pass
  - Apply power scaling factor (0.4) to probabilities
  - Formula: Q = -10 × log10(1 - P_correct) × scale + shift
  - Scale and shift parameters from model config (typically scale=1.0, shift=0.0)
- Clamp quality scores to range [1, 50]
- Convert to Phred+33 ASCII: Q_char = char(33.5 + Q)

### 3.3 Chunk Stitching and FASTQ Writing

The CPU shall stitch all decoded chunks for each read and write each read to a FASTQ file for secondary analysis.

The functionality for this shall be implemented in `./cpu/stream_basecaller.py`.

## Testing Strategy

- Compare output with actual Dorado on small test POD5
- Verify sequence identity (should match Dorado closely when using same beam search parameters)
- Check quality score distribution
- **Expected Accuracy:** Should match Dorado accuracy when using identical beam search parameters (beam_width=32, beam_cut=100.0, blank_score=2.0)
- **Acceptable Differences:** Minor differences may arise from numerical precision or implementation details, but overall accuracy should be comparable
