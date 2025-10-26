<!-- a79d8b46-c29b-49f1-b99f-38e7b102b724 7367baaa-70fe-4a34-911c-b08cb1672dac -->

# Basecalling Workflow Outline

## Overview

Basecalling is the process of converting raw electrical signals from a nanopore sequencer to an ACTG sequence and quality score string.

This plan lays out a CPU and FPGA real-time basecalling architecture that aims to be fast and power-efficient. The high-level workflow is based on the [Dorado](https://github.com/nanoporetech/dorado) basecaller, which converts `.pod5` data to a `.fastq` file.

## 1. POD5 parsing, signal normalization, and chunking on CPU

A CPU shall receive POD5 data from the sequencer. The raw signal data and relevant metadata shall be extracted following POD5's format specification. The signal data shall be normalized similar to Dorado's implementation, chunked, and sent to the FPGA via USB or PCIe.

The functionality for this shall be implemented in `./cpu/stream_basecaller.py`.

**Relevant Resources:**

- [Python POD5 Package](https://pypi.org/project/pod5/)
- [Dorado's Normalization Implementation](https://github.com/nanoporetech/dorado/blob/f9443bb8695f075dadc60bf4d1d92d8fd4361668/dorado/read_pipeline/nodes/ScalerNode.cpp#L271)


## 2. Basecalling Inference and Decoding on FPGA

The FPGA shall receive the following data from each chunk from the CPU:

```vhdl
-- FPGA input interface
type chunk_input is record
    read_id       : std_logic_vector(127 downto 0);  -- UUID
    chunk_id      : std_logic_vector(31 downto 0);
    actual_length : std_logic_vector(15 downto 0);   -- Real samples (≤10000)
    is_last       : std_logic;
    signal_data   : signal_array(0 to 9999);         -- Always 10k, but only use first 'actual_length'
end record;
```

For each chunk, the FPGA shall input the signal data into a neural network and decode the output to a sequence and quality score.

The functionality for this shall be implemented in `./fpga/`.

### 2.1 Neural Network Architecture

**Relevant Resources:**
- [LSTM on FPGA](https://vast.cs.ucla.edu/sites/default/files/publications/ASP-DAC2017-1352-11.pdf)
- [Dorado DNA model](./model/dna_r10.4.1_e8.2_400bps_hac@v5.2.0/)

The FPGA shall implement a CRF (Conditional Random Field) neural network with the following architecture:

**Input Processing:**

- Input: Raw signal data (10,000 samples, 16-bit signed integers)
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

**Linear CRF Layer:**

- Input: 384 features
- Output: 1024 classes (4^4 \* 4 = 4^5 = 1024, representing 4-base k-mers with 4 possible transitions)
- Activation: Clamp to [-5.0, 5.0] range
- Output shape: [1, 1024, 1667]

### 2.2 Neural Network Output

The neural network produces a tensor of shape [1, 1024, 1667] containing:

- **Scores**: Log-probability scores for each possible k-mer transition at each time step
- **Classes**: 1024 possible states representing:
  - 4^4 = 256 possible 4-base k-mers (AAAA, AAAC, ..., TTTT)
  - 4 possible transitions (stay, emit A, emit C, emit G, emit T)
  - Total: 256 × 4 = 1024 states

### 2.3 Greedy Decoding Process

The FPGA shall implement greedy decoding to convert neural network scores into base sequences:

**Step 1: Forward Pass (Viterbi-like)**

- For each time step t, compute the best path ending at each state
- Use transition scores to determine optimal predecessor states
- Track backpointers for path reconstruction

**Step 2: Backward Pass (Path Reconstruction)**

- Start from the final time step with the highest-scoring state
- Follow backpointers to reconstruct the optimal path
- Extract move sequence (0=stay, 1=emit base) and state sequence

**Step 3: Sequence Generation**

- Convert state sequence to base sequence using the following mapping:
  - State % 4 = 0 → 'A', 1 → 'C', 2 → 'G', 3 → 'T'
- Apply move sequence to determine when bases are emitted
- Generate final base sequence of variable length (typically 200-400 bases)

**Step 4: Quality Score Calculation**

- For each emitted base, compute quality score from posterior probabilities
- Formula: Q = -10 × log10(1 - P_correct) × scale + bias
- Scale = 1.05, bias = -0.3 (from model config)
- Clamp quality scores to range [1, 50]
- Convert to Phred+33 ASCII: Q_char = char(33 + Q)

#### 2.4 FPGA Output Interface

The FPGA shall output the following data for each processed chunk:

```vhdl
-- FPGA output interface
type chunk_output is record
    read_id       : std_logic_vector(127 downto 0);  -- UUID (same as input)
    chunk_id      : std_logic_vector(31 downto 0);   -- Chunk ID (same as input)
    sequence_len  : std_logic_vector(15 downto 0);   -- Length of generated sequence
    sequence_data : std_logic_vector(0 to 4095);     -- Base sequence (A=00, C=01, G=10, T=11)
    quality_data  : std_logic_vector(0 to 4095);     -- Quality scores (6 bits per score)
    is_last       : std_logic;                       -- Last chunk flag
    valid         : std_logic;                       -- Output valid flag
end record;
```

**Output Specifications:**

- **Sequence**: Binary encoded bases (2 bits per base: A=00, C=01, G=10, T=11)
- **Quality**: 6-bit quality scores (0-63, representing Phred scores 0-30)
- **Length**: Actual sequence length (typically 200-400 bases)
- **Timing**: Output available within 100ms of input completion

## 3. Chunk stitching and FASTQ writing on CPU

The CPU shall receive the chunked_output data from the FPGA. The CPU shall stitch all the chunks for each read and write each read to a FASTQ file for secondary analysis.

The functionality for this shall be implemented in `./cpu/stream_basecaller.py`.

## Testing Strategy

- Compare output with actual Dorado on small test POD5
- Verify sequence identity (allowing for minor differences in decoding)
- Check quality score distribution
- **Expected Accuracy:** 85-95% of production Dorado accuracy
- **Acceptable Differences:** 5-15% accuracy loss due to greedy decoding
