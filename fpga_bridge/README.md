# CPU Bridge between Sequencer and FGPA 

According to the interface below, Gemini provided two different implementations

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

## C++ Implementation
This code assumes a Linux environment where you have mapped your PCIe BAR to a 
userspace pointer (common with UIO or VFIO drivers).

### Key Considerations for Integration (from Gemini)

1. The .pod5 / .fast5 Library Issue You cannot naively parse these files in real-time inside the inner loop if you want high speed.

Recommendation: Use the official lib_pod5 (C++ API) from Oxford Nanopore. It reads the "Arrow" columns very efficiently.

Optimization: Pre-load the whole file into RAM (if <32GB) or mmap the file itself so you can cast the pointers directly without malloc.

2. Memory Alignment (The packed struct) In the C++ code, I used __attribute__((packed)).

Without this, the C++ compiler might pad the struct to align uint128_t to a 16-byte boundary.

If your SystemVerilog packed struct has no holes, your C++ struct must have no holes.

Check: Use static_assert(sizeof(ChunkInputHeader) == EXPECTED_SIZE, "Struct mismatch"); to compile-fail if they don't match.

3. PCIe Write Ordering (The data_ready flag) The CPU is an out-of-order execution machine. It might decide to write the data_ready flag before the signal_buffer data because it thinks they are unrelated memory addresses.

The Fix: You must use a Memory Barrier (like _mm_sfence() on x86 or C++ std::atomic_thread_fence) between step 3 (write data) and step 5 (write header). This forces the CPU to finish writing the payload before it writes the doorbell.

4. Endianness

x86 CPUs (your laptop/server) are Little Endian.

Network/PCIe protocols are often Big Endian, but internal FPGA logic is whatever you designed it to be.

If your SystemVerilog reads read_id[127:0] and expects the MSB at index 127, ensure your C++ struct order (lo, hi) matches how the FPGA reconstructs it.


## Python Implementation

Easier in Python because libraries like pod5 and h5py (for fast5) handle the complex file parsing for you.

To bridge Python to the low-level FPGA memory, we use the struct library to pack bits exactly how the hardware expects them, and mmap to talk to the PCIe address space.

install these libraries

```bash
pip install pod5 h5py numpy
```
### Why Python is great for this (and where it fails)
#### The Good:

Parsing is Trivial: Reading .pod5 in C++ requires building the Arrow libraries which is a nightmare. In Python, it's just pip install pod5.

Slicing is Fast: signal_data[start:end] in NumPy creates a "view" (it doesn't copy memory), so the chunking step is instant.

#### The Bad:

The Global Interpreter Lock (GIL): If you are trying to stream 500 MB/s of data to the FPGA, Python might choke on the loop overhead.

Memory Mapping: Python's mmap is good, but it doesn't give you the same fine-grained control over CPU caching that C++ does.

Risk: The CPU might cache your write to data_ready and not flush it to the FPGA immediately.

Fix: I added the os.O_SYNC flag in the open call. This forces Linux to write directly to hardware, bypassing the cache.

#### A Note on the Struct Layout
In the code above, I used HEADER_FMT = "<QQIIHIBB".

BB corresponds to is_last (1 byte) and data_ready (1 byte).

Warning: Your SystemVerilog used packed, which technically compresses fields down to the individual bit.

If your FPGA designer literally packed is_last and data_ready into 2 bits next to each other, Python's byte-aligned write will result in garbage data.

However, in 99% of PCIe designs, we pad flags to be a full byte (logic [7:0] data_ready) specifically so software can write them easily. If your FPGA code is strict 1-bit, you need to change the Python to pack bits manually (e.g., flags = (is_last << 1) | data_ready).

