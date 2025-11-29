# CPU Bridge between Sequencer and FGPA accoridng to the interface below

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

