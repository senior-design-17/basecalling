import mmap
import struct
import numpy as np
import pod5
import h5py
import os

# ==========================================
# 1. Hardware Definitions
# ==========================================
MAX_SAMPLES = 10000
BYTES_PER_SAMPLE = 2
BUFFER_SIZE = MAX_SAMPLES * BYTES_PER_SAMPLE

# Struct format for 'chunk_input_header'
# < = Little Endian (Standard for x86 CPUs)
# Q = unsigned long long (64 bit) - used twice for 128-bit UUID
# I = unsigned int (32 bit)
# H = unsigned short (16 bit)
# B = unsigned char (8 bit) - used for logic flags
#
# Total Layout: 
#   read_id (16 bytes), chunk_id (4), actual_len (2), data_len (4), is_last (1), data_ready (1)
# Note: This assumes the FPGA expects the flags to occupy a full byte each (standard SW/HW bridge).
HEADER_FMT = "<QQIIHIBB" 
HEADER_SIZE = struct.calcsize(HEADER_FMT)

class FpgaBridge:
    def __init__(self, pcie_resource_path='/dev/uio0', map_size=1024*1024):
        """
        Initializes the connection to the FPGA via memory mapping.
        In simulation, we can use a temporary file instead of /dev/uio0.
        """
        self.pcie_path = pcie_resource_path
        
        # Open the PCIe BAR (represented as a file in Linux)
        # os.O_SYNC ensures writes hit the hardware immediately, not cached.
        try:
            self.f = os.open(self.pcie_path, os.O_RDWR | os.O_SYNC)
            self.mem = mmap.mmap(self.f, map_size)
            print(f"[System] PCIe mapped successfully at {self.pcie_path}")
        except FileNotFoundError:
            print("[System] PCIe device not found. Running in simulation mode (writing to local buffer).")
            self.mem = bytearray(map_size) # Simulation buffer

    def write_chunk(self, chunk_id, signal_chunk, read_id_parts, is_last):
        """
        Writes a single chunk (Header + Data) to the FPGA.
        """
        n_samples = len(signal_chunk)
        n_bytes = n_samples * BYTES_PER_SAMPLE
        
        # 1. Calculate Offsets
        # As per your spec: BASE + (chunk_id * BUFFER_SIZE)
        # We need to decide where the Header goes. 
        # usually: Header is at offset 0, Data is at offset sizeof(Header).
        base_offset = chunk_id * BUFFER_SIZE
        header_offset = base_offset
        data_offset = base_offset + HEADER_SIZE

        # 2. Pack the Header
        # read_id is split into two 64-bit integers (lo, hi)
        header_binary = struct.pack(
            HEADER_FMT,
            read_id_parts[0], read_id_parts[1], # 128-bit ID split
            chunk_id,
            n_samples,          # actual_length
            n_bytes,            # data_length
            1 if is_last else 0,
            1                   # data_ready (The "Go" signal)
        )

        # 3. Write Data Payload (The Shipping Container)
        # We convert the numpy array to raw bytes
        data_bytes = signal_chunk.tobytes()
        self.mem[data_offset : data_offset + n_bytes] = data_bytes

        # 4. Write Header (The Manifesto / Doorbell)
        # STRICT ORDERING: Write header last so 'data_ready' triggers only when data is safe.
        self.mem[header_offset : header_offset + HEADER_SIZE] = header_binary
        
        # debug info
        # print(f"  -> Wrote Chunk {chunk_id}: {n_samples} samples.")

    def close(self):
        if isinstance(self.mem, mmap.mmap):
            self.mem.close()
            os.close(self.f)

def process_file(filepath, fpga_bridge):
    """
    Detects file type, extracts signal, and chunks it for the FPGA.
    """
    signal_data = None
    read_id_uuid = None

    print(f"\n[Processing] {filepath}")

    # --- PARSING ---
    if filepath.endswith('.pod5'):
        with pod5.Reader(filepath) as reader:
            # Taking the first read for demonstration
            record = next(reader.reads())
            signal_data = record.signal
            read_id_uuid = record.read_id # UUID object
            
    elif filepath.endswith('.fast5'):
        with h5py.File(filepath, 'r') as f:
            # HDF5 structure varies; this is a common path for ONT files
            # You might need to adjust the group path based on your specific MinKNOW version
            try:
                read_name = list(f['Raw/Reads'].keys())[0]
                signal_dataset = f[f'Raw/Reads/{read_name}/Signal']
                signal_data = np.array(signal_dataset, dtype=np.int16)
                # Create a dummy UUID for fast5 or extract from metadata
                import uuid
                read_id_uuid = uuid.uuid4()
            except KeyError:
                print("Error: Could not find Signal path in Fast5.")
                return

    if signal_data is None:
        return

    # --- CHUNKING ---
    total_samples = len(signal_data)
    
    # Convert UUID to two 64-bit integers for the struct
    # uuid.int is a 128-bit integer. We mask it to get low/high parts.
    uuid_int = read_id_uuid.int
    uuid_lo = uuid_int & 0xFFFFFFFFFFFFFFFF
    uuid_hi = (uuid_int >> 64) & 0xFFFFFFFFFFFFFFFF
    
    # Calculate number of chunks
    num_chunks = (total_samples + MAX_SAMPLES - 1) // MAX_SAMPLES
    
    print(f"  Total Samples: {total_samples}")
    print(f"  Total Chunks:  {num_chunks}")

    for i in range(num_chunks):
        start = i * MAX_SAMPLES
        end = min(start + MAX_SAMPLES, total_samples)
        
        chunk_data = signal_data[start:end]
        is_last = (i == num_chunks - 1)
        
        fpga_bridge.write_chunk(
            chunk_id=i, 
            signal_chunk=chunk_data, 
            read_id_parts=(uuid_lo, uuid_hi), 
            is_last=is_last
        )

# ==========================================
# Main Execution
# ==========================================
if __name__ == "__main__":
    # 1. Setup Bridge (Simulated)
    # In real usage, use: bridge = FpgaBridge('/dev/xdma0_user') or similar
    bridge = FpgaBridge(pcie_resource_path='dummy_pcie.bin')

    # 2. Run on a file (Make sure you have a file or comment this out)
    # process_file("test_data.pod5", bridge)
    
    # 3. cleanup
    bridge.close()
