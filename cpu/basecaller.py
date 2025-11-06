import pod5
import numpy as np
import os
from queue import Queue
from threading import Thread
import time
import torch
from pathlib import Path
from fpga_neural_network import create_fpga_network
from crf_layer_implementation import CRFEncoder, load_crf_encoder_with_weights
from beam_search_decode import beam_search_decode
from forward_backward import compute_forward_backward_posterior
import threading
import logging
import argparse
from datetime import datetime

# Configuration
CHUNK_SIZE = 10000
OVERLAP = 500
STRIDE = 6  # Model stride from config (conv3 has stride=6)
STANDARDISE_MEAN = 94.0
STANDARDISE_STDEV = 24.0
POD5_FILE = "path/to/your/file.pod5"
MODEL_DIR = Path(__file__).parent.parent / "model" / "dna_r10.4.1_e8.2_400bps_hac@v5.2.0"
OUTPUT_FASTQ = "output.fastq"  # Output FASTQ file path

# Beam search parameters
MAX_BEAM_WIDTH = 32
BEAM_CUT = 100.0
FIXED_STAY_SCORE = 2.0
Q_SHIFT = 0.0
Q_SCALE = 1.0

# FPGA simulation delays (in seconds)
# PCIe input transfer: ~0.1ms for 20KB (10000 samples * 2 bytes)
PCIe_INPUT_DELAY = 0.0001  # 0.1ms
# FPGA processing: ~50-100ms for neural network inference
FPGA_PROCESSING_DELAY = 0.075  # 75ms (average)
# PCIe output transfer: ~2-5ms for 2.56MB (1667 * 384 * 4 bytes)
PCIe_OUTPUT_DELAY = 0.003  # 3ms

# Queues for pipelining
preprocessing_queue = Queue(maxsize=100)
fpga_input_queue = Queue(maxsize=100)
fpga_output_queue = Queue(maxsize=100)

# Track processed reads to avoid reprocessing
processed_reads = set()

# Thread-safe file writing lock
fastq_lock = threading.Lock()

# Global logger (will be initialized by setup_logging)
logger = None

# Metrics tracking
metrics_lock = threading.Lock()
metrics = {
    'start_time': None,
    'end_time': None,
    'read_timings': {},  # {read_id: {'start': time, 'end': time, 'bases': int}}
    'total_bases': 0,
    'total_reads': 0
}


class WorkerFormatter(logging.Formatter):
    """Custom formatter that extracts worker name from logger name."""
    def format(self, record):
        # Extract worker name from logger name (e.g., 'basecaller.monitor' -> 'monitor')
        logger_name = record.name
        if '.' in logger_name:
            worker_name = logger_name.split('.')[-1]
        else:
            worker_name = logger_name
        record.worker_name = worker_name
        return super().format(record)


def setup_logging(verbose=False):
    """
    Setup logging configuration.
    
    Args:
        verbose: If True, enable logging. If False, disable all logging.
    """
    global logger
    
    if verbose:
        # Clear any existing handlers to avoid duplicates
        root_logger = logging.getLogger()
        root_logger.handlers.clear()
        
        # Configure root logger to output to stdout with custom format
        handler = logging.StreamHandler(__import__('sys').stdout)
        formatter = WorkerFormatter(
            fmt='%(asctime)s [%(worker_name)s] %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        handler.setFormatter(formatter)
        
        root_logger.setLevel(logging.INFO)
        root_logger.addHandler(handler)
        
        logger = logging.getLogger(__name__)
    else:
        # Disable all logging
        root_logger = logging.getLogger()
        root_logger.handlers.clear()
        root_logger.setLevel(logging.CRITICAL + 1)
        logger = logging.getLogger(__name__)
        logger.disabled = True


def get_worker_logger(worker_name):
    """
    Get a logger for a specific worker thread.
    
    Args:
        worker_name: Name of the worker (e.g., 'monitor', 'preprocess', 'fpga', 'assembler', 'main')
    
    Returns:
        Logger instance with worker name
    """
    if logger is None or logger.disabled:
        # Return a disabled logger if logging is not enabled
        disabled_logger = logging.getLogger(f'{__name__}.{worker_name}')
        disabled_logger.disabled = True
        return disabled_logger
    return logging.getLogger(f'{__name__}.{worker_name}')


def monitor_pod5_file(pod5_path, output_queue, poll_interval=1.0):
    """
    Continuously monitor POD5 file for new reads.
    In real-time, the sequencer appends reads to the file.
    
    Args:
        pod5_path: Path to the POD5 file to monitor
        output_queue: Queue to put new read records into
        poll_interval: Time in seconds between checks for new reads
    """
    global processed_reads
    worker_logger = get_worker_logger('monitor')
    
    while True:
        try:
            # Check if file exists
            if not os.path.exists(pod5_path):
                time.sleep(poll_interval)
                continue
            
            # Open POD5 file and read all reads
            with pod5.Reader(pod5_path) as reader:
                # Iterate through all reads in the file
                for read_record in reader.reads():
                    # Get read ID as string for tracking
                    read_id = str(read_record.read_id)
                    
                    # Skip if already processed
                    if read_id in processed_reads:
                        continue
                    
                    # Mark as processed
                    processed_reads.add(read_id)
                    
                    # Extract all necessary data while reader is still open
                    # This prevents "ArrowTableHandle has been closed!" errors
                    try:
                        signal = read_record.signal  # numpy array, dtype=int16
                        calibration = read_record.calibration
                        calibration_scale = calibration.scale
                        calibration_offset = calibration.offset
                        
                        # Create a dictionary with all necessary data
                        read_data = {
                            'read_id': read_id,
                            'signal': signal,
                            'calibration_scale': calibration_scale,
                            'calibration_offset': calibration_offset
                        }
                        
                        # Record start time for this read
                        with metrics_lock:
                            if metrics['start_time'] is None:
                                metrics['start_time'] = time.time()
                            metrics['read_timings'][read_id] = {
                                'start': time.time(),
                                'end': None,
                                'bases': 0
                            }
                        
                        # Put read data into output queue
                        output_queue.put(read_data)
                        
                        worker_logger.info(f"New read detected: {read_id}")
                    except Exception as e:
                        worker_logger.error(f"Error extracting data for read {read_id}: {e}")
                        continue
        
        except FileNotFoundError:
            # File doesn't exist yet, wait and retry
            time.sleep(poll_interval)
            continue
        
        except Exception as e:
            worker_logger.error(f"Error monitoring POD5 file: {e}")
            time.sleep(poll_interval)
            continue
        
        # Wait before checking again
        time.sleep(poll_interval)


def generate_chunks(num_samples, chunk_size, stride, overlap):
    """
    Generate overlapping chunk offsets for signal data.
    
    Based on Dorado's generate_chunks implementation.
    
    Args:
        num_samples: Total number of samples in the signal
        chunk_size: Size of each chunk
        stride: Model stride (for alignment)
        overlap: Overlap between chunks
        
    Returns:
        List of chunk start offsets
    """
    if num_samples == 0:
        return []
    
    offsets = [0]
    
    if num_samples <= chunk_size:
        return offsets
    
    # Calculate last chunk start position, aligned to stride
    last_offset = num_samples - chunk_size
    if last_offset % stride != 0:
        last_offset += stride - (last_offset % stride)
    
    chunk_step = chunk_size - overlap
    offset = 0
    
    while offset + chunk_size < num_samples:
        offset = min(offset + chunk_step, last_offset)
        offsets.append(offset)
    
    return offsets


def preprocess_worker(input_queue, output_queue):
    """
    Worker thread: preprocess reads and chunk them.
    
    Based on Dorado's ScalerNode implementation:
    1. Extract signal from POD5 read_record (int16)
    2. Apply PA scaling with standardisation
    3. Convert to float16
    4. Generate overlapping chunks
    5. Send chunks to FPGA input queue
    """
    worker_logger = get_worker_logger('preprocess')
    
    while True:
        read_data = input_queue.get()
        
        if read_data is None:  # Poison pill for shutdown
            break
        
        try:
            # Extract data from dictionary (already extracted while reader was open)
            read_id = read_data['read_id']
            signal = read_data['signal']  # numpy array, dtype=int16
            calibration_scale = read_data['calibration_scale']
            calibration_offset = read_data['calibration_offset']
            
            if signal is None or len(signal) == 0:
                worker_logger.warning(f"Empty signal for read {read_id}")
                continue
            
            # Apply PA scaling with standardisation (based on ScalerNode.cpp)
            # For PA strategy with standardisation:
            # scale = stdev / calibration_scale
            # shift = (mean / calibration_scale) - calibration_offset
            # Then: (signal - shift) / scale
            # This is equivalent to: ((signal + calibration_offset) * calibration_scale - mean) / stdev
            
            scale = STANDARDISE_STDEV / calibration_scale
            shift = (STANDARDISE_MEAN / calibration_scale) - calibration_offset
            
            # Convert to float32 for computation, then apply shift/scale
            signal_f32 = signal.astype(np.float32)
            signal_normalized = (signal_f32 - shift) / scale
            
            # Convert to float16 (as per Dorado's implementation)
            signal_f16 = signal_normalized.astype(np.float16)
            
            # Generate chunk offsets
            chunk_offsets = generate_chunks(len(signal_f16), CHUNK_SIZE, STRIDE, OVERLAP)
            
            if len(chunk_offsets) == 0:
                worker_logger.warning(f"No chunks generated for read {read_id}")
                continue
            
            # Create and send chunks
            for chunk_idx, offset in enumerate(chunk_offsets):
                # Extract chunk
                chunk_end = min(offset + CHUNK_SIZE, len(signal_f16))
                chunk_signal = signal_f16[offset:chunk_end]
                
                # Pad if necessary (shouldn't happen for middle chunks, but last chunk might be short)
                if len(chunk_signal) < CHUNK_SIZE:
                    # Repeat-pad the chunk (as per Dorado's BasecallerNode)
                    n_repeats = CHUNK_SIZE // len(chunk_signal)
                    remainder = CHUNK_SIZE % len(chunk_signal)
                    chunk_signal = np.concatenate([
                        np.tile(chunk_signal, n_repeats),
                        chunk_signal[:remainder]
                    ])
                
                # Determine if this is the last chunk
                is_last = (chunk_idx == len(chunk_offsets) - 1)
                
                # Create chunk data structure
                chunk_data = {
                    'read_id': read_id,
                    'chunk_id': chunk_idx,
                    'signal': chunk_signal,  # numpy array, shape=(CHUNK_SIZE,), dtype=float16
                    'offset': offset,  # Original offset in the full signal
                    'is_last': is_last
                }
                
                # Send to FPGA input queue
                output_queue.put(chunk_data)
                
        except Exception as e:
            read_id_str = read_data.get('read_id', 'unknown') if isinstance(read_data, dict) else 'unknown'
            worker_logger.error(f"Preprocessing error for read {read_id_str}: {e}")
            import traceback
            traceback.print_exc()
        
        finally:
            input_queue.task_done()


def fpga_sender_worker(input_queue, output_queue, model_dir=None):
    """
    Worker thread: send chunks to FPGA and collect results.
    
    Simulates FPGA-CPU communication by:
    1. Simulating PCIe input transfer delay
    2. Running neural network inference (simulating FPGA processing)
    3. Simulating PCIe output transfer delay
    
    Args:
        input_queue: Queue containing chunk_data dictionaries from preprocessor
        output_queue: Queue to put processed chunk results
        model_dir: Path to model directory (defaults to MODEL_DIR)
    """
    worker_logger = get_worker_logger('fpga')
    
    # Lazy initialization of FPGA network
    fpga_network = None
    
    if model_dir is None:
        model_dir = MODEL_DIR
    
    while True:
        chunk_data = input_queue.get()
        
        if chunk_data is None:  # Poison pill
            break
        
        try:
            # Initialize network on first use
            if fpga_network is None:
                worker_logger.info(f"Initializing FPGA neural network from {model_dir}")
                if not model_dir.exists():
                    raise FileNotFoundError(f"Model directory not found: {model_dir}")
                fpga_network = create_fpga_network(str(model_dir))
                worker_logger.info("FPGA neural network initialized")
            
            read_id = chunk_data['read_id']
            chunk_id = chunk_data['chunk_id']
            signal = chunk_data['signal']  # numpy array, shape=(CHUNK_SIZE,), dtype=float16
            offset = chunk_data['offset']
            is_last = chunk_data['is_last']
            
            # Simulate PCIe input transfer delay
            time.sleep(PCIe_INPUT_DELAY)
            
            # Convert signal to torch tensor
            # Input shape: [1, 10000] or [1, 1, 10000]
            # The network expects [batch, time] or [batch, channels, time]
            signal_tensor = torch.from_numpy(signal.astype(np.float32))
            signal_tensor = signal_tensor.unsqueeze(0)  # [1, 10000]
            
            # Run inference through FPGA neural network
            # The network handles standardization internally
            # Measure actual inference time and add delay to simulate FPGA processing time
            inference_start = time.time()
            with torch.no_grad():
                features = fpga_network(signal_tensor)  # Output: [1, 384, 1667]
            inference_time = time.time() - inference_start
            
            # Add additional delay to reach target FPGA processing time
            # This simulates the fixed processing time of an FPGA implementation
            remaining_delay = max(0, FPGA_PROCESSING_DELAY - inference_time)
            if remaining_delay > 0:
                time.sleep(remaining_delay)
            
            # Convert output to numpy array
            # Shape: [1, 384, 1667] -> [1667, 384] (time_steps, feature_dim)
            features_np = features.squeeze(0).permute(1, 0).cpu().numpy()  # [1667, 384]
            features_np = features_np.astype(np.float32)  # Ensure float32
            
            # Simulate PCIe output transfer delay
            time.sleep(PCIe_OUTPUT_DELAY)
            
            # Create output chunk data structure
            # Format matches what the assembler expects
            result_data = {
                'read_id': read_id,
                'chunk_id': chunk_id,
                'features': features_np,  # numpy array, shape=(1667, 384), dtype=float32
                'offset': offset,
                'is_last': is_last,
                'time_steps': features_np.shape[0],  # 1667
                'feature_dim': features_np.shape[1]  # 384
            }
            
            # Send to output queue
            output_queue.put(result_data)
            
        except Exception as e:
            read_id_str = chunk_data.get('read_id', 'unknown') if chunk_data else 'unknown'
            chunk_id_str = chunk_data.get('chunk_id', 'unknown') if chunk_data else 'unknown'
            worker_logger.error(f"FPGA communication error for read {read_id_str}, chunk {chunk_id_str}: {e}")
            import traceback
            traceback.print_exc()
        
        finally:
            input_queue.task_done()


def basecall_assembler_worker(input_queue, model_dir=None, output_fastq=None):
    """
    Worker thread: for post-processing of chunks.
    
    Processes chunks through CRF layer and beam search decoding, then stitches
    chunks together and writes to FASTQ file.
    
    Args:
        input_queue: Queue containing processed chunk results from FPGA worker
        model_dir: Path to model directory (defaults to MODEL_DIR)
        output_fastq: Path to output FASTQ file (defaults to OUTPUT_FASTQ)
    """
    worker_logger = get_worker_logger('assembler')
    
    # Lazy initialization of CRF encoder
    crf_encoder = None
    
    if model_dir is None:
        model_dir = MODEL_DIR
    if output_fastq is None:
        output_fastq = OUTPUT_FASTQ
    
    # Track chunks by read_id: {read_id: {chunk_id: chunk_data}}
    read_chunks = {}  # {read_id: {chunk_id: {'sequence': str, 'qstring': str, 'moves': list, 'offset': int}}}
    expected_chunks = {}  # {read_id: num_chunks}
    
    while True:
        result = input_queue.get()
        
        if result is None:  # Poison pill
            break
        
        try:
            # Initialize CRF encoder on first use
            if crf_encoder is None:
                worker_logger.info(f"Initializing CRF encoder from {model_dir}")
                if not model_dir.exists():
                    raise FileNotFoundError(f"Model directory not found: {model_dir}")
                crf_encoder = load_crf_encoder_with_weights(str(model_dir))
                crf_encoder.eval()
                worker_logger.info("CRF encoder initialized")
            
            read_id = result['read_id']
            chunk_id = result['chunk_id']
            features = result['features']  # numpy array, shape=(1667, 384), dtype=float32
            offset = result['offset']
            is_last = result['is_last']
            
            # Convert features to torch tensor: [T, C] -> [1, T, C] for batch processing
            features_t = torch.from_numpy(features).unsqueeze(0)  # [1, 1667, 384]
            
            # Pass through CRF layer: [1, 1667, 384] -> [1, 1667, 1024]
            with torch.no_grad():
                crf_scores = crf_encoder(features_t)  # [1, 1667, 1024]
            
            # Transpose to [T, N, C] format for forward/backward computation
            # crf_scores is [1, 1667, 1024], we need [1667, 1, 1024]
            crf_scores_TNC = crf_scores.transpose(0, 1)  # [1667, 1, 1024]
            
            # Compute forward, backward, and posterior probabilities
            fwd, bwd, posts = compute_forward_backward_posterior(
                crf_scores_TNC,
                fixed_stay_score=FIXED_STAY_SCORE
            )
            
            # Beam search decoding
            # bwd is used as back_guide, shape [T+1, N, num_states]
            sequence, qstring, moves = beam_search_decode(
                crf_scores_TNC.squeeze(1),  # [1667, 1024] - remove batch dimension
                bwd.squeeze(1),  # [1667+1, num_states] - backward guide
                posts.squeeze(1),  # [1667+1, num_states] - posterior probabilities
                max_beam_width=MAX_BEAM_WIDTH,
                beam_cut=BEAM_CUT,
                fixed_stay_score=FIXED_STAY_SCORE,
                q_shift=Q_SHIFT,
                q_scale=Q_SCALE
            )
            
            # Store chunk results
            if read_id not in read_chunks:
                read_chunks[read_id] = {}
                # Estimate number of chunks (we'll update when we see is_last=True)
                expected_chunks[read_id] = chunk_id + 1 if is_last else None
            
            read_chunks[read_id][chunk_id] = {
                'sequence': sequence,
                'qstring': qstring,
                'moves': moves,
                'offset': offset
            }
            
            # Update expected chunks count if this is the last chunk
            if is_last:
                expected_chunks[read_id] = chunk_id + 1
            
            # Check if all chunks for this read are collected
            if read_id in expected_chunks and expected_chunks[read_id] is not None:
                num_expected = expected_chunks[read_id]
                if len(read_chunks[read_id]) == num_expected:
                    # All chunks collected, stitch and write
                    chunks_list = [read_chunks[read_id][i] for i in range(num_expected)]
                    stitched_seq, stitched_qstring = stitch_chunks(chunks_list, STRIDE)
                    
                    # Write to FASTQ file
                    write_fastq(read_id, stitched_seq, stitched_qstring, output_fastq)
                    
                    # Record end time and update metrics
                    num_bases = len(stitched_seq)
                    with metrics_lock:
                        if read_id in metrics['read_timings']:
                            metrics['read_timings'][read_id]['end'] = time.time()
                            metrics['read_timings'][read_id]['bases'] = num_bases
                        metrics['total_bases'] += num_bases
                        metrics['total_reads'] += 1
                    
                    worker_logger.info(f"Basecalled read {read_id}: {num_bases} bases")
                    
                    # Clean up
                    del read_chunks[read_id]
                    del expected_chunks[read_id]

        except Exception as e:
            read_id_str = result.get('read_id', 'unknown') if result else 'unknown'
            chunk_id_str = result.get('chunk_id', 'unknown') if result else 'unknown'
            worker_logger.error(f"Assembly error for read {read_id_str}, chunk {chunk_id_str}: {e}")
            import traceback
            traceback.print_exc()
        
        finally:
            input_queue.task_done()


def stitch_chunks(chunks, model_stride):
    """
    Merge overlapping basecalled chunks.
    
    Based on Dorado's stitch.cpp implementation.
    
    Args:
        chunks: List of chunk dictionaries, each containing:
            - 'sequence': str - base sequence
            - 'qstring': str - quality string
            - 'moves': list - move sequence (0=stay, 1=emit)
            - 'offset': int - original signal offset
        model_stride: Model stride (for overlap calculation)
    
    Returns:
        Tuple of (stitched_sequence, stitched_qstring)
    """
    if len(chunks) == 0:
        return "", ""
    
    if len(chunks) == 1:
        # Single chunk - return as is
        return chunks[0]['sequence'], chunks[0]['qstring']
    
    # Calculate overlap in base space
    # Overlap in signal space is OVERLAP samples
    # After downsampling by model_stride, overlap in base space is OVERLAP / model_stride
    overlap_bases = OVERLAP // model_stride
    
    stitched_seq = []
    stitched_qstring = []
    stitched_moves = []
    
    start_pos = 0
    mid_point_front = 0
    
    # Process all chunks except the last one
    for i in range(len(chunks) - 1):
        current_chunk = chunks[i]
        next_chunk = chunks[i + 1]
        
        # Calculate overlap size in base space
        current_offset = current_chunk['offset']
        next_offset = next_chunk['offset']
        overlap_size = (current_offset + CHUNK_SIZE) - next_offset
        overlap_downsampled = overlap_size // model_stride
        
        # Midpoint of overlap region
        mid_point_rear = overlap_downsampled // 2
        
        # Count bases to trim from end of current chunk
        current_moves = current_chunk['moves']
        current_chunk_bases_to_trim = sum(current_moves[-mid_point_rear:])
        
        # Extract sequence and quality from current chunk
        current_seq = current_chunk['sequence']
        current_qstring = current_chunk['qstring']
        current_seq_len = len(current_seq)
        
        # End position in current chunk (before trimming)
        end_pos = current_seq_len - current_chunk_bases_to_trim
        
        # Extract trimmed portion
        trimmed_len = end_pos - start_pos
        if trimmed_len > 0:
            stitched_seq.append(current_seq[start_pos:end_pos])
            stitched_qstring.append(current_qstring[start_pos:end_pos])
            stitched_moves.extend(current_moves[mid_point_front:-mid_point_rear])
        
        # Update start position for next chunk
        mid_point_front = overlap_downsampled - mid_point_rear
        start_pos = 0
        for j in range(mid_point_front):
            if j < len(next_chunk['moves']):
                start_pos += next_chunk['moves'][j]
    
    # Append the final chunk
    last_chunk = chunks[-1]
    last_seq = last_chunk['sequence']
    last_qstring = last_chunk['qstring']
    last_moves = last_chunk['moves']
    
    if len(chunks) > 1:
        # Add remaining portion of last chunk
        stitched_moves.extend(last_moves[mid_point_front:])
        if start_pos < len(last_seq):
            stitched_seq.append(last_seq[start_pos:])
            stitched_qstring.append(last_qstring[start_pos:])
    else:
        # Single chunk case (already handled above, but keep for safety)
        stitched_seq.append(last_seq)
        stitched_qstring.append(last_qstring)
        stitched_moves.extend(last_moves)
    
    return ''.join(stitched_seq), ''.join(stitched_qstring)


def print_metrics():
    """
    Calculate and print performance metrics including throughput, latency, and total time.
    """
    with metrics_lock:
        if metrics['start_time'] is None:
            print("No metrics available (no reads processed)")
            return
        
        end_time = metrics['end_time'] if metrics['end_time'] is not None else time.time()
        total_time = end_time - metrics['start_time']
        
        # Calculate throughput
        if total_time > 0:
            throughput_bases_per_sec = metrics['total_bases'] / total_time
            throughput_reads_per_sec = metrics['total_reads'] / total_time
        else:
            throughput_bases_per_sec = 0.0
            throughput_reads_per_sec = 0.0
        
        # Calculate latency statistics
        latencies = []
        for read_id, timing in metrics['read_timings'].items():
            if timing['end'] is not None and timing['start'] is not None:
                latency = timing['end'] - timing['start']
                latencies.append(latency)
        
        if latencies:
            avg_latency = sum(latencies) / len(latencies)
            min_latency = min(latencies)
            max_latency = max(latencies)
        else:
            avg_latency = 0.0
            min_latency = 0.0
            max_latency = 0.0
        
        # Print metrics
        print("\n" + "="*60)
        print("PERFORMANCE METRICS")
        print("="*60)
        print(f"Total Time:        {total_time:.2f} seconds ({total_time/60:.2f} minutes)")
        print(f"Total Reads:       {metrics['total_reads']}")
        print(f"Total Bases:       {metrics['total_bases']:,}")
        print(f"Throughput:        {throughput_bases_per_sec:.2f} bases/second")
        print(f"                   {throughput_reads_per_sec:.2f} reads/second")
        print(f"Average Latency:   {avg_latency:.3f} seconds/read")
        print(f"Min Latency:       {min_latency:.3f} seconds/read")
        print(f"Max Latency:       {max_latency:.3f} seconds/read")
        print("="*60 + "\n")


def write_fastq(read_id, sequence, qstring, output_file):
    """
    Write basecalled sequence to FASTQ file.
    
    FASTQ format:
    @read_id
    sequence
    +
    quality_string
    
    Args:
        read_id: Read identifier
        sequence: Base sequence string
        qstring: Quality string
        output_file: Path to output FASTQ file
    """
    with fastq_lock:
        with open(output_file, 'a') as f:
            f.write(f"@{read_id}\n")
            f.write(f"{sequence}\n")
            f.write("+\n")
            f.write(f"{qstring}\n")


# Main real-time pipeline
def run_realtime_basecalling(pod5_path, output_fastq=None):
    """
    Start real-time basecalling pipeline with multiple worker threads.
    
    Args:
        pod5_path: Path to the POD5 file to monitor for new reads
        output_fastq: Path to output FASTQ file (defaults to OUTPUT_FASTQ)
    """
    if output_fastq is None:
        output_fastq = OUTPUT_FASTQ
    main_logger = get_worker_logger('main')
    
    # Start thread to monitor POD5 file for new reads
    monitor_thread = Thread(
        target=monitor_pod5_file,
        args=(pod5_path, preprocessing_queue),
        daemon=True
    )
    monitor_thread.start()
    main_logger.info("Started POD5 monitoring thread")
    
    # Start thread to preprocess reads and chunk them
    preprocess_thread = Thread(
        target=preprocess_worker,
        args=(preprocessing_queue, fpga_input_queue),
        daemon=True
    )
    preprocess_thread.start()
    main_logger.info("Started preprocessing worker thread")
    
    # Start thread to send chunks to FPGA and collect results
    fpga_thread = Thread(
        target=fpga_sender_worker,
        args=(fpga_input_queue, fpga_output_queue),
        daemon=True
    )
    fpga_thread.start()
    main_logger.info("Started FPGA sender worker thread")
    
    # Start thread to collect basecalled chunks and assemble final sequences
    assembler_thread = Thread(
        target=basecall_assembler_worker,
        args=(fpga_output_queue, MODEL_DIR, output_fastq),
        daemon=True
    )
    assembler_thread.start()
    main_logger.info("Started basecall assembler worker thread")
    
    main_logger.info(f"Real-time basecalling pipeline started. Monitoring {pod5_path}")
    main_logger.info("Press Ctrl+C to stop...")
    
    try:
        # Keep main thread alive
        while True:
            time.sleep(1)
            
            # Log pipeline stats
            main_logger.info(f"Queue depths: "
                  f"preproc={preprocessing_queue.qsize()}, "
                  f"fpga_in={fpga_input_queue.qsize()}, "
                  f"fpga_out={fpga_output_queue.qsize()}")
    
    except KeyboardInterrupt:
        main_logger.info("Shutting down...")
        
        # Record end time for metrics
        with metrics_lock:
            metrics['end_time'] = time.time()
        
        # Send poison pills to stop workers
        # Note: monitor_thread doesn't need a poison pill as it's daemon
        preprocessing_queue.put(None)
        fpga_input_queue.put(None)
        fpga_output_queue.put(None)
        
        # Wait for worker threads to finish processing
        main_logger.info("Waiting for workers to finish...")
        preprocess_thread.join(timeout=5.0)
        fpga_thread.join(timeout=5.0)
        assembler_thread.join(timeout=5.0)
        
        # Update end time again after workers finish (in case more reads completed)
        with metrics_lock:
            metrics['end_time'] = time.time()
        
        # Print performance metrics
        print_metrics()
        
        main_logger.info("Pipeline shutdown complete")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Real-time basecalling pipeline')
    parser.add_argument('pod5_file', type=str,
                        help='Path to input POD5 file')
    parser.add_argument('output_fastq', type=str,
                        help='Path to output FASTQ file')
    parser.add_argument('-v', '--verbose', action='store_true',
                        help='Enable verbose logging')
    args = parser.parse_args()
    
    # Setup logging based on verbose flag
    setup_logging(verbose=args.verbose)
    
    run_realtime_basecalling(args.pod5_file, args.output_fastq)
