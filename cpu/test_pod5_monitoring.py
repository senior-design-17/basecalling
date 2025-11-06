"""
Test script for POD5 file monitoring functionality.

This script tests the monitor_pod5_file function by:
1. Creating a sample POD5 file with initial reads
2. Running the monitoring function in a separate thread
3. Appending new reads to simulate real-time sequencing
4. Verifying that new reads are detected and queued
"""

import pod5
import numpy as np
import os
import time
import tempfile
import threading
from queue import Queue
from uuid import uuid4
from basecaller import monitor_pod5_file, processed_reads


def create_sample_pod5_file(file_path, num_reads=3, signal_length=1000):
    """
    Create a sample POD5 file with test reads.
    
    Args:
        file_path: Path where to create the POD5 file
        num_reads: Number of reads to create
        signal_length: Length of signal data for each read
    """
    # Create run info
    run_info = pod5.RunInfo(
        acquisition_id=str(uuid4()),
        acquisition_start_time=int(time.time()),
        adc_max=8192,
        adc_min=-8192,
        sample_rate=4000.0,
        flow_cell_id="TEST-FC-001",
        device_id="TEST-DEV-001",
        exp_start_time=int(time.time()),
        flow_cell_product_code="FLO-MIN106",
        protocol_group_id="test_group",
        protocol_run_id="test_run",
        protocol_start_time=int(time.time()),
        sample_id="test_sample",
        sequencing_kit="SQK-LSK109",
        software="MinKNOW",
        software_version="21.06.0",
        tracking_id={"test_key": "test_value"},
    )
    
    # Create reads
    reads = []
    for i in range(num_reads):
        # Generate sample signal data (simulated current measurements)
        signal = np.random.randint(-1000, 1000, size=signal_length, dtype=np.int16)
        
        read = pod5.Read(
            read_id=uuid4(),
            end_reason=pod5.EndReason(
                name=pod5.EndReasonEnum.SIGNAL_POSITIVE,
                forced=False
            ),
            calibration=pod5.Calibration(offset=0.0, scale=1.0),
            pore=pod5.Pore(channel=i+1, well=i+1, pore_type="not_set"),
            run_info=run_info,
            signal=signal,
        )
        reads.append(read)
    
    # Write to file
    with pod5.Writer(file_path) as writer:
        for read in reads:
            writer.add_read(read)
    
    print(f"Created POD5 file with {num_reads} reads: {file_path}")
    return [str(read.read_id) for read in reads]


def append_reads_to_pod5(file_path, num_reads=2, signal_length=1000):
    """
    Append new reads to an existing POD5 file.
    This simulates real-time sequencing where reads are appended.
    
    Args:
        file_path: Path to the POD5 file
        num_reads: Number of reads to append
        signal_length: Length of signal data for each read
    """
    # Read existing file to get run info
    with pod5.Reader(file_path) as reader:
        # Get run info from first read
        first_read = next(reader.reads())
        run_info = first_read.run_info
    
    # Create new reads
    new_read_ids = []
    reads_to_append = []
    
    for i in range(num_reads):
        # Generate sample signal data
        signal = np.random.randint(-1000, 1000, size=signal_length, dtype=np.int16)
        
        read = pod5.Read(
            read_id=uuid4(),
            end_reason=pod5.EndReason(
                name=pod5.EndReasonEnum.SIGNAL_POSITIVE,
                forced=False
            ),
            calibration=pod5.Calibration(offset=0.0, scale=1.0),
            pore=pod5.Pore(channel=100+i, well=100+i, pore_type="not_set"),
            run_info=run_info,
            signal=signal,
        )
        reads_to_append.append(read)
        new_read_ids.append(str(read.read_id))
    
    # Append to file
    # Note: pod5 library doesn't support direct appending, so we need to
    # read all existing reads and write them back with new reads
    existing_reads = []
    with pod5.Reader(file_path) as reader:
        for read in reader.reads():
            existing_reads.append(read)
    
    # Write all reads (existing + new) to a new file, then replace
    temp_path = file_path + ".tmp"
    with pod5.Writer(temp_path) as writer:
        for read in existing_reads:
            writer.add_read(read)
        for read in reads_to_append:
            writer.add_read(read)
    
    # Replace original file
    os.replace(temp_path, file_path)
    
    print(f"Appended {num_reads} new reads to POD5 file")
    return new_read_ids


def test_monitor_pod5_file():
    """Test the monitor_pod5_file function."""
    # Create temporary file
    with tempfile.NamedTemporaryFile(suffix='.pod5', delete=False) as tmp_file:
        test_file_path = tmp_file.name
    
    try:
        # Clear processed reads set
        processed_reads.clear()
        
        # Create initial POD5 file with 3 reads
        initial_read_ids = create_sample_pod5_file(test_file_path, num_reads=3)
        print(f"Initial read IDs: {initial_read_ids}")
        
        # Create queue for monitoring output
        output_queue = Queue()
        
        # Start monitoring in a separate thread
        monitor_thread = threading.Thread(
            target=monitor_pod5_file,
            args=(test_file_path, output_queue),
            kwargs={'poll_interval': 0.5},  # Check every 0.5 seconds
            daemon=True
        )
        monitor_thread.start()
        
        # Wait a bit for initial reads to be detected
        time.sleep(2)
        
        # Check that initial reads were detected
        detected_reads = []
        while not output_queue.empty():
            read_record = output_queue.get()
            detected_reads.append(str(read_record.read_id))
            print(f"Detected read: {read_record.read_id}")
        
        # Verify initial reads were detected
        assert len(detected_reads) == 3, f"Expected 3 reads, got {len(detected_reads)}"
        assert set(detected_reads) == set(initial_read_ids), "Read IDs don't match"
        print("✓ Initial reads detected correctly")
        
        # Append new reads to simulate real-time sequencing
        time.sleep(1)
        new_read_ids = append_reads_to_pod5(test_file_path, num_reads=2)
        print(f"New read IDs: {new_read_ids}")
        
        # Wait for new reads to be detected
        time.sleep(2)
        
        # Check that new reads were detected
        newly_detected_reads = []
        while not output_queue.empty():
            read_record = output_queue.get()
            newly_detected_reads.append(str(read_record.read_id))
            print(f"Newly detected read: {read_record.read_id}")
        
        # Verify new reads were detected
        assert len(newly_detected_reads) == 2, f"Expected 2 new reads, got {len(newly_detected_reads)}"
        assert set(newly_detected_reads) == set(new_read_ids), "New read IDs don't match"
        print("✓ New reads detected correctly")
        
        # Verify no duplicate processing
        assert len(processed_reads) == 5, f"Expected 5 processed reads, got {len(processed_reads)}"
        print("✓ No duplicate processing detected")
        
        # Test that already processed reads are not reprocessed
        time.sleep(1)
        # Check queue again - should be empty
        assert output_queue.empty(), "Queue should be empty (no reprocessing)"
        print("✓ No reprocessing of existing reads")
        
        print("\n✅ All tests passed!")
        
    finally:
        # Cleanup
        if os.path.exists(test_file_path):
            os.remove(test_file_path)
        print(f"Cleaned up test file: {test_file_path}")


def test_monitor_nonexistent_file():
    """Test monitoring a file that doesn't exist yet."""
    # Create queue for monitoring output
    output_queue = Queue()
    
    # Use a non-existent file path
    nonexistent_file = "/tmp/nonexistent_test_file.pod5"
    
    # Clear processed reads
    processed_reads.clear()
    
    # Start monitoring in a separate thread
    monitor_thread = threading.Thread(
        target=monitor_pod5_file,
        args=(nonexistent_file, output_queue),
        kwargs={'poll_interval': 0.1},
        daemon=True
    )
    monitor_thread.start()
    
    # Wait a bit - should not crash
    time.sleep(1)
    
    # Create the file now
    create_sample_pod5_file(nonexistent_file, num_reads=1)
    
    # Wait for read to be detected
    time.sleep(1)
    
    # Check that read was detected once file was created
    detected_reads = []
    while not output_queue.empty():
        read_record = output_queue.get()
        detected_reads.append(str(read_record.read_id))
    
    assert len(detected_reads) == 1, f"Expected 1 read after file creation, got {len(detected_reads)}"
    print("✓ File creation detection works correctly")
    
    # Cleanup
    if os.path.exists(nonexistent_file):
        os.remove(nonexistent_file)


if __name__ == "__main__":
    print("Testing POD5 file monitoring...")
    print("=" * 50)
    
    print("\nTest 1: Basic monitoring with initial and appended reads")
    test_monitor_pod5_file()
    
    print("\nTest 2: Monitoring non-existent file")
    test_monitor_nonexistent_file()
    
    print("\n" + "=" * 50)
    print("All tests completed successfully!")
