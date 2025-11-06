#!/usr/bin/env python3
"""
Tester script for comparing basecaller.py with Dorado.

This script:
1. Runs basecaller.py on a POD5 file and measures execution time
2. Runs Dorado on the same POD5 file with the same model and measures execution time
3. Compares the outputs (sequences) for accuracy
4. Reports timing and accuracy metrics
"""

import argparse
import subprocess
import time
import sys
import os
import tempfile
from pathlib import Path
from collections import defaultdict
from typing import Dict, Tuple, List
import difflib


def parse_fastq(fastq_path: Path) -> Dict[str, Tuple[str, str]]:
    """
    Parse a FASTQ file and return a dictionary mapping read_id to (sequence, quality).
    
    Args:
        fastq_path: Path to the FASTQ file
        
    Returns:
        Dictionary mapping read_id (str) to tuple of (sequence, quality_string)
    """
    reads = {}
    
    if not fastq_path.exists():
        return reads
    
    with open(fastq_path, 'r') as f:
        lines = f.readlines()
    
    i = 0
    while i < len(lines):
        if lines[i].startswith('@'):
            # Read ID is on the line starting with @
            read_id_line = lines[i].strip()
            read_id = read_id_line[1:].split()[0]  # Remove @ and get first token
            
            if i + 3 < len(lines):
                sequence = lines[i + 1].strip()
                quality = lines[i + 3].strip()
                reads[read_id] = (sequence, quality)
                i += 4
            else:
                break
        else:
            i += 1
    
    return reads


def calculate_edit_distance(seq1: str, seq2: str) -> Tuple[int, float]:
    """
    Calculate edit distance (Levenshtein distance) between two sequences.
    
    Args:
        seq1: First sequence
        seq2: Second sequence
        
    Returns:
        Tuple of (edit_distance, similarity_ratio)
    """
    if len(seq1) == 0 and len(seq2) == 0:
        return 0, 1.0
    
    if len(seq1) == 0:
        return len(seq2), 0.0
    
    if len(seq2) == 0:
        return len(seq1), 0.0
    
    # Use difflib for similarity ratio
    similarity = difflib.SequenceMatcher(None, seq1, seq2).ratio()
    
    # Calculate edit distance using dynamic programming
    m, n = len(seq1), len(seq2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    
    # Initialize base cases
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j
    
    # Fill the DP table
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if seq1[i - 1] == seq2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = 1 + min(
                    dp[i - 1][j],      # deletion
                    dp[i][j - 1],      # insertion
                    dp[i - 1][j - 1]   # substitution
                )
    
    edit_distance = dp[m][n]
    return edit_distance, similarity


def compare_fastq_files(
    fastq1_path: Path,
    fastq2_path: Path,
    fastq1_name: str = "basecaller.py",
    fastq2_name: str = "Dorado"
) -> Dict:
    """
    Compare two FASTQ files and return accuracy metrics.
    
    Args:
        fastq1_path: Path to first FASTQ file
        fastq2_path: Path to second FASTQ file
        fastq1_name: Name for first file (for reporting)
        fastq2_name: Name for second file (for reporting)
        
    Returns:
        Dictionary containing comparison metrics
    """
    reads1 = parse_fastq(fastq1_path)
    reads2 = parse_fastq(fastq2_path)
    
    metrics = {
        'total_reads_1': len(reads1),
        'total_reads_2': len(reads2),
        'common_reads': 0,
        'only_in_1': [],
        'only_in_2': [],
        'matching_sequences': 0,
        'total_edit_distance': 0,
        'total_similarity': 0.0,
        'sequence_lengths_1': [],
        'sequence_lengths_2': [],
        'per_read_metrics': []
    }
    
    # Find common reads
    common_read_ids = set(reads1.keys()) & set(reads2.keys())
    metrics['common_reads'] = len(common_read_ids)
    metrics['only_in_1'] = list(set(reads1.keys()) - set(reads2.keys()))
    metrics['only_in_2'] = list(set(reads2.keys()) - set(reads1.keys()))
    
    # Compare sequences for common reads
    for read_id in common_read_ids:
        seq1, qual1 = reads1[read_id]
        seq2, qual2 = reads2[read_id]
        
        metrics['sequence_lengths_1'].append(len(seq1))
        metrics['sequence_lengths_2'].append(len(seq2))
        
        if seq1 == seq2:
            metrics['matching_sequences'] += 1
            edit_dist = 0
            similarity = 1.0
        else:
            edit_dist, similarity = calculate_edit_distance(seq1, seq2)
            metrics['total_edit_distance'] += edit_dist
            metrics['total_similarity'] += similarity
        
        metrics['per_read_metrics'].append({
            'read_id': read_id,
            'length_1': len(seq1),
            'length_2': len(seq2),
            'edit_distance': edit_dist,
            'similarity': similarity,
            'sequences_match': seq1 == seq2
        })
    
    # Calculate average metrics
    if metrics['common_reads'] > 0:
        metrics['avg_edit_distance'] = metrics['total_edit_distance'] / metrics['common_reads']
        metrics['avg_similarity'] = metrics['total_similarity'] / metrics['common_reads']
        metrics['match_rate'] = metrics['matching_sequences'] / metrics['common_reads']
    else:
        metrics['avg_edit_distance'] = 0
        metrics['avg_similarity'] = 0.0
        metrics['match_rate'] = 0.0
    
    return metrics


def count_reads_in_pod5(pod5_path: Path) -> int:
    """Count the number of reads in a POD5 file."""
    try:
        import pod5
        with pod5.Reader(pod5_path) as reader:
            count = sum(1 for _ in reader.reads())
        return count
    except Exception as e:
        print(f"Warning: Could not count reads in POD5 file: {e}")
        return 0


def run_basecaller(
    pod5_path: Path,
    output_fastq: Path,
    model_dir: Path,
    verbose: bool = False
) -> Tuple[float, bool]:
    """
    Run basecaller.py on a POD5 file.
    
    Note: basecaller.py runs in a continuous monitoring loop, so we need to
    detect when processing is complete by monitoring the output file.
    
    Args:
        pod5_path: Path to input POD5 file
        output_fastq: Path to output FASTQ file
        model_dir: Path to model directory
        verbose: Whether to enable verbose logging
        
    Returns:
        Tuple of (execution_time_seconds, success)
    """
    # Clear output file if it exists
    if output_fastq.exists():
        output_fastq.unlink()
    
    # Count reads in POD5 file
    expected_reads = count_reads_in_pod5(pod5_path)
    print(f"POD5 file contains {expected_reads} reads")
    
    # Get the basecaller.py script path
    script_dir = Path(__file__).parent
    basecaller_script = script_dir / "basecaller.py"
    
    # Build command
    cmd = [
        sys.executable,
        str(basecaller_script),
        str(pod5_path),
        str(output_fastq)
    ]
    
    if verbose:
        cmd.append('-v')
    
    print(f"Running basecaller.py...")
    print(f"Command: {' '.join(cmd)}")
    
    start_time = time.time()
    process = None
    
    try:
        # Start the process
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE if not verbose else None,
            stderr=subprocess.PIPE if not verbose else None,
            text=True
        )
        
        # Monitor output file to detect when processing is complete
        max_wait_time = 3600  # 1 hour max
        check_interval = 2.0  # Check every 2 seconds
        stable_periods = 3  # Number of consecutive stable checks before considering done
        stable_count = 0
        last_read_count = 0
        last_file_size = 0
        
        print("Monitoring basecaller.py output...")
        
        while True:
            elapsed = time.time() - start_time
            
            # Check if process has terminated
            if process.poll() is not None:
                # Process has ended
                if process.returncode != 0:
                    stderr = process.stderr.read() if process.stderr else ""
                    print(f"Error: basecaller.py exited with code {process.returncode}")
                    if stderr:
                        print(f"Stderr: {stderr}")
                    return elapsed, False
                break
            
            # Check timeout
            if elapsed > max_wait_time:
                print(f"Timeout: Killing basecaller.py process after {elapsed:.2f} seconds")
                process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
                return elapsed, False
            
            # Check output file
            if output_fastq.exists():
                current_file_size = output_fastq.stat().st_size
                current_reads = len(parse_fastq(output_fastq))
                
                # Check if we've processed all expected reads
                if expected_reads > 0 and current_reads >= expected_reads:
                    print(f"All {expected_reads} reads processed. Waiting for process to finish...")
                    # Give it a bit more time to finish writing
                    time.sleep(5)
                    if process.poll() is None:
                        process.terminate()
                        try:
                            process.wait(timeout=10)
                        except subprocess.TimeoutExpired:
                            process.kill()
                    break
                
                # Check if output is stable (no change in read count or file size)
                if current_reads == last_read_count and current_file_size == last_file_size:
                    stable_count += 1
                    if stable_count >= stable_periods:
                        print(f"Output appears stable ({current_reads} reads). Terminating process...")
                        process.terminate()
                        try:
                            process.wait(timeout=10)
                        except subprocess.TimeoutExpired:
                            process.kill()
                        break
                else:
                    stable_count = 0
                
                last_read_count = current_reads
                last_file_size = current_file_size
                
                if verbose and current_reads > 0:
                    print(f"  Processed {current_reads} reads so far...")
            
            time.sleep(check_interval)
        
        elapsed_time = time.time() - start_time
        
        # Final check
        if not output_fastq.exists() or output_fastq.stat().st_size == 0:
            print(f"Warning: Output file {output_fastq} was not created or is empty")
            return elapsed_time, False
        
        final_read_count = len(parse_fastq(output_fastq))
        print(f"✓ basecaller.py completed in {elapsed_time:.2f} seconds ({final_read_count} reads)")
        return elapsed_time, True
        
    except KeyboardInterrupt:
        if process:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
        elapsed_time = time.time() - start_time
        print(f"\n✗ Interrupted after {elapsed_time:.2f} seconds")
        return elapsed_time, False
    except Exception as e:
        if process:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
        elapsed_time = time.time() - start_time
        print(f"✗ Error running basecaller.py: {e}")
        import traceback
        traceback.print_exc()
        return elapsed_time, False


def run_dorado(
    pod5_path: Path,
    output_fastq: Path,
    model_path: str,
    dorado_bin: Path,
    device: str = "cpu",
    verbose: bool = False
) -> Tuple[float, bool]:
    """
    Run Dorado on a POD5 file.
    
    Args:
        pod5_path: Path to input POD5 file
        output_fastq: Path to output FASTQ file
        model_path: Model path (can be model name like "hac" or path to model directory)
        dorado_bin: Path to Dorado executable
        device: Device to use (cpu, auto, metal, cuda:0, etc.)
        verbose: Whether to enable verbose logging
        
    Returns:
        Tuple of (execution_time_seconds, success)
    """
    # Clear output file if it exists
    if output_fastq.exists():
        output_fastq.unlink()
    
    # Build command
    cmd = [
        str(dorado_bin),
        "basecaller",
        model_path,
        str(pod5_path),
        "--emit-fastq",
        "--device", device
    ]
    
    if verbose:
        cmd.append("-v")
    
    print(f"Running Dorado...")
    print(f"Command: {' '.join(cmd)}")
    
    start_time = time.time()
    try:
        with open(output_fastq, 'w') as f:
            result = subprocess.run(
                cmd,
                stdout=f,
                stderr=subprocess.PIPE,
                text=True,
                timeout=3600  # 1 hour timeout
            )
        
        elapsed_time = time.time() - start_time
        
        if result.returncode != 0:
            print(f"Error running Dorado:")
            print(result.stderr)
            return elapsed_time, False
        
        if not output_fastq.exists() or output_fastq.stat().st_size == 0:
            print(f"Warning: Output file {output_fastq} was not created or is empty")
            print(f"Dorado stderr: {result.stderr}")
            return elapsed_time, False
        
        print(f"✓ Dorado completed in {elapsed_time:.2f} seconds")
        return elapsed_time, True
        
    except subprocess.TimeoutExpired:
        elapsed_time = time.time() - start_time
        print(f"✗ Dorado timed out after {elapsed_time:.2f} seconds")
        return elapsed_time, False
    except Exception as e:
        elapsed_time = time.time() - start_time
        print(f"✗ Error running Dorado: {e}")
        return elapsed_time, False


def print_comparison_report(metrics: Dict, time1: float, time2: float):
    """
    Print a formatted comparison report.
    
    Args:
        metrics: Comparison metrics dictionary
        time1: Execution time for basecaller.py
        time2: Execution time for Dorado
    """
    print("\n" + "=" * 80)
    print("COMPARISON REPORT")
    print("=" * 80)
    
    print("\n📊 TIMING METRICS")
    print("-" * 80)
    print(f"basecaller.py execution time: {time1:.2f} seconds")
    print(f"Dorado execution time:       {time2:.2f} seconds")
    if time2 > 0:
        speedup = time1 / time2
        print(f"Speedup ratio:                {speedup:.2f}x ({'faster' if speedup > 1 else 'slower'})")
    
    print("\n📈 READ COUNTS")
    print("-" * 80)
    print(f"basecaller.py reads:          {metrics['total_reads_1']}")
    print(f"Dorado reads:                 {metrics['total_reads_2']}")
    print(f"Common reads:                 {metrics['common_reads']}")
    
    if metrics['only_in_1']:
        print(f"\nReads only in basecaller.py:  {len(metrics['only_in_1'])}")
        if len(metrics['only_in_1']) <= 5:
            for read_id in metrics['only_in_1']:
                print(f"  - {read_id}")
    
    if metrics['only_in_2']:
        print(f"\nReads only in Dorado:         {len(metrics['only_in_2'])}")
        if len(metrics['only_in_2']) <= 5:
            for read_id in metrics['only_in_2']:
                print(f"  - {read_id}")
    
    if metrics['common_reads'] > 0:
        print("\n🎯 ACCURACY METRICS")
        print("-" * 80)
        print(f"Exact sequence matches:       {metrics['matching_sequences']} / {metrics['common_reads']} ({metrics['match_rate']*100:.2f}%)")
        print(f"Average edit distance:        {metrics['avg_edit_distance']:.2f} bases")
        print(f"Average similarity:           {metrics['avg_similarity']*100:.2f}%")
        
        if metrics['sequence_lengths_1']:
            avg_len_1 = sum(metrics['sequence_lengths_1']) / len(metrics['sequence_lengths_1'])
            avg_len_2 = sum(metrics['sequence_lengths_2']) / len(metrics['sequence_lengths_2'])
            print(f"\nAverage sequence length:")
            print(f"  basecaller.py:              {avg_len_1:.1f} bases")
            print(f"  Dorado:                      {avg_len_2:.1f} bases")
        
        # Show per-read details for mismatches
        mismatches = [m for m in metrics['per_read_metrics'] if not m['sequences_match']]
        if mismatches:
            print(f"\n⚠️  Mismatched reads: {len(mismatches)}")
            print("\nTop 10 mismatched reads (by edit distance):")
            sorted_mismatches = sorted(mismatches, key=lambda x: x['edit_distance'], reverse=True)
            for i, m in enumerate(sorted_mismatches[:10], 1):
                print(f"  {i}. Read {m['read_id'][:36]}...")
                print(f"     Edit distance: {m['edit_distance']}, Similarity: {m['similarity']*100:.2f}%")
                print(f"     Lengths: {m['length_1']} vs {m['length_2']}")
    else:
        print("\n⚠️  WARNING: No common reads found for comparison!")
    
    print("\n" + "=" * 80)


def main():
    parser = argparse.ArgumentParser(
        description='Compare basecaller.py with Dorado basecaller',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Test with a POD5 file using default model
  python test_basecaller_comparison.py test.pod5
  
  # Test with specific model path
  python test_basecaller_comparison.py test.pod5 --model-path ./model/dna_r10.4.1_e8.2_400bps_hac@v5.2.0
  
  # Test with GPU (if available)
  python test_basecaller_comparison.py test.pod5 --device auto
        """
    )
    
    parser.add_argument(
        'pod5_file',
        type=Path,
        help='Path to input POD5 file'
    )
    
    parser.add_argument(
        '--model-path',
        type=str,
        default=None,
        help='Path to model directory (default: ./model/dna_r10.4.1_e8.2_400bps_hac@v5.2.0)'
    )
    
    parser.add_argument(
        '--dorado-bin',
        type=Path,
        default=None,
        help='Path to Dorado executable (default: ./dorado-1.2.0-osx-arm64/bin/dorado)'
    )
    
    parser.add_argument(
        '--device',
        type=str,
        default='cpu',
        help='Device for Dorado (cpu, auto, metal, cuda:0, etc.) (default: cpu)'
    )
    
    parser.add_argument(
        '--output-dir',
        type=Path,
        default=None,
        help='Directory for output FASTQ files (default: temporary directory)'
    )
    
    parser.add_argument(
        '-v', '--verbose',
        action='store_true',
        help='Enable verbose logging'
    )
    
    args = parser.parse_args()
    
    # Validate input file
    if not args.pod5_file.exists():
        print(f"Error: POD5 file not found: {args.pod5_file}")
        sys.exit(1)
    
    # Set default model path
    if args.model_path is None:
        script_dir = Path(__file__).parent
        project_root = script_dir.parent
        args.model_path = str(project_root / "model" / "dna_r10.4.1_e8.2_400bps_hac@v5.2.0")
    
    # Validate model path
    model_path_obj = Path(args.model_path)
    if not model_path_obj.exists():
        print(f"Error: Model directory not found: {args.model_path}")
        sys.exit(1)
    
    # Set default Dorado binary path
    if args.dorado_bin is None:
        script_dir = Path(__file__).parent
        project_root = script_dir.parent
        args.dorado_bin = project_root / "dorado-1.2.0-osx-arm64" / "bin" / "dorado"
    
    # Validate Dorado binary
    if not args.dorado_bin.exists():
        print(f"Error: Dorado executable not found: {args.dorado_bin}")
        print("Please specify the correct path with --dorado-bin")
        sys.exit(1)
    
    # Set up output directory
    if args.output_dir is None:
        output_dir = Path(tempfile.mkdtemp(prefix="basecaller_test_"))
    else:
        output_dir = args.output_dir
        output_dir.mkdir(parents=True, exist_ok=True)
    
    basecaller_output = output_dir / "basecaller_output.fastq"
    dorado_output = output_dir / "dorado_output.fastq"
    
    print("=" * 80)
    print("BASECALLER COMPARISON TEST")
    print("=" * 80)
    print(f"Input POD5 file:  {args.pod5_file}")
    print(f"Model path:       {args.model_path}")
    print(f"Dorado binary:    {args.dorado_bin}")
    print(f"Device:           {args.device}")
    print(f"Output directory: {output_dir}")
    print("=" * 80)
    
    # Run basecaller.py
    print("\n[1/3] Running basecaller.py...")
    time1, success1 = run_basecaller(
        args.pod5_file,
        basecaller_output,
        model_path_obj,
        verbose=args.verbose
    )
    
    if not success1:
        print("✗ basecaller.py failed. Cannot continue comparison.")
        sys.exit(1)
    
    # Run Dorado
    print("\n[2/3] Running Dorado...")
    time2, success2 = run_dorado(
        args.pod5_file,
        dorado_output,
        args.model_path,
        args.dorado_bin,
        device=args.device,
        verbose=args.verbose
    )
    
    if not success2:
        print("✗ Dorado failed. Cannot continue comparison.")
        sys.exit(1)
    
    # Compare outputs
    print("\n[3/3] Comparing outputs...")
    metrics = compare_fastq_files(
        basecaller_output,
        dorado_output,
        fastq1_name="basecaller.py",
        fastq2_name="Dorado"
    )
    
    # Print report
    print_comparison_report(metrics, time1, time2)
    
    # Save detailed report to file
    report_file = output_dir / "comparison_report.txt"
    with open(report_file, 'w') as f:
        f.write("BASECALLER COMPARISON REPORT\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Input POD5 file: {args.pod5_file}\n")
        f.write(f"Model path: {args.model_path}\n")
        f.write(f"Device: {args.device}\n\n")
        
        f.write("TIMING METRICS\n")
        f.write("-" * 80 + "\n")
        f.write(f"basecaller.py: {time1:.2f} seconds\n")
        f.write(f"Dorado:        {time2:.2f} seconds\n")
        if time2 > 0:
            f.write(f"Speedup:       {time1/time2:.2f}x\n\n")
        
        f.write("READ COUNTS\n")
        f.write("-" * 80 + "\n")
        f.write(f"basecaller.py reads: {metrics['total_reads_1']}\n")
        f.write(f"Dorado reads:        {metrics['total_reads_2']}\n")
        f.write(f"Common reads:        {metrics['common_reads']}\n\n")
        
        if metrics['common_reads'] > 0:
            f.write("ACCURACY METRICS\n")
            f.write("-" * 80 + "\n")
            f.write(f"Exact matches:      {metrics['matching_sequences']}/{metrics['common_reads']} ({metrics['match_rate']*100:.2f}%)\n")
            f.write(f"Avg edit distance:  {metrics['avg_edit_distance']:.2f}\n")
            f.write(f"Avg similarity:     {metrics['avg_similarity']*100:.2f}%\n\n")
            
            f.write("PER-READ METRICS\n")
            f.write("-" * 80 + "\n")
            for m in metrics['per_read_metrics']:
                f.write(f"Read {m['read_id']}:\n")
                f.write(f"  Match: {m['sequences_match']}\n")
                f.write(f"  Edit distance: {m['edit_distance']}\n")
                f.write(f"  Similarity: {m['similarity']*100:.2f}%\n")
                f.write(f"  Lengths: {m['length_1']} vs {m['length_2']}\n\n")
    
    print(f"\n📄 Detailed report saved to: {report_file}")
    print(f"📁 Output files:")
    print(f"   basecaller.py: {basecaller_output}")
    print(f"   Dorado:        {dorado_output}")


if __name__ == "__main__":
    main()

