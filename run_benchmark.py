#!/usr/bin/env python3
"""
Batch processing script for running MASA-OA tracker on benchmark sequences.

This script automatically processes all sequences in a benchmark dataset without
requiring manual code modifications.

Usage:
    python run_benchmark.py --benchmark MOT15 [options]
"""

import os
import sys
import argparse
import subprocess
from pathlib import Path
import glob


def find_sequences(benchmark_dir):
    """
    Find all valid sequences in a benchmark directory.

    A valid sequence must have:
    - A video file (.mp4)
    - A ground truth file (gt/gt.txt)

    Args:
        benchmark_dir: Path to benchmark directory (e.g., benchmarks/MOT15)

    Returns:
        List of tuples (sequence_name, video_path, gt_path)
    """
    sequences = []

    # Look for all subdirectories
    benchmark_path = Path(benchmark_dir)
    if not benchmark_path.exists():
        print(f"Error: Benchmark directory not found: {benchmark_dir}")
        return sequences

    for seq_dir in sorted(benchmark_path.iterdir()):
        if not seq_dir.is_dir():
            continue

        seq_name = seq_dir.name

        # Look for video file
        video_path = seq_dir / f"{seq_name}.mp4"
        if not video_path.exists():
            print(f"Warning: Skipping {seq_name} - video file not found")
            continue

        # Look for ground truth file
        gt_path = seq_dir / "gt" / "gt.txt"
        if not gt_path.exists():
            print(f"Warning: Skipping {seq_name} - ground truth file not found")
            continue

        sequences.append((seq_name, str(video_path), str(gt_path)))

    return sequences


def run_tracker_on_sequence(seq_name, video_path, gt_path, args):
    """
    Run the tracker on a single sequence.

    Args:
        seq_name: Name of the sequence
        video_path: Path to the video file
        gt_path: Path to the ground truth detections file
        args: Command-line arguments

    Returns:
        True if successful, False otherwise
    """
    # Construct output paths
    benchmark_name = Path(args.benchmark_dir).name
    output_dir = Path(args.output_dir) / benchmark_name / seq_name
    output_dir.mkdir(parents=True, exist_ok=True)

    output_mot_txt = output_dir / f"{seq_name}_{args.tracker_variant}.txt"

    # Optionally generate GT visualization video
    output_gt_video = None
    if args.generate_gt_video:
        output_gt_video = output_dir / f"{seq_name}_GT_bboxes.mp4"

    # Build command
    cmd = [
        sys.executable,  # Use the same Python interpreter
        "demo/video_demo_with_text.py",
        "--masa_config", args.masa_config,
        "--masa_checkpoint", args.masa_checkpoint,
        "--det_file", gt_path,
        "--input_video", video_path,
        "--output_mot_txt", str(output_mot_txt),
        "--score-thr", str(args.score_thr),
        "--device", args.device,
    ]

    if output_gt_video:
        cmd.extend(["--output_gt_video", str(output_gt_video)])

    if args.show_fps:
        cmd.append("--show_fps")

    if args.fp16:
        cmd.append("--fp16")

    if args.no_post:
        cmd.append("--no-post")

    # Print command for debugging
    print(f"\n{'='*80}")
    print(f"Processing: {seq_name}")
    print(f"{'='*80}")
    print(f"Video: {video_path}")
    print(f"Detections: {gt_path}")
    print(f"Output: {output_mot_txt}")
    if output_gt_video:
        print(f"GT Video: {output_gt_video}")
    print(f"\nCommand: {' '.join(cmd)}\n")

    # Run the command
    try:
        result = subprocess.run(cmd, check=True, capture_output=False, text=True)
        print(f"\n✓ Successfully processed {seq_name}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n✗ Error processing {seq_name}")
        print(f"Error: {e}")
        return False
    except KeyboardInterrupt:
        print(f"\n✗ Interrupted by user")
        raise


def parse_args():
    parser = argparse.ArgumentParser(
        description='Batch process benchmark sequences with MASA-OA tracker',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process all MOT15 sequences with default settings
  python run_benchmark.py --benchmark benchmarks/MOT15

  # Process specific sequences only
  python run_benchmark.py --benchmark benchmarks/MOT15 --sequences Venice-2 KITTI-13

  # Generate GT visualization videos
  python run_benchmark.py --benchmark benchmarks/MOT15 --generate-gt-video

  # Use different tracker variant
  python run_benchmark.py --benchmark benchmarks/MOT15 --tracker-variant masa-hung-kf
        """
    )

    # Required arguments
    parser.add_argument('--benchmark', '--benchmark-dir', dest='benchmark_dir', required=True,
                        help='Path to benchmark directory (e.g., benchmarks/MOT15)')

    # Model configuration
    parser.add_argument('--masa-config', dest='masa_config',
                        default='configs/masa-one/masa_r50_plug_and_play.py',
                        help='Path to MASA config file')
    parser.add_argument('--masa-checkpoint', dest='masa_checkpoint',
                        default='saved_models/masa_models/masa_r50.pth',
                        help='Path to MASA checkpoint file')

    # Output configuration
    parser.add_argument('--output-dir', dest='output_dir', default='GCH_eval',
                        help='Output directory for tracking results')
    parser.add_argument('--tracker-variant', dest='tracker_variant',
                        default='masa-hung-kf-OcclusionAware',
                        help='Tracker variant name (for output filename)')

    # Sequence selection
    parser.add_argument('--sequences', nargs='+',
                        help='Specific sequences to process (default: all)')

    # Processing options
    parser.add_argument('--generate-gt-video', dest='generate_gt_video',
                        action='store_true',
                        help='Generate visualization video with ground truth bboxes')
    parser.add_argument('--score-thr', dest='score_thr', type=float, default=0.2,
                        help='Bbox score threshold')
    parser.add_argument('--device', default='cuda:0',
                        help='Device used for inference')
    parser.add_argument('--show-fps', dest='show_fps', action='store_true',
                        help='Show FPS during processing')
    parser.add_argument('--fp16', action='store_true',
                        help='Use FP16 mode')
    parser.add_argument('--no-post', dest='no_post', action='store_true',
                        help='Do not post-process the results')

    # Execution control
    parser.add_argument('--continue-on-error', dest='continue_on_error',
                        action='store_true',
                        help='Continue processing even if a sequence fails')

    return parser.parse_args()


def main():
    args = parse_args()

    print("="*80)
    print("MASA-OA Benchmark Batch Processing")
    print("="*80)
    print(f"Benchmark: {args.benchmark_dir}")
    print(f"MASA Config: {args.masa_config}")
    print(f"MASA Checkpoint: {args.masa_checkpoint}")
    print(f"Tracker Variant: {args.tracker_variant}")
    print(f"Output Directory: {args.output_dir}")
    print(f"Device: {args.device}")
    print("="*80)

    # Find all sequences
    all_sequences = find_sequences(args.benchmark_dir)

    if not all_sequences:
        print("\nError: No valid sequences found!")
        return 1

    # Filter sequences if specific ones are requested
    if args.sequences:
        requested_seqs = set(args.sequences)
        sequences = [(name, vid, gt) for name, vid, gt in all_sequences
                     if name in requested_seqs]

        # Check if all requested sequences were found
        found_seqs = {name for name, _, _ in sequences}
        missing_seqs = requested_seqs - found_seqs
        if missing_seqs:
            print(f"\nWarning: Requested sequences not found: {', '.join(missing_seqs)}")

        if not sequences:
            print("\nError: None of the requested sequences were found!")
            return 1
    else:
        sequences = all_sequences

    print(f"\nFound {len(sequences)} sequence(s) to process:")
    for seq_name, _, _ in sequences:
        print(f"  - {seq_name}")
    print()

    # Process each sequence
    successful = []
    failed = []

    try:
        for i, (seq_name, video_path, gt_path) in enumerate(sequences, 1):
            print(f"\n[{i}/{len(sequences)}] Processing {seq_name}...")

            success = run_tracker_on_sequence(seq_name, video_path, gt_path, args)

            if success:
                successful.append(seq_name)
            else:
                failed.append(seq_name)
                if not args.continue_on_error:
                    print("\nStopping due to error (use --continue-on-error to process remaining sequences)")
                    break

    except KeyboardInterrupt:
        print("\n\nProcessing interrupted by user!")

    # Print summary
    print("\n" + "="*80)
    print("PROCESSING SUMMARY")
    print("="*80)
    print(f"Total sequences: {len(sequences)}")
    print(f"Successful: {len(successful)}")
    print(f"Failed: {len(failed)}")

    if successful:
        print(f"\n✓ Successfully processed ({len(successful)}):")
        for seq in successful:
            print(f"  - {seq}")

    if failed:
        print(f"\n✗ Failed ({len(failed)}):")
        for seq in failed:
            print(f"  - {seq}")

    print("="*80)

    return 0 if not failed else 1


if __name__ == '__main__':
    sys.exit(main())
