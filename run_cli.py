"""
Command-line interface for anomaly detection.
Usage:
    python run_cli.py path/to/video.mp4
    python run_cli.py path/to/video.mp4 --output ./results --threshold 0.12
"""

import argparse
import sys
import os
from detector import process_video


def main():
    parser = argparse.ArgumentParser(
        description="Machine Vision Anomaly Detector — CLI mode"
    )
    parser.add_argument("video", help="Path to input video file")
    parser.add_argument("--output", default="./output", help="Output directory (default: ./output)")
    parser.add_argument("--threshold", type=float, default=0.15,
                        help="Anomaly detection threshold 0.0–1.0 (default: 0.15)")
    args = parser.parse_args()

    if not os.path.isfile(args.video):
        print(f"[ERROR] Video file not found: {args.video}")
        sys.exit(1)

    print(f"\n🎥  Input   : {args.video}")
    print(f"📁  Output  : {args.output}")
    print(f"⚙️  Threshold: {args.threshold}")
    print()

    def progress(done, total):
        pct = int(done / total * 100)
        bar = "█" * (pct // 5) + "░" * (20 - pct // 5)
        print(f"\r  [{bar}] {pct}% — frame {done}/{total}", end="", flush=True)

    print("  Processing…")
    summary, vid, heatmap, json_path = process_video(
        args.video, args.output, progress_callback=progress
    )
    print("\n")
    print("─" * 50)
    print(f"  ✅ Done!")
    print(f"  Total frames : {summary['total_frames']}")
    print(f"  Alerts raised: {summary['total_alerts']}")
    print(f"  Baseline mean: {summary['baseline_mean']:.5f}" if summary['baseline_mean'] else "  Baseline: insufficient frames")
    print(f"\n  Outputs saved to: {args.output}/")
    print(f"    • {os.path.basename(vid)}")
    print(f"    • {os.path.basename(heatmap)}")
    print(f"    • {os.path.basename(json_path)}")
    print()

    if summary["total_alerts"] > 0:
        print("  ⚠️  Top alerts:")
        for a in summary["alerts"][:5]:
            ts = a["timestamp_sec"]
            m, s = divmod(int(ts), 60)
            print(f"    Frame {a['frame']:5d}  [{m:02d}:{s:02d}]  score={a['anomaly_score']:.3f}")
    else:
        print("  ✔  No anomalies detected.")
    print()


if __name__ == "__main__":
    main()
