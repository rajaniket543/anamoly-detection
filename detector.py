"""
Machine Vision-Based Video Anomaly Detection
Core detection engine using OpenCV background subtraction + temporal analysis
"""

import cv2
import numpy as np
import json
import os
import subprocess
from datetime import datetime
from collections import deque


class AnomalyDetector:
    def __init__(self, config=None):
        cfg = config or {}

        # Background subtractor (MOG2 = Mixture of Gaussians v2)
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(
            history=cfg.get("history", 200),
            varThreshold=cfg.get("var_threshold", 40),
            detectShadows=cfg.get("detect_shadows", True),
        )

        # Morphological kernel for noise cleanup
        self.kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

        # Temporal motion history (sliding window)
        self.window_size = cfg.get("window_size", 30)
        self.motion_history = deque(maxlen=self.window_size)

        # Anomaly threshold (fraction of frame that is "moving")
        self.anomaly_threshold = cfg.get("anomaly_threshold", 0.15)
        self.min_baseline_std = cfg.get("min_baseline_std", 0.002)
        self.min_region_area = cfg.get("min_region_area", 150)
        self.min_anomaly_regions = cfg.get("min_anomaly_regions", 1)
        self.fps = cfg.get("fps", 30) or 30

        # Baseline: mean motion level during normal period
        self.baseline_mean = None
        self.baseline_std = None
        self.baseline_window = cfg.get("baseline_frames", 100)
        self.baseline_buffer = []
        self.baseline_ready = False

        # Heatmap accumulator
        self.heatmap_accum = None

        # Stats
        self.frame_count = 0
        self.alerts = []

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _preprocess(self, frame):
        """Denoise and convert to grayscale."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        return blurred

    def _get_motion_mask(self, frame):
        """Apply background subtraction and clean up mask."""
        preprocessed = self._preprocess(frame)
        fg_mask = self.bg_subtractor.apply(preprocessed)

        # Remove shadows (value 127 in MOG2 output)
        _, fg_mask = cv2.threshold(fg_mask, 200, 255, cv2.THRESH_BINARY)

        # Morphological ops: remove noise, fill holes
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, self.kernel)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, self.kernel)
        fg_mask = cv2.dilate(fg_mask, self.kernel, iterations=2)

        return fg_mask

    def _motion_score(self, mask):
        """Fraction of frame pixels that are moving (0.0 – 1.0)."""
        total = mask.shape[0] * mask.shape[1]
        moving = cv2.countNonZero(mask)
        return moving / total if total > 0 else 0.0

    def _update_baseline(self, score):
        """Collect motion scores until baseline is established."""
        if self.baseline_ready:
            return
        self.baseline_buffer.append(score)
        if len(self.baseline_buffer) >= self.baseline_window:
            arr = np.array(self.baseline_buffer)
            self.baseline_mean = float(np.mean(arr))
            self.baseline_std = max(float(np.std(arr)), self.min_baseline_std)
            self.baseline_ready = True

    def _compute_anomaly_score(self, score):
        """
        Z-score of current motion vs baseline.
        Clamped to [0, 1] for display.
        """
        if not self.baseline_ready:
            return 0.0
        z = (score - self.baseline_mean) / self.baseline_std
        # Map z-score > 0 to (0, 1] using sigmoid-like clamp
        normalized = min(1.0, max(0.0, z / 6.0))
        return round(normalized, 4)

    def _get_motion_regions(self, mask):
        """Find bounding boxes of moving regions."""
        contours, _ = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        regions = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < self.min_region_area:  # ignore tiny noise blobs
                continue
            x, y, w, h = cv2.boundingRect(cnt)
            regions.append({"x": int(x), "y": int(y), "w": int(w), "h": int(h), "area": int(area)})
        return regions

    def _update_heatmap(self, mask, shape):
        """Accumulate motion into a persistent heatmap."""
        if self.heatmap_accum is None:
            self.heatmap_accum = np.zeros((shape[0], shape[1]), dtype=np.float32)
        motion_float = mask.astype(np.float32) / 255.0
        self.heatmap_accum += motion_float

    def _render_heatmap(self, shape):
        """Normalize heatmap and return color image."""
        if self.heatmap_accum is None:
            return np.zeros((shape[0], shape[1], 3), dtype=np.uint8)
        norm = cv2.normalize(self.heatmap_accum, None, 0, 255, cv2.NORM_MINMAX)
        norm_u8 = norm.astype(np.uint8)
        colored = cv2.applyColorMap(norm_u8, cv2.COLORMAP_JET)
        return colored

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def process_frame(self, frame):
        """
        Process one frame. Returns a result dict.
        """
        self.frame_count += 1
        h, w = frame.shape[:2]

        mask = self._get_motion_mask(frame)
        score = self._motion_score(mask)
        regions = self._get_motion_regions(mask)

        self._update_baseline(score)
        self._update_heatmap(mask, (h, w))
        self.motion_history.append(score)

        anomaly_score = self._compute_anomaly_score(score)
        is_anomaly = (
            anomaly_score > self.anomaly_threshold
            and self.baseline_ready
            and len(regions) >= self.min_anomaly_regions
        )

        if is_anomaly:
            self.alerts.append({
                "frame": self.frame_count,
                "timestamp_sec": round(self.frame_count / self.fps, 2),
                "anomaly_score": anomaly_score,
                "motion_score": round(score, 4),
                "region_count": len(regions),
            })

        return {
            "frame_num": self.frame_count,
            "motion_score": round(score, 4),
            "anomaly_score": anomaly_score,
            "is_anomaly": is_anomaly,
            "baseline_ready": self.baseline_ready,
            "regions": regions,
            "mask": mask,
        }

    def get_heatmap(self, shape):
        return self._render_heatmap(shape)

    def get_motion_history(self):
        return list(self.motion_history)

    def get_summary(self):
        return {
            "total_frames": self.frame_count,
            "total_alerts": len(self.alerts),
            "baseline_mean": round(self.baseline_mean, 5) if self.baseline_mean is not None else None,
            "baseline_std": round(self.baseline_std, 5) if self.baseline_std is not None else None,
            "alerts": self.alerts,
        }


# ------------------------------------------------------------------
# Video processing pipeline
# ------------------------------------------------------------------

def _transcode_for_web(src_path, dst_path):
    """Convert OpenCV output into a browser-friendly H.264 MP4 when ffmpeg is available."""
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        src_path,
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        "-an",
        dst_path,
    ]
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        os.remove(src_path)
        return dst_path
    except (FileNotFoundError, subprocess.CalledProcessError):
        if src_path != dst_path and os.path.exists(src_path):
            try:
                os.replace(src_path, dst_path)
            except OSError:
                pass
        return dst_path

def process_video(video_path, output_dir, progress_callback=None):
    """
    Full pipeline: read video → detect anomalies → write annotated video + heatmap.
    Returns a summary dict.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    os.makedirs(output_dir, exist_ok=True)

    base = os.path.splitext(os.path.basename(video_path))[0]
    raw_out_video_path = os.path.join(output_dir, f"{base}_annotated_raw.mp4")
    out_video_path = os.path.join(output_dir, f"{base}_annotated.mp4")
    out_heatmap_path = os.path.join(output_dir, f"{base}_heatmap.jpg")
    out_json_path = os.path.join(output_dir, f"{base}_results.json")

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(raw_out_video_path, fourcc, fps, (width, height))

    detector = AnomalyDetector({"fps": fps})

    frame_results = []
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        result = detector.process_frame(frame)
        frame_results.append({
            "frame": result["frame_num"],
            "motion_score": result["motion_score"],
            "anomaly_score": result["anomaly_score"],
            "is_anomaly": result["is_anomaly"],
        })

        # --- Annotate frame ---
        annotated = frame.copy()

        # Draw motion region boxes
        for r in result["regions"]:
            color = (0, 0, 220) if result["is_anomaly"] else (0, 200, 80)
            cv2.rectangle(annotated, (r["x"], r["y"]),
                          (r["x"] + r["w"], r["y"] + r["h"]), color, 2)

        # Overlay: anomaly score bar
        bar_w = int(result["anomaly_score"] * 200)
        bar_color = (0, 0, 220) if result["is_anomaly"] else (0, 200, 80)
        cv2.rectangle(annotated, (10, 10), (210, 32), (30, 30, 30), -1)
        cv2.rectangle(annotated, (10, 10), (10 + bar_w, 32), bar_color, -1)
        cv2.putText(annotated, f"Anomaly: {result['anomaly_score']:.2f}",
                    (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)

        # Status label
        status = "ANOMALY" if result["is_anomaly"] else ("LEARNING..." if not result["baseline_ready"] else "NORMAL")
        color_txt = (0, 0, 220) if result["is_anomaly"] else (0, 200, 80)
        cv2.putText(annotated, status, (10, 58),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color_txt, 2)

        # Frame counter
        cv2.putText(annotated, f"Frame {result['frame_num']}/{total_frames}",
                    (width - 180, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        writer.write(annotated)

        frame_idx += 1
        if progress_callback and frame_idx % 10 == 0:
            progress_callback(frame_idx, total_frames)

    cap.release()
    writer.release()
    out_video_path = _transcode_for_web(raw_out_video_path, out_video_path)

    # Save heatmap
    heatmap_img = detector.get_heatmap((height, width))
    cv2.imwrite(out_heatmap_path, heatmap_img)

    # Build summary
    summary = detector.get_summary()
    summary["fps"] = fps
    summary["width"] = width
    summary["height"] = height
    summary["total_frames"] = total_frames
    summary["video_path"] = os.path.basename(out_video_path)
    summary["heatmap_path"] = os.path.basename(out_heatmap_path)
    summary["frame_results"] = frame_results
    summary["processed_at"] = datetime.now().isoformat()

    with open(out_json_path, "w") as f:
        json.dump(summary, f, indent=2)

    return summary, out_video_path, out_heatmap_path, out_json_path
