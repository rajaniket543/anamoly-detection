# Machine Vision–Based Video Anomaly Detection System

A full Python + OpenCV project that detects anomalous motion events in surveillance video — no labels, no action recognition, purely temporal motion analysis.

---

## Project Structure

```
anomaly_detection/
├── app.py              ← Flask web app (main entry point)
├── detector.py         ← Core detection engine (OpenCV)
├── run_cli.py          ← Command-line interface
├── requirements.txt    ← Python dependencies
├── templates/
│   └── index.html      ← Web UI
├── uploads/            ← Auto-created: uploaded videos stored here
└── output/             ← Auto-created: annotated video, heatmap, JSON
```

---

## Quick Setup (Windows / macOS / Linux)

### Step 1 — Install Python
Make sure you have Python 3.9 or newer.
Check with: `python --version`

### Step 2 — Create a virtual environment (recommended)
```bash
python -m venv venv

# Windows:
venv\Scripts\activate

# macOS / Linux:
source venv/bin/activate
```

### Step 3 — Install dependencies
```bash
pip install -r requirements.txt
```

### Step 4 — Run the web app
```bash
python app.py
```

Then open your browser at → **http://localhost:5000**

---

## Using the Web App

1. Open http://localhost:5000
2. Drop or click to upload any surveillance video (MP4, AVI, MOV, MKV, WEBM)
3. The system will process it frame-by-frame in the background
4. When done, you'll see:
   - **Annotated video** — with bounding boxes (green = normal, red = anomaly) and live anomaly score bar
   - **Anomaly score chart** — score over time for every frame
   - **Motion heatmap** — color-coded map of where movement occurred most
   - **Alert log** — table of every anomaly frame with timestamp and severity

---

## Using the CLI (no browser needed)

```bash
# Basic usage
python run_cli.py path/to/video.mp4

# Custom output folder and threshold
python run_cli.py path/to/video.mp4 --output ./my_results --threshold 0.12
```

Outputs saved to `./output/`:
- `<name>_annotated.mp4` — annotated video
- `<name>_heatmap.jpg`   — motion heatmap image
- `<name>_results.json`  — full frame-by-frame data

---

## How It Works

| Stage | What happens |
|-------|-------------|
| Frame extraction | Video decoded at original FPS |
| Preprocessing | Gaussian blur removes sensor noise |
| Background subtraction | MOG2 model separates foreground (moving) from background |
| Region formation | Morphological ops clean mask; contours grouped into blobs |
| Temporal analysis | Motion scores tracked over a sliding window of 30 frames |
| Baseline modeling | First ~100 frames establish normal motion mean + std dev |
| Anomaly scoring | Z-score of current motion vs baseline, normalized 0–1 |
| Output | Annotated video + heatmap + JSON report |

### Adjustable Parameters (in detector.py)

| Parameter | Default | Effect |
|-----------|---------|--------|
| `history` | 200 | Background model memory (frames) |
| `var_threshold` | 40 | Sensitivity to pixel change |
| `window_size` | 30 | Temporal sliding window |
| `anomaly_threshold` | 0.15 | Score needed to trigger alert |
| `baseline_frames` | 100 | Frames used to build normal model |

---

## Output Files

### Annotated Video
- Green boxes = normal motion regions
- Red boxes = anomalous motion regions
- Top-left bar = real-time anomaly score (0–1)
- Status label: `LEARNING...` → `NORMAL` → `ANOMALY`

### Heatmap (JET colormap)
- Blue = rarely active regions
- Green/Yellow = moderately active
- Red = hotspots (most motion over time)

### JSON Results
```json
{
  "total_frames": 1500,
  "total_alerts": 12,
  "baseline_mean": 0.00312,
  "baseline_std": 0.00089,
  "alerts": [
    { "frame": 423, "timestamp_sec": 14.1, "anomaly_score": 0.72, ... }
  ],
  "frame_results": [
    { "frame": 1, "motion_score": 0.0021, "anomaly_score": 0.0, "is_anomaly": false },
    ...
  ]
}
```

---

## Troubleshooting

**`ModuleNotFoundError: No module named 'cv2'`**
→ Run `pip install opencv-python`

**Video won't open / black output**
→ Try converting your video to MP4 H.264 first using HandBrake or ffmpeg:
`ffmpeg -i input.avi -c:v libx264 output.mp4`

**Too many false positives**
→ Increase `anomaly_threshold` (try 0.25) or `var_threshold` (try 60) in detector.py

**Misses real anomalies**
→ Lower `anomaly_threshold` (try 0.08) or `var_threshold` (try 25)

---

## Limitations

- Cannot identify specific actions (outputs motion deviation, not "person fainted")
- Sensitive to lighting changes — avoid scenes with flickering lights
- Best for relatively static camera and stable backgrounds
- Multiple overlapping objects may merge into one motion region
