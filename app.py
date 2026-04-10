"""
Machine Vision Anomaly Detection — Flask Web App
Run: python app.py
Then open http://localhost:5000
"""

import os
import json
import threading
from flask import Flask, request, jsonify, render_template, send_from_directory
from werkzeug.utils import secure_filename
from detector import process_video

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 500 * 1024 * 1024  # 500 MB max upload

UPLOAD_DIR = os.path.join(os.path.dirname(__file__), "uploads")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "output")
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

ALLOWED_EXTENSIONS = {"mp4", "avi", "mov", "mkv", "webm"}

# In-memory job tracker
jobs = {}  # job_id -> { status, progress, total, summary, error }


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def run_job(job_id, video_path):
    """Background thread: process video and update job status."""
    jobs[job_id]["status"] = "processing"

    def progress_cb(done, total):
        jobs[job_id]["progress"] = done
        jobs[job_id]["total"] = total

    try:
        summary, vid, heatmap, json_path = process_video(
            video_path, OUTPUT_DIR, progress_callback=progress_cb
        )
        jobs[job_id]["status"] = "done"
        jobs[job_id]["summary"] = summary
        jobs[job_id]["output_video"] = os.path.basename(vid)
        jobs[job_id]["heatmap"] = os.path.basename(heatmap)
    except Exception as e:
        jobs[job_id]["status"] = "error"
        jobs[job_id]["error"] = str(e)


# ------------------------------------------------------------------
# Routes
# ------------------------------------------------------------------

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/upload", methods=["POST"])
def upload():
    if "video" not in request.files:
        return jsonify({"error": "No file part"}), 400
    f = request.files["video"]
    if f.filename == "":
        return jsonify({"error": "No file selected"}), 400
    if not allowed_file(f.filename):
        return jsonify({"error": "Unsupported file type. Use mp4, avi, mov, mkv, webm"}), 400

    filename = secure_filename(f.filename)
    save_path = os.path.join(UPLOAD_DIR, filename)
    f.save(save_path)

    import uuid
    job_id = str(uuid.uuid4())[:8]
    jobs[job_id] = {
        "status": "queued",
        "progress": 0,
        "total": 0,
        "filename": filename,
        "summary": None,
        "error": None,
    }

    t = threading.Thread(target=run_job, args=(job_id, save_path), daemon=True)
    t.start()

    return jsonify({"job_id": job_id})


@app.route("/status/<job_id>")
def status(job_id):
    job = jobs.get(job_id)
    if not job:
        return jsonify({"error": "Job not found"}), 404
    resp = {
        "status": job["status"],
        "progress": job["progress"],
        "total": job["total"],
        "error": job.get("error"),
    }
    if job["status"] == "done":
        s = job["summary"]
        resp["summary"] = {
            "total_frames": s["total_frames"],
            "total_alerts": s["total_alerts"],
            "fps": s["fps"],
            "baseline_mean": s["baseline_mean"],
            "baseline_std": s["baseline_std"],
            "output_video": job.get("output_video"),
            "heatmap": job.get("heatmap"),
            "alerts": s["alerts"][:50],  # cap for JSON response
            "frame_results": s["frame_results"],
        }
    return jsonify(resp)


@app.route("/output/<filename>")
def output_file(filename):
    return send_from_directory(OUTPUT_DIR, filename)


@app.route("/uploads/<filename>")
def uploaded_file(filename):
    return send_from_directory(UPLOAD_DIR, filename)


if __name__ == "__main__":
    print("\n🎥  Anomaly Detection System")
    print("   Open your browser at → http://localhost:5000\n")
    app.run(debug=True, port=5000)
