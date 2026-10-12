# AI RoadGuard

AI RoadGuard is a computer vision application for road-scene analysis using OpenCV, YOLOv8, and Streamlit. It detects objects in road images, analyzes traffic videos, tracks vehicles, estimates traffic levels, and generates preliminary review alerts.

The project is designed as a prototype for road-scene monitoring. Its alerts are intended to support human review, not replace professional traffic assessment.

## Features

### 1. Road Object Detection
- Upload a road image through the Streamlit interface.
- Detect road-scene objects using YOLOv8.
- Configure detection confidence and inference image size.
- View annotated detection results and object information.

### 2. Video Traffic Analytics
- Upload a road video for analysis.
- Detect and track objects across video frames.
- Estimate traffic levels using detected vehicle counts.
- Configure a virtual crossing line to analyze vehicle crossings.
- Generate an annotated video and a JSON summary.

### 3. Rule-Based Review Alerts
- Flag high detected vehicle counts for manual congestion review.
- Flag unexpected detections, such as a clock in a road scene.
- Identify highly overlapping detection boxes for inspection.
- Save a structured JSON report containing detection statistics, alerts, and limitations.

### 4. Result Visualization
- Generate visual road-scene review images.
- Display detection summaries and preliminary alerts.
- Provide downloadable results through the application where supported.

## Technology Stack

- Python
- OpenCV
- Ultralytics YOLOv8
- PyTorch
- Streamlit
- NumPy
- Boto3

## Project Structure

```text
AI-RoadGuard/
├── docs/
├── outputs/                 # Generated files; ignored by Git
├── samples/                 # Sample road images
├── src/
│   ├── app.py               # Streamlit application
│   ├── main.py              # Basic OpenCV image processing
│   ├── detect_objects.py    # YOLO image detection
│   ├── video_analyzer.py    # Video analytics and tracking
│   ├── analyze_hazards.py   # Rule-based review alerts
│   └── visualize_results.py # Review image generation
├── tests/
│   └── test_hazard_logic.py
│   └── test_opencv.py
├── .gitignore
├── requirements.txt
└── README.md
```

## Requirements

- Python installed on your system
- Git
- Internet access for installing dependencies and obtaining the YOLO model weights
- Sufficient disk space for video processing and generated outputs

## Installation

Clone the repository after it has been published to GitHub:

```bash
git clone YOUR_REPOSITORY_URL
cd AI-RoadGuard
```

Create and activate a virtual environment on Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Install dependencies:

```powershell
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

The application expects `yolov8n.pt` in the project root. Place the model file there before running the application if it is not already present.

## Run the Application

Start the Streamlit interface:

```powershell
python -m streamlit run src/app.py
```

Open the local URL displayed in the terminal, usually `http://localhost:8501`.

Choose **Image Detection** or **Video Traffic Analytics** from the sidebar and follow the on-screen instructions.

## Run the Tests

Install pytest if needed:

```powershell
python -m pip install pytest
```

Run the automated tests:

```powershell
python -m pytest -q
```

The current test suite checks bounding-box intersection-over-union calculations and selected rule-based alert behavior.

## Generated Outputs

Analysis results are saved under `outputs/`. Depending on the analysis performed, generated files can include annotated images, JSON summaries, hazard-review reports, and processed videos.

Generated outputs and model weights are excluded from Git to avoid accidentally committing large files.

## Limitations

- Detection accuracy depends on image quality, lighting, camera angle, and model performance.
- Traffic levels are heuristic estimates based on detected vehicle counts; they are not calibrated traffic measurements.
- Vehicle detections and crossing-line counts may be inaccurate when objects are occluded or tracking IDs change.
- High vehicle counts do not prove congestion.
- Overlapping bounding boxes do not necessarily indicate incorrect detections.
- Review alerts are preliminary and require human verification.
- The application does not confirm collisions or predict accidents.

## Future Improvements

- Evaluate detection and tracking performance on diverse road-scene datasets.
- Improve traffic-level estimation with calibrated camera views and longer observation windows.
- Add configurable alert thresholds and richer evaluation metrics.
- Provide reproducible sample results and screenshots.

## License

Add a license file before describing this project as open source under a specific license.
