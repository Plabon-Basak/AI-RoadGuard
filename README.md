# AI RoadGuard

AI RoadGuard is a computer vision application for road-scene analysis built with OpenCV, YOLOv8, and Streamlit. It detects objects in road images, tracks objects across video frames, visualizes apparent motion, estimates traffic levels, analyzes virtual-line crossings, and generates preliminary review alerts.

The project is a prototype for road-scene monitoring. Its results are intended to support human review, not replace professional traffic assessment or calibrated traffic monitoring systems.

## Features

### 1. Road Object Detection

- Upload a road image through the Streamlit interface.
- Detect road-scene objects using YOLOv8.
- Configure detection confidence and inference image size.
- View annotated detection results and object information.

### 2. Video Traffic Analytics

- Upload a road video for analysis.
- Detect and track objects across video frames using YOLOv8 and ByteTrack.
- Count tracked vehicle crossings over a configurable virtual line.
- Estimate traffic levels using detected vehicle counts.
- Generate an annotated video and a structured JSON summary.

### 3. OpenCV Optical Flow Analysis

- Calculate dense optical flow using OpenCV's Farnebäck algorithm.
- Visualize apparent frame-to-frame movement using magenta motion arrows.
- Display the average apparent motion magnitude for each analyzed frame pair.
- Include optical-flow statistics in the video analysis JSON report.
- Process a reduced-resolution grayscale copy for optical-flow calculations.

**Important:** Optical flow measures apparent pixel movement at the analysis resolution. Its score is not a direct measurement of vehicle speed, and camera movement, lighting changes, shadows, and moving backgrounds can affect the result.

### 4. Rule-Based Review Alerts

- Flag high detected vehicle counts for manual congestion review.
- Flag unexpected detections, such as a clock in a road scene.
- Identify highly overlapping detection boxes for inspection.
- Generate structured reports containing detection statistics, preliminary alerts, and limitations.

These alerts are rule-based indicators for human review, not confirmed road hazards.

### 5. Result Visualization and Reporting

- Generate annotated road-scene images.
- Display detection summaries and preliminary alerts.
- Generate downloadable results through the application where supported.
- Save processed videos and JSON analysis reports in the output directory.

## Technology Stack

- Python
- OpenCV
- Ultralytics YOLOv8
- PyTorch
- ByteTrack
- Streamlit
- NumPy
- Boto3

## Project Structure

```text
AI-RoadGuard/
├── docs/
├── outputs/                  # Generated files; ignored by Git
├── samples/                  # Sample road images
├── src/
│   ├── app.py                # Streamlit application
│   ├── main.py               # Basic OpenCV image processing
│   ├── detect_objects.py     # YOLO image detection
│   ├── video_analyzer.py     # Video analytics, tracking, optical flow
│   ├── analyze_hazards.py    # Rule-based review alerts
│   └── visualize_results.py  # Review image generation
├── tests/
│   ├── test_hazard_logic.py
│   └── test_opencv.py
├── .gitignore
├── requirements.txt
└── README.md
```

The tree above describes the expected project layout. Check the actual repository if any filenames differ.

## Requirements

- Python
- Git
- Internet access for installing dependencies and obtaining model weights
- Sufficient disk space for video processing and generated outputs

Video analysis can require substantial memory and storage, particularly for high-resolution videos.

## Installation

Clone the repository:

```powershell
git clone https://github.com/Plabon-Basak/AI-RoadGuard.git
cd AI-RoadGuard
```

Create and activate a virtual environment on Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Install the project dependencies:

```powershell
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

The application expects `yolov8n.pt` in the project root. Ensure the model weights are available there before running the application. If they are not already present, obtain them from the official Ultralytics model distribution.

## Run the Application

Start the Streamlit interface:

```powershell
python -m streamlit run src/app.py
```

Open the local URL displayed in the terminal, usually:

`http://localhost:8501`

Choose **Image Detection** or **Video Traffic Analytics** in the interface and follow the on-screen instructions.

For video analysis, the application generates an annotated video and a JSON summary when processing completes successfully.

## Run the Tests

Install pytest if needed:

```powershell
python -m pip install pytest
```

Run the automated tests:

```powershell
python -m pytest -q
```

The current test suite includes checks for selected OpenCV functionality, bounding-box intersection-over-union calculations, and rule-based alert behavior.

## Generated Outputs

Analysis results are saved under `outputs/`. Depending on the operation, generated files may include:

- Annotated images
- Processed road videos
- `video_detection_summary.json`
- Hazard-review reports
- Detection statistics and per-frame summaries

The video analysis report includes vehicle crossing counts, detected object statistics, heuristic congestion estimates, optical-flow measurements, and known limitations.

Generated outputs and model weights are excluded from Git to avoid accidentally committing large files.

## Methodology

### Object Detection and Tracking

YOLOv8 detects road-scene objects, while ByteTrack associates detections across frames to maintain object track IDs.

### Virtual-Line Crossing Analysis

A configurable horizontal line is placed in the video frame. Tracked vehicles that move across the line are counted according to the implemented crossing logic.

Tracking errors, occlusion, and changes in track IDs may affect the counts.

### Optical Flow

Farnebäck dense optical flow estimates apparent pixel displacement between consecutive grayscale frames. The analysis uses a reduced-width image to reduce the computational cost of optical-flow calculations.

Motion vectors are drawn over the annotated video, and per-frame motion magnitudes are recorded in the JSON report.

The motion values depend on the analysis resolution and frame rate. They should not be interpreted as calibrated physical velocity.

### Traffic-Level Estimation

Traffic levels are estimated using the number of detected vehicles in each frame:

- **Low:** 0–5 vehicles
- **Moderate:** 6–12 vehicles
- **High:** More than 12 vehicles

These are configurable prototype thresholds. They are not calibrated measurements of road occupancy, traffic density, or congestion.

## Limitations

- Detection accuracy depends on image quality, lighting, camera angle, occlusion, and model performance.
- Vehicle counts and crossing counts may be inaccurate when tracking IDs change or objects are missed.
- High vehicle counts do not independently establish congestion.
- Optical flow measures apparent image motion and can be affected by camera movement, shadows, lighting changes, and moving backgrounds.
- Motion magnitude is not a direct measure of vehicle speed.
- Overlapping bounding boxes do not necessarily indicate incorrect detections.
- Review alerts are preliminary and require human verification.
- The application does not confirm collisions or predict accidents.
- Performance and reliability must be evaluated on representative road-scene data before operational use.

## Future Improvements

- Evaluate detection and tracking performance on diverse road-scene datasets.
- Add quantitative evaluation metrics for detection, tracking, and crossing counts.
- Improve traffic estimation using calibrated camera views and longer observation windows.
- Investigate camera-motion compensation for more robust optical-flow measurements.
- Add configurable alert thresholds and reproducible benchmark results.
- Provide screenshots, sample videos, and documented evaluation results.
- Explore reproducible cloud deployment when appropriate infrastructure is available.

## License

A license file has not yet been specified. Add a license before describing the project as open source under a particular license.
