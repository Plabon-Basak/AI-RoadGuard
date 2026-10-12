# AI RoadGuard

AI RoadGuard is a computer vision prototype for road-scene analysis built with OpenCV, YOLOv8, ByteTrack, and Streamlit. It analyzes road images and videos to detect objects, track vehicles, visualize apparent motion, estimate traffic levels, count virtual-line crossings, and generate preliminary review alerts.

The project is designed to support human review of road scenes. It does not replace professional traffic assessment, calibrated traffic monitoring, or validated road-safety systems.

## Features

### 1. Road Object Detection

- Upload road images through the Streamlit interface.
- Detect objects using YOLOv8.
- Configure detection confidence and inference image size.
- View annotated images and detection information.

### 2. Video Traffic Analytics

- Upload a road video for analysis.
- Detect and track objects across frames using YOLOv8 and ByteTrack.
- Count tracked vehicle crossings over a configurable virtual line.
- Estimate traffic levels using detected vehicle counts.
- Generate an annotated video and a structured JSON summary.

### 3. Road-Relevant Class Counts

- Summarize detections for road-relevant classes: cars, buses, trucks, motorcycles, bicycles, and people.
- Filter out unrelated classes from the road-relevant summary.
- Present the summary in the dashboard when the analysis data is available.

**Important:** These counts represent detections across processed frames, not necessarily unique objects or vehicles. A single object can be counted multiple times across different frames.

### 4. OpenCV Optical Flow Analysis

- Calculate dense optical flow using OpenCV's Farnebäck algorithm.
- Visualize apparent frame-to-frame movement using motion arrows.
- Calculate average apparent motion magnitude for analyzed frame pairs.
- Include optical-flow statistics in the video analysis summary.
- Process reduced-resolution grayscale frames to reduce computational cost.

**Important:** Optical flow measures apparent pixel movement, not actual vehicle speed. Camera movement, lighting changes, shadows, and moving backgrounds can affect the measurements.

### 5. Rule-Based Review Alerts

- Flag high detected vehicle counts for manual congestion review.
- Flag unexpected detections for inspection.
- Identify highly overlapping detection boxes for review.
- Generate preliminary alerts and supporting detection statistics.

These alerts are rule-based indicators for human review, not confirmed road hazards or verified incidents.

### 6. Visualization and Reporting

- Display annotated detection results.
- Visualize road-relevant class counts and traffic analytics.
- Generate processed videos and JSON analysis summaries.
- Produce evaluation reports from available analysis results.

## Technology Stack

- Python
- OpenCV
- Ultralytics YOLOv8
- PyTorch
- ByteTrack
- Streamlit
- NumPy
- pytest

Boto3 is relevant only if the current implementation uses AWS functionality; it is not required merely to run the local Streamlit application.

## Project Structure

```text
AI-RoadGuard/
├── docs/
├── outputs/                  # Generated results; ignored by Git
├── samples/                  # Sample road images
├── src/
│   ├── app.py                # Streamlit application
│   ├── main.py               # Basic OpenCV image processing
│   ├── detect_objects.py     # YOLO image detection
│   ├── video_analyzer.py     # Tracking, video analytics, optical flow
│   ├── analyze_hazards.py    # Rule-based review alerts
│   ├── evaluate_results.py   # Evaluation report generation
│   └── visualize_results.py  # Result visualization
├── tests/
│   ├── test_hazard_logic.py
│   ├── test_opencv.py
│   └── test_evaluate_results.py
├── .gitignore
├── requirements.txt
└── README.md
```

This tree should match the actual repository. Remove or adjust entries if your current files differ.

## Requirements

- Python
- Git
- Internet access for installing dependencies and obtaining model weights
- Sufficient disk space and memory for video processing

Video analysis can require substantial storage and memory, especially for high-resolution videos.

## Installation

Clone the repository:

```powershell
git clone https://github.com/Plabon-Basak/AI-RoadGuard.git
cd AI-RoadGuard
```

Create and activate a virtual environment in Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Install dependencies:

```powershell
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

The application expects `yolov8n.pt` in the project root. Ensure the model weights are available before running the application. If they are missing, obtain them through the official Ultralytics model distribution.

## Run the Application

Start Streamlit:

```powershell
python -m streamlit run src/app.py
```

Open the local URL displayed in the terminal, usually:

http://localhost:8501

Choose **Image Detection** or **Video Traffic Analytics** and follow the instructions in the application.

For video analysis, the application generates an annotated video and a JSON summary when processing completes successfully.

## Run the Tests

Run the automated test suite:

```powershell
python -m pytest -q
```

The current test suite has **15 passing tests** in the latest local verification.

The tests cover selected OpenCV functionality, optical flow, road-relevant class filtering, bounding-box intersection-over-union calculations, rule-based alert behavior, and evaluation-report logic.

To check Python syntax:

```powershell
python -m py_compile .\src\video_analyzer.py .\src\app.py
```

To check for whitespace errors in pending Git changes:

```powershell
git diff --check
```

## Generated Outputs

Analysis results are saved under `outputs/`. Depending on the operation, these may include:

- Annotated images
- Processed road videos
- `video_detection_summary.json`
- Hazard-review reports
- Detection statistics and per-frame summaries
- Evaluation reports

The video analysis summary can include object-class counts, vehicle crossing counts, heuristic traffic estimates, optical-flow measurements, and documented limitations.

Generated outputs and model weights are excluded from Git to avoid accidentally committing large files.

## Methodology

### Object Detection and Tracking

YOLOv8 detects objects in road scenes, while ByteTrack associates detections across video frames to maintain track IDs.

Detection and tracking performance depends on scene conditions, occlusion, model confidence, and tracking stability.

### Virtual-Line Crossing Analysis

A configurable virtual line is placed in the video frame. Tracked vehicles crossing the line are counted according to the implemented crossing logic.

Missed detections, occlusion, and changes in track IDs can affect crossing counts.

### Optical Flow

Farnebäck dense optical flow estimates apparent pixel displacement between consecutive grayscale frames. A reduced-resolution image is used to reduce computational cost.

Motion vectors are visualized over the processed video, and motion magnitude statistics are recorded in the analysis summary.

These measurements depend on image resolution, frame rate, and camera conditions. They are not calibrated physical velocity measurements.

### Traffic-Level Estimation

Traffic levels are estimated using the number of detected vehicles in each frame:

- **Low:** 0–5 vehicles
- **Moderate:** 6–12 vehicles
- **High:** More than 12 vehicles

These are prototype thresholds, not calibrated measurements of road occupancy, traffic density, or congestion.

### Preliminary Hazard Review

Rule-based checks flag selected conditions, such as high vehicle counts, unexpected classes, and overlapping detection boxes.

These conditions indicate cases that may deserve inspection. They do not establish that a collision, hazard, or traffic violation has occurred.

## Evaluation and Testing

The project includes automated unit tests and an evaluation-reporting module.

The latest local test run completed with 15 passing tests. This confirms that the tested code paths pass their current checks; it does not establish real-world detection accuracy or traffic-monitoring reliability.

For meaningful performance evaluation, use representative road videos and ground-truth annotations to measure detection, tracking, and line-crossing performance. Report the test data, evaluation procedure, and limitations alongside any quantitative results.

## Limitations

- Detection accuracy depends on image quality, lighting, camera angle, occlusion, and model performance.
- Class counts across frames are not equivalent to unique object counts.
- Vehicle and line-crossing counts may be inaccurate when objects are missed or track IDs change.
- High vehicle counts alone do not establish congestion.
- Optical flow can be affected by camera movement, shadows, lighting changes, and moving backgrounds.
- Optical-flow magnitude is not a direct measurement of vehicle speed.
- Overlapping bounding boxes do not necessarily indicate incorrect detections.
- Rule-based alerts require human verification.
- The application does not confirm collisions or predict accidents.
- The prototype requires evaluation on representative road-scene data before operational use.

## Future Improvements

- Evaluate detection and tracking on diverse road-scene datasets.
- Add quantitative metrics for detection, tracking, and crossing counts.
- Improve traffic estimation using calibrated camera views and longer observation windows.
- Investigate camera-motion compensation for optical-flow analysis.
- Add configurable alert thresholds and reproducible benchmark results.
- Include representative screenshots, sample inputs, and documented evaluation results.
- Explore cloud deployment when suitable infrastructure is available.

## License

A license has not yet been specified. Add a `LICENSE` file before distributing the project under a particular open-source license.
