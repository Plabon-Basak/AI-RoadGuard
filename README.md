# AI RoadGuard

**AI-powered road-scene analysis using OpenCV, YOLOv8, and ByteTrack.**

AI RoadGuard is a computer vision prototype that analyzes road images and videos to detect road users, track vehicles, estimate traffic levels, visualize apparent motion, and generate preliminary road-scene review alerts.

**Live Demo:** [Open AI RoadGuard](https://ai-roadguard-plabonbasak.streamlit.app/)

**GitHub Repository:** [Plabon-Basak/AI-RoadGuard](https://github.com/Plabon-Basak/AI-RoadGuard)

## Project Overview

AI RoadGuard combines object detection, object tracking, optical flow, and rule-based analysis to demonstrate how computer vision can support preliminary road-scene monitoring.

Users can upload an image or video through the Streamlit interface and inspect the resulting detections and analytics.

The project is a prototype for educational and hackathon purposes. Its outputs are intended to support human review, not replace professional traffic monitoring or road-safety decisions.

## Features

### 1. Image Detection
- Upload a road-scene image.
- Detect supported objects using YOLOv8.
- Display detected classes and confidence scores.
- View the annotated image with detection bounding boxes.

### 2. Video Analysis
- Upload a road-scene video.
- Detect road users across video frames.
- Track detected objects using ByteTrack.
- Generate an annotated output video.
- Produce a JSON summary of the analysis.

### 3. Vehicle Tracking
- Assign tracking IDs to detected objects across frames.
- Visualize tracking results in the annotated video.
- Support virtual line-crossing analysis.

Tracking IDs are model-generated identifiers and may change when tracking is interrupted or an object is lost.

### 4. Line-Crossing Analysis
- Count tracked objects crossing a configured virtual line.
- Use line-crossing results as an additional road-scene analysis signal.

The counts depend on tracking quality, line placement, video quality, and the selected detection settings.

### 5. Traffic-Level Estimation
AI RoadGuard provides a preliminary traffic-level estimate based on the number of detected vehicles.

The current heuristic uses these thresholds:

| Detected vehicles | Estimated traffic level |
|---|---|
| 0–5 | Low |
| 6–12 | Moderate |
| More than 12 | High |

These thresholds are configurable prototype heuristics. They do not represent calibrated traffic density, congestion, or road capacity.

### 6. Optical Flow Visualization
- Use OpenCV's Farnebäck optical flow to estimate apparent pixel motion between video frames.
- Visualize motion with magenta arrows.
- Provide an additional visual cue for movement in the scene.

Optical flow measures apparent image motion. It does not directly calculate real-world vehicle speed.

### 7. Road-User Class Counts
Display detections for supported road-relevant classes, including:
- Cars
- Buses
- Trucks
- Motorcycles
- Bicycles
- People

Counts accumulated across video frames are detection counts, not necessarily unique-object counts.

### 8. Preliminary Review Alerts
The prototype includes rule-based checks that can flag conditions such as:
- High vehicle detection counts.
- Unexpected detected classes.
- Overlapping detection bounding boxes.

These alerts are preliminary indicators for human review. They do not establish that an accident, hazard, or traffic violation has occurred.

## Technology Stack

| Technology | Purpose |
|---|---|
| Python | Main programming language |
| OpenCV | Image and video processing, optical flow |
| Ultralytics YOLOv8 | Object detection |
| ByteTrack | Multi-object tracking |
| Streamlit | Interactive web application |
| NumPy | Numerical operations |
| Pytest | Automated testing |
| Git and GitHub | Version control and source hosting |
| Streamlit Community Cloud | Application deployment |

## Project Structure

```text
AI-RoadGuard/
├── docs/
├── outputs/                  # Generated files; ignored by Git
├── samples/
├── screenshots/
│   ├── image_detection_screenshots/
│   └── video_analytics_screenshots/
├── src/
│   ├── app.py
│   ├── main.py
│   ├── detect_objects.py
│   ├── video_analyzer.py
│   ├── analyze_hazards.py
│   ├── evaluate_results.py
│   └── visualize_results.py
├── tests/
│   ├── test_hazard_logic.py
│   ├── test_opencv.py
│   └── test_evaluate_results.py
├── .gitignore
├── requirements.txt
└── README.md
```

## Getting Started

### Prerequisites

- Python 3.10 or a compatible Python version supported by the installed dependencies.
- Git.
- An internet connection for installing packages and downloading YOLOv8n weights when they are not already available locally.

### 1. Clone the Repository

```powershell
git clone https://github.com/Plabon-Basak/AI-RoadGuard.git
cd AI-RoadGuard
```

### 2. Create a Virtual Environment

```powershell
python -m venv .venv
```

### 3. Activate the Virtual Environment

For Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

If PowerShell blocks activation, run this command in the current terminal session and try again:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned
```

### 4. Upgrade pip

```powershell
python -m pip install --upgrade pip
```

### 5. Install Dependencies

```powershell
python -m pip install -r requirements.txt
```

The application dependencies include:

- `opencv-python-headless`
- `ultralytics`
- `streamlit`
- `boto3`
- `lap`

The headless OpenCV package is suitable for a cloud deployment without desktop GUI dependencies.

### 6. Run the Application

```powershell
python -m streamlit run src/app.py
```

Streamlit will display a local URL in the terminal. Open that URL in your browser.

### Model Weights

AI RoadGuard uses the YOLOv8n model. If `yolov8n.pt` exists in the project root, the application can load it from there. Otherwise, Ultralytics attempts to download the weights automatically, which requires internet access. The initial model download may take additional time.

## Using the Application

### Image Analysis
1. Open the application.
2. Select the image-analysis workflow.
3. Upload a road-scene image.
4. Run detection.
5. Inspect the annotated image, detected classes, and confidence scores.

### Video Analysis
1. Open the video-analysis workflow.
2. Upload a road-scene video.
3. Run the analysis.
4. Review the annotated video and tracking results.
5. Inspect line-crossing counts, road-user class counts, traffic-level estimates, optical-flow visualization, and preliminary alerts.
6. Review or download generated results when the interface makes them available.

Processing time depends on video length, resolution, model execution speed, and available computing resources.

## Screenshots

Screenshots are included in the repository to demonstrate the application's workflows and results.

### Image Detection

| Screenshot | Description |
|---|---|
| `screenshots/image_detection_screenshots/01_image_upload.png` | Image upload interface |
| `screenshots/image_detection_screenshots/02_image_detection_results.png` | Detection results |
| `screenshots/image_detection_screenshots/03_annotated_image.png` | Annotated image output |

### Video Analytics

| Screenshot | Description |
|---|---|
| `screenshots/video_analytics_screenshots/01_video_upload.png` | Video upload interface |
| `screenshots/video_analytics_screenshots/02_video_detection.png` | Video detection output |
| `screenshots/video_analytics_screenshots/03_vehicle_tracking.png` | Vehicle tracking visualization |
| `screenshots/video_analytics_screenshots/04_line_crossing.png` | Virtual line-crossing analysis |
| `screenshots/video_analytics_screenshots/05_road_class_counts.png` | Road-user class counts |
| `screenshots/video_analytics_screenshots/06_traffic_analytics.png` | Traffic analytics and summary |

## Testing

The project includes automated tests for selected application logic and OpenCV functionality.

Run the test suite from the project root:

```powershell
python -m pytest -v
```

The current project test suite has reported **15 passing tests** in the development environment.

Passing tests demonstrate that the tested code paths behave as expected. They do not establish real-world detection accuracy, tracking reliability, or traffic-analysis validity.

## Generated Outputs

Depending on the selected workflow, analysis may produce:
- Annotated images.
- Annotated videos.
- Detection and tracking information.
- JSON analysis summaries.
- Road-user class counts.
- Line-crossing counts.
- Preliminary review alerts.

Generated files are stored in the configured output directory, such as `outputs/`. These files are excluded from version control where configured in `.gitignore`.

## Methodology

AI RoadGuard uses a combination of computer vision and rule-based analysis:

1. **Input processing:** Load the uploaded image or video.
2. **Object detection:** Use YOLOv8 to identify supported objects.
3. **Object tracking:** Apply ByteTrack to associate detections across video frames.
4. **Line-crossing analysis:** Evaluate tracked-object movement relative to a virtual line.
5. **Optical flow:** Estimate apparent pixel motion between consecutive frames using Farnebäck optical flow.
6. **Traffic-level estimation:** Apply threshold-based rules to vehicle detection counts.
7. **Preliminary review:** Apply simple rules to identify conditions that may require human inspection.
8. **Output generation:** Present visual results and save configured analysis summaries.

## Limitations

AI RoadGuard is an experimental prototype and has several important limitations:

- **Detection performance:** Results depend on the model, lighting, camera angle, image quality, occlusion, and environmental conditions.
- **Tracking reliability:** Tracking IDs and line-crossing counts may be affected by missed detections, occlusion, and interrupted tracks.
- **Traffic estimation:** Traffic levels use simple detection-count thresholds rather than calibrated density or congestion measurements.
- **Frame-based counts:** Counts accumulated across frames should not be interpreted as unique vehicles unless a dedicated unique-object counting method is used.
- **Optical flow:** Motion vectors represent apparent pixel movement and are not calibrated real-world speeds.
- **Review alerts:** Rule-based alerts may produce false positives and false negatives and require human verification.
- **Evaluation:** Automated tests do not replace evaluation on representative, independently labeled road-scene datasets.
- **Deployment resources:** Processing large or high-resolution videos may be slow under limited cloud resources.

The system should not be used as the sole basis for traffic enforcement, emergency response, or safety-critical decisions.

## Future Improvements

Potential improvements include:
- Evaluate detection performance using labeled road-scene datasets.
- Measure precision, recall, and other appropriate evaluation metrics.
- Improve unique-vehicle counting and tracking robustness.
- Calibrate traffic-level estimates against real traffic measurements.
- Explore road-specific hazard detection models.
- Improve performance for long and high-resolution videos.
- Add structured evaluation reports and more comprehensive logging.

These are potential future improvements and should not be interpreted as completed features.

## Competition Demo

Try the deployed application:

**[Launch AI RoadGuard](https://ai-roadguard-plabonbasak.streamlit.app/)**

Explore the image and video workflows to see how computer vision can support preliminary road-scene analysis.

## Author

**Plabon Basak**

- GitHub: [Plabon-Basak](https://github.com/Plabon-Basak)
- Project Repository: [AI-RoadGuard](https://github.com/Plabon-Basak/AI-RoadGuard)
- Live Demo: [AI RoadGuard on Streamlit Community Cloud](https://ai-roadguard-plabonbasak.streamlit.app/)

## License

No license has been specified for this repository yet. Add a `LICENSE` file if you choose to distribute the project under an open-source license.
