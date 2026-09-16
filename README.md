# VisionTraffic: Intelligent Traffic Video Analytics Using Computer Vision

## Objective
VisionTraffic is a comprehensive, academically structured Computer Vision project that analyzes traffic video streams. It explicitly demonstrates the integration of classical CV geometry (Hough Transform, RANSAC, DLT Homography) with modern deep-learning-based perception (YOLO detection, Multi-Object Tracking).

## Features
- Vehicle detection and classification (YOLOv8)
- Persistent Multi-Object Tracking (BoT-SORT / ByteTrack)
- Virtual line-crossing vehicle counting
- Geometrically accurate traffic density estimation via Homography (Bird's-Eye View)
- Classical Lane Detection (Canny, Hough, explicitly implemented RANSAC)
- Geometric anomaly detection (wrong-direction, lane-crossing)
- Detailed CLI and automated evaluation reporting

## Computer Vision Syllabus Traceability

| CV Course Topic | Where it appears in this project |
|---|---|
| Filtering / convolution | Gaussian blur preprocessing (`src/lane_detector.py`) |
| Edge detection | Canny edge detector (`src/lane_detector.py`) |
| Hough Transform | Candidate lane line extraction logic (`src/lane_detector.py`) |
| RANSAC | Robust lane line fitting from scratch (`src/lane_detector.py`) |
| Homography / DLT | Ground-plane calibration via SVD from scratch (`src/calibration.py`) |
| Camera calibration concepts | ROI-to-ground-plane point correspondence setup |
| Deep-learning CV | YOLO vehicle detector (`src/detector.py`, `src/tracker.py`) |

## Project Architecture
The system follows a modular architecture separating Video I/O, Deep Learning Perception (YOLO), Classical/Geometric CV (DLT/RANSAC), Analytics, and Visualization. 

## Installation
```bash
git clone <repository>
cd VisionTraffic
python -m venv venv
# Windows
venv\Scripts\activate
# Linux/macOS
# source venv/bin/activate
pip install -r requirements.txt
```

## Usage
Run the main pipeline:
```bash
python main.py --input data/sample/traffic.mp4 --mode full
```

Run tests:
```bash
pytest
```

Generate the academic report:
```bash
python generate_report.py
```

## Limitations
**Epipolar Geometry:** True epipolar geometry and depth estimation require two or more camera views of the same scene (stereo vision). Because the input here is monocular video (a single camera view), epipolar geometry is out of scope. We rely on planar homography mapping assuming a flat road surface. 

## Future Enhancements
A future enhancement could involve adding a secondary camera view to enable stereo depth estimation and epipolar-constrained matching, allowing for true 3D spatial analysis rather than 2D planar assumption.
