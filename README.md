<div align="center">

# 🚦 VisionTraffic
### Intelligent Traffic Video Analytics Using Computer Vision

![Status](https://img.shields.io/badge/Status-Complete-success?style=for-the-badge)
![Domain](https://img.shields.io/badge/Domain-Computer%20Vision-blue?style=for-the-badge)
![Python](https://img.shields.io/badge/Python-3.10%2B-blue?style=for-the-badge)
![License](https://img.shields.io/badge/License-Academic-lightgrey?style=for-the-badge)

*An academic project mapping standard 2D camera perspectives to a mathematically accurate ground-plane via Direct Linear Transform (DLT) for true spatial traffic analysis.*

</div>

---

## 📖 1. Overview
**VisionTraffic** is a comprehensive, academically structured Computer Vision project designed to analyze traffic video streams. It explicitly demonstrates the powerful integration of **Classical/Geometric Computer Vision** (Hough Transforms, RANSAC, DLT Homography) with **Modern Deep Learning** (YOLOv8 Detection, BoT-SORT Multi-Object Tracking). 

By grounding AI perception inside a mathematically rigorous geometric space, this system is capable of calculating true, perspective-corrected traffic density and trajectories.

## 🎯 2. Problem Statement
Traditional traffic monitoring solutions often rely on proprietary sensors or deep learning methods that operate solely in the pixel space. Analyzing traffic purely based on 2D pixels suffers from heavy **perspective distortion** (cars far away appear smaller and closer together). VisionTraffic solves this by mathematically mapping the camera's view to a flat ground plane, allowing for highly accurate, geometrically meaningful analysis.

## ✨ 3. Features
- 🚘 **Vehicle Detection & Classification:** Uses `YOLOv8` to identify cars, trucks, buses, and motorcycles.
- 🔗 **Persistent Multi-Object Tracking:** Uses `BoT-SORT` (ByteTrack) to track vehicles across frames and assign unique IDs.
- 📏 **Virtual Line-Crossing Counting:** Uses vector intersection mathematics to count vehicles crossing a user-defined threshold.
- 🗺️ **Geometrically Accurate Density Estimation:** Calculates traffic density dynamically on the rectified ground plane (bird's-eye view).
- 🛣️ **Classical Lane Detection:** Implements image processing, edge detection, and an explicitly coded `RANSAC` algorithm to robustly fit lane lines.
- 📐 **Geometric Calibration (DLT):** Computes a `3x3 Homography matrix` from scratch using Singular Value Decomposition (SVD) to warp the perspective.

## 🛠️ 4. Technologies & Tools Used
- **Programming Language:** Python 3.10+
- **Deep Learning Framework:** Ultralytics (YOLOv8)
- **Computer Vision Libraries:** OpenCV (`cv2`) for image transformations and video I/O
- **Mathematical Computation:** NumPy (for linear algebra, SVD, and vector math)
- **Testing Framework:** Pytest (for verifying mathematical logic and robust statistics)
- **Configuration Management:** PyYAML

## 📸 5. Screenshots
> **Note:** Insert your screenshots here after running the project on your evaluation video.

| Original Perspective View | Bird's-Eye View (Geometrically Mapped) |
| :---: | :---: |
| *(Insert image here)* | *(Insert image here)* |

## 📚 6. Computer Vision Syllabus Traceability
This project was strictly designed to represent major topics taught in a standard Computer Vision curriculum.

| CV Course Topic | Where it appears in this project |
|:---|:---|
| **Image Enhancement / Filtering** | Gaussian blur preprocessing applied in `src/lane_detector.py` to remove high-frequency noise. |
| **Edge Detection** | Canny edge detector applied in `src/lane_detector.py` for structural extraction. |
| **Hough Transform** | Candidate line structure extraction logic for lane boundaries. |
| **Robust Statistics (RANSAC)** | From-scratch implementation of RANSAC in `src/lane_detector.py` to perfectly fit lines while ignoring outlier noise. |
| **Projective Geometry (Homography)** | Ground-plane perspective calibration using the Direct Linear Transform (DLT) matrix solved via SVD in `src/calibration.py`. |
| **Camera Calibration Concepts** | ROI-to-ground-plane physical point correspondence setup via `config.yaml`. |
| **Deep-Learning Perception** | YOLOv8 object detection (`src/detector.py`) and temporal tracking (`src/tracker.py`). |

## 🏗️ 7. System Architecture & Workflow
The system processes data linearly through separated, modular components:

1. **Video Processor:** Reads frames, standardizes resolution.
2. **Perception Layer:** YOLO detects bounding boxes; BoT-SORT assigns trajectory IDs.
3. **Classical Layer:** Edge detection and RANSAC calculate lane boundaries.
4. **Geometric Layer:** DLT Homography maps all coordinates to a flat ground plane.
5. **Analytics Engine:** Vector math evaluates line crossings and calculates spatial density.
6. **Visualization Engine:** Overlays bounding boxes on the perspective view and generates a separate Bird's-Eye View video.

## 🚀 8. Steps to Install & Run the Project

### Installation Steps
```bash
# 1. Clone the repository
git clone https://github.com/YOUR-USERNAME/VisionTraffic.git
cd VisionTraffic

# 2. Create a virtual environment
python -m venv venv

# 3. Activate the environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

# 4. Install dependencies
pip install -r requirements.txt
```

### Running the Project
Before running, place your traffic video inside `data/sample/traffic.mp4` and update `config/config.yaml` with the 4 point corners for the DLT calibration.
```bash
# Run the main pipeline (No GUI required)
python main.py --input data/sample/traffic.mp4 --mode full

# Generate the academic traceability report
python generate_report.py
```

## 🧪 9. Instructions for Testing
To prove that the classical math algorithms (like DLT Homography and RANSAC) are working accurately, you can run the automated test suite.
```bash
# Run the mathematical unit tests
pytest
```
> If the tests pass, it means the from-scratch SVD matrix math is successfully calculating the same homographies as standard industry libraries.

## 📁 10. Output Description
After execution, check the `outputs/` folder:
- 🎬 **`result.mp4`**: The raw perspective video with all CV annotations overlaid.
- 🚁 **`result_birdseye.mp4`**: The perspective-warped top-down view proving the DLT homography implementation.
- 📊 **`vehicle_statistics.json`**: Machine-readable metrics containing accurate counts and spatial densities.
- 📄 **`traffic_report.txt`**: A human-readable academic report summarizing the outcome.

## ⚠️ 11. Limitations & Future Enhancements
- **Monocular Limitation (Epipolar Geometry):** True epipolar geometry and absolute depth estimation require a stereo camera setup. Because this system processes standard monocular video, epipolar-constrained matching is out of scope. The system instead utilizes planar homography, assuming the road is a flat surface.

## 🎓 12. Academic Integrity
All classic CV mathematical functions (DLT formulation, SVD extraction, RANSAC sampling/consensus loops) are written **explicitly from scratch** in the `src/` modules to demonstrate core understanding, specifically avoiding the use of black-box library calls for the primary academic logic.
