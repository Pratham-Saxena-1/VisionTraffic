# Problem Statement
Modern traffic management requires automated, accurate, and scalable video analytics to monitor traffic flow, count vehicles, and detect anomalies. Traditional solutions often rely on proprietary sensors or purely deep learning methods without grounding in geometric reality. 

# Scope
VisionTraffic is a software application designed to process monocular traffic video feeds. It applies both classical geometric computer vision algorithms (DLT, RANSAC, Hough) and modern deep learning (YOLO, BoT-SORT) to extract meaningful analytics and visual overlays.

# Target Users
- Traffic Management Authorities
- Computer Vision Students and Evaluators
- Urban Planners

# Objectives
1. Accurately detect and track vehicles in a video stream.
2. Provide a robust, from-scratch implementation of classical CV geometry (Homography via DLT, robust line fitting via RANSAC).
3. Compute geometrically accurate traffic density and count vehicles.
4. Provide clear, academic traceability of all implemented CV techniques.

# High-Level Features
- Deep-learning based vehicle detection and tracking.
- Classical lane detection using explicit RANSAC.
- DLT-based Homography computation for bird's-eye perspective correction.
- Virtual line-crossing vehicle counting.
- Automated anomaly detection based on ground-plane trajectories.
- Comprehensive report generation.

# Expected Inputs
- A monocular traffic video file (`.mp4`, `.avi`, etc.)
- A YAML configuration file defining regions of interest and point correspondences.

# Expected Outputs
- Raw perspective annotated video (`result.mp4`).
- Bird's-eye perspective annotated video (`result_birdseye.mp4`).
- JSON and CSV metrics data.
- Academic summary report (`traffic_report.txt`).

# Limitations
Because the system processes monocular video, true epipolar geometry and depth estimation are out of scope. The perspective rectification relies entirely on the assumption that the observed road is a planar surface, allowing the use of a 2D-to-2D homography rather than a fundamental matrix and stereo triangulation.
