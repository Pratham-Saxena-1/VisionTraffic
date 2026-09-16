<div align="center">

# 📄 VisionTraffic: Problem Statement & Scope

*A formal academic definition of the VisionTraffic Computer Vision project.*

</div>

---

## 🛑 1. Problem Statement
Modern traffic management and urban planning require automated, accurate, and scalable video analytics to monitor traffic flow, count vehicles, and detect anomalous behavior. Traditional solutions often rely either on expensive proprietary physical sensors (like induction loops) or purely deep-learning-based perception methods that lack grounding in geometric reality. 

Without geometric calibration, extracting meaningful real-world metrics (such as true spatial density or physical trajectories) from a 2D camera perspective remains highly inaccurate due to **perspective distortion**. There is a critical need for a hybrid system that marries the robust perception of modern AI with the mathematical rigor of classical projective geometry.

## 🔭 2. Scope
**VisionTraffic** is a comprehensive software application designed to process monocular traffic video feeds. The scope of this project is explicitly academic and demonstrative, aiming to implement and showcase a full spectrum of Computer Vision (CV) techniques. 

It applies both **classical geometric computer vision algorithms** (such as the Direct Linear Transform for homography, and RANSAC for robust line fitting) and **modern deep learning** (YOLOv8 and BoT-SORT) to extract mathematically grounded analytics and visual overlays.

## 👥 3. Target Users
- 🎓 **Computer Vision Evaluators and Researchers:** To assess the correct implementation of fundamental CV mathematical concepts alongside modern deep learning frameworks.
- 🚦 **Traffic Management Authorities:** To derive actionable, geometrically corrected insights regarding traffic density and lane discipline.
- 🏙️ **Urban Planners:** To gather historical traffic flow data for infrastructure planning.

## 🎯 4. Objectives
1. **Accurate Perception:** Detect and persistently track vehicles in a continuous video stream under varying conditions.
2. **Mathematical Rigor:** Provide a robust, from-scratch implementation of classical CV geometry. This includes computing Homography matrices via the Direct Linear Transform (DLT) using Singular Value Decomposition (SVD), and performing robust lane line fitting via a custom RANSAC implementation.
3. **Geometric Analytics:** Compute traffic density and count vehicles in a geometrically accurate, perspective-corrected ground plane (bird's-eye view) rather than in raw, distorted pixel space.
4. **Academic Traceability:** Provide clear documentation and automated reporting that maps every generated metric and visual output directly to the foundational Computer Vision technique that produced it.

## ✨ 5. High-Level Features
- 🤖 **Deep-Learning Perception:** Real-time vehicle detection and multi-object tracking.
- 🛣️ **Classical Lane Detection:** Image enhancement, edge detection, and robust mathematical lane fitting (RANSAC).
- 📐 **Geometric Calibration:** DLT-based Homography computation to map the camera's perspective to a 2D top-down ground plane.
- 📏 **Virtual Line-Crossing:** Vector-math-based detection of vehicles crossing a designated threshold for accurate counting.
- 🚨 **Spatial Anomaly Detection:** Automated detection of wrong-direction driving or illegal lane crossings based on ground-plane trajectories.
- 📊 **Automated Academic Reporting:** Generation of structured markdown and JSON reports detailing the CV methodologies used and the resulting metrics.

## 📥 6. Expected Inputs
- A standard monocular traffic video file (`.mp4`, `.avi`, etc.) captured from a stationary, elevated viewpoint.
- A YAML configuration file (`config.yaml`) defining the environmental parameters, including Regions of Interest (ROI) and, most importantly, four point correspondences mapping the image plane to the ground plane.

## 📤 7. Expected Outputs
- 🎬 **Raw Perspective Annotated Video:** The original video feed overlaid with bounding boxes, lane lines, counting tripwires, and live statistics.
- 🚁 **Bird's-Eye Perspective Annotated Video:** A geometrically transformed video feed showing the traffic from a top-down, perspective-corrected map view.
- 📈 **Metrics Data:** Machine-readable JSON and CSV files containing vehicle counts and traffic density metrics.
- 📑 **Academic Summary Report:** A generated text document tracing the outputs to the CV syllabus.

## ⚠️ 8. Limitations & Constraints
> **Epipolar Geometry Constraint:** True epipolar geometry and absolute depth estimation require two or more camera views of the exact same scene simultaneously (stereo vision). 

Because this system is designed to operate on widely available *monocular* (single camera) video feeds, stereo epipolar geometry is out of scope. The system relies on the assumption that the observed road surface is approximately planar (flat). This allows the use of a 2D-to-2D planar homography to rectify the perspective, rather than relying on essential or fundamental matrices for 3D triangulation.
