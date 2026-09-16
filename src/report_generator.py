import yaml
import json
import os
from datetime import datetime

class ReportGenerator:
    """
    Generates the final academic report, explicitly linking 
    output metrics to the Computer Vision theory topics.
    """
    def __init__(self, config_path, metrics_path):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
            
        with open(metrics_path, 'r') as f:
            self.metrics = json.load(f)

    def generate(self):
        report_path = self.config['paths']['report_txt']
        os.makedirs(os.path.dirname(report_path), exist_ok=True)
        
        lines = []
        lines.append("==================================================")
        lines.append(" VisionTraffic: CV Academic Evaluation Report ")
        lines.append("==================================================")
        lines.append(f"Generated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append("\n1. SYSTEM CONFIGURATION")
        lines.append(f"- Input Resolution: {self.config['model']['input_resolution']}")
        lines.append(f"- Bird's-Eye Resolution: {self.config['calibration']['birdseye_resolution']}")
        
        lines.append("\n2. CV SYLLABUS TRACEABILITY")
        lines.append("| CV Course Topic            | Applied Module / Result                              |")
        lines.append("|----------------------------|------------------------------------------------------|")
        lines.append("| Image Enhancement          | Gaussian blur in src/lane_detector.py                |")
        lines.append("| Edge Detection             | Canny edge detector (lane extraction)                |")
        lines.append("| Hough Transform            | Candidate line structure extraction                  |")
        lines.append("| RANSAC Robust Estimation   | Robust lane line fitting amidst outlier noise        |")
        lines.append("| Homography & DLT           | Ground-plane perspective calibration (DLT SVD logic) |")
        lines.append("| Deep-Learning CV           | YOLOv8 vehicle detection & BoT-SORT tracking         |")
        
        lines.append("\n3. TRAFFIC ANALYTICS (from pixel geometry)")
        lines.append(f"- Total Vehicles Counted: {self.metrics['counts']['total']}")
        lines.append("- Class Breakdown:")
        for cls, count in self.metrics['counts']['classes'].items():
            lines.append(f"  - Class ID {cls}: {count}")
            
        lines.append("\n4. GEOMETRIC ANALYTICS (from ground-plane/rectified geometry)")
        lines.append(f"- Peak Density (vehicles per area): {self.metrics['density']['peak_density']:.2f}")
        lines.append(f"- Average Density: {self.metrics['density']['average_density']:.2f}")
        
        lines.append("\n5. DETECTED GEOMETRIC ANOMALIES")
        if not self.metrics['violations']:
            lines.append("- None detected.")
        else:
            for v in self.metrics['violations']:
                lines.append(f"- {v}")
                
        lines.append("\n6. LIMITATIONS")
        lines.append("- Epipolar Geometry: True epipolar geometry and depth estimation require a stereo")
        lines.append("  camera pair. This system operates on monocular video and utilizes planar ")
        lines.append("  homography (assuming a flat road) rather than essential/fundamental matrices.")
        
        report_content = "\n".join(lines)
        
        with open(report_path, 'w') as f:
            f.write(report_content)
            
        print(f"[*] Academic report generated successfully at {report_path}")
