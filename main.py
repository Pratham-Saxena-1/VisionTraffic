import argparse
import yaml
import time
import json
import cv2
import os

from src.video_processor import VideoProcessor
from src.tracker import MultiObjectTracker
from src.calibration import GeometricCalibration
from src.lane_detector import LaneDetector
from src.vehicle_counter import VehicleCounter
from src.traffic_analyzer import TrafficAnalyzer
from src.violation_detector import ViolationDetector
from src.visualization import VisualizationEngine

def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def main():
    parser = argparse.ArgumentParser(description="VisionTraffic: Intelligent Traffic Video Analytics")
    parser.add_argument('--input', type=str, required=True, help='Path to input video')
    parser.add_argument('--config', type=str, default='config/config.yaml', help='Path to config file')
    parser.add_argument('--mode', type=str, default='full', choices=['detection', 'tracking', 'calibration', 'analytics', 'full'], help='Operation mode')
    args = parser.parse_args()

    config = load_config(args.config)
    
    # Initialize modules
    print("[*] Initializing modules...")
    video_proc = VideoProcessor(args.input, config['paths']['video_output'], config['paths']['birdseye_video_output'], config)
    calibration = GeometricCalibration(config)
    tracker = MultiObjectTracker(config)
    lane_detector = LaneDetector(config)
    counter = VehicleCounter(config)
    analyzer = TrafficAnalyzer(config, calibration)
    violation_det = ViolationDetector(calibration)
    visualizer = VisualizationEngine(config, calibration, lane_detector)
    
    print(f"[*] Starting processing in mode: {args.mode}")
    frame_idx = 0
    start_time = time.time()
    
    while True:
        ret, frame = video_proc.read_frame()
        if not ret:
            break
            
        frame_idx += 1
        t0 = time.time()
        
        # 1. Detection and Tracking (Deep Learning CV)
        tracked_objects = []
        if args.mode in ['tracking', 'analytics', 'full', 'detection']:
            # In 'detection' mode, we could just run detection without tracking, 
            # but our tracker module does both. We'll use it for all object-based modes.
            tracked_objects = tracker.update(frame)
            
        # 2. Lane Detection (Classical CV: Canny, Hough, RANSAC)
        lanes = []
        if args.mode in ['analytics', 'full']:
            lanes, _ = lane_detector.detect(frame)
            
        # 3. Analytics (Counting, Density, Violations)
        current_density = 0
        if args.mode in ['analytics', 'full']:
            counter.update(tracked_objects)
            current_density, _ = analyzer.update(tracked_objects)
            violation_det.update(tracked_objects, frame_idx)
            
        # 4. Visualization (Geometric Calibration / Birdseye)
        fps_val = 1.0 / (time.time() - t0)
        
        raw_annotated = frame
        birdseye_annotated = None
        
        if args.mode in ['calibration', 'analytics', 'full']:
            # Draw on perspective image
            raw_annotated = visualizer.draw_raw_frame(
                frame, tracked_objects, lanes, counter.get_counts(), current_density, fps_val)
            
            # Generate bird's-eye view using DLT homography
            birdseye_annotated = visualizer.draw_birdseye_frame(frame, tracked_objects, lanes)
            
        video_proc.write_frame(raw_annotated, birdseye_annotated)
        
        if frame_idx % 30 == 0:
            print(f"Processed {frame_idx}/{video_proc.total_frames} frames... FPS: {fps_val:.2f}")

    total_time = time.time() - start_time
    print(f"[*] Processing complete. Total time: {total_time:.2f}s. Avg FPS: {frame_idx/total_time:.2f}")
    
    # Save Metrics
    if args.mode in ['analytics', 'full']:
        os.makedirs(os.path.dirname(config['paths']['metrics_json']), exist_ok=True)
        stats = {
            'counts': counter.get_counts(),
            'density': analyzer.get_statistics(),
            'violations': violation_det.get_violations()
        }
        with open(config['paths']['metrics_json'], 'w') as f:
            json.dump(stats, f, indent=4)
        print(f"[*] Saved metrics to {config['paths']['metrics_json']}")
        
    video_proc.release()
    print("[*] Released resources.")

if __name__ == '__main__':
    main()
