import cv2
import numpy as np

class VisualizationEngine:
    """
    Handles drawing overlays, bounding boxes, trajectories, and the 
    perspective warp for the bird's eye view.
    """
    def __init__(self, config: dict, calibration, lane_detector):
        self.config = config
        self.calibration = calibration
        self.lane_detector = lane_detector
        self.counting_line = config['analytics']['counting_line']
        self.density_roi = np.array(config['regions']['density_roi'], np.int32)
        self.birdseye_res = tuple(config['calibration']['birdseye_resolution'])
        
        # Color mapping for classes (e.g. 2: car, 3: motorcycle, 5: bus, 7: truck)
        self.colors = {
            2: (0, 255, 0),
            3: (0, 0, 255),
            5: (255, 0, 0),
            7: (0, 255, 255)
        }

    def draw_raw_frame(self, frame, tracked_objects, lanes, counts, current_density, fps_val):
        """Draws annotations on the original perspective frame."""
        vis_frame = frame.copy()
        
        # 1. Draw ROI for density
        cv2.polylines(vis_frame, [self.density_roi], isClosed=True, color=(255, 255, 0), thickness=2)
        
        # 2. Draw Counting Line
        x1, y1, x2, y2 = self.counting_line
        cv2.line(vis_frame, (x1, y1), (x2, y2), (0, 0, 255), 3)
        
        # 3. Draw Lanes
        if lanes is not None:
            vis_frame = self.lane_detector.draw_lanes(vis_frame, lanes)
            
        # 4. Draw Tracked Objects
        for obj in tracked_objects:
            x1, y1, x2, y2 = obj['bbox']
            track_id = obj['track_id']
            cls_id = obj['class']
            color = self.colors.get(cls_id, (255, 255, 255))
            
            # Bounding box
            cv2.rectangle(vis_frame, (x1, y1), (x2, y2), color, 2)
            
            # Label
            label = f"ID:{track_id} C:{cls_id}"
            cv2.putText(vis_frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
            # Center point
            cx, cy = obj['center']
            cv2.circle(vis_frame, (cx, cy), 4, color, -1)
            
        # 5. Draw Stats HUD
        cv2.putText(vis_frame, f"Total Count: {counts['total']}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
        cv2.putText(vis_frame, f"Density: {current_density:.2f}", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
        cv2.putText(vis_frame, f"FPS: {fps_val:.1f}", (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
        
        return vis_frame

    def draw_birdseye_frame(self, frame, tracked_objects, lanes):
        """
        Creates the bird's eye view and draws ground-plane mapped entities.
        """
        # Warp the original frame to birdseye view using DLT homography
        birdseye = self.calibration.warp_to_birdseye(frame, self.birdseye_res)
        
        # Draw tracked objects in bird's eye
        for obj in tracked_objects:
            cx, cy = obj['center']
            cls_id = obj['class']
            color = self.colors.get(cls_id, (255, 255, 255))
            
            # Transform point
            gx, gy = self.calibration.image_point_to_ground((cx, cy))
            
            # Draw circle in birdseye view
            cv2.circle(birdseye, (gx, gy), 6, color, -1)
            cv2.putText(birdseye, str(obj['track_id']), (gx+10, gy), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            
        # Draw lanes mapped to birdseye (approximate by mapping endpoints)
        y_max = frame.shape[0]
        y_min = int(frame.shape[0] * 0.5)
        for line in (lanes or []):
            pts = self.lane_detector.get_line_endpoints(line, y_min, y_max)
            if pts:
                g_pt1 = self.calibration.image_point_to_ground(pts[0])
                g_pt2 = self.calibration.image_point_to_ground(pts[1])
                cv2.line(birdseye, g_pt1, g_pt2, (0, 255, 0), 3)
                
        return birdseye
