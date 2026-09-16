import cv2
import numpy as np

class TrafficAnalyzer:
    """
    Computes geometrically meaningful traffic density in the rectified
    ground-plane view. Demonstrates applied geometric CV.
    """
    def __init__(self, config: dict, calibration):
        self.roi_points = np.array(config['regions']['density_roi'], np.int32)
        self.calibration = calibration
        self.density_history = []
        
        # Calculate area in the ground plane
        # 1. Project ROI points to ground plane
        self.roi_ground = []
        for pt in self.roi_points:
            g_pt = self.calibration.image_point_to_ground(pt)
            self.roi_ground.append(g_pt)
            
        self.roi_ground = np.array(self.roi_ground, dtype=np.int32)
        
        # 2. Calculate area of the ground polygon (in square pixels of the birdseye view)
        # We can use cv2.contourArea
        self.ground_area = cv2.contourArea(self.roi_ground)
        if self.ground_area == 0:
            self.ground_area = 1.0 # Prevent division by zero

    def update(self, tracked_objects):
        """
        Calculates density for the current frame.
        Density = Count of vehicles inside ROI / Area of ROI in ground plane.
        """
        vehicles_in_roi = 0
        
        for obj in tracked_objects:
            center = tuple(obj['center'])
            # Check if center is in the original image ROI
            if cv2.pointPolygonTest(self.roi_points, center, False) >= 0:
                vehicles_in_roi += 1
                
        # Geometrically meaningful density
        density = vehicles_in_roi / self.ground_area
        # Scale up by 1000 for readability (e.g. vehicles per 1000 sq birdseye pixels)
        density = density * 1000 
        
        self.density_history.append(density)
        return density, vehicles_in_roi

    def get_statistics(self):
        return {
            'peak_density': max(self.density_history) if self.density_history else 0,
            'average_density': sum(self.density_history) / len(self.density_history) if self.density_history else 0,
            'current_density': self.density_history[-1] if self.density_history else 0
        }
