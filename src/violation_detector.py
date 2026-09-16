class ViolationDetector:
    """
    Detects potential traffic events/violations using tracked trajectories 
    mapped to the geometric ground plane.
    """
    def __init__(self, calibration):
        self.calibration = calibration
        self.history = {}
        self.violations = []
        
    def update(self, tracked_objects, frame_idx):
        """
        Updates the state and checks for anomalous movement.
        """
        for obj in tracked_objects:
            track_id = obj['track_id']
            center = obj['center']
            
            # Map center to ground plane for accurate geometric analysis
            g_pt = self.calibration.image_point_to_ground(center)
            
            if track_id not in self.history:
                self.history[track_id] = []
                
            self.history[track_id].append((frame_idx, g_pt))
            
            # Check for wrong direction (e.g. moving 'up' the birdseye image instead of 'down')
            # Assuming normal traffic flows from top to bottom (y increases)
            # This is highly dependent on scene configuration, but serves as a demonstration.
            if len(self.history[track_id]) > 10:
                past_y = self.history[track_id][-10][1][1]
                current_y = g_pt[1]
                
                # If they moved significantly in the wrong direction
                if past_y - current_y > 20: 
                    # They are moving UP the screen (decreasing Y)
                    event = f"Frame {frame_idx}: Vehicle {track_id} wrong direction (delta Y: {current_y - past_y})"
                    if event not in self.violations: # basic deduplication
                        self.violations.append(event)
                        
    def get_violations(self):
        return self.violations
