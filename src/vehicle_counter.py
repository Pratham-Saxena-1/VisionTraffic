import numpy as np

class VehicleCounter:
    """
    Counts vehicles as they cross a configured virtual line.
    Uses vector math (cross product) to detect line crossing.
    """
    def __init__(self, config: dict):
        # Line represented as [x1, y1, x2, y2]
        self.line = config['analytics']['counting_line']
        self.counted_ids = set()
        self.total_count = 0
        self.class_counts = {}
        
        # We need to track the previous position of each ID to detect a crossing
        self.previous_positions = {}
        
    def _ccw(self, A, B, C):
        """Returns True if points A, B, C are in counter-clockwise order."""
        return (C[1] - A[1]) * (B[0] - A[0]) > (B[1] - A[1]) * (C[0] - A[0])

    def _intersect(self, A, B, C, D):
        """Returns True if line segment AB intersects CD."""
        return self._ccw(A, C, D) != self._ccw(B, C, D) and self._ccw(A, B, C) != self._ccw(A, B, D)

    def update(self, tracked_objects):
        """
        Updates counts based on tracked objects' current positions.
        """
        A = (self.line[0], self.line[1])
        B = (self.line[2], self.line[3])
        
        for obj in tracked_objects:
            track_id = obj['track_id']
            center = obj['center']
            cls_id = obj['class']
            
            if track_id in self.previous_positions:
                prev_center = self.previous_positions[track_id]
                
                # Check if the segment from prev_center to current center intersects the counting line
                if track_id not in self.counted_ids:
                    if self._intersect(A, B, prev_center, center):
                        self.counted_ids.add(track_id)
                        self.total_count += 1
                        self.class_counts[cls_id] = self.class_counts.get(cls_id, 0) + 1
                        
            self.previous_positions[track_id] = center
            
    def get_counts(self):
        return {
            'total': self.total_count,
            'classes': self.class_counts
        }
