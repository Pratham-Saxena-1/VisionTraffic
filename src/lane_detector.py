import cv2
import numpy as np

class LaneDetector:
    """
    Implements classical Computer Vision pipeline for lane detection:
    Grayscale -> Blur -> Canny -> ROI Mask -> Hough -> Custom RANSAC.
    """
    def __init__(self, config: dict):
        self.roi_points = np.array(config['regions']['lane_roi'], np.int32)
        
        # RANSAC Params
        self.ransac_inlier_thresh = config['ransac']['inlier_threshold']
        self.ransac_max_iters = config['ransac']['max_iterations']
        
        # Hough Params
        self.h_rho = config['hough']['rho']
        self.h_theta = config['hough']['theta']
        self.h_thresh = config['hough']['threshold']
        self.h_min_len = config['hough']['min_line_length']
        self.h_max_gap = config['hough']['max_line_gap']

    def _preprocess(self, frame):
        """Grayscale, Blur, Canny, and ROI masking."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blur, 50, 150)
        
        # Apply ROI mask
        mask = np.zeros_like(edges)
        cv2.fillPoly(mask, [self.roi_points], 255)
        masked_edges = cv2.bitwise_and(edges, mask)
        
        return masked_edges

    def fit_line_ransac(self, points):
        """
        Custom explicitly implemented RANSAC for line fitting.
        Fits a line equation (ax + by + c = 0) to a set of 2D points.
        
        Args:
            points: (N, 2) numpy array of edge point coordinates [x, y]
        Returns:
            best_line: (a, b, c) line parameters
            best_inliers: boolean array of size N
        """
        N = len(points)
        if N < 2:
            return None, None
            
        best_inliers = np.zeros(N, dtype=bool)
        best_inlier_count = 0
        best_line = None
        
        # Convert points to homogeneous for easy dot product
        pts_h = np.hstack((points, np.ones((N, 1))))
        
        for _ in range(self.ransac_max_iters):
            # 1. Randomly sample 2 points
            idx = np.random.choice(N, 2, replace=False)
            p1, p2 = points[idx[0]], points[idx[1]]
            
            # Avoid degenerate cases (same point sampled twice)
            if np.all(p1 == p2):
                continue
                
            # 2. Fit model to the sample (Cross product of homogeneous points gives line)
            # l = p1 x p2
            p1_h = np.array([p1[0], p1[1], 1.0])
            p2_h = np.array([p2[0], p2[1], 1.0])
            line = np.cross(p1_h, p2_h)
            
            # Normalize line parameters (a, b, c) such that a^2 + b^2 = 1
            # so the dot product gives point-to-line distance directly.
            norm = np.linalg.norm(line[:2])
            if norm == 0:
                continue
            line = line / norm
            
            # 3. Calculate distance from all points to this line
            # Distance = |a*x + b*y + c|
            distances = np.abs(np.dot(pts_h, line))
            
            # 4. Count inliers
            inliers = distances < self.ransac_inlier_thresh
            inlier_count = np.sum(inliers)
            
            # 5. Keep the best model
            if inlier_count > best_inlier_count:
                best_inlier_count = inlier_count
                best_inliers = inliers
                best_line = line
                
        return best_line, best_inliers

    def detect(self, frame):
        """Runs the lane detection pipeline."""
        edges = self._preprocess(frame)
        
        # Optionally, we can run Hough Transform just to demonstrate it
        # and then extract endpoints, OR directly use Canny edge points for RANSAC.
        # The prompt says: "RANSAC-based robust lane fitting on edge point sets 
        # (implemented explicitly...)"
        # So we will extract coordinates of all white pixels in the Canny image
        y_coords, x_coords = np.where(edges == 255)
        edge_points = np.column_stack((x_coords, y_coords))
        
        detected_lanes = []
        if len(edge_points) > 100:
            # 1. Find the first dominant line (e.g. Right Lane)
            line1, inliers1 = self.fit_line_ransac(edge_points)
            if line1 is not None and np.sum(inliers1) > 50:
                detected_lanes.append(line1)
                
                # Remove inliers to find the second line
                remaining_points = edge_points[~inliers1]
                
                # 2. Find the second dominant line (e.g. Left Lane)
                if len(remaining_points) > 100:
                    line2, inliers2 = self.fit_line_ransac(remaining_points)
                    if line2 is not None and np.sum(inliers2) > 50:
                        detected_lanes.append(line2)
                        
        return detected_lanes, edges
        
    def get_line_endpoints(self, line, y_min, y_max):
        """Converts (a,b,c) line eq into [x1, y1, x2, y2] within y bounds."""
        a, b, c = line
        # ax + by + c = 0 -> x = -(by + c)/a
        if abs(a) < 1e-6:
            # Horizontal line, should not happen for lanes but handle it
            return None
            
        x1 = int(-(b * y_max + c) / a)
        x2 = int(-(b * y_min + c) / a)
        return (x1, int(y_max)), (x2, int(y_min))

    def draw_lanes(self, frame, lanes):
        """Draws detected lane lines on the frame."""
        vis_frame = frame.copy()
        y_max = frame.shape[0]
        # Usually lane is visible up to mid screen roughly
        y_min = int(frame.shape[0] * 0.5) 
        
        for line in lanes:
            pts = self.get_line_endpoints(line, y_min, y_max)
            if pts:
                cv2.line(vis_frame, pts[0], pts[1], (0, 255, 0), 4)
                
        return vis_frame
