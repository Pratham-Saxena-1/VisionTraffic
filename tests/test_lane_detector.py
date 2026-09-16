import pytest
import numpy as np

# Mock config
config = {
    'regions': {'lane_roi': [[0,0], [1,0], [1,1], [0,1]]},
    'ransac': {
        'inlier_threshold': 2.0,
        'max_iterations': 100,
        'min_points': 2
    },
    'hough': {
        'rho': 1, 'theta': 0.017, 'threshold': 50, 'min_line_length': 50, 'max_line_gap': 20
    }
}

from src.lane_detector import LaneDetector

def test_ransac_fitting():
    """Tests if our explicitly implemented RANSAC correctly ignores outliers."""
    detector = LaneDetector(config)
    
    # Create a synthetic dataset: a perfect line (y = 2x) + some random outliers
    x_inliers = np.linspace(0, 100, 50)
    y_inliers = 2 * x_inliers
    inliers = np.column_stack((x_inliers, y_inliers))
    
    outliers = np.random.uniform(0, 200, (20, 2))
    
    points = np.vstack((inliers, outliers))
    
    line, inlier_mask = detector.fit_line_ransac(points)
    
    assert line is not None
    # Ensure most of our known inliers were caught
    assert np.sum(inlier_mask) >= 45 
