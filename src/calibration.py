import numpy as np
import cv2

class GeometricCalibration:
    """
    Implements Geometric Computer Vision techniques, primarily the 
    Direct Linear Transform (DLT) for computing planar homographies.
    """
    def __init__(self, config: dict):
        self.image_points = np.array(config['calibration']['image_points'], dtype=np.float32)
        self.ground_points = np.array(config['calibration']['ground_points'], dtype=np.float32)
        
        # Scale and offset ground points to birdseye pixel coordinates
        scale = config['calibration']['birdseye_scale']
        offset_x = config['calibration']['birdseye_offset_x']
        offset_y = config['calibration']['birdseye_offset_y']
        
        # Our target for the homography is not the physical ground points (in meters)
        # but the pixel coordinates in the birdseye view image.
        self.birdseye_points = np.zeros_like(self.ground_points)
        self.birdseye_points[:, 0] = self.ground_points[:, 0] * scale + offset_x
        # Invert Y so that lower Y in ground (closer) corresponds to higher Y in image (bottom)
        # or just map it directly. Let's map directly and handle orientation later.
        self.birdseye_points[:, 1] = config['calibration']['birdseye_resolution'][1] - (self.ground_points[:, 1] * scale + offset_y)
        
        self.H_dlt = None
        self.H_cv2 = None
        self._calibrate()

    def compute_homography_dlt(self, pts_src, pts_dst):
        """
        Computes the Homography matrix using the Direct Linear Transform (DLT).
        This explicitly demonstrates classical projective geometry.
        
        Args:
            pts_src: (N, 2) array of source points
            pts_dst: (N, 2) array of destination points
        Returns:
            H: (3, 3) Homography matrix
        """
        num_points = pts_src.shape[0]
        if num_points < 4:
            raise ValueError("DLT requires at least 4 point correspondences.")
            
        A = []
        for i in range(num_points):
            x, y = pts_src[i, 0], pts_src[i, 1]
            u, v = pts_dst[i, 0], pts_dst[i, 1]
            # Form the 2x9 design matrix block for this correspondence
            A.append([-x, -y, -1, 0, 0, 0, x*u, y*u, u])
            A.append([0, 0, 0, -x, -y, -1, x*v, y*v, v])
            
        A = np.array(A, dtype=np.float64)
        
        # Solve Ah = 0 using SVD
        # A = U * S * V^T
        U, S, Vt = np.linalg.svd(A)
        
        # The solution is the eigenvector corresponding to the smallest singular value,
        # which is the last row of V^T (or last column of V).
        H_flat = Vt[-1, :]
        H = H_flat.reshape((3, 3))
        
        # Normalize so that H[2, 2] = 1 (if not close to 0)
        if abs(H[2, 2]) > 1e-10:
            H = H / H[2, 2]
            
        return H

    def _calibrate(self):
        """Runs the DLT calibration and validates against OpenCV."""
        self.H_dlt = self.compute_homography_dlt(self.image_points, self.birdseye_points)
        self.H_cv2, _ = cv2.findHomography(self.image_points, self.birdseye_points)
        
        # Calculate reprojection error to ensure correctness
        error = self._calculate_reprojection_error(self.H_dlt, self.image_points, self.birdseye_points)
        print(f"Geometric Calibration initialized. DLT reprojection error: {error:.6f} pixels.")

    def _calculate_reprojection_error(self, H, pts_src, pts_dst):
        """Calculates Mean Squared Error (MSE) of reprojection."""
        pts_src_h = np.hstack([pts_src, np.ones((pts_src.shape[0], 1))])
        pts_proj = (H @ pts_src_h.T).T
        # Convert homogeneous back to 2D
        pts_proj_2d = pts_proj[:, :2] / pts_proj[:, 2:]
        error = np.mean(np.linalg.norm(pts_dst - pts_proj_2d, axis=1)**2)
        return error

    def get_homography(self):
        """Returns the DLT-computed homography matrix."""
        return self.H_dlt

    def warp_to_birdseye(self, image, resolution):
        """
        Warps the original image into the bird's-eye view using the homography.
        Demonstrates geometric perspective correction.
        """
        # We use cv2.warpPerspective for efficient pixel interpolation during the actual warp
        return cv2.warpPerspective(image, self.H_dlt, tuple(resolution))

    def image_point_to_ground(self, point):
        """
        Transforms a point (x, y) from the image plane to the rectified bird's eye plane.
        """
        p = np.array([point[0], point[1], 1.0])
        p_proj = self.H_dlt @ p
        return (int(p_proj[0] / p_proj[2]), int(p_proj[1] / p_proj[2]))
