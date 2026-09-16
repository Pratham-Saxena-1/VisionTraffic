import cv2
import os

class VideoProcessor:
    """
    Handles video I/O, frame extraction, and coordinate scaling.
    """
    def __init__(self, input_path, output_path, birdseye_output_path, config: dict):
        self.input_path = input_path
        self.output_path = output_path
        self.birdseye_output_path = birdseye_output_path
        self.target_resolution = tuple(config['model']['input_resolution'])
        self.birdseye_resolution = tuple(config['calibration']['birdseye_resolution'])
        
        self.cap = cv2.VideoCapture(input_path)
        if not self.cap.isOpened():
            raise FileNotFoundError(f"Cannot open video file: {input_path}")
            
        # Get video properties
        self.fps = int(self.cap.get(cv2.CAP_PROP_FPS))
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Initialize writers
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.out_raw = cv2.VideoWriter(output_path, fourcc, self.fps, self.target_resolution)
        self.out_birdseye = cv2.VideoWriter(birdseye_output_path, fourcc, self.fps, self.birdseye_resolution)

    def read_frame(self):
        """Reads and resizes the next frame."""
        ret, frame = self.cap.read()
        if not ret:
            return False, None
            
        frame_resized = cv2.resize(frame, self.target_resolution)
        return True, frame_resized

    def write_frame(self, raw_annotated, birdseye_annotated):
        """Writes the annotated frames to the output videos."""
        if self.out_raw is not None and raw_annotated is not None:
            self.out_raw.write(raw_annotated)
            
        if self.out_birdseye is not None and birdseye_annotated is not None:
            self.out_birdseye.write(birdseye_annotated)

    def release(self):
        """Releases video resources."""
        self.cap.release()
        if self.out_raw is not None:
            self.out_raw.release()
        if self.out_birdseye is not None:
            self.out_birdseye.release()
