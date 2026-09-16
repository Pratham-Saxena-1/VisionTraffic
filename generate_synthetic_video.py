import cv2
import numpy as np
import os

def create_synthetic_video(output_path, num_frames=60, width=1280, height=720):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, 30.0, (width, height))
    
    # Car state: [x, y, width, height, speed_y]
    cars = [
        [300, 100, 80, 150, 10], # Moving down lane 1
        [600, 50, 80, 150, 12],  # Moving down lane 2
        [900, 400, 80, 150, 15]  # Moving down lane 3
    ]
    
    for f in range(num_frames):
        # Create road background (gray)
        frame = np.ones((height, width, 3), dtype=np.uint8) * 100
        
        # Draw lane lines (white)
        cv2.line(frame, (450, 0), (450, height), (255, 255, 255), 10)
        cv2.line(frame, (750, 0), (750, height), (255, 255, 255), 10)
        
        # Draw cars
        for car in cars:
            x, y, w, h, sy = car
            # Draw car body (blue-ish)
            cv2.rectangle(frame, (int(x), int(y)), (int(x+w), int(y+h)), (200, 50, 50), -1)
            # Update position
            car[1] += sy
            
        out.write(frame)
        
    out.release()
    print(f"Created synthetic video at {output_path}")

if __name__ == '__main__':
    create_synthetic_video('data/sample/traffic.mp4')
