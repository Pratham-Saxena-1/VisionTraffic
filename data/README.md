# VisionTraffic Dataset Directory

This directory is intended to store sample traffic videos for evaluation.

## Acquiring a Sample Video
To run the evaluation, you can download any public traffic video (e.g., from the MIO-TCD dataset, DETRAC, or YouTube traffic cams).
Save the file as `data/sample/traffic.mp4`.

## Configuring Calibration Points
If you use a new video, you MUST configure the point correspondences in `config/config.yaml`.
1. Open the first frame of your video in an image viewer.
2. Select 4 points forming a rectangle on the ground (e.g., a section of a lane).
3. Record their `(x, y)` pixel coordinates.
4. Update `calibration: image_points` in the config file.
5. Provide the estimated real-world geometric layout for those points in `ground_points` (e.g., treating the rectangle as 10m wide and 30m long).
