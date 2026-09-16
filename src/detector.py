from ultralytics import YOLO
import cv2

class VehicleDetector:
    """
    Wraps a YOLOv8 model for detecting vehicles in frames.
    Demonstrates integration of modern deep-learning Computer Vision.
    """
    def __init__(self, config: dict):
        model_path = config['model']['path']
        self.model = YOLO(model_path)
        self.conf = config['model']['confidence_threshold']
        self.iou = config['model']['iou_threshold']
        self.classes = config['model']['vehicle_classes']
        self.imgsz = config['model']['input_resolution']

    def detect(self, frame):
        """
        Runs inference on a single frame.
        
        Returns:
            results: A list of dicts with 'bbox', 'class', 'conf'
        """
        # Run inference
        preds = self.model(frame, conf=self.conf, iou=self.iou, classes=self.classes, imgsz=self.imgsz, verbose=False)
        
        detections = []
        if len(preds) > 0:
            boxes = preds[0].boxes
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                conf = float(box.conf[0].cpu().numpy())
                cls_id = int(box.cls[0].cpu().numpy())
                
                detections.append({
                    'bbox': [int(x1), int(y1), int(x2), int(y2)],
                    'conf': conf,
                    'class': cls_id
                })
                
        return detections
