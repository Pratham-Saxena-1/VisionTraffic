from ultralytics import YOLO

class MultiObjectTracker:
    """
    Wraps the ByteTrack algorithm (via Ultralytics) to assign persistent IDs 
    to detected vehicles.
    """
    def __init__(self, config: dict):
        model_path = config['model']['path']
        self.model = YOLO(model_path)
        self.conf = config['model']['confidence_threshold']
        self.iou = config['model']['iou_threshold']
        self.classes = config['model']['vehicle_classes']
        self.imgsz = config['model']['input_resolution']

    def update(self, frame):
        """
        Runs detection and tracking on the frame.
        (Since YOLOv8 integrates ByteTrack tightly, we run both here for efficiency,
        conceptually acting as the tracker module consuming the frame).
        
        Returns:
            tracked_objects: List of dicts with 'track_id', 'bbox', 'class', 'conf'
        """
        # Run tracking (ByteTrack is default in YOLOv8 tracker='bytetrack.yaml')
        preds = self.model.track(frame, persist=True, conf=self.conf, iou=self.iou, 
                                 classes=self.classes, imgsz=self.imgsz, verbose=False,
                                 tracker="botsort.yaml") # using botsort or bytetrack
        
        tracked_objects = []
        if len(preds) > 0 and preds[0].boxes is not None and preds[0].boxes.id is not None:
            boxes = preds[0].boxes
            for i in range(len(boxes)):
                x1, y1, x2, y2 = boxes.xyxy[i].cpu().numpy()
                conf = float(boxes.conf[i].cpu().numpy())
                cls_id = int(boxes.cls[i].cpu().numpy())
                
                # Check if ID exists (sometimes it loses track briefly)
                if boxes.id is not None and i < len(boxes.id):
                    track_id = int(boxes.id[i].cpu().numpy())
                else:
                    track_id = -1
                    
                if track_id != -1:
                    tracked_objects.append({
                        'track_id': track_id,
                        'bbox': [int(x1), int(y1), int(x2), int(y2)],
                        'center': [int((x1+x2)/2), int((y1+y2)/2)],
                        'conf': conf,
                        'class': cls_id
                    })
                
        return tracked_objects
