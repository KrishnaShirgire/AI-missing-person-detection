# YOLO for Person detection and tracking using ByteTrack
from ultralytics import YOLO

# Load YOLO model
model = YOLO("yolov8n.pt") # load the pretrained YOLOv8n model

# Run YOLO + tracking on webcam
model.track( # take some input and perform object detection and tracking
    source=0, # use computer webcam as input
    show=True, # show detection results in a window
    conf=0.5, # only show detections with confidence >= 0.5
    tracker="bytetrack.yaml", # use the bytetrack tracker for tracking objects across frames
    persist=True # keep the tracker state across frames
)