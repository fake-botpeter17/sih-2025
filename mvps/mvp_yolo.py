# Install if not already:
# pip install ultralytics opencv-python

import cv2
from ultralytics import YOLO

# --- Load YOLOv8 model for vehicle detection ---
model = YOLO("models/yolov8n.pt")  # small, fast model for demo

# --- Load traffic video ---
video_path = "c:/Users/win11/Downloads/2103099-uhd_3840_2160_30fps.mp4"
cap = cv2.VideoCapture(video_path)
frame_count = 0
frame_skip = 20

while True:
    ret, frame = cap.read()
    frame_count += 1
    print(f"{frame_count = }")
    if not ret:
        break
    
    # Resize frame for faster processing
    frame = cv2.resize(frame, (960, 540))
    
    condition = True #frame_count % frame_skip == 0

    if condition:
    # Run YOLO detection
        results = model(frame)[0]
        
        vehicle_count = 0  # per frame
        
        for box in results.boxes:
            cls_id = int(box.cls[0])
            label = model.names[cls_id]
            
            # Only detect vehicles
            if label in ["car", "truck", "bus", "motorbike"]:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0), 2)
                
                # Put vehicle label on bounding box
                cv2.putText(frame, label, (x1, y1-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 2)
                
                vehicle_count += 1
    

    
    # Display total vehicles detected in this frame
    cv2.putText(frame, f"Vehicles Detected: {vehicle_count if condition else "Skip"}", (20,30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
    
    cv2.imshow("Smart Traffic MVP - YOLO", frame)
    
    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
