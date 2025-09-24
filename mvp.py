import cv2
import numpy as np

# --- 1. Load a traffic video ---
video_path = "c:/Users/win11/Downloads/2103099-uhd_3840_2160_30fps.mp4"
cap = cv2.VideoCapture(video_path)

# Background subtractor (for detecting moving vehicles)
fgbg = cv2.createBackgroundSubtractorMOG2(history=100, varThreshold=40)

vehicle_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break  # End of video
    
    # Resize for faster processing
    frame = cv2.resize(frame, (640, 360))
    
    # --- 2. Apply background subtraction ---
    fgmask = fgbg.apply(frame)
    
    # Clean the mask (remove noise)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5,5))
    fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel)
    fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_DILATE, kernel)
    
    # --- 3. Find contours (moving objects) ---
    contours, _ = cv2.findContours(fgmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    vehicles_detected = 0
    for cnt in contours:
        # Ignore small contours (noise)
        area = cv2.contourArea(cnt)
        if area > 500:  
            x, y, w, h = cv2.boundingRect(cnt)
            cv2.rectangle(frame, (x,y), (x+w, y+h), (0,255,0), 2)
            vehicles_detected += 1
    
    # --- 4. Estimate density ---
    if vehicles_detected < 5:
        density = "Low"
        suggestion = "Keep Green Short"
    elif vehicles_detected < 15:
        density = "Medium"
        suggestion = "Normal Cycle"
    else:
        density = "High"
        suggestion = "Extend Green Light"
    
    # --- 5. Display info ---
    cv2.putText(frame, f"Vehicles: {vehicles_detected}", (20,30), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
    cv2.putText(frame, f"Traffic Density: {density}", (20,70), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2)
    cv2.putText(frame, f"Signal Suggestion: {suggestion}", (20,110), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,255), 2)
    
    cv2.imshow("Traffic Management MVP", frame)
    
    # Press 'q' to quit
    if cv2.waitKey(30) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
