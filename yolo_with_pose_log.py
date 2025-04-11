import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
from ultralytics import YOLO
import time
import os
import datetime

# Initialize video capture
cap = cv2.VideoCapture(0)

# Initialize MediaPipe Pose
mpPose = mp.solutions.pose
pose = mpPose.Pose()
mpDraw = mp.solutions.drawing_utils

# Load pre-trained models
action_model = tf.keras.models.load_model("violent_actions_model.h5")
yolo_model = YOLO("best.pt")  # Load your trained YOLO model

# Configure logging with current date
current_date = datetime.datetime.now().strftime("%Y-%m-%d")
log_file_path = f"logs/pose_log_{current_date}.txt"

# Create log file header if it doesn't exist or is empty
if not os.path.exists(log_file_path) or os.path.getsize(log_file_path) == 0:
    with open(log_file_path, "a") as log_file:
        log_file.write("Timestamp,Action,Object_Detections,Confidence,Keypoints\n")

lm_list = []
label = "normal"
neutral_label = "normal"

def make_landmark_timestep(results):
    c_lm = []
    for lm in results.pose_landmarks.landmark:
        c_lm.append(lm.x)
        c_lm.append(lm.y)
        c_lm.append(lm.z)
        c_lm.append(lm.visibility)
    return c_lm

def get_keypoints_string(results):
    """Convert pose landmarks to a compact string representation for logging"""
    keypoints = []
    if results.pose_landmarks:
        for idx, lm in enumerate(results.pose_landmarks.landmark):
            keypoints.append(f"{idx}:({lm.x:.3f},{lm.y:.3f},{lm.z:.3f})")
    return "|".join(keypoints)

def log_detection(action_label, object_detections, keypoints_str):
    """Log detection results to file"""
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
    
    # Format object detections as a string
    objects_str = []
    for obj in object_detections:
        objects_str.append(f"{obj['class']}:{obj['confidence']:.2f}")
    objects_formatted = "|".join(objects_str) if objects_str else "None"
    
    # Write to log file
    with open(log_file_path, "a") as log_file:
        log_file.write(f"{timestamp},{action_label},{objects_formatted},{keypoints_str}\n")

def draw_landmark_on_image(mpDraw, results, frame):
    mpDraw.draw_landmarks(frame, results.pose_landmarks, mpPose.POSE_CONNECTIONS)
    for lm in results.pose_landmarks.landmark:
        h, w, _ = frame.shape
        cx, cy = int(lm.x * w), int(lm.y * h)
        cv2.circle(frame, (cx, cy), 3, (0, 255, 0), cv2.FILLED)
    return frame

def draw_class_on_image(label, img):
    font = cv2.FONT_HERSHEY_SIMPLEX
    bottomLeftCornerOfText = (10, 30)
    fontScale = 1
    fontColor = (0, 255, 0) if label == neutral_label else (0, 0, 255)
    thickness = 2
    lineType = 2
    cv2.putText(img, str(label),
                bottomLeftCornerOfText,
                font,
                fontScale,
                fontColor,
                thickness,
                lineType)
    return img

def detect(model, lm_list):
    global label
    lm_list = np.array(lm_list)
    lm_list = np.expand_dims(lm_list, axis=0)
    result = model.predict(lm_list)
   
    # Map indices to labels
    label_map = {0: "normal", 1: "kicking", 2: "punching"}
    predicted_index = np.argmax(result[0])
    label = label_map[predicted_index]
   
    return label

# Warm-up frames to stabilize detection
i = 0
warm_up_frames = 60

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # Create separate frames for each detection system
    pose_frame = frame.copy()
    object_frame = frame.copy()
    
    # Run YOLO object detection
    yolo_results = yolo_model(object_frame)
    
    # Process the frame for pose detection
    frameRGB = cv2.cvtColor(pose_frame, cv2.COLOR_BGR2RGB)
    pose_results = pose.process(frameRGB)
   
    i += 1
    if i > warm_up_frames:
        # Process object detection results
        object_detections = []
        for detection in yolo_results:
            boxes = detection.boxes.cpu().numpy()
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0].astype(int)
                confidence = float(box.conf[0])
                class_id = int(box.cls[0])
                class_name = detection.names[class_id]
                
                # Store detection info for logging
                object_detections.append({
                    "class": class_name,
                    "confidence": confidence,
                    "bbox": (x1, y1, x2, y2)
                })
                
                # Draw bounding box for the detected object
                cv2.rectangle(object_frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                
                # Put class name and confidence score
                label_text = f"{class_name}: {confidence:.2f}"
                cv2.putText(object_frame, label_text, (x1, y1 - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
        
        # Handle pose detection and action classification
        keypoints_str = get_keypoints_string(pose_results)
        
        if pose_results.pose_landmarks:
            lm = make_landmark_timestep(pose_results)
            lm_list.append(lm)
           
            if len(lm_list) == 20:
                label = detect(action_model, lm_list)
                lm_list = []
                
                # Log detection results with pose and objects
                log_detection(label, object_detections, keypoints_str)
           
            # Draw pose bounding box
            x_coordinate = []
            y_coordinate = []
            for lm in pose_results.pose_landmarks.landmark:
                h, w, _ = pose_frame.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                x_coordinate.append(cx)
                y_coordinate.append(cy)
           
            if x_coordinate and y_coordinate:  # Check if lists are not empty
                cv2.rectangle(pose_frame,
                            (min(x_coordinate), max(y_coordinate)),
                            (max(x_coordinate), min(y_coordinate) - 25),
                            (0, 255, 0),
                            1)
            pose_frame = draw_landmark_on_image(mpDraw, pose_results, pose_frame)
        
        # Draw action classification label on pose frame
        pose_frame = draw_class_on_image(label, pose_frame)
       
        # Display both frames in separate windows
        cv2.imshow("Pose Detection", pose_frame)
        cv2.imshow("Object Detection", object_frame)
       
        if cv2.waitKey(1) == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()