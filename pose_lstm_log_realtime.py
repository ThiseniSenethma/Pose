import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
import time
import datetime
import os

# Create logs directory if it doesn't exist
if not os.path.exists("logs"):
    os.makedirs("logs")

# Create or open log file with current date
current_date = datetime.datetime.now().strftime("%Y-%m-%d")
log_file = open(f"logs/log_{current_date}.txt", "a")
log_file.write(f"\n--- New Session Started at {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ---\n")

cap = cv2.VideoCapture(0)

mpPose = mp.solutions.pose
pose = mpPose.Pose()
mpDraw = mp.solutions.drawing_utils

# Load pre-trained model
model = tf.keras.models.load_model("violent_actions_model.h5")

lm_list = []
label = "normal"
previous_label = "normal"  # To track label changes
neutral_label = "normal"
detection_start_time = None

def make_landmark_timestep(results):
    c_lm = []
    for lm in results.pose_landmarks.landmark:
        c_lm.append(lm.x)
        c_lm.append(lm.y)
        c_lm.append(lm.z)
        c_lm.append(lm.visibility)
    return c_lm

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

def log_detection(action, confidence=None):
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
    log_entry = f"[{timestamp}] Detected: {action}"
    if confidence:
        log_entry += f" (Confidence: {confidence:.2f})"
    log_file.write(log_entry + "\n")
    log_file.flush()  # Ensure it's written immediately
    print(log_entry)  # Also print to console

# Warm-up frames to stabilize detection
i = 0
warm_up_frames = 60
fps_counter = 0
fps_start_time = time.time()
fps = 0

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Calculate FPS
        fps_counter += 1
        if time.time() - fps_start_time >= 1:
            fps = fps_counter
            fps_counter = 0
            fps_start_time = time.time()

        frameRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(frameRGB)
        
        i += 1
        if i > warm_up_frames:
            if results.pose_landmarks:
                lm = make_landmark_timestep(results)
                lm_list.append(lm)
                
                if len(lm_list) == 20:
                    previous_label = label
                    label = detect(model, lm_list)
                    
                    # Log when detection changes
                    if label != previous_label:
                        if label != "normal":
                            # Start of violent action
                            detection_start_time = time.time()
                            log_detection(label)
                        else:
                            # End of violent action
                            if detection_start_time is not None:
                                duration = time.time() - detection_start_time
                                log_detection(f"{previous_label} ended", duration)
                                detection_start_time = None
                    
                    lm_list = []
                
                # Draw bounding box
                x_coordinate = []
                y_coordinate = []
                for lm in results.pose_landmarks.landmark:
                    h, w, _ = frame.shape
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    x_coordinate.append(cx)
                    y_coordinate.append(cy)
                
                cv2.rectangle(frame,
                            (min(x_coordinate), max(y_coordinate)),
                            (max(x_coordinate), min(y_coordinate) - 25),
                            (0, 255, 0),
                            1)

                frame = draw_landmark_on_image(mpDraw, results, frame)
            
            frame = draw_class_on_image(label, frame)
            
            # Display FPS
            cv2.putText(frame, f"FPS: {fps}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 
                        0.7, (0, 255, 0), 2)
            
            cv2.imshow("Violence Detection", frame)
            
            if cv2.waitKey(1) == ord('q'):
                break

except Exception as e:
    log_file.write(f"[{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] ERROR: {str(e)}\n")
    print(f"Error: {str(e)}")

finally:
    # Clean up
    if detection_start_time is not None:
        duration = time.time() - detection_start_time
        log_file.write(f"[{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {label} ended (Duration: {duration:.2f}s)\n")
    
    log_file.write(f"--- Session Ended at {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ---\n\n")
    log_file.close()
    cap.release()
    cv2.destroyAllWindows()