import cv2
import os
import numpy as np
import torch
from ultralytics import YOLO
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Load YOLOv8 models for pitch and ball detection
pitch_model = YOLO("Yolov8_model.pt")
ball_model = YOLO("runs/detect/ball_detection/weights/best.pt")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
pitch_model.to(device)
ball_model.to(device)



def detect_pitch(frame, model):
    results = model(frame)
    for result in results:
        for pred in result.boxes:
            if model.names[int(pred.cls)] == "Cricket_pitch" and pred.conf > 0.75:
                return pred.xyxy[0].cpu().numpy()
    return None

def detect_ball(frame, model):
    results = model(frame)
    for result in results:
        for pred in result.boxes:
            if model.names[int(pred.cls)] == "Cricket-Ball" and pred.conf > 0.50:
                return pred.xyxy[0].cpu().numpy()
    return None

def track_ball(video_path, pitch_model, ball_model):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logging.error("Error: Could not open video.")
        return

    frame_number = 0
    ball_trajectory = []
    bounce_point = None
    pitch_coords = None
    output_dir = "output_frames"
    os.makedirs(output_dir, exist_ok=True)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Detect pitch first
        pitch_coords = detect_pitch(frame, pitch_model)
        if pitch_coords is not None:
            x1, y1, x2, y2 = map(int, pitch_coords)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)

            # Detect ball only if pitch is detected
            ball_coords = detect_ball(frame, ball_model)
            if ball_coords is not None:
                ball_trajectory.append((frame_number, ball_coords))
                x1, y1, x2, y2 = map(int, ball_coords)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                # Detect bounce point (simplified logic)
                if len(ball_trajectory) > 1:
                    prev_y = ball_trajectory[-2][1][1]
                    curr_y = ball_coords[1]
                    if curr_y > prev_y:  # Assuming bounce when y-coordinate increases
                        bounce_point = ball_coords

        frame_number += 1
        output_file = os.path.join(output_dir, f"frame_{frame_number}.jpg")
        cv2.imwrite(output_file, frame)

    cap.release()
    return pitch_coords, bounce_point, ball_trajectory


def classify_length(bounce_point, pitch_coords):
    if bounce_point is None or pitch_coords is None:
        return "Unknown"

    x1, y1, x2, y2 = pitch_coords
    pitch_height = y2 - y1
    bounce_y = bounce_point[1]

    if bounce_y < y1 + 0.3 * pitch_height:
        return "short"
    elif y1 + 0.3 * pitch_height <= bounce_y <= y1 + 0.7 * pitch_height:
        return "good length"
    else:
        return "full length"

def process_video(video_file):
    # Track the ball and detect pitch
    pitch_coords, bounce_point, ball_trajectory = track_ball(video_file, pitch_model, ball_model)

    # Classify the length of the delivery
    length_classification = classify_length(bounce_point, pitch_coords)
    logging.info(f"Bowler's length classification: {length_classification}")

    return length_classification

if __name__ == "__main__":
    video_file = '2_ball_match.mp4'
    try:
        length_classification = process_video(video_file)
    except Exception as e:
        logging.error(f"Error processing video: {e}")