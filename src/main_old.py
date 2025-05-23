import cv2
import pytesseract
import re
import os
from collections import defaultdict
from ultralytics import YOLO
from transformers import VideoMAEForVideoClassification, VivitImageProcessor, VivitForVideoClassification
import torch
from torchvision.transforms import Compose, Normalize, ToTensor
import numpy as np
from concurrent.futures import ThreadPoolExecutor
import logging

# from torchvision.transforms import Compose
# from pytorchvideo.transforms import UniformTemporalSubsample, Normalize
# from torchvision.transforms import Resize, CenterCrop
# from torchvision.io import read_video



# torch.cuda.empty_cache()
# torch.cuda.reset_peak_memory_stats()

# Load pitch and score_bar multi detection YOLOv8 model
model = YOLO("Yolov8_model.pt")
device = torch.device("cpu")
model.to(device)

ball_model = YOLO('Crciket_ball_tracking/Cricket-ball-tracking-updated-main/runs/detect/train2/weights/best.pt')



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

local_model_dir_MAE = "./videomae-base-finetuned-Custom_Dataset_Finetune"
# local_model_vivit = "./Cricket_Shot_Detection_vivit_finetuned_1"
local_model_vivit = "./vivit-b-16x2-kinetics400-Finetune_10Shots"

pytesseract.pytesseract.tesseract_cmd = r'C:/Program Files/Tesseract-OCR/tesseract.exe'  # Update this path as needed

# 2. Load the processor
processor_load = VivitImageProcessor.from_pretrained(local_model_vivit)




# Load the model from the local directory
model_load_VideoMAE = VideoMAEForVideoClassification.from_pretrained(local_model_dir_MAE).to(device)
model_load_vivit = VivitForVideoClassification.from_pretrained(local_model_vivit).to(device)

# mean = image_processor.image_mean
# std = image_processor.image_std

# # Ensure mean and std have the correct shape
# mean = torch.tensor(mean).view(1, -1, 1, 1)
# std = torch.tensor(std).view(1, -1, 1, 1)

# video_transform = Compose([
#     UniformTemporalSubsample(16),   # Sample 16 frames
#     Resize((224, 224)),            # Resize frames
#     CenterCrop(224),               # Center crop
#     Normalize(mean, std)           # Normalize
# ])

# optionally, change the confidence and overlap thresholds
# values are percentages
# Set confidence and overlap thresholds
model.confidence = 70  # Increased confidence threshold
model.overlap = 10      # Decreased overlap threshold

def run_inference(model, frames):
    """Utility to run inference given a model and test video.

    The video is assumed to be preprocessed already.
    """
    # frames = video.unsqueeze(0)

    # 4. Prepare input for the model
    # inputs = processor(frames, return_tensors="pt", sampling_rate=25)
    model.eval()
    # inputs = {
    #     "pixel_values": video.unsqueeze(0),
    #     # "labels": torch.tensor([sample_test_video["label"]]),  # this can be skipped if you don't have labels available.
    # }
    device = torch.device("cpu")
    model = model.to(device)
    # forward pass
    with torch.no_grad():
        outputs = model(pixel_values=frames)
        logits = outputs.logits

    return logits




def get_predicted_class(model, logits):
    """Utility to get the predicted class from logits."""
    # Get the class with the highest logit value
    # probabilities = torch.nn.functional.softmax(logits, dim=-1)
    # predicted_class_idx = torch.argmax(probabilities, dim=-1).item()
    # # Get the class label from the model's configuration
    # class_label = model.config.id2label[predicted_class_idx]
    probabilities = torch.nn.functional.softmax(logits, dim=-1)
    predicted_class_idx = torch.argmax(probabilities, dim=-1).item()
    confidence = probabilities[0, predicted_class_idx].item()
    class_label = model.config.id2label[predicted_class_idx]
    return class_label, predicted_class_idx, confidence




# # Example usage
# ocr_text = "'ENG. 31-1 Ty 42/29 Toss SA Salt 3 (3) / Bairstowi30) -y:) Nortje 1-9 0 |\\n'"
# runs, wickets = parse_score_and_wickets(ocr_text)
# print(f"Extracted Runs: {runs}, Extracted Wickets: {wickets}")




def convert_frames_to_videos(frames_directory, video_name, video_output_directory):
    frames = [f for f in os.listdir(frames_directory) if f.endswith('.jpg')]
    frames.sort(key=lambda x: int(x.split('_')[-1].split('.')[0]))

    sequence_start = 0
    sequence_length = 0
    video_count = 0
    frame_ranges = []
    max_gap = 20
    video_paths = []  # List to store all video paths

    for i in range(len(frames)):
        if i == 0 or (int(frames[i].split('_')[-1].split('.')[0]) - int(frames[i-1].split('_')[-1].split('.')[0]) <= max_gap + 1):
            sequence_length += 1
        else:
            if sequence_length > 20:
                start_frame = int(frames[sequence_start].split('_')[-1].split('.')[0])
                end_frame = int(frames[i-1].split('_')[-1].split('.')[0])
                frame_ranges.append((start_frame, end_frame))
                video_path = create_video_from_sequence(frames_directory, frames[sequence_start:i], video_name, video_count, video_output_directory)
                video_paths.append(video_path)  # Add the video path to the list
                video_count += 1
            sequence_start = i
            sequence_length = 1

    if sequence_length > 20:
        start_frame = int(frames[sequence_start].split('_')[-1].split('.')[0])
        end_frame = int(frames[-1].split('_')[-1].split('.')[0])
        frame_ranges.append((start_frame, end_frame))
        video_path = create_video_from_sequence(frames_directory, frames[sequence_start:], video_name, video_count, video_output_directory)
        video_paths.append(video_path)  # Add the video path to the list

    save_frame_ranges(video_name, frame_ranges, video_output_directory)

    return frame_ranges, video_count, video_paths

def create_video_from_sequence(frames_directory, frames, video_name, video_count, video_output_directory):

    # public_dir = "../public/videos"
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Change codec to mp4v for MP4 format
    frame = cv2.imread(os.path.join(frames_directory, frames[0]))
    if frame is None:
        logging.error(f"Error reading frame: {os.path.join(frames_directory, frames[0])}")
        raise ValueError(f"Error reading frame: {os.path.join(frames_directory, frames[0])}")
    height, width, layers = frame.shape
    video_output_path = os.path.join(video_output_directory, f"{video_name}_output_{video_count}.mp4")
    # public_dir = os.path.join(video_output_directory, f"{video_name}_output_{video_count}.mp4")
    out = cv2.VideoWriter(video_output_path, fourcc, 12.0, (width, height))
    # out = cv2.VideoWriter(public_dir, fourcc, 12.0, (width, height))


    for frame_file in frames:
        frame = cv2.imread(os.path.join(frames_directory, frame_file))
        if frame is None:
            logging.error(f"Error reading frame: {os.path.join(frames_directory, frame_file)}")
            raise ValueError(f"Error reading frame: {os.path.join(frames_directory, frame_file)}")
        out.write(frame)

    out.release()
    logging.debug(f"Video {video_count} has been created and saved as {video_output_path}")
    return video_output_path

def save_frame_ranges(video_name, frame_ranges, video_output_directory):
    # Save frame range information to a text file
    frame_ranges_file = os.path.join(video_output_directory, f"{video_name}_frame_ranges.txt")
    with open(frame_ranges_file, 'w') as f:
        for i, (start_frame, end_frame) in enumerate(frame_ranges):
            f.write(f"Video {i}: Start Frame = {start_frame}, End Frame = {end_frame}\n")
    print(f"Frame range information has been saved to {frame_ranges_file}")

def uniform_sampling(frames, num_frames=16):
    if not frames:
        raise ValueError("The frames list is empty. No frames were extracted from the video.")

    total_frames = len(frames)
    if total_frames > num_frames:
        frame_indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
        frames = [frames[i] for i in frame_indices]
    elif total_frames < num_frames:
        padding = num_frames - total_frames
        last_frame = frames[-1]
        frames.extend([last_frame] * padding)
    return frames

def pitch_coordinates(results):
    """
    Function to detected pitch area in the frame using Roboflow API.
    """
    for result in results:
        boxes = result.boxes  # Boxes object for bounding box outputs
        for box in boxes:
            x, y, w, h = box.xywh[0]  # xywh returns a tensor, so we take the first element
            x, y, w, h = int(x), int(y), int(w), int(h)  # Convert to integers
            x1 = int(x - w / 2)
            y1 = int(y - h / 2)
            x2 = x1 + w
            y2 = y1 + h
            return x1, y1, x2, y2
    else:
        print("Error: No pitch detected.")
        return None, None, None, None

# Extract bounding box coordinates for the pitch
    # if results['predictions']:
    #     pitch = results['predictions'][0]
    #     x, y, w, h = pitch['x'], pitch['y'], pitch['width'], pitch['height']
    #     x1 = int(x - w / 2)
    #     y1 = int(y - h / 2)
    #     x2 = x1 + w
    #     y2 = y1 + h
    #     return x1, y1, x2, y2
def detect_ball(ball_model, frame):
    # Perform ball detection
    result = ball_model(frame)
    return result

def draw_pitch_length_annotations(frame, pitch_x1, pitch_y1, pitch_x2, pitch_y2):
    """
    Draw pitch length annotations on the frame based on detected pitch coordinates.
    """
    # Define colors for each region
    colors = {
        "Short": (0, 0, 255),   # Red
        "Good": (0, 255, 0),    # Green
        "Full": (255, 0, 0),    # Blue
        "Yorker": (0, 255, 255) # Yellow
    }

    # Ensure the coordinates are integers
    pitch_x1, pitch_y1, pitch_x2, pitch_y2 = map(int, [pitch_x1, pitch_y1, pitch_x2, pitch_y2])

    # Calculate y-coordinates for the length annotations based on pitch height
    yorker_length_y = int(pitch_y1 + 0.10 * (pitch_y2 - pitch_y1))
    full_length_y = int(pitch_y1 + 0.17 * (pitch_y2 - pitch_y1))
    good_length_y = int(pitch_y1 + 0.25 * (pitch_y2 - pitch_y1))
    short_length_y = int(pitch_y1 + 0.40 * (pitch_y2 - pitch_y1))

    # Draw pitch length regions on the frame
    cv2.rectangle(frame, (pitch_x1, short_length_y), (pitch_x2, pitch_y2), colors["Short"], 2)
    cv2.rectangle(frame, (pitch_x1, good_length_y), (pitch_x2, short_length_y), colors["Good"], 2)
    cv2.rectangle(frame, (pitch_x1, full_length_y), (pitch_x2, good_length_y), colors["Full"], 2)
    cv2.rectangle(frame, (pitch_x1, yorker_length_y), (pitch_x2, full_length_y), colors["Yorker"], 2)

    # Add text labels
    cv2.putText(frame, "Short", (pitch_x1 + 10, short_length_y + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, colors["Short"], 2)
    cv2.putText(frame, "Good", (pitch_x1 + 10, good_length_y + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, colors["Good"], 2)
    cv2.putText(frame, "Full", (pitch_x1 + 10, full_length_y + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, colors["Full"], 2)
    cv2.putText(frame, "Yorker", (pitch_x1 + 10, yorker_length_y + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, colors["Yorker"], 2)

def classify_bounce(ball_y, pitch_y1, pitch_y2):

    """
    Classify the bounce position based on the y-coordinate relative to the pitch.
    Returns both a general classification and a more specific length description.
    """
    yorker_length_y = int(pitch_y1 + 0.10 * (pitch_y2 - pitch_y1))
    full_length_y = int(pitch_y1 + 0.20 * (pitch_y2 - pitch_y1))
    good_length_y = int(pitch_y1 + 0.35 * (pitch_y2 - pitch_y1))
    short_length_y = int(pitch_y1 + 0.50 * (pitch_y2 - pitch_y1))

    if ball_y >= short_length_y:
        return "Short", "Very Short"
    elif ball_y >= good_length_y:
        if ball_y >= (good_length_y + short_length_y) / 2:
            return "Short", "Back of a Length"
        else:
            return "Good", "Good Length"
    elif ball_y >= full_length_y:
        if ball_y >= (full_length_y + good_length_y) / 2:
            return "Good", "Full of a Length"
        else:
            return "Full", "Full"
    elif ball_y >= yorker_length_y:
        return "Yorker", "Yorker"
    else:
        return "Beyond Yorker", "Full Toss"


def extract_frames(video_file, main_output_directory):
    cap = cv2.VideoCapture(video_file)
    if not cap.isOpened():
        raise ValueError(f"Error opening video file {video_file}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"FPS: {fps}")
    frame_rate = fps if fps <= 30 else 30

    output_path = 'annotated_output_with_continuous_display14.mp4'
    # Initialize VideoWriter with MP4 codec
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Use 'mp4v' codec for MP4 format
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    # Initialize variables to track the ball's position and detect bounce
    prev_ball_y = None
    bounce_detected = False

    # Ball delivery tracking
    ball_counter = 0
    current_ball_frames = []
    pitch_data = {}  # Stores {ball_number: (pitch_coords, annotation_frame)}

    # Detection flags
    new_ball_triggered = False
    ball_detected = False

    pitch_coords = None
    prev_velocity = None
    bounce_frame = None
    pitch_x1, pitch_y1, pitch_x2, pitch_y2 = None, None, None, None

    # New variables to store the last detected bounce information
    last_bounce_classification = "N/A"
    last_length_description = "N/A"
    last_bounce_coordinates = None

    # Find the highest confidence box among all detections
    highest_confidence_box = None
    highest_avg_confidence = -1

    frame_count = 0
    # Get the video file's name without extension
    video_name = os.path.splitext(os.path.basename(video_file))[0]

    # Create an output folder with a name corresponding to the video
    video_output_directory = os.path.join(main_output_directory, video_name)
    final_frames_directory = os.path.join(video_output_directory, "final_frames")
    os.makedirs(video_output_directory, exist_ok=True)
    os.makedirs(final_frames_directory, exist_ok=True)
    TARGET_CLASS_1 = "Cricket_pitch"
    TARGET_CLASS_2 = "score_bar"
    CONFIDENCE_THRESHOLD_Pitch = 0.75
    CONFIDENCE_THRESHOLD_ScoreBar = 0.75

    frames = []
    ball_trajectory = []
    additional_frames_count = 0  # Initialize additional frames count
    detection_active = False  # Flag to track if detection is active

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        if fps > 0 and frame_rate > 0:
            frame_interval = int(fps / frame_rate)
            if frame_interval > 0 and frame_count % frame_interval == 0:
                # Only extract frames at the desired frame rate
                output_file = f"{video_output_directory}/frame_{frame_count}.jpg"
                # img = cv2.imwrite(output_file, frame)
                # predict on a local image
                results = model(frame)

                # Check if the prediction contains both target classes with confidence greater than the threshold
                class_1_present = any(model.names[int(pred.cls)] == TARGET_CLASS_1 and pred.conf > CONFIDENCE_THRESHOLD_Pitch for pred in results[0].boxes)
                class_2_present = any(model.names[int(pred.cls)] == TARGET_CLASS_2 and pred.conf > CONFIDENCE_THRESHOLD_ScoreBar for pred in results[0].boxes)

                if class_1_present and class_2_present:
                    detection_active = True
                    desired_outputfile = f"{final_frames_directory}/desired_frame_{frame_count}.jpg"
                    cv2.imwrite(desired_outputfile, frame)
                    frames.append(frame)
                    if not new_ball_triggered:
                        new_ball_triggered = True
                        ball_counter += 1
                        pitch_data[ball_counter] = {'coords': None, 'first_frame': None}
                        print(f"New ball detected: Ball #{ball_counter}")

                    # Process current ball delivery
                    if new_ball_triggered:
                        # Detect pitch coordinates once per ball
                        if not pitch_data[ball_counter]['coords']:
                            temp_coords = pitch_coordinates(results)
                            if temp_coords and None not in temp_coords:
                                pitch_data[ball_counter]['coords'] = temp_coords
                                pitch_data[ball_counter]['first_frame'] = frame_count
                                print(f"Ball #{ball_counter} pitch coordinates: {temp_coords}")
                                # Save initial annotated frame
                                draw_pitch_length_annotations(frame, *temp_coords)
                                cv2.imwrite(os.path.join(main_output_directory, f"ball_{ball_counter}_pitch_frame.jpg"), frame)
                        _, pitch_y1, _, pitch_y2 = temp_coords
                        ball_result = detect_ball(ball_model, frame)
                        if ball_result[0].boxes and temp_coords:
                            # Get highest confidence ball detection
                            sorted_boxes = sorted(ball_result[0].boxes, key=lambda x: x.conf, reverse=True)
                            best_box = sorted_boxes[0]
                            x1, y1, x2, y2 = map(int, best_box.xyxy[0])
                            ball_y = (y1 + y2) // 2
                            # Bounce detection logic using initial pitch coordinates
                            if prev_ball_y is not None:
                                current_velocity = ball_y - prev_ball_y

                                if prev_velocity is not None:
                                    if prev_velocity > 0 and current_velocity < 0:  # Downward -> Upward
                                        pitch_y1 = temp_coords[1]
                                        pitch_y2 = temp_coords[3]

                                        if pitch_y1 <= ball_y <= pitch_y2:
                                            classification, desc = classify_bounce(ball_y, pitch_y1, pitch_y2)
                                            print(f"Bounce detected! Frame {frame_count}: {classification}")

                                            # Draw bounce-specific annotations
                                            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                                            cv2.putText(frame, f"{classification} ({desc})", (x1, y1-10),
                                                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

                                            last_bounce_info = (classification, desc)
                                prev_velocity = current_velocity
                            prev_ball_y = ball_y
                    additional_frames_count = 0  # Reset additional frames count
                    print(f"Frame {frame_count} has been extracted and saved as {desired_outputfile}")
                    out.write(frame)
                else:
                    if detection_active:
                        additional_frames_count += 1
                        if additional_frames_count <= 10:
                            desired_outputfile = f"{final_frames_directory}/desired_frame_{frame_count}.jpg"
                            cv2.imwrite(desired_outputfile, frame)
                            frames.append(frame)
                            print(f"Additional frame {frame_count} has been extracted and saved as {desired_outputfile}")
                        else:
                            detection_active = False  # Stop adding frames if 10 additional frames are added without detecting the pitch and score bar
                    else:
                        print(f"Frame {frame_count} does not contain both target classes {TARGET_CLASS_1} and {TARGET_CLASS_2}")

    cap.release()
    cv2.destroyAllWindows()

    if not frames:
        raise ValueError(f"No frames were extracted from the video {video_file}")

    # Convert the sequence of frames back into videos if the sequence is greater than 40 frames
    frame_ranges, video_count, video_path = convert_frames_to_videos(final_frames_directory, video_name, video_output_directory)
    return frames, frame_ranges, video_count, video_path

# 2. Define a function to preprocess the video
def preprocess_video(video_path, num_frames=16, frame_size=224):
    """
    Extract frames from a video, resize, normalize, and format for ViViT input.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video file: {video_path}")

    frames = []
    frame_count = 0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Total frames in video: {total_frames}")
    sampling_interval = max(total_frames // num_frames, 1)  # Evenly sample frames

    while frame_count < num_frames:
        frame_id = frame_count * sampling_interval
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id)  # Jump to the frame
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, (frame_size, frame_size))
        frames.append(frame)
        frame_count += 1
    if frame_count < num_frames:
        padding = num_frames - total_frames
        last_frame = frames[-1]
        frames.extend([last_frame] * padding)

    cap.release()

    if len(frames) == 0:
        raise ValueError("No frames were extracted from the video.")

    # Normalize frames using the processor's mean and std
    transform = Compose([
        ToTensor(),
        Normalize(mean=processor_load.image_mean, std=processor_load.image_std),
    ])
    frames = torch.stack([transform(frame) for frame in frames])  # Shape: (num_frames, C, H, W)
    return frames, total_frames




def classify_videos(video_name, main_output_directory):
    video_name = video_name + "_output"
    video_output_directory = os.path.join(main_output_directory, video_name.split('_output')[0])
    videos = [f for f in os.listdir(video_output_directory) if f.startswith(video_name) and f.endswith('.mp4')]
    videos.sort()
    
    results = []
    
    
    if not videos:
        logging.debug(f"No videos found for classification in {video_output_directory}")
        return results

    def process_video(video):
        video_path = os.path.join(video_output_directory, video)
        frames_VideoMAE, total_frame_count = preprocess_video(video_path, num_frames=16, frame_size=224)
        frames_VideoMAE = frames_VideoMAE.unsqueeze(0)
        frames_VideoMAE = frames_VideoMAE

        logits_VideoMAE = run_inference(model_load_VideoMAE, frames_VideoMAE)
        # logits_vivit = run_inference(model_load_vivit, frames_VideoMAE)
        class_label_VideoMAE, predicted_class_VideoMAE, confidence_VideoMAE = get_predicted_class(model_load_VideoMAE, logits_VideoMAE)
        # class_label_Vivit, predicted_class_Vivit, confidence_Vivit = get_predicted_class(model_load_vivit, logits_vivit)
        logging.debug(f"Classification results for {video}: {class_label_VideoMAE} with confidence {confidence_VideoMAE}")
        # logging.debug(f"Classification results for {video}: {class_label_Vivit} with confidence {confidence_Vivit}")

        frame_ranges_file = os.path.join(video_output_directory, f"{video_name.split('_output')[0]}_frame_ranges.txt")
        # with open(frame_ranges_file, 'r') as f:
        #     lines = f.readlines()
        #     for line in lines:
        #         if video in line:
        #             frame_range = line.split(":")[1].strip()
        #             print(frame_range)
        #             break

        results.append({
            "video": video,
            "predicted_class": predicted_class_VideoMAE,
            "class_label" : class_label_VideoMAE
            })
        # results.append({
        #     "video": video,
        #     "predicted_class": predicted_class_Vivit,
        #     "class_label" : class_label_Vivit
        #     })
   
        
    
    with ThreadPoolExecutor() as executor:
        executor.map(process_video, videos)
    return results
    
# def classify_videos(video_name, main_output_directory):
#     video_name = video_name+"_output"
#     video_output_directory = os.path.join(main_output_directory, video_name.split('_output')[0])
#     videos = [f for f in os.listdir(video_output_directory) if f.startswith(video_name) and f.endswith('.mp4')]
#     videos.sort()
#     results = []

#     if not videos:
#         print(f"No videos found for classification in {video_output_directory}")
#         return
#     for video in videos:
        
#         def process_video(video):
#             video_path = os.path.join(video_output_directory, video)

#             # Preprocess for ViViT
#             # frames_vivit = preprocess_video(video_path, num_frames=32, frame_size=224)
#             # frames_vivit = frames_vivit.unsqueeze(0)
#             # frames_vivit = frames_vivit.to(device)

#             # Preprocess for VideoMAE
#             frames_VideoMAE = preprocess_video(video_path, num_frames=16, frame_size=224)
#             frames_VideoMAE = frames_VideoMAE.unsqueeze(0)
#             frames_VideoMAE = frames_VideoMAE.to(device)
#             print("preprocessed")
#             # logits_vivit = run_inference(model_load_vivit, frames_vivit)
#             logits_VideoMAE = run_inference(model_load_VideoMAE, frames_VideoMAE)
#             print("Run Inference")
#             # class_label_vivit, predicted_class_vivit, confidence_vivit = get_predicted_class(model_load_vivit, logits_vivit)
#             class_label_VideoMAE, predicted_class_VideoMAE, confidence_VideoMAE = get_predicted_class(model_load_VideoMAE, logits_VideoMAE)
#             print("Get Prediction")
#             # print(f"Classification results for {video}: {class_label_vivit} with confidence {confidence_vivit}")
#             print(f"Classification results for {video}: {class_label_VideoMAE} with confidence {confidence_VideoMAE}")

#             frame_ranges_file = os.path.join(video_output_directory, f"{video_name.split('_output')[0]}_frame_ranges.txt")
#             with open(frame_ranges_file, 'r') as f:
#                 lines = f.readlines()
#                 for line in lines:
#                     if video in line:
#                         frame_range = line.split(":")[1].strip()
#                         break
#             results.append({
#                 "video": video,
#                 "predicted_class": class_label_VideoMAE,
#                 "frame_range": frame_range
#             })

    # with ThreadPoolExecutor() as executor:
    #     executor.map(process_video, video)
    # return results


if __name__ == "__main__":
    video_file = r'2_ball_match.mp4'
    main_output_directory = "extracted_videos_output"
    os.makedirs(main_output_directory, exist_ok=True)

    try:
        frames, frame_ranges, video_count, video_path= extract_frames(video_file, main_output_directory)
        classify_videos(os.path.splitext(os.path.basename(video_file))[0], main_output_directory)
    except ValueError as e:
        print(f"Error: {e}")