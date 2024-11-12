from flask import Flask, render_template, request,  send_file, session, redirect, url_for
from ultralytics import YOLO
from PIL import Image
from scipy.spatial.distance import cosine
import cv2
import os
import numpy as np
import torchreid
import time
import torch
import gdown

# Initialize Flask app
app = Flask(__name__)

# Set a secret key for session management
app.secret_key = os.urandom(24)

# Load your YOLOv8 model
model = YOLO('Large_v1_last.pt')

# Initialize the ReID model
reid_model = torchreid.models.build_model(
    name='osnet_x1_0', 
    num_classes=1000, 
    pretrained=True
).eval()

# Path to save output images and videos
output_dir = os.path.join(app.root_path, 'static', 'output')

# Ensure the output directory exists
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Route to render the homepage (upload page)
@app.route('/')
def home():
    return render_template('home.html')

@app.route('/restart', methods=['POST'])
def restart():
    # Method 1: Update file timestamp to trigger auto-reloader (for development)
    script_path = 'C:\\Users\\david\\yolov8_env\\web_app_botsort_with_timer_and_reid.py'
    os.utime(script_path, (time.time(), time.time()))
    
    # Method 2: Exit the process to trigger a full restart (for deployment)
    # os._exit(0)

    return "Restarting...", 200

@app.route('/upload', methods=['POST'])
def upload_file():

    # Get the uploaded file
    file = request.files['file']
    original_file_name = file.filename  # Get the original file name
    file_extension = os.path.splitext(original_file_name)[1].lower()
    base_name = os.path.splitext(original_file_name)[0]
    
    counts = { 'Cattle': 0, 'Chicken': 0, 'Goat': 0 }
    file_type = ''
    tracked_ids = set()  # Initialize the set for tracking unique object IDs

    if file_extension in ['.jpg', '.jpeg', '.png']:
        # Process image
        file_type = 'image'
        image = Image.open(file)
        
        # Convert PIL image to OpenCV format
        img_cv2 = np.array(image)
        img_cv2 = cv2.cvtColor(img_cv2, cv2.COLOR_RGB2BGR)
        
        # Run the YOLOv8 model on the image
        results = model(img_cv2)
        
        # Draw results on the image
        img_cv2, counts = draw_results_on_image(img_cv2, results, counts)
        
        # Save the output image (keep original name with same extension)
        output_path = os.path.join(output_dir, original_file_name)
        cv2.imwrite(output_path, img_cv2)

    elif file_extension in ['.mp4', '.avi']:
        # Process video
        file_type = 'video'
        input_video_path = os.path.join(output_dir, original_file_name)
        file.save(input_video_path)  # Save the uploaded video file
        frame_count = 0
        last_seen = {}
        last_confirmed = {}

        # Open video file
        cap = cv2.VideoCapture(input_video_path)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 3)
        fourcc = cv2.VideoWriter_fourcc(*'avc1')
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Ensure the output video is saved as .mov regardless of input extension
        output_video_path = os.path.join(output_dir, base_name + '.mov')
        out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                print("Error reading frame or end of video.")
                break
            # Process the frame...

            # Run the YOLOv8 model on each frame with tracking (BotSort)
            frame_count += 1
            results = model.track(frame, persist=True, tracker="botsort.yaml") # Use built-in tracker
            frame_with_results, counts, tracked_ids, last_seen, last_confirmed = draw_results_on_video(frame, results, counts, tracked_ids, frame_count, last_seen, last_confirmed)
            out.write(frame_with_results)  # Write the processed frame to output video
            print("frame_count =", frame_count, "\nlast_seen =", last_seen, "\nlast_confirmed =", last_confirmed)

        cap.release()
        out.release()

        # Update the original file name to point to the .mov file
        original_file_name = base_name + '.mov'

    elif file_extension in ['.mov','.avi']:
        # Process video
        file_type = 'video'
        input_video_path = os.path.join(output_dir, original_file_name)
        file.save(input_video_path) 
        frame_count = 0
        last_seen = {}
        last_confirmed = {}
        
        # Open video file
        cap = cv2.VideoCapture(input_video_path)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 3) 
        fourcc = cv2.VideoWriter_fourcc(*'avc1')  
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Ensure the output video is saved as .mp4 regardless of input extension
        output_video_path = os.path.join(output_dir, base_name + '.mp4')
        out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                print("Error reading frame or end of video.")
                break
            # Process the frame...

            # Run the YOLOv8 model on each frame with tracking (BotSort)
            frame_count += 1
            results = model.track(frame, persist=True, tracker="botsort.yaml") # Use built-in tracker
            frame_with_results, counts, tracked_ids, last_seen, last_confirmed = draw_results_on_video(frame, results, counts, tracked_ids, frame_count, last_seen, last_confirmed)
            out.write(frame_with_results)  # Write the processed frame to output video
            print("frame_count =", frame_count, "\nlast_seen =", last_seen, "\nlast_confirmed =", last_confirmed)

        cap.release()
        out.release()

        # Update the original file name to point to the .mp4 file
        original_file_name = base_name + '.mp4'

    else:
        return 'Unsupported file type', 200

    # Pass the updated file name and file type to the result template
    return render_template('result.html', 
                           original_file_name=original_file_name, 
                           counts=counts, 
                           file_type=file_type)


def draw_results_on_image(img_cv2, results, counts):
    # Extracting information from YOLO results
    detections = results[0].boxes  # Extract bounding boxes
    class_names = ['Cattle', 'Chicken', 'Goat']
    colors = [(0, 255, 0), (255, 0, 0), (0, 0, 255)]  # Colors for each class

    # Set thickness and font scale based on image size
    height, width, _ = img_cv2.shape
    thickness = max(1, width // 400) 
    font_scale = width / 1000
    box_font = width / 1500

    for box in detections:
        class_index = int(box.cls)  # YOLOv8 stores class index in the 'cls' attribute
        bbox = box.xyxy[0].numpy()  # Get the bounding box
        confidence = box.conf.item()  # Extract the confidence score
        x1, y1, x2, y2 = map(int, bbox)

        # Draw bounding box and label with confidence score
        color = colors[class_index]
        label = class_names[class_index]
        label_with_conf = f"{label} {confidence:.2f}" 
        
        cv2.rectangle(img_cv2, (x1, y1), (x2, y2), color, thickness)
        cv2.putText(img_cv2, f"{label_with_conf} ID-{counts[label]}", 
                    (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, box_font, color, thickness)

        # Increment the count for the class
        counts[class_names[class_index]] += 1

    # Add the counts to the image at the bottom center
    text = f'Cattle: {counts["Cattle"]}, Chicken: {counts["Chicken"]}, Goat: {counts["Goat"]}'
    text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)[0]
    text_x = (img_cv2.shape[1] - text_size[0]) // 2
    text_y = img_cv2.shape[0] - 20
    
    # Draw black background rectangle for the text
    cv2.rectangle(img_cv2, (text_x - 9, text_y - text_size[1] - 9), 
                  (text_x + text_size[0] + 9, text_y + 9), (0, 0, 0), -1)
    # Add the text on top of the rectangle
    cv2.putText(img_cv2, text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), thickness)

    return img_cv2, counts

def draw_results_on_video(img_cv2, results, counts, tracked_ids, frame_count, last_seen, last_confirmed, frames_to_confirm=30, embeddings={}, similarity_threshold=0.10):
    detections = results[0].boxes  # Extract bounding boxes
    class_names = ['Cattle', 'Chicken', 'Goat']
    colors = [(0, 255, 0), (255, 0, 0), (0, 0, 255)]  # Colors for each class

    for box in detections:
        class_index = int(box.cls)
        bbox = box.xyxy[0].numpy()
        confidence = box.conf.item()
        x1, y1, x2, y2 = map(int, bbox)

        # Generate embedding using ReID model
        cropped_obj = img_cv2[y1:y2, x1:x2]  # Crop the detected object
        cropped_obj_tensor = torch.tensor(cropped_obj).float().unsqueeze(0).permute(0, 3, 1, 2)  # Prepare for model
        embedding = reid_model(cropped_obj_tensor).detach().cpu().numpy().flatten()

        tracking_id = box.id if hasattr(box, 'id') else None
        if tracking_id is not None:
            tracking_id_value = int(tracking_id.item())
            
            # Initialize tracking ID in first frame
            if frame_count == 1:
                last_seen[tracking_id_value] = 1
                counts[class_names[class_index]] += 1
                last_confirmed[tracking_id_value] = last_seen[tracking_id_value]
                tracked_ids.add(tracking_id_value)

            elif tracking_id_value in last_seen:
                # Confirm if ID has been seen long enough and not already confirmed
                if (last_seen[tracking_id_value] >= frames_to_confirm and 
                    tracking_id_value not in last_confirmed):
                    # Confirm detection and save embedding
                    counts[class_names[class_index]] += 1
                    last_confirmed[tracking_id_value] = last_seen[tracking_id_value]  # Mark as confirmed
                    tracked_ids.add(tracking_id_value)  # Track confirmed ID
                    embeddings[tracking_id_value] = embedding  # Store confirmed embedding

                # Update last_seen count
                last_seen[tracking_id_value] += 1

            else:
                matched_id = None
                min_similarity = float('inf')

                # Check similarity to confirm re-appearance of known object
                for stored_id, stored_embedding in embeddings.items():
                    similarity = cosine(embedding, stored_embedding)
                    if similarity < similarity_threshold and similarity < min_similarity:
                        min_similarity = similarity
                        matched_id = stored_id
                        print("\nSimilarity =", similarity, "\nStored ID =", stored_id)

                # Assign matched ID or initialize new one if no match
                if matched_id is not None:
                    tracking_id_value = matched_id
                    last_seen[tracking_id_value] = last_seen.get(matched_id, 1)
                    print(f"ReID match found for ID {tracking_id_value} -> {matched_id}")

                else:
                    # Initialize tracking for a new ID
                    last_seen[tracking_id_value] = 1
                    embeddings[tracking_id_value] = embedding

        # Draw bounding box and label
        color = colors[class_index]
        label = class_names[class_index]
        label_with_conf = f"{label} {confidence:.2f} ID: {tracking_id_value}" if tracking_id else label
        cv2.rectangle(img_cv2, (x1, y1), (x2, y2), color, 2)
        cv2.putText(img_cv2, label_with_conf, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

    # Display counts on the video frame
    text = f'Cattle: {counts["Cattle"]}, Chicken: {counts["Chicken"]}, Goat: {counts["Goat"]}'
    text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)[0]
    text_x = (img_cv2.shape[1] - text_size[0]) // 2
    text_y = img_cv2.shape[0] - 20
    cv2.rectangle(img_cv2, (text_x - 10, text_y - text_size[1] - 10), 
                  (text_x + text_size[0] + 10, text_y + 10), (0, 0, 0), -1)
    cv2.putText(img_cv2, text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

    return img_cv2, counts, tracked_ids, last_seen, last_confirmed


# Route to download the processed image or video
@app.route('/download')
def download_file():
    file_name = request.args.get('file_name')
    path = os.path.join(output_dir, file_name)
    
    if os.path.exists(path):
        return send_file(path, as_attachment=True)
    else:
        return "File not found", 200

# Run the app
if __name__ == '__main__':
    app.run(debug=True)

