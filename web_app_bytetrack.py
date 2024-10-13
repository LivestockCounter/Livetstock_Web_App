from flask import Flask, render_template, request,  send_file, session, redirect, url_for
from ultralytics import YOLO
from PIL import Image
import cv2
import os
import numpy as np

# Initialize Flask app
app = Flask(__name__)

# Set a secret key for session management (this should be a random, secure value)
app.secret_key = os.urandom(24)

# Load your YOLOv8 model
model = YOLO('C:/Users/david/yolov8_env/webapp/version_1_last.pt')

# Path to save output images and videos
output_dir = os.path.join(app.root_path, 'static', 'output')

# Ensure the output directory exists
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Route to render the homepage (upload form)
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/cancel', methods=['POST'])
def cancel_processing():
    session['cancel'] = True  # Set the cancel flag in the session
    return 'Processing canceled', 200

@app.route('/upload', methods=['POST'])
def upload_file():
    session['cancel'] = False

    if 'file' not in request.files:
        return 'No file uploaded'
    
    # Get the uploaded file
    file = request.files['file']
    original_file_name = file.filename  # Get the original file name
    file_extension = os.path.splitext(original_file_name)[1].lower()
    base_name = os.path.splitext(original_file_name)[0]
    
    counts = { 'Cattle': 0, 'Chicken': 0, 'Goat': 0 }
    file_type = ''
    tracked_ids = set()  # Initialize the set for tracking unique object IDs

    # Check if the process was canceled early
    if session.get('cancel'):
        return 'Processing canceled early', 200

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
        
        # Open video file
        cap = cv2.VideoCapture(input_video_path)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 3)  # Set buffer size to 3 frames
        fourcc = cv2.VideoWriter_fourcc(*'avc1')  # H.264 codec
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
            
            # Check for cancel flag in each loop iteration
            if session.get('cancel'):
                cap.release()
                out.release()
                return 'Processing canceled', 200

            # Run the YOLOv8 model on each frame with tracking (Bytetrack)
            results = model.track(frame, persist=True, tracker="bytetrack.yaml") # Use built-in tracker
            frame_with_results, counts, tracked_ids = draw_results_on_video(frame, results, counts, tracked_ids)
            out.write(frame_with_results)  # Write the processed frame to output video

        cap.release()
        out.release()

        # Update the original file name to point to the .mov file
        original_file_name = base_name + '.mov'

    elif file_extension in ['.mov','.avi']:
        # Process video
        file_type = 'video'
        input_video_path = os.path.join(output_dir, original_file_name)
        file.save(input_video_path)  # Save the uploaded video file
        
        # Open video file
        cap = cv2.VideoCapture(input_video_path)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 3)  # Set buffer size to 3 frames
        fourcc = cv2.VideoWriter_fourcc(*'avc1')  # H.264 codec
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
            
            # Check for cancel flag in each loop iteration
            if session.get('cancel'):
                cap.release()
                out.release()
                return 'Processing canceled', 200

            # Run the YOLOv8 model on each frame with tracking (ByteTrack)
            results = model.track(frame, persist=True, tracker="bytetrack.yaml") # Use built-in tracker
            frame_with_results, counts, tracked_ids = draw_results_on_video(frame, results, counts, tracked_ids)
            out.write(frame_with_results)  # Write the processed frame to output video

        cap.release()
        out.release()

        # Update the original file name to point to the .mp4 file
        original_file_name = base_name + '.mp4'

    else:
        return 'Unsupported file type', 400

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

    for box in detections:
        class_index = int(box.cls)  # YOLOv8 stores class index in the 'cls' attribute
        bbox = box.xyxy[0].numpy()  # Get the bounding box
        confidence = box.conf.item()  # Extract the confidence score
        x1, y1, x2, y2 = map(int, bbox)

        # Draw bounding box and label with confidence score
        color = colors[class_index]
        label = class_names[class_index]
        label_with_conf = f"{label} {confidence:.2f}"  # Label with confidence score
        
        cv2.rectangle(img_cv2, (x1, y1), (x2, y2), color, 2)
        cv2.putText(img_cv2, f"{label_with_conf} ID-{counts[label]}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

        # Increment the count for the class
        counts[class_names[class_index]] += 1

    # Add the counts to the image at the bottom center
    text = f'Cattle: {counts["Cattle"]}, Chicken: {counts["Chicken"]}, Goat: {counts["Goat"]}'
    text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)[0]
    text_x = (img_cv2.shape[1] - text_size[0]) // 2
    text_y = img_cv2.shape[0] - 20
    # Draw black background rectangle for the text
    cv2.rectangle(img_cv2, (text_x - 10, text_y - text_size[1] - 10), 
                  (text_x + text_size[0] + 10, text_y + 10), (0, 0, 0), -1)
    # Add the text on top of the rectangle
    cv2.putText(img_cv2, text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

    return img_cv2, counts

def draw_results_on_video(img_cv2, results, counts, tracked_ids):
    # Extracting information from YOLO results
    detections = results[0].boxes  # Extract bounding boxes
    class_names = ['Cattle', 'Chicken', 'Goat']
    colors = [(0, 255, 0), (255, 0, 0), (0, 0, 255)]  # Colors for each class

    for box in detections:
        class_index = int(box.cls)  # YOLOv8 stores class index in the 'cls' attribute
        bbox = box.xyxy[0].numpy()  # Get the bounding box
        confidence = box.conf.item()  # Extract the confidence score
        x1, y1, x2, y2 = map(int, bbox)
        
        # Check for tracking ID (if any)
        tracking_id = box.id if hasattr(box, 'id') else None
        if tracking_id is not None:
            # Extract the actual ID value from the tensor
            tracking_id_value = int(tracking_id.item())  
            
            # Add the tracking ID to the tracked_ids set if it's new
            if tracking_id_value not in tracked_ids:
                tracked_ids.add(tracking_id_value)
                # Increment the count for the detected class
                counts[class_names[class_index]] += 1

        # Draw bounding box and label with confidence score
        color = colors[class_index]
        label = class_names[class_index]
        label_with_conf = f"{label} {confidence:.2f}"  # Label with confidence score
        
        if tracking_id is not None:
            label_with_conf += f" ID: {tracking_id_value}"  # Add the tracking ID to the label if it exists
        
        cv2.rectangle(img_cv2, (x1, y1), (x2, y2), color, 2)
        cv2.putText(img_cv2, f"{label_with_conf}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

    # Add the counts to the image at the bottom center
    text = f'Cattle: {counts["Cattle"]}, Chicken: {counts["Chicken"]}, Goat: {counts["Goat"]}'
    text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)[0]
    text_x = (img_cv2.shape[1] - text_size[0]) // 2
    text_y = img_cv2.shape[0] - 20
    # Draw black background rectangle for the text
    cv2.rectangle(img_cv2, (text_x - 10, text_y - text_size[1] - 10), 
                  (text_x + text_size[0] + 10, text_y + 10), (0, 0, 0), -1)
    # Add the text on top of the rectangle
    cv2.putText(img_cv2, text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

    return img_cv2, counts, tracked_ids


# Route to download the processed image or video
@app.route('/download')
def download_file():
    file_name = request.args.get('file_name')
    path = os.path.join(output_dir, file_name)
    
    if os.path.exists(path):
        return send_file(path, as_attachment=True)
    else:
        return "File not found", 404

# Run the app
if __name__ == '__main__':
    app.run(debug=True)
