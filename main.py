import cv2
import os
import pandas as pd
from ultralytics import YOLO
import xgboost as xgb
import numpy as np
import cvzone
import smtplib
from email.mime.text import MIMEText
from email.mime.image import MIMEImage
from email.mime.multipart import MIMEMultipart
from datetime import datetime
import json

# Load email configuration from config file
def load_config():
    try:
        with open('config.json', 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        print("Error: config.json not found. Please create a config file with your email settings.")
        return None

# Email configuration
config = load_config()
if config:
    SMTP_SERVER = config.get('SMTP_SERVER', 'smtp.gmail.com')
    SMTP_PORT = config.get('SMTP_PORT', 587)
    SENDER_EMAIL = config.get('SENDER_EMAIL')
    SENDER_PASSWORD = config.get('SENDER_PASSWORD')
    RECEIVER_EMAIL = config.get('RECEIVER_EMAIL')
else:
    # Fallback to environment variables if config file is not available
    import os
    SMTP_SERVER = os.getenv('SMTP_SERVER', 'smtp.gmail.com')
    SMTP_PORT = int(os.getenv('SMTP_PORT', 587))
    SENDER_EMAIL = os.getenv('SENDER_EMAIL')
    SENDER_PASSWORD = os.getenv('SENDER_PASSWORD')
    RECEIVER_EMAIL = os.getenv('RECEIVER_EMAIL')

def send_email_with_image(frame, timestamp):
    if not all([SMTP_SERVER, SMTP_PORT, SENDER_EMAIL, SENDER_PASSWORD, RECEIVER_EMAIL]):
        print("Error: Email configuration is incomplete. Please check your config.json or environment variables.")
        return

    try:
        # Create message container
        msg = MIMEMultipart()
        msg['Subject'] = f'Suspicious Activity Detected - {timestamp}'
        msg['From'] = SENDER_EMAIL
        msg['To'] = RECEIVER_EMAIL

        # Add text message with more details
        text = MIMEText(f'''
        Suspicious activity was detected at {timestamp}
        
        This is an automated alert from your security system.
        Please review the attached image for details.
        
        Best regards,
        Security System
        ''')
        msg.attach(text)

        # Save the frame temporarily with a unique filename
        temp_image_path = f'suspicious_activity_{timestamp.replace(":", "_")}.jpg'
        cv2.imwrite(temp_image_path, frame)

        # Attach the image
        with open(temp_image_path, 'rb') as f:
            img = MIMEImage(f.read())
            img.add_header('Content-Disposition', 'attachment', filename=temp_image_path)
            msg.attach(img)

        # Send the email
        with smtplib.SMTP(SMTP_SERVER, SMTP_PORT) as server:
            server.starttls()
            server.login(SENDER_EMAIL, SENDER_PASSWORD)
            server.send_message(msg)

        # Remove the temporary image
        os.remove(temp_image_path)
        print(f"Email sent successfully at {timestamp}")
    except Exception as e:
        print(f"Error sending email: {str(e)}")

# Define the path to the video file
video_path = "susp1.mp4"

def detect_shoplifting(video_path):
    # Load YOLOv8 model (replace with the actual path to your YOLOv8 model)
    model_yolo = YOLO('yolo11s-pose.pt')

    # Load the trained XGBoost model (replace with the actual path to your XGBoost model)
    model = xgb.Booster()
    model.load_model('trained_model.json')

    # Open the video
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("Error: Could not open video.")
        return

    print(f"Total Frames: {cap.get(cv2.CAP_PROP_FRAME_COUNT)}")

    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    frame_tot = 0
    count = 0
    last_email_time = 0  # To prevent too frequent emails

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            print("Warning: Frame could not be read. Skipping.")
            break

        count += 1
        if count % 5 != 0:  # Changed from 3 to 2 to process more frames
            continue

        # Resize the frame
        frame = cv2.resize(frame, (1018, 600))

        # Run YOLOv8 on the frame
        results = model_yolo(frame, verbose=False)

        # Visualize the YOLO results on the frame
        annotated_frame = results[0].plot(boxes=False)

        for r in results:
            bound_box = r.boxes.xyxy
            conf = r.boxes.conf.tolist()
            keypoints = r.keypoints.xyn.tolist()

            print(f'Frame {frame_tot}: Detected {len(bound_box)} bounding boxes')

            for index, box in enumerate(bound_box):
                if conf[index] > 0.55:
                    x1, y1, x2, y2 = box.tolist()

                    data = {}
                    for j in range(len(keypoints[index])):
                        data[f'x{j}'] = keypoints[index][j][0]
                        data[f'y{j}'] = keypoints[index][j][1]

                    df = pd.DataFrame(data, index=[0])
                    dmatrix = xgb.DMatrix(df)
                    sus = model.predict(dmatrix)
                    binary_predictions = (sus > 0.5).astype(int)
                    print(f'Prediction: {binary_predictions}')

                    if binary_predictions == 0:  # Suspicious
                        cv2.rectangle(annotated_frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)
                        cvzone.putTextRect(annotated_frame,f"{'Suspicious'}",(int(x1),(int(y1) +50)),1,1)
                        
                        # Send email if enough time has passed since last email (e.g., 30 seconds)
                        current_time = datetime.now()
                        if (current_time.timestamp() - last_email_time) > 30:
                            send_email_with_image(annotated_frame, current_time.strftime("%Y-%m-%d %H:%M:%S"))
                            last_email_time = current_time.timestamp()
                    else:  # Normal
                        cv2.rectangle(annotated_frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                        cvzone.putTextRect(annotated_frame,f"{'Normal'}",(int(x1),(int(y1) +50)),1,1)

        cv2.imshow('Frame', annotated_frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

# Call the function with the video path
detect_shoplifting(video_path)
