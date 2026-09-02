import os
import cv2
import pandas as pd
from ultralytics import YOLO
import cvzone
from twilio.rest import Client
import time
from dotenv import load_dotenv

load_dotenv()

# ---------------- TWILIO CONFIG ----------------
account_sid = os.getenv("TWILIO_ACCOUNT_SID", "")
auth_token = os.getenv("TWILIO_AUTH_TOKEN", "")
twilio_phone_number = os.getenv("TWILIO_FROM_NUMBER", "")
recipient_phone_number = os.getenv("TWILIO_TO_NUMBER", "")

client = Client(account_sid, auth_token) if account_sid and auth_token else None

def send_alert():
    if not client:
        print("Twilio credentials are not configured. Set TWILIO_ACCOUNT_SID and TWILIO_AUTH_TOKEN in your environment or .env file.")
        return

    try:
        client.messages.create(
            body='🚨 ALERT!!! Accident Detected! Send response team immediately!',
            from_=twilio_phone_number,
            to=recipient_phone_number
        )
        print("SMS alert sent!")
    except Exception as e:
        print("Error sending SMS:", e)

# ---------------- MODEL LOADING ----------------
model = YOLO("best.pt")

# Read class names
with open("classes.txt", "r") as f:
    class_list = f.read().split("\n")

# ---------------- VIDEO CAPTURE ----------------
cap = cv2.VideoCapture("cr.mp4")

cv2.namedWindow("RGB")

accident_detected = False
last_alert_time = 0
alert_cooldown = 5   # seconds

frame_skip = 3
count = 0

while True:
    ret, frame = cap.read()

    # ---------------- CHECK FOR EMPTY FRAME ----------------
    if not ret or frame is None:
        print("⚠️ No frame received. Ending video.")
        break

    count += 1
    if count % frame_skip != 0:
        continue

    # Safe resize
    try:
        frame = cv2.resize(frame, (1020, 500))
    except:
        print("⚠️ Resize failed, skipping frame.")
        continue

    # ---------------- YOLO PREDICTION ----------------
    results = model.predict(frame, verbose=False)

    if len(results[0].boxes) == 0:
        cv2.imshow("RGB", frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break
        continue

    # Convert detections to DataFrame
    px = pd.DataFrame(results[0].boxes.data).astype(float)

    accident_found = False

    for _, row in px.iterrows():
        x1, y1, x2, y2 = map(int, row[:4])
        class_id = int(row[5])
        class_name = class_list[class_id]

        # Draw bounding boxes
        if "accident" in class_name.lower():
            color = (0, 0, 255)  # RED
            accident_found = True
        else:
            color = (0, 255, 0)  # GREEN

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cvzone.putTextRect(frame, f"{class_name}", (x1, y1), 1, 1)

    # ---------------- ALERT MECHANISM ----------------
    current_time = time.time()

    if accident_found and (current_time - last_alert_time > alert_cooldown):
        send_alert()
        last_alert_time = current_time

    # ---------------- DISPLAY FRAME ----------------
    cv2.imshow("RGB", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
