from ultralytics import YOLO
import cv2
import pandas as pd
from datetime import datetime
import os
import time
import winsound
from twilio.rest import Client

# ==============================
# 🔥 إعداد Twilio
# ==============================

account_sid = os.environ.get("TWILIO_ACCOUNT_SID")  # ✅ safe
auth_token = os.environ.get("TWILIO_AUTH_TOKEN")  # ✅ safe

client = Client(account_sid, auth_token)

def send_alert(violation, time_now):
    try:
        print("Sending WhatsApp alert...")

        message = client.messages.create(
            from_='whatsapp:+14155238886',
            body=f"""🚨 Safety Alert!

Violation: {violation}
Time: {time_now.strftime('%Y-%m-%d %H:%M:%S')}
Location: Camera 1
""",
            to='whatsapp:+201026432243'
        )

        print("✅ WhatsApp alert sent:", message.sid)

    except Exception as e:
        print("❌ WhatsApp Error:", e)

# ==============================
# 🔥 تحميل الموديلات
# ==============================
#person_model = YOLO("yolov8n.pt")
person_model = YOLO("models/person.pt")
fire_model = YOLO("models/fire.pt")
helmet_model = YOLO("models/helmet.pt")

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

# ==============================
# 📁 إعداد البيانات
# ==============================
os.makedirs("data/images", exist_ok=True)

file_name = "data/logs.csv"

if os.path.exists(file_name):
    df = pd.read_csv(file_name)
else:
    df = pd.DataFrame(columns=["time", "violation", "image"])

last_saved_time = 0

# ==============================
# 🔴 المناطق المحظورة
# ==============================
restricted_areas = []
drawing = False
start_point = None

def draw_area(event, x, y, flags, param):
    global drawing, start_point, restricted_areas

    if event == cv2.EVENT_RBUTTONDOWN:
        drawing = True
        start_point = (x, y)
        restricted_areas.clear()

    elif event == cv2.EVENT_RBUTTONUP:
        drawing = False
        end_point = (x, y)
        restricted_areas.append((start_point, end_point))
        print("New restricted area set:", start_point, end_point)

cv2.namedWindow("Detection")
cv2.setMouseCallback("Detection", draw_area)

print("Right click & drag to draw restricted areas")
print("Press 'd' to delete last area | 'q' to quit")

# ==============================
# 🔄 التشغيل
# ==============================
while True:
    ret, frame = cap.read()
    if not ret:
        print("Camera error")
        break

    person_results = person_model(frame)
    fire_results = fire_model(frame)
    helmet_results = helmet_model(frame)

    current_time = time.time()

    # ==============================
    # 🔥 Fire Detection
    # ==============================
    fire_detected = False

    for r in fire_results:
        for box in r.boxes:
            fire_detected = True
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 3)
            cv2.putText(frame, "FIRE", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    if fire_detected and (current_time - last_saved_time > 3):
        last_saved_time = current_time

        winsound.Beep(2000, 800)

        violation = "Fire Detected 🔥"
        time_now = datetime.now()

        print("🔥 FIRE DETECTED!")

        image_name = os.path.join(
            "data/images",
            f"{time_now.strftime('%Y%m%d_%H%M%S_%f')}.jpg"
        )

        cv2.imwrite(image_name, frame)

        df.loc[len(df)] = {
            "time": time_now,
            "violation": violation,
            "image": image_name
        }

        df.to_csv(file_name, index=False)

        # send_alert(violation, time_now)

    # ==============================
    # 🔴 رسم المناطق
    # ==============================
    for area in restricted_areas:
        cv2.rectangle(frame, area[0], area[1], (0, 0, 255), 2)

    # ==============================
    # 👤 Person Detection + Zones
    # ==============================
    for r in person_results:
        for box in r.boxes:
            cls = int(box.cls[0])

            if cls == 0:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cx = int((x1 + x2) / 2)
                cy = int((y1 + y2) / 2)

                inside_restricted = False

                for area in restricted_areas:
                    (ax1, ay1), (ax2, ay2) = area
                    if ax1 < cx < ax2 and ay1 < cy < ay2:
                        inside_restricted = True
                        break

                color = (0, 255, 255) if inside_restricted else (0, 255, 0)

                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.circle(frame, (cx, cy), 5, color, -1)

                if inside_restricted and (current_time - last_saved_time > 3):
                    last_saved_time = current_time

                    winsound.Beep(1000, 500)

                    violation = "Restricted Area دخول"
                    time_now = datetime.now()

                    print("🚨 Restricted Area Violation")

                    image_name = os.path.join(
                        "data/images",
                        f"{time_now.strftime('%Y%m%d_%H%M%S_%f')}.jpg"
                    )

                    cv2.imwrite(image_name, frame)

                    df.loc[len(df)] = {
                        "time": time_now,
                        "violation": violation,
                        "image": image_name
                    }

                    df.to_csv(file_name, index=False)

                if inside_restricted:
                    cv2.putText(frame, "WARNING!", (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    # ==============================
    # 🪖 Helmet Detection
    # ==============================
    no_helmet_detected = False

    for r in helmet_results:
        for box in r.boxes:
            cls = int(box.cls[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            if cls == 0:
                label = "Helmet"
                color = (0, 255, 0)
            else:
                label = "No Helmet"
                color = (0, 0, 255)
                no_helmet_detected = True

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    if no_helmet_detected and (current_time - last_saved_time > 3):
        last_saved_time = current_time

        winsound.Beep(1500, 700)

        violation = "No Helmet 🚨"
        time_now = datetime.now()

        print("🚨 NO HELMET!")

        image_name = os.path.join(
            "data/images",
            f"{time_now.strftime('%Y%m%d_%H%M%S_%f')}.jpg"
        )

        cv2.imwrite(image_name, frame)

        df.loc[len(df)] = {
            "time": time_now,
            "violation": violation,
            "image": image_name
        }

        df.to_csv(file_name, index=False)

        # send_alert(violation, time_now)

    # ==============================
    # 🎥 عرض
    # ==============================
    cv2.imshow("Detection", frame)

    key = cv2.waitKey(1) & 0xFF

    if key == ord('d'):
        if restricted_areas:
            restricted_areas.pop()
            print("Last area removed")

    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()