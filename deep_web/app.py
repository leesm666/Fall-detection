import cv2
from ultralytics import YOLO
import numpy as np
from flask import Flask, Response, render_template, request, redirect, url_for
from flask_sqlalchemy import SQLAlchemy
from twilio.rest import Client
import threading
import time

app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///guardian.db'
db = SQLAlchemy(app)

class Guardian(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    number = db.Column(db.String(20), unique=True, nullable=False)

    def __init__(self, number):
        self.number = number

with app.app_context():
    db.create_all()

# Twilio 설정
account_sid = ''
auth_token = ''
twilio_number = ''
client = Client(account_sid, auth_token)

# YOLO 모델 로드
model = YOLO("D:\\fall_dataset\\runs\\detect\\train\\weights\\best.pt")

# 웹캠 캡처 객체 생성
cap = cv2.VideoCapture(0)

# 클래스별 색상 정의
colors = {
    0: (255, 0, 0),  # 걷기
    1: (0, 255, 0),  # 넘어짐
    2: (0, 0, 255),  # 앉기
}

last_fall_detection_time = 0
fall_detection_cooldown = 30  # 30초 쿨다운

def send_fall_alert(guardian_number):
    try:
        message = client.messages.create(
            body="낙상이 감지되었습니다. 확인이 필요합니다.",
            from_=twilio_number,
            to=guardian_number
        )
        print(f"Fall alert sent to {guardian_number}. Message SID: {message.sid}")
    except Exception as e:
        print(f"Failed to send message: {str(e)}")

def process_frame():
    global last_fall_detection_time
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture frame")
            break

        frame = cv2.resize(frame, dsize=(640, 360))
        results = model(frame, conf=0.85)

        annotated_frame = frame.copy()
        for result in results:
            boxes = result.boxes.cpu().numpy()
            
            for box in boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = float(box.conf)
                cls = int(box.cls)
                label = f"{model.names[cls]} {conf:.2f}"
                
                color = colors.get(cls, (255, 255, 255))
                
                
                
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(annotated_frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                print(f"Detected class: {cls}, Confidence: {conf}")
                
                # 낙상 감지 및 알림 
                if cls == 0 and conf >= 0.85:
                    print(f"Fall detected with confidence {conf}")
                    current_time = time.time()
                    print(f"Time since last detection: {current_time - last_fall_detection_time}")
                    if current_time - last_fall_detection_time > fall_detection_cooldown:
                        print("Cooldown period passed, attempting to send alert")
                        with app.app_context():
                            guardian = Guardian.query.first()
                            if guardian:
                                print(f"Guardian found: {guardian.number}")
                                send_fall_alert(guardian.number)
                                last_fall_detection_time = current_time
                            else:
                                print("No guardian number set")
                    else:
                        print("Still in cooldown period")

        ret, buffer = cv2.imencode('.jpg', annotated_frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    guardian = Guardian.query.first()
    if guardian:
        return render_template('index.html', guardian_number=guardian.number)
    else:
        return redirect(url_for('set_guardian'))

@app.route('/set_guardian', methods=['GET', 'POST'])
def set_guardian():
    if request.method == 'POST':
        number = request.form['number']
        guardian = Guardian.query.first()
        if guardian:
            guardian.number = number
        else:
            guardian = Guardian(number)
            db.session.add(guardian)
        db.session.commit()
        return redirect(url_for('index'))
    return render_template('set_guardian.html')

@app.route('/video_feed')
def video_feed():
    return Response(process_frame(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True)
