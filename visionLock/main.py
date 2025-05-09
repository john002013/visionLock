import cv2
from ultralytics import YOLO
import math
import numpy as np
import serial
import sys
import time
import pickle
import os
from call_message import upload, send_message
from mtcnn.mtcnn import MTCNN
from keras_facenet import FaceNet


port = 'COM10'
ser = serial.Serial(port, 9600, timeout=1)
data = 1
data2 = 2
intruder = 0
last_time = 0
cooldown_time = 10

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
#ClassName = model.names

model = YOLO("yolo11n")
ClassName = model.names 
start_time = time.time()
Masking = cv2.imread("Mask.jpg")

face_detector = MTCNN()
face_Embedder = FaceNet()

Known_embeddings = []
Known_labels = []

#Loading image and creating embeddings
if os.path.exists("face_embeddings.pkl"):
    with open("face_embeddings.pkl", "rb") as f:
        Known_embeddings, Known_labels = pickle.load(f)
else:
#The path "known_faces" contains the image of the faces to be authenticated and registered
    for file in os.listdir("Known_faces"):
        if file.endswith(".jpg") or file.endswith(".png"):
            img = cv2.imread(os.path.join("Known_faces", file))
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            detections = face_detector.detect_faces(img_rgb)

            if detections:
                x, y, w, h = detections[0]['box']
                face = img_rgb[y:y+h, x:x+w]
                face = cv2.resize(face, (160, 160))
                embedding = face_Embedder.embeddings([face])[0]
                label = file.split('1')[0].split('2')[0].split('.')[0]
                Known_embeddings.append(embedding)
                Known_labels.append(label)

with open("face_embeddings.pkl", "wb") as f:
        pickle.dump((Known_embeddings, Known_labels), f)


while time.time() - start_time < 130:
    Success, img = cap.read()
    #mask = cv2.bitwise_and(img, Masking)
    #results = model(mask, stream=True)
    stage2_display = np.zeros_like(img)

    if ser.in_waiting > 0:
        ser.reset_input_buffer()
        arduino_msg = ser.readline().decode().strip()

        if arduino_msg == "Detect":
            Success, img = cap.read()
            mask = cv2.bitwise_and(img, Masking)
            results = model(mask, stream=True)
            

#1st Stage Human detection
            for r in results:
                boxes = r.boxes
                for box in boxes:
                    x1,y1,x2,y2 = box.xyxy[0]
                    x1, y1, x2, y2 = int(x1-40), int(y1-40), int(x2+40), int(y2+40)
                        
                    conf = math.ceil((box.conf[0]*100))/100
                    cls = int(box.cls[0])
                    currentClass = ClassName[cls]

                    if currentClass == "person" and conf > 0.4:
                        cv2.rectangle(img, (x1, y1), (x2, y2), (255, 165, 0), 1)
                        cv2.putText(img, f' {currentClass} {conf}', (x1 - 10, y1 - 10), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 165, 0), 1)
                        Cropped = mask[y1:y2, x1:x2]

#Stage 2 Validation
                        img_rgb2 = cv2.cvtColor(Cropped, cv2.COLOR_BGR2RGB)
                        detection2 = face_detector.detect_faces(img_rgb2)

                        if detection2:
                            cx, cy, cw, ch = detection2[0]['box']
                            face2 = img_rgb2[cy:cy+ch, cx:cx+cw]
                            face2 = cv2.resize(face2, (160, 160))
                            embedding2 = face_Embedder.embeddings([face2])[0]

                            def cosine_similarity(a, b):
                                return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

                            best_score = -1
                            best_label = "unknown"
                            threshold = 0.7

                            for idx, known_embedding in enumerate(Known_embeddings):
                                score = cosine_similarity(embedding2, known_embedding)
                                if score > best_score:
                                    best_score = score
                                    best_label = Known_labels[idx]

                            if best_score > threshold:
                                cv2.putText(Cropped, f"Access Granted ", (40, 30),
                                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 1)
                                time.sleep(2)
                                ser.write(f"{data}\n".encode())
                       

                            else:
                                cv2.putText(Cropped, f"Access denied ", (60, 40),
                                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 1)
                                #cv2.rectangle(Cropped, (cx, cy), (cw, ch), (255, 165, 0), 1)    
                                current_time = time.time()
                                if current_time - last_time > cooldown_time:
                                    intruder += 1
                                    filename = f"intruder{intruder}.jpg"
                                    time.sleep(2)
                                    cv2.imwrite(filename, Cropped)
                                    time.sleep(20)
                                    ser.write(f"{data2}\n".encode())
                                    image_url = upload(filename)
                                        
                                    if image_url:
                                        send_message(image_url)
                                        
                        if 'Cropped' in locals() and Cropped.shape[:2] != img.shape[:2]:
                            stage2_display = cv2.resize(Cropped, (img.shape[1], img.shape[0]))

            stacked = np.hstack((img, stage2_display))
            cv2.imshow("VisionLock Multistage Detection", stacked)
            cv2.waitKey(1)

cv2.destroyAllWindows()
ser.close()
sys.exit()
