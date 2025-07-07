import cv2
from ultralytics import YOLO

# YOLOv5 Modell laden (Standard: yolov8n.pt)
model = YOLO("yolov8n.pt")  # oder "yolov5s.pt" wenn du ein altes YOLOv5 Modell nutzt

# Haarcascades für Gesichtserkennung
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Kamera starten
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # YOLO-Objekterkennung
    yolo_results = model(frame, verbose=False)[0]

    for r in yolo_results.boxes:
        x1, y1, x2, y2 = map(int, r.xyxy[0])
        label = model.names[int(r.cls[0])]
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (0, 255, 0), 2)

    # Gesichtserkennung mit Haarcascade
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        cv2.putText(frame, "Gesicht", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (255, 0, 0), 2)

    # Zeige das Bild
    cv2.imshow("YOLO & Gesichtserkennung", frame)

    # Beenden mit Taste 'q'
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
