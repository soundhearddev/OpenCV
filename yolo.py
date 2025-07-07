import cv2
from ultralytics import YOLO

# -------------------- Konfiguration --------------------
MODEL_PATH = "yolov8n.pt"        # YOLOv8-Modell (leicht & schnell)
CAM_INDEX = 0                    # 0 = integrierte Webcam, 1+ = externe Kamera
ALLOWED_CLASSES = None           # z.B. ['bottle', 'laptop'], oder None für alle

# -------------------- Initialisierung --------------------
model = YOLO(MODEL_PATH)
cap = cv2.VideoCapture(CAM_INDEX)

if not cap.isOpened():
    raise RuntimeError("❌ Kamera konnte nicht geöffnet werden.")

# -------------------- Schleife --------------------
while True:
    ret, frame = cap.read()
    if not ret:
        print("⚠️ Kein Bild von der Kamera.")
        break

    # Objekterkennung mit YOLOv8
    results = model(frame, verbose=False)[0]

    for box in results.boxes:
        cls_id = int(box.cls[0])
        label = model.names[cls_id]

        # Optional: Nur bestimmte Klassen anzeigen
        if ALLOWED_CLASSES and label not in ALLOWED_CLASSES:
            continue

        x1, y1, x2, y2 = map(int, box.xyxy[0])
        conf = float(box.conf[0]) * 100
        text = f"{label} {conf:.1f}%"

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, text, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # Bild anzeigen
    cv2.imshow("YOLOv8 Kamera-Objekterkennung", frame)

    # Beenden mit 'q'
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# -------------------- Aufräumen --------------------
cap.release()
cv2.destroyAllWindows()
