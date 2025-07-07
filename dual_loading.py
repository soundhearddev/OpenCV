import cv2
from PIL import Image
import torch
from transformers import BlipProcessor, BlipForConditionalGeneration
import threading
import queue
import time

device = "cuda" if torch.cuda.is_available() else "cpu"

processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base", use_fast=True)
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base").to(device)
model.eval()

frame_queue = queue.Queue(maxsize=2)
result_queue = queue.Queue(maxsize=2)

def capture_frames():
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if not frame_queue.full():
            frame_queue.put(frame)
    cap.release()

def process_frames():
    while True:
        if not frame_queue.empty():
            frame = frame_queue.get()
            # Bild skalieren, PIL Konvertierung
            small_frame = cv2.resize(frame, (224, 224))
            pil_img = Image.fromarray(cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB))

            inputs = processor(pil_img, return_tensors="pt").to(device)

            with torch.no_grad():
                out = model.generate(**inputs)

            caption = processor.decode(out[0], skip_special_tokens=True)
            if not result_queue.full():
                result_queue.put((frame, caption))

# Threads starten
threading.Thread(target=capture_frames, daemon=True).start()
threading.Thread(target=process_frames, daemon=True).start()

fps = 0
frame_count = 0
start_time = time.time()

while True:
    if not result_queue.empty():
        frame, caption = result_queue.get()
        frame_count += 1
        cv2.putText(frame, f"Caption: {caption}", (10,25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
        cv2.putText(frame, f"FPS: {fps:.2f}", (10,50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
        cv2.imshow("Live Caption", frame)
        print(caption)

    # FPS berechnen alle 30 Frames
    if frame_count % 30 == 0 and frame_count > 0:
        fps = 30 / (time.time() - start_time)
        start_time = time.time()

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
