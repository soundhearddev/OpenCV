import cv2
from PIL import Image
import torch
from transformers import BlipProcessor, BlipForConditionalGeneration
import time

device = "cuda" if torch.cuda.is_available() else "cpu"

processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base", use_fast=True)
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base").to(device)

def describe_pil_image(pil_img):
    inputs = processor(pil_img, return_tensors="pt").to(device)
    out = model.generate(**inputs)
    caption = processor.decode(out[0], skip_special_tokens=True)
    return caption

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)

frame_count = 0
last_time = time.time()
fps_display = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1

    # Prozessiere nur jeden 3. Frame
    if frame_count % 3 == 0:
        pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        caption = describe_pil_image(pil_img)
    # Sonst Caption behalten oder leer lassen
    # Für Demo hier einfach die letzte Caption benutzen
    else:
        caption = caption if 'caption' in locals() else ""

    # FPS berechnen
    if frame_count % 10 == 0:
        now = time.time()
        fps_display = 10 / (now - last_time)
        last_time = now

    # Caption + FPS anzeigen
    cv2.putText(frame, f"Caption: {caption}", (10,25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
    cv2.putText(frame, f"FPS: {fps_display:.2f}", (10,50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

    cv2.imshow("Live Caption", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
