import cv2
import time
# Kamera öffnen (0 = Standardkamera)
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

if not cap.isOpened():
    print("Kamera konnte nicht geöffnet werden")
    exit()

# Videoeigenschaften holen
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)
if fps == 0:
    fps = 30  # Falls FPS nicht erkannt wird, Standard auf 30 setzen

# VideoWriter zum Speichern einrichten (Datei: output.avi, Codec, fps, Größe)
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
file = time.time()
file= "/home/admin/Videos/Records/" + str(file) + ".mp4"
out = cv2.VideoWriter(file, fourcc, fps, (frame_width, frame_height))

print("Starte Aufnahme. Drücke 'q' zum Beenden.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Kein Frame erhalten, breche ab.")
        break

    # Frame ins Fenster anzeigen
    cv2.imshow('Kamera Aufnahme', frame)

    # Frame in Datei speichern
    out.write(frame)

    # Warte auf Tastendruck
    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("Beenden gedrückt")
        break

# Ressourcen freigeben
cap.release()
out.release()
cv2.destroyAllWindows()
print(f"Video gespeichert als '{file}'")
