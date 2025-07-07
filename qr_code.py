import cv2
import webbrowser
        
# Initialize the webcam
cap = cv2.VideoCapture(0)

# Initialize QR Code detector
detector = cv2.QRCodeDetector()

decoded_text = None

print("Scanning for QR Code. Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Detect and decode the QR code
    data, bbox, _ = detector.detectAndDecode(frame)

    # If there is a QR code
    if data:
        decoded_text = data
        print(f"[QR Code Detected]: {decoded_text}")
        
        path = "templates/" + decoded_text
        
        webbrowser.open_new_tab(path)
        

        

        break
        




    # Display the camera feed
    cv2.imshow("QR Code Scanner", frame)

    # Exit on 'q' key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("Exiting without QR code.")
        break





cap.release()
cv2.destroyAllWindows()
