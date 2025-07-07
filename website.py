from flask import Flask, Response
import cv2
import threading

app = Flask(__name__)

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("❌ Kamera konnte nicht geöffnet werden!")
    exit()
else:
    print("✅ Kamera erfolgreich geöffnet.")

def generate_frames():
    while True:
        success, frame = cap.read()
        if not success:
            break
        else:
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/cam')
def cam():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/')
def index():
    return '''
    <html><body>
    <h1>Kamerahhhh</h1>
    <img src="/cam" width="640" height="480">
    </body></html>
    '''

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5005)
