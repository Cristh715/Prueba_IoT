import cv2
import torch
import numpy as np
import time
import threading
import sys
from flask import Flask, Response
sys.path.append('/home/admin/yolov5')

# Importar desde las carpetas de yolov5
from models.common import DetectMultiBackend
from utils.general import non_max_suppression, scale_boxes
from utils.augmentations import letterbox

# Configuración
FRAME_WIDTH = 640
FRAME_HEIGHT = 480
FRAME_SKIP_COUNT = 30  # Procesar 1 frame cada ~1 segundo a 30fps
YOLO_PATH = '/home/pi/mi_proyecto/yolov5/yolov5n.pt'  # Ruta modificada para Raspberry Pi
CLASES_VEHICULOS = [2, 3, 5, 7]  # coche, moto, autobús, camión
CONF_THRESH = 0.4

# Modelo
device = torch.device('cpu')  # Asegúrate de que tu Raspberry Pi tenga suficiente potencia
model = DetectMultiBackend(YOLO_PATH, device=device)
model.eval()
model.conf = CONF_THRESH

# Variables
cap = cv2.VideoCapture(0)  # Asegúrate de que tu cámara esté conectada correctamente
cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

output_frame = None
lock = threading.Lock()
frame_index = 0
vehiculo_detectado = False
last_detected_time = 0
VEHICLE_TIMEOUT = 5  # segundos

# Detección Vehículos
def detectar_vehiculo(frame):
    img_resized = letterbox(frame, new_shape=320, stride=model.stride, auto=True)[0]
    img_resized = img_resized.transpose((2, 0, 1))[::-1]
    img_resized = np.ascontiguousarray(img_resized)
    img_tensor = torch.from_numpy(img_resized).to(device).float() / 255.0
    if len(img_tensor.shape) == 3:
        img_tensor = img_tensor[None]

    pred = model(img_tensor)
    pred = non_max_suppression(pred, conf_thres=model.conf, iou_thres=0.45, classes=CLASES_VEHICULOS)

    hay_vehiculo = False
    for det in pred:
        if len(det):
            det[:, :4] = scale_boxes(img_tensor.shape[2:], det[:, :4], frame.shape).round()
            for *xyxy, conf, cls in det:
                label = f'{model.names[int(cls)]} {conf:.2f}'
                xyxy = tuple(map(int, xyxy))
                cv2.rectangle(frame, xyxy[:2], xyxy[2:], (0, 255, 0), 2)
                cv2.putText(frame, label, (xyxy[0], xyxy[1] - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            hay_vehiculo = True

    return hay_vehiculo, frame

# Hilo de captura
def capturar():
    global output_frame, frame_index, vehiculo_detectado, last_detected_time

    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        frame_index += 1
        if frame_index % FRAME_SKIP_COUNT == 0:
            hay_vehiculo, frame = detectar_vehiculo(frame)
            if hay_vehiculo:
                vehiculo_detectado = True
                last_detected_time = time.time()
            elif time.time() - last_detected_time > VEHICLE_TIMEOUT:
                vehiculo_detectado = False
        else:
            if vehiculo_detectado:
                # dibuja texto para debug
                cv2.putText(frame, "Transmitiendo (vehículo activo)", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        with lock:
            output_frame = frame.copy()

# MJPEG
app = Flask(__name__)

def generar_frames():
    global output_frame
    while True:
        with lock:
            if output_frame is None:
                continue
            _, buffer = cv2.imencode('.jpg', output_frame)
            frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(generar_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

# Main
if __name__ == '__main__':
    print("Iniciando captura...")
    t = threading.Thread(target=capturar)
    t.daemon = True
    t.start()

    print("Servidor Flask iniciado en http://0.0.0.0:5000/video_feed")
    app.run(host='0.0.0.0', port=5000, threaded=True)

    cap.release()
