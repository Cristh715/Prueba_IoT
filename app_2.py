import cv2
import torch
import numpy as np
import time
import threading
import sys
from flask import Flask, Response
import requests

sys.path.append('/home/admin/yolov5')

from models.common import DetectMultiBackend
from utils.general import non_max_suppression, scale_boxes
from utils.augmentations import letterbox

# Configuración
FRAME_WIDTH = 640
FRAME_HEIGHT = 480
FRAME_SKIP_COUNT = 30 
YOLO_PATH = '/home/admin/yolov5/yolov5n.pt' 
CLASES_VEHICULOS = [2, 3, 5, 7]
CONF_THRESH = 0.4
SERVER_URL = 'http://192.168.18.176:5000/video_feed'

# Modelo
device = torch.device('cpu')
model = DetectMultiBackend(YOLO_PATH, device=device)
model.eval()
model.conf = CONF_THRESH

# Variables
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

output_frame = None
lock = threading.Lock()
frame_index = 0
vehiculo_detectado = False
last_detected_time = 0
VEHICLE_TIMEOUT = 5

# Detección de vehículos
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
            hay_vehiculo = True

    return hay_vehiculo, frame

# Hilo de captura de frames
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
                enviar_stream(frame)
            elif time.time() - last_detected_time > VEHICLE_TIMEOUT:
                vehiculo_detectado = False
        else:
            if vehiculo_detectado:
                cv2.putText(frame, "Transmitiendo (vehículo activo)", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        with lock:
            output_frame = frame.copy()

# Enviar el fotograma al servidor
def enviar_stream(frame):
    _, buffer = cv2.imencode('.jpg', frame)
    frame_data = buffer.tobytes()
    try:
        response = requests.post(SERVER_URL, data=frame_data)
        if response.status_code == 200:
            print("Frame enviado correctamente al servidor.")
        else:
            print("Error al enviar el frame al servidor.")
    except Exception as e:
        print("Error enviando el frame:", e)

# Main
if __name__ == '__main__':
    print("Iniciando captura de video...")
    t = threading.Thread(target=capturar)
    t.daemon = True
    t.start()

    print("Captura en marcha. Esperando detecciones para enviar frames al servidor.")

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nDeteniendo la aplicación por interrupción del usuario...")
    except Exception as e:
        print(f"Se produjo un error inesperado en el hilo principal: {e}")
    finally:
        if cap.isOpened():
            cap.release()
            print("Recursos de video liberados.")
