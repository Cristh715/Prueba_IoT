import cv2
import torch
import numpy as np
import time
import threading
import sys
import requests

# Añade tu ruta de yolov5 al path
sys.path.append('/home/admin/yolov5')

from models.common import DetectMultiBackend
from utils.general import non_max_suppression, scale_boxes
from utils.augmentations import letterbox

# Configuración
FRAME_WIDTH = 640
FRAME_HEIGHT = 480
FRAME_SKIP_COUNT = 30 
YOLO_PATH = '/home/admin/yolov5/yolov5n.pt' 
CLASSES_VEHICLES = [2, 3, 5, 7]
CONF_THRESH = 0.4
SERVER_URL = 'http://192.168.18.176:5000/video_feed'
HEADERS = {'Content-Type': 'image/jpeg'}

# Inicializar modelo en CPU
device = torch.device('cpu')
model = DetectMultiBackend(YOLO_PATH, device=device)
model.eval()
model.conf = CONF_THRESH

# Captura de vídeo
cap = cv2.VideoCapture('plates.mp4')
cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

# Estado compartido
output_frame = None
lock = threading.Lock()
frame_index = 0
vehicle_detected = False
last_detected_time = 0
VEHICLE_TIMEOUT = 5  # segundos

def detectar_vehiculo(frame):
    img, _, _ = letterbox(frame, new_shape=320, stride=model.stride, auto=True)
    img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3xHxW
    img = np.ascontiguousarray(img)
    tensor = torch.from_numpy(img).to(device).float() / 255.0
    if tensor.ndim == 3:
        tensor = tensor.unsqueeze(0)

    pred = model(tensor)
    pred = non_max_suppression(pred, conf_thres=model.conf, iou_thres=0.45, classes=CLASSES_VEHICLES)

    hay_vehiculo = False
    for det in pred:
        if det is not None and len(det):
            # Escalado de cajas
            det[:, :4] = scale_boxes(tensor.shape[2:], det[:, :4], frame.shape).round()
            hay_vehiculo = True
    return hay_vehiculo

def enviar_stream(frame):
    _, buf = cv2.imencode('.jpg', frame)
    try:
        r = requests.post(SERVER_URL, data=buf.tobytes(), headers=HEADERS, timeout=2)
        if r.status_code == 200:
            print("Frame enviado correctamente.")
        else:
            print(f"Error HTTP al enviar: {r.status_code}")
    except Exception as e:
        print(f"Excepción enviando frame: {e}")

def capturar():
    global output_frame, frame_index, vehicle_detected, last_detected_time

    while True:
        ret, frame = cap.read()
        if not ret:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            time.sleep(0.1)
            continue

        frame_index += 1

        if frame_index % FRAME_SKIP_COUNT == 0:
            print(f"[DEBUG] Procesando frame {frame_index}")
            if detectar_vehiculo(frame):
                print("[DEBUG] Vehículo detectado")
                vehicle_detected = True
                last_detected_time = time.time()
                enviar_stream(frame)
        else:
            if vehicle_detected and (time.time() - last_detected_time) <= VEHICLE_TIMEOUT:
                cv2.putText(frame, "Transmitiendo (vehículo activo)", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            else:
                vehicle_detected = False

        with lock:
            output_frame = frame.copy()

        time.sleep(0.005)

if __name__ == '__main__':
    print("Iniciando captura de video...")
    hilo = threading.Thread(target=capturar, daemon=True)
    hilo.start()

    print("Captura en marcha. Ctrl+C para detener.")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nInterrupción por usuario. Cerrando...")
    finally:
        cap.release()
        print("Recursos de vídeo liberados.")
