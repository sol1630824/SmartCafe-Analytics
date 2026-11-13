import cv2
import time
from ultralytics import YOLO

# =====================================================================
# 1. CONFIGURACIÓN DEL SISTEMA (CALIBRACIÓN)
# =====================================================================
FUENTE_VIDEO = 0                # 0 para webcam en vivo (o cambia a "video.mp4" si usas archivo)
TIEMPO_LIMITE_ALERTA = 10.0     # Segundos máximos antes de activar alerta visual de demora

# Coordenadas (x1, y1, x2, y2) de la Zona de Atención / Mostrador
# Actualiza estos valores con el resultado de calibrar_roi.py
ROI_MOSTRADOR = (350, 180, 720, 520)

# Cargar modelo YOLOv8 Nano (ligero y rápido para CPU)
modelo = YOLO("yolov8n.pt")

# =====================================================================
# 2. VARIABLES DE ESTADO Y MÉTRICAS
# =====================================================================
clientes_unicos_totales = set()   # IDs únicos detectados durante toda la sesión
tiempos_inicio_mostrador = {}     # Diccionario {id_cliente: timestamp_entrada}
historial_tiempos_atencion = []   # Lista con tiempos completados de atención

def punto_en_roi(px, py, roi):
    """Verifica si el punto (px, py) está dentro del rectángulo ROI (x1, y1, x2, y2)."""
    x1, y1, x2, y2 = roi
    return x1 <= px <= x2 and y1 <= py <= y2

# =====================================================================
# 3. BUCLE PRINCIPAL DE PROCESAMIENTO DE VIDEO
# =====================================================================
cap = cv2.VideoCapture(FUENTE_VIDEO)

if not cap.isOpened():
    print(f"Error: No se pudo abrir la fuente de video '{FUENTE_VIDEO}'.")
    exit()

print("Iniciando SmartCafé Analytics... Presiona 'q' o 'ESC' en la ventana de video para salir.")

while True:
    ret, frame = cap.read()

    if not ret:
        if isinstance(FUENTE_VIDEO, str):
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue
        else:
            print("Error al leer fotograma de la webcam.")
            break

    # Estandarizar resolución para mantener consistencia con la calibración ROI
    frame = cv2.resize(frame, (1020, 600))
    tiempo_actual = time.time()

    # Detección + seguimiento (ByteTrack) solo para personas (classes=[0])
    # imgsz=320 optimiza la velocidad en computadoras sin GPU dedicada
    resultados = modelo.track(frame, persist=True, classes=[0], conf=0.45, imgsz=320, verbose=False)

    clientes_en_escena = 0
    clientes_en_mostrador_ahora = 0
    alerta_demora_activa = False
    ids_presentes_en_roi = set()

    if resultados[0].boxes is not None and resultados[0].boxes.id is not None:
        cajas = resultados[0].boxes.xyxy.cpu().numpy().astype(int)
        ids = resultados[0].boxes.id.cpu().numpy().astype(int)

        clientes_en_escena = len(ids)

        for box, track_id in zip(cajas, ids):
            x1, y1, x2, y2 = box
            clientes_unicos_totales.add(track_id)

            # Punto de referencia del cliente (centro del cuerpo para funcionar bien frente a cámara)
            cx = int((x1 + x2) / 2)
            cy = int((y1 + y2) / 2)

            en_zona = punto_en_roi(cx, cy, ROI_MOSTRADOR)

            if en_zona:
                ids_presentes_en_roi.add(track_id)
                clientes_en_mostrador_ahora += 1

                # Iniciar cronómetro si acaba de ingresar al mostrador
                if track_id not in tiempos_inicio_mostrador:
                    tiempos_inicio_mostrador[track_id] = tiempo_actual

                tiempo_espera = tiempo_actual - tiempos_inicio_mostrador[track_id]

                # Verificar si supera el tiempo límite de espera
                if tiempo_espera > TIEMPO_LIMITE_ALERTA:
                    color_caja = (0, 0, 255)  # Rojo (Alerta)
                    alerta_demora_activa = True
                else:
                    color_caja = (0, 255, 255)  # Amarillo (En atención)

                etiqueta = f"ID:{track_id} | Espera: {tiempo_espera:.1f}s"
            else:
                color_caja = (0, 255, 0)  # Verde (Fuera de mostrador)
                etiqueta = f"Cliente #{track_id}"

            # Dibujar cuadro delimitador y punto central del cliente
            cv2.rectangle(frame, (x1, y1), (x2, y2), color_caja, 2)
            cv2.circle(frame, (cx, cy), 6, color_caja, -1)
            cv2.putText(frame, etiqueta, (x1, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color_caja, 2)

    # Registrar tiempo total cuando un cliente sale del mostrador
    ids_que_salieron = [cid for cid in tiempos_inicio_mostrador if cid not in ids_presentes_en_roi]
    for cid in ids_que_salieron:
        duracion_total = tiempo_actual - tiempos_inicio_mostrador.pop(cid)
        if duracion_total >= 1.5:  # Ignorar cruces accidentales menores a 1.5s
            historial_tiempos_atencion.append(duracion_total)

    # =====================================================================
    # 4. INTERFAZ GRÁFICA / DASHBOARD EN PANTALLA
    # =====================================================================
    rx1, ry1, rx2, ry2 = ROI_MOSTRADOR
    color_roi = (0, 0, 255) if alerta_demora_activa else (255, 180, 0)
    cv2.rectangle(frame, (rx1, ry1), (rx2, ry2), color_roi, 2)
    cv2.putText(frame, "ZONA DE ATENCION (ROI)", (rx1, ry1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color_roi, 2)

    # Panel semitransparente de métricas
    overlay = frame.copy()
    cv2.rectangle(overlay, (15, 15), (410, 175), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)

    # Cálculo del tiempo promedio de atención
    if historial_tiempos_atencion:
        promedio_espera = sum(historial_tiempos_atencion) / len(historial_tiempos_atencion)
    elif tiempos_inicio_mostrador:
        tiempos_activos = [tiempo_actual - t for t in tiempos_inicio_mostrador.values()]
        promedio_espera = sum(tiempos_activos) / len(tiempos_activos)
    else:
        promedio_espera = 0.0

    # Renderizar métricas en el dashboard
    cv2.putText(frame, "SMARTCAFE ANALYTICS - MONITOREO", (25, 42),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.putText(frame, f"Flujo Total Clientes: {len(clientes_unicos_totales)}", (25, 72),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)
    cv2.putText(frame, f"Clientes en Local Ahora: {clientes_en_escena}", (25, 100),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
    cv2.putText(frame, f"En Mostrador (ROI): {clientes_en_mostrador_ahora}", (25, 128),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)
    cv2.putText(frame, f"Tiempo Prom. Atencion: {promedio_espera:.1f} seg", (25, 156),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 200, 100), 2)

    # Banner de alerta por demora
    if alerta_demora_activa:
        cv2.rectangle(frame, (430, 15), (1005, 70), (0, 0, 200), -1)
        cv2.putText(frame, f"ALERTA: DEMORA EN MOSTRADOR (> {int(TIEMPO_LIMITE_ALERTA)}s)", (445, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)

    cv2.imshow("SmartCafe Analytics", frame)

    # Salir al presionar 'q' o 'ESC'
    tecla = cv2.waitKey(1) & 0xFF
    if tecla == ord('q') or tecla == 27:
        break

cap.release()
cv2.destroyAllWindows()