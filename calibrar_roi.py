import cv2

# 0 utiliza la webcam principal de la computadora
FUENTE_VIDEO = 0

cap = cv2.VideoCapture(FUENTE_VIDEO)

if not cap.isOpened():
    print("Error: No se pudo acceder a la webcam. Verifica que esté conectada y libre.")
    exit()

# Leer unos fotogramas iniciales para que la cámara ajuste su iluminación
for _ in range(10):
    ret, frame = cap.read()

cap.release()

if not ret:
    print("Error: No se pudo capturar imagen de la webcam.")
    exit()

# Redimensionar a un tamaño estándar para que coincida exactamente con main.py
frame = cv2.resize(frame, (1020, 600))

print("\nINSTRUCCIONES:")
print("1. Arrastra el mouse sobre la ventana para dibujar el rectángulo del mostrador.")
print("2. Presiona ENTER o ESPACIO para confirmar.\n")

roi = cv2.selectROI("Calibrador de ROI - SmartCafe", frame, fromCenter=False, showCrosshair=True)
cv2.destroyAllWindows()

# Convertir (x, y, ancho, alto) a coordenadas (x1, y1, x2, y2)
x1, y1, w, h = roi
x2, y2 = x1 + w, y1 + h

print("=" * 55)
print("Copia y pega esta línea en tu archivo main.py:")
print(f"ROI_MOSTRADOR = ({x1}, {y1}, {x2}, {y2})")
print("=" * 55 + "\n")