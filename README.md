# SmartCafé Analytics

Prototipo de visión artificial desarrollado en Python con OpenCV y YOLOv8 para el monitoreo en tiempo real del flujo de clientes, cronometraje de tiempos de atención en barra y alertas de demora en cafeterías.

## Características Principales
- **Detección y Seguimiento de Clientes:** Uso de YOLOv8 + ByteTrack para detectar personas y asignarles un ID único, contabilizando el flujo total sin duplicados.
- **Cronometraje en Zona de Interés (ROI):** Cálculo automático del tiempo de permanencia desde que un cliente ingresa al mostrador hasta que finaliza su atención.
- **Dashboard y Alertas Visuales:** Superposición de métricas en vivo (flujo total, clientes actuales, ocupación en barra, tiempo promedio de atención) y alerta visual ante tiempos excesivos de espera.

## Requisitos e Instalación
1. Clonar este repositorio y abrir la carpeta del proyecto.
2. Crear y activar un entorno virtual:
   ```bash
   python -m venv venv
   .\venv\Scripts\activate