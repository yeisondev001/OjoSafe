# OjoSafe 👁️

Sistema de visión por computadora para seguridad industrial: detecta si los trabajadores usan su **EPP** (Equipo de Protección Personal): casco, chaleco, etc.

## Demo (MVP)

```powershell
# 1. Crear entorno virtual (solo la primera vez)
python -m venv venv
.\venv\Scripts\pip.exe install -r requirements.txt

# 2. Descargar el modelo PPE (solo la primera vez)
# https://huggingface.co/Hansung-Cho/yolov8-ppe-detection (best.pt -> models/ppe_yolov8n.pt)

# 3. Ejecutar
.\venv\Scripts\python.exe src\main.py --source 0          # webcam
.\venv\Scripts\python.exe src\main.py --source 0 --solo-casco  # webcam, solo revisa el casco (demo)
.\venv\Scripts\python.exe src\main.py --source video.mp4  # video de fabrica
.\venv\Scripts\python.exe src\main.py --source rtsp://usuario:pass@ip/stream  # camara IP (Fase 2)
```

Controles: **`q`** para salir.

### Alertas por Telegram

1. Crea tu bot con [@BotFather](https://t.me/BotFather) (`/newbot`) y copia `.env.example` a `.env` con tu `TELEGRAM_TOKEN` y `TELEGRAM_CHAT_ID`.
2. Pon el nombre y la descripcion de OjoSafe al bot (una sola vez):

```powershell
.\venv\Scripts\python.exe tests\telegram_bot_perfil.py
```

## Que detecta

| Color | Clase | Significado |
|-------|-------|-------------|
| 🟠 | Person | Persona detectada |
| 🟢 | Hardhat / Safety Vest | EPP correcto |
| 🔴 | NO-Hardhat / NO-Safety Vest | **Infraccion** -> alerta + captura |

- HUD con FPS y contadores OK / infracciones
- Espera 5 segundos continuos de infraccion antes de guardar y enviar una alerta.
- La alerta de Telegram y el registro SQLite incluyen la fecha/hora en que comenzo la infraccion.
- Envia una alerta por episodio; se rearma tras 5 segundos sin infracciones (configurable con `--save-interval`).

## Tecnologias

- Python 3.10+ / OpenCV / Ultralytics YOLOv8n
- Modelo PPE preentrenado (mAP@0.50: 0.744) — corre en CPU, no requiere GPU
- Compatible con webcam, videos y camaras RTSP

## Estructura

```
OjoSafe/
├── src/main.py    # App de deteccion
├── models/        # Modelos YOLO (.pt)
├── data/          # Videos de prueba
├── violations/    # Capturas de infracciones (auto)
├── docs/          # Documentacion
└── tests/
```

## Roadmap

- [x] MVP: deteccion en tiempo real (webcam/video)
- [ ] Multiples camaras RTSP con hilos
- [ ] Alertas por Telegram
- [ ] Registro de infracciones (SQLite)
- [ ] Dashboard web (FastAPI)
- [ ] Modelo propio entrenado con datos de la fabrica
