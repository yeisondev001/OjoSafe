# SafeVision 🦺

Sistema de visión por computadora para seguridad industrial: detecta si los trabajadores usan su **EPP** (Equipo de Protección Personal): casco, chaleco, gafas, guantes, etc.

## Descripción

SafeVision analiza video en tiempo real (o imágenes) de cámaras en la fábrica y detecta:
- ✅ Trabajadores con casco
- ✅ Trabajadores con chaleco reflectante
- ⚠️ Trabajadores **sin** EPP → genera alerta

## Tecnologías

- Python 3.10+
- OpenCV (captura de video)
- YOLOv8 / YOLOv11 (detección de objetos)
- Ultralytics (entrenamiento e inferencia)

## Estructura del proyecto

```
SafeVision/
├── src/          # Código fuente
├── models/       # Modelos entrenados (.pt)
├── data/         # Dataset (imágenes + etiquetas)
├── docs/         # Documentación
├── tests/        # Pruebas
└── requirements.txt
```

## Instalación

```bash
git clone https://github.com/TU_USUARIO/SafeVision.git
cd SafeVision
pip install -r requirements.txt
```

## Uso

```bash
python src/main.py --source 0        # webcam
python src/main.py --source video.mp4
```

## Roadmap

- [ ] Detección con modelo preentrenado (COCO)
- [ ] Entrenar modelo con dataset de EPP
- [ ] Alertas sonoras / notificaciones
- [ ] Panel de monitoreo (dashboard)
- [ ] Guardar evidencia (capturas de infracciones)
