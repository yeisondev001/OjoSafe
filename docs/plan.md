# Documentación SafeVision

## Objetivo
Detectar el uso correcto de EPP en la fábrica mediante visión por computadora.

## Clases a detectar
| Clase | Descripción |
|-------|-------------|
| helmet | Casco puesto |
| no_helmet | Sin casco |
| vest | Chaleco puesto |
| no_vest | Sin chaleco |
| person | Persona detectada |

## Datasets recomendados
- Roboflow: buscar "Hard Hat Workers Dataset" o "PPE Detection"
- Kaggle: "Hard Hat Detection Dataset"

## Modelo
YOLOv8n (nano) para tiempo real en CPU; usar yolov8s/m si hay GPU.
