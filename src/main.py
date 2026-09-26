"""SafeVision - Deteccion de EPP (casco, chaleco) con vision por computadora."""

import argparse
import os
import time
from datetime import datetime

import cv2
from ultralytics import YOLO

from telegram_alerts import alert as send_alert
from database import log_violation

MODEL_PATH = os.path.join("models", "ppe_yolov8n.pt")
VIOLATIONS_DIR = "violations"

GREEN = (0, 200, 0)
RED = (0, 0, 220)
ORANGE = (0, 140, 255)
WHITE = (255, 255, 255)

VIOLATION_CLASSES = {"NO-Hardhat", "NO-Safety Vest"}
OK_CLASSES = {"Hardhat", "Safety Vest"}
PERSON_CLASSES = {"Person"}
# Clases que se ignoran por completo (no se dibujan ni alertan)
IGNORE_CLASSES = {"Mask", "NO-Mask", "Safety Cone", "machinery", "vehicle"}

# Confianza minima para que una infraccion dispare alerta/captura
ALERT_CONF = 0.45
# Area minima del box (fraccion del frame) para ignorar detecciones diminutas
MIN_BOX_AREA = 0.003
# Frames consecutivos con infraccion antes de alertar (evita falsos avisos)
PERSISTENCE = 3


def draw_label(frame, text, x, y, color):
    """Etiqueta con fondo solido para que siempre se lea."""
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
    y = max(y, th + 6)
    cv2.rectangle(frame, (x, y - th - 6), (x + tw + 4, y), color, -1)
    cv2.putText(frame, text, (x + 2, y - 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, WHITE, 2)


def draw_hud(frame, ok_count, violations, fps):
    color = GREEN if violations == 0 else RED
    cv2.rectangle(frame, (0, 0), (frame.shape[1], 34), (30, 30, 30), -1)
    cv2.putText(frame, f"SafeVision  |  FPS: {fps:.1f}", (10, 24),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, WHITE, 2)
    text = f"OK: {ok_count}  Infracciones: {violations}"
    (tw, _), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
    cv2.putText(frame, text, (frame.shape[1] - tw - 10, 24),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)


def save_violation(frame, detail):
    os.makedirs(VIOLATIONS_DIR, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = os.path.join(VIOLATIONS_DIR, f"violation_{ts}.jpg")
    cv2.imwrite(path, frame)
    print(f"[ALERTA] {detail} -> captura guardada en {path}")
    return path


def annotate_boxes(frame, results, model_names):
    ok_count = 0
    violation_count = 0
    persons = 0
    h, w = frame.shape[:2]
    min_area = MIN_BOX_AREA * h * w

    # Personas primero (caja grande), EPP despues (etiquetas dentro de su caja)
    for pass_name in ("person", "ppe"):
        for box in results.boxes:
            cls_name = model_names[int(box.cls)]
            conf = float(box.conf)
            if cls_name in IGNORE_CLASSES:
                continue
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            if (x2 - x1) * (y2 - y1) < min_area:
                continue

            if pass_name == "person":
                if cls_name in PERSON_CLASSES:
                    persons += 1
                    cv2.rectangle(frame, (x1, y1), (x2, y2), ORANGE, 2)
                    draw_label(frame, f"Persona {conf:.0%}", x1, y1 - 4, ORANGE)
            else:
                if cls_name in OK_CLASSES:
                    ok_count += 1
                    cv2.rectangle(frame, (x1, y1), (x2, y2), GREEN, 2)
                    draw_label(frame, f"{cls_name} {conf:.0%}", x1, y1 + 16, GREEN)
                elif cls_name in VIOLATION_CLASSES:
                    strong = conf >= ALERT_CONF
                    color = RED if strong else (0, 120, 180)
                    if strong:
                        violation_count += 1
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    mark = "" if strong else " ?"
                    draw_label(frame, f"{cls_name}{mark} {conf:.0%}", x1, y1 + 16, color)

    return ok_count, violation_count, persons


def run(source, conf_threshold, save_interval):
    if not os.path.exists(MODEL_PATH):
        raise SystemExit(f"No se encontro el modelo en {MODEL_PATH}")

    model = YOLO(MODEL_PATH)
    names = model.names

    if source.isdigit():
        cap = cv2.VideoCapture(int(source))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    else:
        cap = cv2.VideoCapture(source)

    if not cap.isOpened():
        raise SystemExit(
            f"No se pudo abrir la fuente: {source}\n"
            "  - Webcam: prueba --source 1 o cierra apps que la usen\n"
            "  - Video: verifica que la ruta exista\n"
            "  - RTSP: revisa usuario/contraseña/IP"
        )

    source_is_camera = source.isdigit()
    is_video_file = not source_is_camera and not str(source).lower().startswith("rtsp")
    fps = 0.0
    prev_time = time.time()
    frame_skip = 2  # procesar 1 de cada 2 frames (CPU sin GPU)
    frame_idx = 0
    last_results = None
    violation_streak = 0
    last_violation_time = 0.0
    can_alert = True  # 1 alerta por episodio de infraccion

    print("[SafeVision] Corriendo. Presiona 'q' para salir.")

    while True:
        ok, frame = cap.read()
        if not ok:
            print("[SafeVision] Fin del video/fuente.")
            break

        frame_idx += 1
        if frame_idx % frame_skip == 0:
            last_results = model(frame, conf=conf_threshold, imgsz=480, verbose=False)[0]
        results = last_results if last_results is not None else model(frame, conf=conf_threshold, verbose=False)[0]
        ok_count, viol_count, persons = annotate_boxes(frame, results, names)

        # La infraccion debe persistir varios frames para alertar
        now = time.time()
        if viol_count > 0:
            violation_streak += 1
            last_violation_time = now
        else:
            violation_streak = 0
            # El episodio termina cuando estuvo limpio un tiempo prolongado
            if now - last_violation_time > save_interval:
                can_alert = True

        if viol_count > 0 and violation_streak >= PERSISTENCE and can_alert:
            detail = ", ".join(
                f"{names[int(b.cls)]} {float(b.conf):.0%}"
                for b in results.boxes
                if names[int(b.cls)] in VIOLATION_CLASSES and float(b.conf) >= ALERT_CONF
            )
            path = save_violation(frame, detail)
            log_violation(detail, path, source)
            send_alert(path, f"⚠️ SafeVision — Infracción detectada: {detail}")
            can_alert = False
        fps = 0.9 * fps + 0.1 * (1.0 / max(now - prev_time, 1e-6))
        prev_time = now

        draw_hud(frame, ok_count, viol_count, fps)
        cv2.imshow("SafeVision - Deteccion de EPP", frame)

        if cv2.waitKey(1 if source_is_camera else 30) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(description="SafeVision - Deteccion de EPP")
    parser.add_argument("--source", default="0",
                        help="0/1 webcam, ruta de video, o URL RTSP")
    parser.add_argument("--conf", type=float, default=0.3,
                        help="Umbral de confianza (0-1)")
    parser.add_argument("--save-interval", type=float, default=5.0,
                        help="Segundos minimos entre capturas de infraccion")
    args = parser.parse_args()
    run(args.source, args.conf, args.save_interval)


if __name__ == "__main__":
    main()