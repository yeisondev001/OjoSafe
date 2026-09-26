"""SafeVision - Deteccion de EPP (casco, chaleco) con vision por computadora."""

import argparse
import os
import time
from datetime import datetime

import cv2
from ultralytics import YOLO

MODEL_PATH = os.path.join("models", "ppe_yolov8n.pt")
VIOLATIONS_DIR = "violations"

GREEN = (0, 200, 0)
RED = (0, 0, 220)
ORANGE = (0, 140, 255)
WHITE = (255, 255, 255)

VIOLATION_CLASSES = {"NO-Hardhat", "NO-Safety Vest"}


def draw_hud(frame, ok_count, violations, fps):
    h = frame.shape[0]
    color = GREEN if violations == 0 else RED
    cv2.rectangle(frame, (0, 0), (frame.shape[1], 34), (30, 30, 30), -1)
    cv2.putText(frame, f"SafeVision  |  FPS: {fps:.1f}", (10, 24),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, WHITE, 2)
    cv2.putText(frame, f"OK: {ok_count}  Infracciones: {violations}",
                (frame.shape[1] - 320, 24),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)


def save_violation(frame, detail):
    os.makedirs(VIOLATIONS_DIR, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = os.path.join(VIOLATIONS_DIR, f"violation_{ts}.jpg")
    cv2.imwrite(path, frame)
    print(f"[ALERTA] {detail} -> captura guardada en {path}")


def annotate_boxes(frame, results, model_names):
    ok_count = 0
    violation_count = 0
    persons = 0

    for box in results.boxes:
        cls_name = model_names[int(box.cls)]
        conf = float(box.conf)
        x1, y1, x2, y2 = map(int, box.xyxy[0])

        if cls_name == "Person":
            persons += 1
            color, label = ORANGE, f"Persona {conf:.0%}"
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, label, (x1, y1 - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        elif cls_name in ("Hardhat", "Safety Vest"):
            ok_count += 1
            color, label = GREEN, f"{cls_name} {conf:.0%}"
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, label, (x1, y1 - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        elif cls_name in VIOLATION_CLASSES:
            violation_count += 1
            color, label = RED, f"{cls_name} {conf:.0%}"
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, label, (x1, y1 - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    return ok_count, violation_count, persons


def run(source, conf_threshold, save_interval):
    if not os.path.exists(MODEL_PATH):
        raise SystemExit(f"No se encontro el modelo en {MODEL_PATH}")

    model = YOLO(MODEL_PATH)
    names = model.names

    if source.isdigit():
        cap = cv2.VideoCapture(int(source))
    else:
        cap = cv2.VideoCapture(source)

    if not cap.isOpened():
        raise SystemExit(f"No se pudo abrir la fuente de video: {source}")

    source_is_camera = source.isdigit()
    last_save = 0.0
    fps = 0.0
    prev_time = time.time()

    print("[SafeVision] Corriendo. Presiona 'q' para salir.")

    while True:
        ok, frame = cap.read()
        if not ok:
            print("[SafeVision] Fin del video/fuente.")
            break

        results = model(frame, conf=conf_threshold, verbose=False)[0]
        ok_count, viol_count, persons = annotate_boxes(frame, results, names)

        if viol_count > 0 and time.time() - last_save > save_interval:
            detail = ", ".join(
                names[int(b.cls)] for b in results.boxes
                if names[int(b.cls)] in VIOLATION_CLASSES
            )
            save_violation(frame, detail)
            last_save = time.time()

        now = time.time()
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
    parser.add_argument("--conf", type=float, default=0.4,
                        help="Umbral de confianza (0-1)")
    parser.add_argument("--save-interval", type=float, default=5.0,
                        help="Segundos minimos entre capturas de infraccion")
    args = parser.parse_args()
    run(args.source, args.conf, args.save_interval)


if __name__ == "__main__":
    main()