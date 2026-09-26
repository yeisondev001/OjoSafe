"""OjoSafe - Deteccion de EPP (casco, chaleco) con vision por computadora."""

import argparse
import os
import time
from datetime import datetime

import cv2
from ultralytics import YOLO

from telegram_alerts import alert as send_alert
from database import log_violation

MODEL_PATH = os.path.join("models", "ppe_yolov8n.onnx")  # ONNX: ~50% mas rapido en CPU
MODEL_FALLBACK = os.path.join("models", "ppe_yolov8n.pt")
VIOLATIONS_DIR = "violations"

GREEN = (0, 200, 0)
RED = (0, 0, 220)
ORANGE = (0, 140, 255)
YELLOW = (0, 200, 255)
WHITE = (255, 255, 255)
DARK = (30, 30, 30)
FONT = cv2.FONT_HERSHEY_DUPLEX

VIOLATION_CLASSES = {"NO-Hardhat", "NO-Safety Vest"}
OK_CLASSES = {"Hardhat", "Safety Vest"}
PERSON_CLASSES = {"Person"}
VEST_CLASSES = {"Safety Vest", "NO-Safety Vest"}
# Clases que se ignoran por completo (no se dibujan ni alertan)
IGNORE_CLASSES = {"Mask", "NO-Mask", "Safety Cone", "machinery", "vehicle"}

# Texto en pantalla y en alertas (OpenCV no dibuja tildes ni emojis)
LABELS = {
    "Hardhat": "CON CASCO",
    "NO-Hardhat": "SIN CASCO",
    "Safety Vest": "CON CHALECO",
    "NO-Safety Vest": "SIN CHALECO",
    "Person": "Persona",
}

# Confianza minima para que una infraccion dispare alerta/captura
ALERT_CONF = 0.45
# Area minima del box (fraccion del frame) para ignorar detecciones diminutas
MIN_BOX_AREA = 0.003
# Frames consecutivos con infraccion antes de alertar (evita falsos avisos)
PERSISTENCE = 3


def label_scale(frame):
    """Tamano de letra proporcional al ancho del video (legible en el celular)."""
    return max(0.6, frame.shape[1] / 900)


def draw_mark(frame, ok, x, y, size):
    """Palomita (ok=True) o X (ok=False) dibujada con lineas."""
    if ok:
        pts = [(x, y + size // 2), (x + size // 3, y + size), (x + size, y)]
        cv2.line(frame, pts[0], pts[1], WHITE, 3, cv2.LINE_AA)
        cv2.line(frame, pts[1], pts[2], WHITE, 3, cv2.LINE_AA)
    else:
        cv2.line(frame, (x, y), (x + size, y + size), WHITE, 3, cv2.LINE_AA)
        cv2.line(frame, (x, y + size), (x + size, y), WHITE, 3, cv2.LINE_AA)


def draw_label(frame, text, x, y, color, mark=None):
    """Etiqueta con fondo solido; mark=True/False agrega palomita o X."""
    scale = label_scale(frame)
    (tw, th), _ = cv2.getTextSize(text, FONT, scale, 2)
    mark_w = th + 8 if mark is not None else 0
    y = max(y, th + 10)
    cv2.rectangle(frame, (x, y - th - 10), (x + tw + mark_w + 10, y + 4), color, -1)
    cv2.putText(frame, text, (x + 5, y - 4), FONT, scale, WHITE, 2, cv2.LINE_AA)
    if mark is not None:
        draw_mark(frame, mark, x + tw + 12, y - th - 4, th)


def draw_hud(frame, fps, solo_casco):
    w = frame.shape[1]
    cv2.rectangle(frame, (0, 0), (w, 34), DARK, -1)
    cv2.putText(frame, f"OjoSafe  |  FPS: {fps:.1f}", (10, 24),
                FONT, 0.6, WHITE, 1, cv2.LINE_AA)
    mode = "Modo: solo casco" if solo_casco else "Modo: casco + chaleco"
    (tw, _), _ = cv2.getTextSize(mode, FONT, 0.6, 1)
    cv2.putText(frame, mode, (w - tw - 10, 24), FONT, 0.6, WHITE, 1, cv2.LINE_AA)


def draw_compliance(frame, ok_count, violations):
    """Barra inferior: porcentaje de EPP correcto entre lo detectado."""
    h, w = frame.shape[:2]
    bar_h = 44
    cv2.rectangle(frame, (0, h - bar_h), (w, h), DARK, -1)

    total = ok_count + violations
    if total == 0:
        pct, color, text = 0.0, WHITE, "Cumplimiento: --"
    else:
        pct = ok_count / total
        color = GREEN if pct >= 0.8 else YELLOW if pct >= 0.5 else RED
        text = f"Cumplimiento: {pct:.0%}"

    (tw, th), _ = cv2.getTextSize(text, FONT, 0.8, 2)
    cv2.putText(frame, text, (10, h - (bar_h - th) // 2), FONT, 0.8, color, 2, cv2.LINE_AA)

    x0, x1 = 10 + tw + 20, w - 10
    y0, y1 = h - bar_h + 14, h - 14
    if x1 > x0:
        cv2.rectangle(frame, (x0, y0), (x1, y1), (90, 90, 90), -1)
        cv2.rectangle(frame, (x0, y0), (x0 + int((x1 - x0) * pct), y1), color, -1)


def save_violation(frame, detail):
    os.makedirs(VIOLATIONS_DIR, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = os.path.join(VIOLATIONS_DIR, f"violation_{ts}.jpg")
    cv2.imwrite(path, frame)
    print(f"[ALERTA] {detail} -> captura guardada en {path}")
    return path


def annotate_boxes(frame, results, model_names, ignore):
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
            if cls_name in ignore:
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
                label = LABELS.get(cls_name, cls_name)
                if cls_name in OK_CLASSES:
                    ok_count += 1
                    cv2.rectangle(frame, (x1, y1), (x2, y2), GREEN, 3)
                    draw_label(frame, label, x1, y1 + 30, GREEN, mark=True)
                elif cls_name in VIOLATION_CLASSES:
                    strong = conf >= ALERT_CONF
                    color = RED if strong else (0, 120, 180)
                    if strong:
                        violation_count += 1
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
                    draw_label(frame, label if strong else f"{label}?", x1, y1 + 30, color,
                               mark=False if strong else None)

    return ok_count, violation_count, persons


def run(source, conf_threshold, save_interval, solo_casco=False):
    model_path = MODEL_PATH if os.path.exists(MODEL_PATH) else MODEL_FALLBACK
    if not os.path.exists(model_path):
        raise SystemExit(f"No se encontro el modelo en {MODEL_PATH} ni {MODEL_FALLBACK}")

    model = YOLO(model_path)
    names = model.names
    # En modo solo casco el chaleco no se dibuja ni cuenta como infraccion
    ignore = IGNORE_CLASSES | VEST_CLASSES if solo_casco else IGNORE_CLASSES

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

    print("[OjoSafe] Corriendo. Presiona 'q' para salir.")

    while True:
        ok, frame = cap.read()
        if not ok:
            print("[OjoSafe] Fin del video/fuente.")
            break

        frame_idx += 1
        if frame_idx % frame_skip == 0:
            last_results = model(frame, conf=conf_threshold, imgsz=480, verbose=False)[0]
        results = last_results if last_results is not None else model(frame, conf=conf_threshold, verbose=False)[0]
        ok_count, viol_count, persons = annotate_boxes(frame, results, names, ignore)

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
                f"{LABELS[names[int(b.cls)]]} {float(b.conf):.0%}"
                for b in results.boxes
                if names[int(b.cls)] in VIOLATION_CLASSES - ignore and float(b.conf) >= ALERT_CONF
            )
            path = save_violation(frame, detail)
            log_violation(detail, path, source)
            send_alert(path, f"Infracción detectada: {detail}")
            can_alert = False
        fps = 0.9 * fps + 0.1 * (1.0 / max(now - prev_time, 1e-6))
        prev_time = now

        draw_hud(frame, fps, solo_casco)
        draw_compliance(frame, ok_count, viol_count)
        cv2.imshow("OjoSafe - Deteccion de EPP", frame)

        if cv2.waitKey(1 if source_is_camera else 30) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(description="OjoSafe - Deteccion de EPP")
    parser.add_argument("--source", default="0",
                        help="0/1 webcam, ruta de video, o URL RTSP")
    parser.add_argument("--conf", type=float, default=0.3,
                        help="Umbral de confianza (0-1)")
    parser.add_argument("--save-interval", type=float, default=5.0,
                        help="Segundos minimos entre capturas de infraccion")
    parser.add_argument("--solo-casco", action="store_true",
                        help="Revisar solo el casco (ignora el chaleco)")
    args = parser.parse_args()
    run(args.source, args.conf, args.save_interval, args.solo_casco)


if __name__ == "__main__":
    main()