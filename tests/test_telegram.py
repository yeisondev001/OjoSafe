"""Prueba de alertas Telegram de SafeVision."""

import cv2
import numpy as np

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from telegram_alerts import alert


def main():
    img = np.full((240, 320, 3), (40, 40, 160), dtype=np.uint8)
    cv2.putText(img, "SafeVision TEST", (40, 130),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv2.imwrite("violations_test.jpg", img)
    ok = alert("violations_test.jpg", "Prueba de alerta SafeVision")
    print("Resultado:", "ENVIADO" if ok else "NO ENVIADO (revisa .env)")


if __name__ == "__main__":
    main()