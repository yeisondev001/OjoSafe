"""SafeVision - Detección de EPP (casco, chaleco) con visión por computadora."""

import argparse


def main():
    parser = argparse.ArgumentParser(description="SafeVision - Detección de EPP")
    parser.add_argument("--source", default="0", help="Ruta de video o índice de webcam")
    parser.add_argument("--model", default="yolov8n.pt", help="Modelo YOLO a usar")
    args = parser.parse_args()

    print(f"[SafeVision] Fuente: {args.source} | Modelo: {args.model}")
    # TODO: cargar modelo YOLO y ejecutar inferencia
    print("Proyecto inicializado. Implementación pendiente.")


if __name__ == "__main__":
    main()