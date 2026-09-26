"""Cambia el nombre y la descripcion del bot de Telegram de OjoSafe.

Usa el TELEGRAM_TOKEN del .env. Ejecutar una sola vez:
    python tests/telegram_bot_perfil.py
"""

import json
import sys
import urllib.parse
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from telegram_alerts import _telegram_url, load_env

PERFIL = {
    "setMyName": {"name": "OjoSafe"},
    "setMyShortDescription": {
        "short_description": "👁️ OjoSafe: alertas de seguridad con vision por computadora."
    },
    "setMyDescription": {
        "description": (
            "👁️ OjoSafe vigila tus camaras con inteligencia artificial.\n\n"
            "Te avisa con foto cuando un trabajador no tiene su equipo de "
            "proteccion (casco, chaleco...) o cuando detecta algo sospechoso."
        )
    },
}


def main():
    token = load_env().get("TELEGRAM_TOKEN", "")
    if not token:
        raise SystemExit("Falta TELEGRAM_TOKEN en el archivo .env")

    for method, params in PERFIL.items():
        data = urllib.parse.urlencode(params).encode()
        try:
            with urllib.request.urlopen(_telegram_url(token, method), data, timeout=10) as r:
                ok = json.load(r).get("ok")
        except Exception as exc:
            ok = False
            print(f"[Telegram] {method}: error -> {exc}")
        print(f"{method}: {'OK' if ok else 'FALLO'}")


if __name__ == "__main__":
    main()
