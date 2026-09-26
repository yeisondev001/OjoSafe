"""Alertas de SafeVision via Telegram (sin dependencias externas)."""

import os
import urllib.parse
import urllib.request

import cv2


def load_env(path=".env"):
    if not os.path.exists(path):
        return {}
    env = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line and "=" in line and not line.startswith("#"):
                key, _, value = line.partition("=")
                env[key.strip()] = value.strip()
    return env


def _telegram_url(token, method):
    return f"https://api.telegram.org/bot{token}/{method}"


def send_message(text):
    env = load_env()
    token = env.get("TELEGRAM_TOKEN", "")
    chat_id = env.get("TELEGRAM_CHAT_ID", "")
    if not token or not chat_id:
        print("[Telegram] No configurado (.env), solo mensaje en consola")
        return False
    data = urllib.parse.urlencode({"chat_id": chat_id, "text": text}).encode()
    try:
        with urllib.request.urlopen(_telegram_url(token, "sendMessage"), data, timeout=10) as r:
            return r.status == 200
    except Exception as exc:
        print(f"[Telegram] Error enviando mensaje: {exc}")
        return False


def send_photo(path, caption=""):
    env = load_env()
    token = env.get("TELEGRAM_TOKEN", "")
    chat_id = env.get("TELEGRAM_CHAT_ID", "")
    if not token or not chat_id:
        return False

    boundary = "----SafeVisionBoundary"
    file_bytes = open(path, "rb").read()

    parts = []
    fields = {"chat_id": chat_id, "caption": caption}
    for key, value in fields.items():
        parts.append(
            f"--{boundary}\r\nContent-Disposition: form-data; name=\"{key}\"\r\n\r\n{value}\r\n".encode()
        )
    parts.append(
        f"--{boundary}\r\nContent-Disposition: form-data; name=\"photo\"; filename=\"{os.path.basename(path)}\"\r\nContent-Type: image/jpeg\r\n\r\n".encode()
        + file_bytes + b"\r\n"
    )
    parts.append(f"--{boundary}--\r\n".encode())
    body = b"".join(parts)

    req = urllib.request.Request(
        _telegram_url(token, "sendPhoto"),
        data=body,
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
    )
    try:
        with urllib.request.urlopen(req, timeout=20) as r:
            return r.status == 200
    except Exception as exc:
        print(f"[Telegram] Error enviando foto: {exc}")
        return False


def alert(violation_path, detail, frame=None):
    caption = f"⚠️ SafeVision: {detail}\n{violation_path}"
    ok = send_photo(violation_path, caption)
    if not ok:
        send_message(caption)
    return ok