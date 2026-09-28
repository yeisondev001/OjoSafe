"""Registro de infracciones de OjoSafe en SQLite."""

import os
import sqlite3
from datetime import datetime

DB_PATH = os.path.join("data", "ojosafe.db")


def _connect():
    os.makedirs("data", exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS violations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            fecha TEXT NOT NULL,
            tipo TEXT NOT NULL,
            confianza REAL NOT NULL,
            fuente TEXT NOT NULL,
            imagen TEXT NOT NULL
        )
        """
    )
    return conn


def log_violation(detail, image_path, source="", occurred_at=None):
    """detail: ej 'NO-Safety Vest 69%'. Inserta la infraccion en la BD."""
    conn = _connect()
    tipo, _, conf = detail.rpartition(" ")
    conf = float(conf.rstrip("%")) / 100 if conf else 0.0
    fecha = (occurred_at or datetime.now()).isoformat(timespec="seconds")
    conn.execute(
        "INSERT INTO violations (fecha, tipo, confianza, fuente, imagen) VALUES (?, ?, ?, ?, ?)",
        (fecha, tipo or detail, conf, str(source), image_path),
    )
    conn.commit()
    conn.close()


def resumen():
    """Devuelve (total, por tipo, hoy) para mostrar en consola."""
    conn = _connect()
    total = conn.execute("SELECT COUNT(*) FROM violations").fetchone()[0]
    hoy = conn.execute(
        "SELECT COUNT(*) FROM violations WHERE date(fecha) = date('now', 'localtime')"
    ).fetchone()[0]
    por_tipo = conn.execute(
        "SELECT tipo, COUNT(*) as n FROM violations GROUP BY tipo ORDER BY n DESC"
    ).fetchall()
    conn.close()
    return total, por_tipo, hoy


if __name__ == "__main__":
    total, por_tipo, hoy = resumen()
    print(f"Total infracciones: {total} (hoy: {hoy})")
    for tipo, n in por_tipo:
        print(f"  {tipo}: {n}")
