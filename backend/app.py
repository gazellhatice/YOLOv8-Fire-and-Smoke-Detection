import os
import sqlite3
from datetime import datetime
from flask_cors import CORS
from flask import Flask, request, jsonify, send_from_directory


ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(ROOT_DIR, "alarms.db")

ALARM_IMAGES_DIR = os.path.join(ROOT_DIR, "..", "alarms")

app = Flask(__name__)
CORS(app)  # React veya başka bir client'tan istek gelirse rahat olsun


def init_db():
    os.makedirs(ROOT_DIR, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute(
        """
        CREATE TABLE IF NOT EXISTS alarms (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            source TEXT,
            label TEXT,
            confidence REAL,
            area_ratio REAL,
            image_path TEXT
        )
        """
    )
    conn.commit()
    conn.close()


@app.route("/api/alarms", methods=["POST"])
def create_alarm():
    data = request.get_json(force=True)

    timestamp = data.get("timestamp") or datetime.utcnow().isoformat()
    source = data.get("source") or "unknown"
    label = data.get("label") or "unknown"
    confidence = float(data.get("confidence") or 0.0)
    area_ratio = float(data.get("area_ratio") or 0.0)
    image_path = data.get("image_path") or ""

    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute(
        """
        INSERT INTO alarms (timestamp, source, label, confidence, area_ratio, image_path)
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        (timestamp, source, label, confidence, area_ratio, image_path),
    )
    conn.commit()
    conn.close()

    return jsonify({"status": "ok"})


@app.route("/api/alarms", methods=["GET"])
def list_alarms():
    """
    Son 100 alarmı dönelim.
    """
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute(
        """
        SELECT id, timestamp, source, label, confidence, area_ratio, image_path
        FROM alarms
        ORDER BY id DESC
        LIMIT 100
        """
    )
    rows = c.fetchall()
    conn.close()

    alarms = [
        {
            "id": r[0],
            "timestamp": r[1],
            "source": r[2],
            "label": r[3],
            "confidence": r[4],
            "area_ratio": r[5],
            "image_path": r[6],
        }
        for r in rows
    ]
    return jsonify(alarms)


@app.route("/api/stats", methods=["GET"])
def stats():
    """
    Basit istatistik: etiket bazlı alarm sayıları.
    """
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute(
        """
        SELECT label, COUNT(*)
        FROM alarms
        GROUP BY label
        """
    )
    rows = c.fetchall()
    conn.close()

    stats = {label: count for label, count in rows}
    return jsonify(stats)

@app.route("/images/<path:filename>", methods=["GET"])
def get_alarm_image(filename):
    """
    YOLO tarafında kaydedilen alarm görüntülerini servis eder.
    realtime_detect.py, image_path olarak sadece dosya adını (ör: alarm_20250101_120000.jpg) gönderiyor.
    """
    return send_from_directory(ALARM_IMAGES_DIR, filename)

if __name__ == "__main__":
    init_db()
    app.run(host="0.0.0.0", port=5000, debug=True)
