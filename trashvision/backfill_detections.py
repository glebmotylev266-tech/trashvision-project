"""Одноразовый пересчёт детекций для уже загруженных фото.

Раньше код оставлял только COCO-классы (bottle, cup, ...), поэтому у обученной
модели best.pt (класс "waste") все находки отбрасывались и в отчёты попадал 0.
Скрипт прогоняет модель по сохранённым оригиналам и заполняет поле detections.
"""
import os
import sqlite3

from ultralytics import YOLO

BASE = os.path.dirname(os.path.abspath(__file__))
DB = os.path.join(BASE, "database.db")

model = YOLO(os.path.join(BASE, "best.pt"))

con = sqlite3.connect(DB)
rows = con.execute("SELECT id, source_type, file_url, detections FROM reports").fetchall()

updated = 0
for rid, source_type, file_url, detections in rows:
    if detections and detections.strip() not in ("", "[]"):
        continue
    if source_type != "image":
        print(f"#{rid}: skip {source_type}")
        continue
    path = file_url.lstrip("/").replace("\\", os.sep)
    if not os.path.isabs(path):
        path = os.path.join(BASE, path)
    if not os.path.exists(path):
        print(f"#{rid}: файл не найден -> {path}")
        continue

    res = model(path, conf=0.35, verbose=False)[0]
    names = res.names or {}
    dets = []
    for box in res.boxes:
        cls_id = int(box.cls[0])
        dets.append({
            "class": names.get(cls_id, str(cls_id)),
            "confidence": round(float(box.conf[0]), 3),
            "bbox": [int(v) for v in box.xyxy[0].tolist()],
        })
    con.execute("UPDATE reports SET detections=? WHERE id=?", (str(dets), rid))
    updated += 1
    print(f"#{rid}: найдено {len(dets)}")

con.commit()
con.close()
print(f"обновлено записей: {updated}")
