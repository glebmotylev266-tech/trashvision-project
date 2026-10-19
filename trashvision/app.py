import csv
import io
import os
import sqlite3
import uuid
from datetime import datetime

import cv2
from flask import (Flask, flash, jsonify, redirect, render_template_string,
                   request, send_file, url_for)
from ultralytics import YOLO

# ---------- конфигурация ----------
app = Flask(__name__)
app.secret_key = "change-me-in-production"
app.config["MAX_CONTENT_LENGTH"] = 200 * 1024 * 1024

DB = "database.db"
UPLOAD_DIR = "static/uploads"
RESULT_DIR = "static/results"
LOG_DIR = "static/logs"

for _d in (UPLOAD_DIR, RESULT_DIR, LOG_DIR):
    os.makedirs(_d, exist_ok=True)

# ---------- YOLO ----------
# Классы берём из самой модели (best.pt -> {0: "waste"}),
# считаем все найденные объекты.
model = YOLO("best.pt")
MODEL_NAMES = getattr(model, "names", {}) or {}


# ---------- база ----------
def init_db():
    con = sqlite3.connect(DB)
    con.execute("""
        CREATE TABLE IF NOT EXISTS reports (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            created_at  TEXT,
            location    TEXT,
            lat         REAL,
            lng         REAL,
            source_type TEXT,
            file_url    TEXT,
            result_url  TEXT,
            detections  TEXT,
            log_url     TEXT,
            status      TEXT DEFAULT 'done'
        )
    """)
    con.commit()
    con.close()


init_db()


# ---------- детекция ----------
def detect_image(image_path: str):
    results = model(image_path, conf=0.35)[0]
    names = results.names or MODEL_NAMES
    detections = []
    for box in results.boxes:
        cls_id = int(box.cls[0])
        conf = float(box.conf[0])
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        detections.append({
            "class": names.get(cls_id, str(cls_id)),
            "confidence": round(conf, 3),
            "bbox": [x1, y1, x2, y2],
        })
    out_name = f"res_{uuid.uuid4().hex[:8]}.jpg"
    out_path = os.path.join(RESULT_DIR, out_name)
    results.save(filename=out_path)
    return detections, f"/{out_path}"


def detect_video(video_path: str, frame_step: int = 30):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    detections = []
    idx = 0

    out_name = f"res_{uuid.uuid4().hex[:8]}.mp4"
    out_path = os.path.join(RESULT_DIR, out_name)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    writer = cv2.VideoWriter(out_path, fourcc, fps, (w, h))

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if idx % frame_step == 0:
            res = model(frame, conf=0.35, verbose=False)[0]
            names = res.names or MODEL_NAMES
            for box in res.boxes:
                cls_id = int(box.cls[0])
                detections.append({
                    "frame": idx,
                    "time_sec": round(idx / fps, 2),
                    "class": names.get(cls_id, str(cls_id)),
                    "confidence": round(float(box.conf[0]), 3),
                })
            frame = res.plot()
        writer.write(frame)
        idx += 1

    cap.release()
    writer.release()

    return {
        "total_frames": total_frames,
        "fps": fps,
        "duration_sec": round(total_frames / fps, 2),
        "detections": detections,
        "result_video": f"/{out_path}",
    }


def write_log(detections, source: str, location: str):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_name = f"log_{datetime.now().strftime('%Y%m%d')}.txt"
    log_path = os.path.join(LOG_DIR, log_name)
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(f"[{ts}] source={source} location={location}\n")
        if not detections:
            f.write("  - объектов не обнаружено (0)\n")
        else:
            f.write(f"  - объектов найдено: {len(detections)}\n")
        for d in detections:
            f.write(f"  * {d.get('class')} conf={d.get('confidence')} "
                    f"bbox={d.get('bbox', d.get('time_sec'))}\n")
        f.write("\n")
    return f"/{log_path}"


# ---------- шаблоны ----------
BASE_HTML = """
<!DOCTYPE html>
<html lang="ru">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{% block title %}TrashVision{% endblock %}</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500;700&display=swap" rel="stylesheet">
<link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css"/>
<style>
:root{
  --bg:#0f0b08;
  --bg2:#160f0a;
  --panel:rgba(26,18,12,.82);
  --panel2:rgba(18,12,9,.9);
  --line:rgba(242,107,15,.22);
  --line2:rgba(242,107,15,.4);
  --text:#f0e4d4;
  --muted:#9c8570;
  --orange:#f26b0f;
  --orange2:#ff8a1f;
  --deep:#7a3a09;
  --green:#f26b0f;
  --red:#ff4a00;
  --blue:#ffb066;
  --accent:linear-gradient(135deg,#ff8a1f,#ef6207);
  --mono:'JetBrains Mono',Consolas,monospace;
  --sans:'Space Grotesk',system-ui,sans-serif;
}
*{box-sizing:border-box}
html,body{height:100%}
body{
  margin:0; font:14px/1.55 var(--sans);
  color:var(--text); min-height:100vh;
  background-color:var(--bg);
  background-image:
    linear-gradient(rgba(242,107,15,.055) 1px,transparent 1px),
    linear-gradient(90deg,rgba(242,107,15,.055) 1px,transparent 1px),
    radial-gradient(1100px 600px at 50% -15%,rgba(242,107,15,.12),transparent 65%),
    radial-gradient(900px 500px at 100% 110%,rgba(122,58,9,.18),transparent 60%);
  background-size:44px 44px,44px 44px,auto,auto;
}
body::after{
  content:""; position:fixed; inset:0; pointer-events:none; z-index:5;
  background:repeating-linear-gradient(0deg,rgba(0,0,0,.16) 0 1px,transparent 1px 3px);
  opacity:.35; mix-blend-mode:multiply;
}
::selection{background:rgba(242,107,15,.35);color:#fff}

.topbar{
  display:flex;align-items:center;gap:26px;padding:12px 26px;
  background:rgba(14,10,7,.86);backdrop-filter:blur(12px);
  border-bottom:1px solid var(--line);position:sticky;top:0;z-index:1000;
}
.topbar::after{content:"";position:absolute;left:0;right:0;bottom:-1px;height:1px;
  background:linear-gradient(90deg,transparent,rgba(242,107,15,.6),transparent);}
.brand{display:flex;gap:14px;align-items:center}
.logo{width:44px;height:44px;color:var(--orange);flex:0 0 auto;
  filter:drop-shadow(0 0 10px rgba(242,107,15,.55));}
.logo svg{display:block;width:100%;height:100%}
.brand h1{margin:0;font-size:17px;font-weight:700;letter-spacing:.22em}
.brand small{display:block;margin-top:2px;color:var(--muted);
  font:9px/1.2 var(--mono);text-transform:uppercase;letter-spacing:.28em}
.nav{display:flex;gap:8px;flex:1;justify-content:center}
.nav a{display:flex;align-items:center;gap:9px;padding:9px 18px;
  border:1px solid transparent;color:var(--muted);text-decoration:none;
  font:10px/1 var(--mono);text-transform:uppercase;letter-spacing:.2em;
  transition:.18s;border-radius:1px;}
.nav a:hover{color:var(--text);background:rgba(242,107,15,.06)}
.nav a.active{color:var(--orange2);border-color:var(--line2);
  background:rgba(242,107,15,.09);box-shadow:inset 0 0 22px rgba(242,107,15,.15)}
.nav .dot{width:6px;height:6px;background:var(--muted);display:inline-block;
  transform:rotate(45deg);transition:.18s}
.nav a.active .dot{background:var(--orange);box-shadow:0 0 8px var(--orange)}
.stats{display:flex;gap:16px;margin-left:auto}
.stat{text-align:right;font-family:var(--mono)}
.stat b{display:block;font-size:18px;color:var(--orange2);font-weight:600;
  text-shadow:0 0 14px rgba(242,107,15,.5)}
.stat span{font-size:9px;text-transform:uppercase;letter-spacing:.22em;color:var(--muted)}

.layout{display:grid;grid-template-columns:390px 1fr;height:calc(100vh - 69px)}
.sidebar{overflow-y:auto;padding:18px;display:flex;flex-direction:column;
  gap:14px;border-right:1px solid var(--line);
  background:rgba(12,8,6,.55);counter-reset:card}
#map{width:100%;height:100%;background:#0a0705}
.leaflet-tile-pane{filter:invert(.9) hue-rotate(180deg) sepia(.62) saturate(1.15) brightness(.72) contrast(1.08)}
.leaflet-control-zoom a{background:#150e09 !important;color:#ff8a1f !important;
  border-color:rgba(242,107,15,.35) !important}
.leaflet-control-zoom a:hover{background:#241408 !important}
.leaflet-control-attribution{background:rgba(10,7,5,.78) !important;color:#9c8570 !important;
  font:9px/1.5 var(--mono) !important}
.leaflet-control-attribution a{color:#ff8a1f !important}

.card{position:relative;background:var(--panel);border:1px solid var(--line);
  padding:18px;border-radius:2px;
  clip-path:polygon(0 0,calc(100% - 14px) 0,100% 14px,100% 100%,0 100%);
  box-shadow:0 14px 40px rgba(0,0,0,.45);counter-increment:card;}
.card::before{content:"0" counter(card);position:absolute;top:12px;right:14px;
  font:10px/1 var(--mono);color:rgba(242,107,15,.45);letter-spacing:.2em}
.card::after{content:"";position:absolute;left:0;top:0;width:56px;height:1px;
  background:var(--orange);opacity:.55}
.cardHead{display:flex;gap:12px;align-items:center;margin-bottom:16px}
.cardHead h2{margin:0;font-size:12px;font-weight:600;text-transform:uppercase;
  letter-spacing:.16em}
.cardHead small{display:block;color:var(--muted);font:9px/1.4 var(--mono);
  letter-spacing:.12em;margin-top:3px;text-transform:uppercase}
.icon{width:34px;height:34px;display:flex;align-items:center;justify-content:center;
  border:1px solid var(--line2);color:var(--orange2);flex:0 0 auto}
.icon svg{width:16px;height:16px}
.icon.photo,.icon.video,.icon.recent{background:rgba(242,107,15,.08);color:var(--orange2)}

input,button,label{font:inherit}
form{display:flex;flex-direction:column;gap:10px}
input[type="text"],input:not([type]){
  width:100%;padding:11px 13px;background:rgba(8,5,4,.72);
  border:1px solid rgba(242,107,15,.28);border-radius:1px;color:var(--text);
  font:12px/1.4 var(--mono);transition:.16s;letter-spacing:.04em;
}
input::placeholder{color:#6f5a47}
input:focus{outline:none;border-color:var(--orange);
  box-shadow:0 0 0 1px rgba(242,107,15,.35),inset 0 0 22px rgba(242,107,15,.14)}
.row2{display:grid;grid-template-columns:1fr 1fr;gap:10px}

.dropzone{display:flex;flex-direction:column;align-items:center;justify-content:center;
  gap:8px;padding:22px 12px;border:1px dashed rgba(242,107,15,.38);border-radius:1px;
  cursor:pointer;color:var(--muted);text-align:center;transition:.18s;
  font:10px/1.4 var(--mono);text-transform:uppercase;letter-spacing:.14em;
  background:rgba(8,5,4,.35)}
.dropzone:hover,.dropzone.hover{border-color:var(--orange);color:var(--text);
  background:rgba(242,107,15,.07);box-shadow:inset 0 0 26px rgba(242,107,15,.13)}
.dropzone input{display:none}
.dropIcon{width:34px;height:34px;border:1px solid var(--line2);border-radius:50%;
  color:var(--orange);display:flex;align-items:center;justify-content:center;
  box-shadow:0 0 16px rgba(242,107,15,.25)}
.dropIcon svg{width:16px;height:16px}

button.primary{padding:12px 16px;border:1px solid var(--orange);
  background:linear-gradient(135deg,#ff8a1f,#ef6207);color:#150c05;
  font:11px/1 var(--mono);font-weight:700;text-transform:uppercase;letter-spacing:.18em;
  cursor:pointer;border-radius:1px;
  clip-path:polygon(0 0,calc(100% - 12px) 0,100% 12px,100% 100%,0 100%);
  box-shadow:0 8px 26px rgba(242,107,15,.22);transition:.16s}
button.primary:hover{background:#ff9a33;box-shadow:0 10px 32px rgba(242,107,15,.4)}
button.primary:active{transform:translateY(1px)}

.recentList{display:flex;flex-direction:column;gap:8px;max-height:320px;overflow:auto}
.recentItem{padding:10px 12px;background:rgba(10,7,5,.6);
  border:1px solid rgba(242,107,15,.12);border-left:2px solid var(--orange);
  font:11px/1.5 var(--mono);border-radius:1px}
.recentItem.video{border-left-color:#ff3d00}
.recentTop{display:flex;justify-content:space-between;align-items:center}
.recentLoc{color:var(--text);margin-top:4px;font-family:var(--sans);
  font-size:12px;font-weight:500}
.recentMeta{color:var(--muted);font-size:10px;margin-top:2px;letter-spacing:.06em}
.recentLink{color:var(--orange2);text-decoration:none;font-size:10px;
  display:inline-block;margin-top:5px;text-transform:uppercase;letter-spacing:.12em}
.recentLink:hover{color:#ffb066}
.empty{color:var(--muted);font:10px/1.5 var(--mono);padding:16px;text-align:center;
  border:1px dashed rgba(242,107,15,.25);border-radius:1px;
  text-transform:uppercase;letter-spacing:.14em}
.badge{font:9px/1 var(--mono);padding:4px 8px;border:1px solid rgba(242,107,15,.4);
  color:var(--orange2);text-transform:uppercase;letter-spacing:.16em;border-radius:1px}
.badge.image{color:#ffb066;border-color:rgba(255,176,102,.45)}
.badge.video{color:#ff6a33;border-color:rgba(255,74,0,.5)}

.page{max-width:1200px;margin:0 auto;padding:36px 28px}
.pageTitle{margin:0 0 6px;font-size:22px;text-transform:uppercase;
  letter-spacing:.14em;padding-left:14px;position:relative}
.pageTitle::before{content:"";position:absolute;left:0;top:2px;bottom:2px;
  width:3px;background:var(--orange)}
.pageSub{color:var(--muted);margin:0 0 26px;font:11px/1.5 var(--mono);
  letter-spacing:.1em;text-transform:uppercase}
.pageHead{display:flex;justify-content:space-between;align-items:flex-start;
  margin-bottom:22px}
.exportBtn{padding:11px 18px;border:1px solid var(--orange);
  background:linear-gradient(135deg,#ff8a1f,#ef6207);color:#150c05;
  text-decoration:none;font:10px/1 var(--mono);font-weight:700;
  text-transform:uppercase;letter-spacing:.16em;
  clip-path:polygon(0 0,calc(100% - 12px) 0,100% 12px,100% 100%,0 100%);
  box-shadow:0 8px 26px rgba(242,107,15,.22)}

.logFile{background:var(--panel);border:1px solid var(--line);
  margin-bottom:16px;overflow:hidden;position:relative}
.logFileHead{display:flex;gap:10px;align-items:center;padding:12px 16px;
  border-bottom:1px solid var(--line);background:rgba(10,7,5,.55);
  font:11px/1 var(--mono);letter-spacing:.08em}
.logDot{width:7px;height:7px;background:var(--orange);transform:rotate(45deg);
  box-shadow:0 0 10px var(--orange)}
.logSize{color:var(--muted);font-size:10px;margin-left:auto;letter-spacing:.12em}
.logFile pre{margin:0;padding:16px;background:#0a0705;color:#d9c8b4;
  font:12px/1.65 var(--mono);overflow-x:auto;max-height:420px}

.kpiRow{display:flex;gap:14px;flex-wrap:wrap;margin-bottom:22px}
.kpi{flex:1;min-width:150px;padding:18px 20px;background:var(--panel);
  border:1px solid var(--line);border-left:2px solid var(--orange);
  display:flex;flex-direction:column;gap:6px}
.kpi b{font:600 28px/1 var(--mono);color:var(--orange2);
  text-shadow:0 0 18px rgba(242,107,15,.45)}
.kpi span{color:var(--muted);font:9px/1.3 var(--mono);
  text-transform:uppercase;letter-spacing:.2em}

.tableWrap{background:var(--panel);border:1px solid var(--line);overflow:hidden}
table{width:100%;border-collapse:collapse;font-size:12px}
th,td{padding:12px 16px;text-align:left;vertical-align:top}
th{background:rgba(10,7,5,.6);font:9px/1 var(--mono);text-transform:uppercase;
  letter-spacing:.2em;color:var(--orange2);font-weight:600;
  border-bottom:1px solid var(--line)}
tr{border-bottom:1px solid rgba(242,107,15,.1)}
tr:last-child{border-bottom:none}
tr:hover td{background:rgba(242,107,15,.05)}
td{font-family:var(--mono);font-size:11px}
td code{background:rgba(8,5,4,.7);padding:3px 8px;
  border:1px solid rgba(242,107,15,.18);font-size:10px;color:#ffb066;
  word-break:break-all;display:inline-block;max-width:420px;overflow:hidden;
  text-overflow:ellipsis;white-space:nowrap;vertical-align:middle}

.flashArea{position:fixed;right:24px;top:84px;z-index:2000;
  display:flex;flex-direction:column;gap:10px}
.toast{padding:12px 18px;background:rgba(16,11,8,.97);
  border:1px solid rgba(242,107,15,.45);color:var(--text);
  font:11px/1.5 var(--mono);letter-spacing:.06em;
  box-shadow:0 14px 40px rgba(0,0,0,.5);animation:slideIn .3s ease;min-width:260px}
.toast.success{border-color:rgba(242,107,15,.7);
  box-shadow:0 0 0 1px rgba(242,107,15,.25),0 0 30px rgba(242,107,15,.18)}
.toast.error{border-color:#ff3d00;color:#ffb3a0}
@keyframes slideIn{from{opacity:0;transform:translateX(20px)}to{opacity:1;transform:none}}

.leaflet-popup-content-wrapper{background:#150e09;color:var(--text);
  border:1px solid var(--line2);border-radius:2px;
  box-shadow:0 14px 40px rgba(0,0,0,.6)}
.leaflet-popup-tip{background:#150e09;border:1px solid var(--line2)}
.leaflet-popup-content{margin:12px 14px;font:11px/1.6 var(--mono)}
.leaflet-popup-content a{color:var(--orange2)}
.leaflet-container{background:#0a0705;font-family:var(--sans)}

.markerDot{width:16px;height:16px;background:var(--orange);transform:rotate(45deg);
  border:1px solid #ffb066;
  box-shadow:0 0 0 4px rgba(242,107,15,.14),0 0 20px rgba(242,107,15,.85);
  animation:markerPulse 2.4s infinite}
.markerDot.video{background:#ff3d00;border-color:#ff8a66;
  box-shadow:0 0 0 4px rgba(255,61,0,.14),0 0 20px rgba(255,61,0,.85)}
@keyframes markerPulse{0%,100%{opacity:1}50%{opacity:.5}}

.dashCard{background:var(--panel);border:1px solid var(--line);padding:18px 20px 14px;
  position:relative;margin-bottom:16px;
  clip-path:polygon(0 0,calc(100% - 14px) 0,100% 14px,100% 100%,0 100%)}
.dashCard::after{content:"";position:absolute;left:0;top:0;width:56px;height:1px;
  background:var(--orange);opacity:.55}
.dashHead{display:flex;align-items:center;justify-content:space-between;
  gap:16px;margin-bottom:10px;flex-wrap:wrap}
.dashHead h3{margin:0;font:600 11px/1 var(--mono);text-transform:uppercase;
  letter-spacing:.2em;color:var(--text)}
.legend{display:flex;gap:16px;flex-wrap:wrap;font:10px/1 var(--mono);
  color:var(--muted);text-transform:uppercase;letter-spacing:.12em}
.legend i{display:inline-block;width:8px;height:8px;margin-right:7px;
  transform:rotate(45deg)}
.legend .r i{background:var(--orange);box-shadow:0 0 8px rgba(242,107,15,.6)}
.legend .o i{background:#ffc46b;box-shadow:0 0 8px rgba(255,196,107,.5)}
.chartBox{position:relative;height:300px}

.donutRow{display:grid;grid-template-columns:repeat(3,1fr);gap:16px;
  margin-bottom:26px}
.donutCard{position:relative;background:var(--panel);border:1px solid var(--line);
  padding:22px 18px 18px;text-align:center;
  background-image:radial-gradient(120% 80% at 50% 0%,var(--tint,rgba(242,107,15,.1)),transparent 72%);
  clip-path:polygon(0 0,calc(100% - 14px) 0,100% 14px,100% 100%,0 100%)}
.donutWrap{position:relative;width:190px;height:190px;margin:0 auto}
.donutWrap canvas{position:absolute;inset:0}
.donutCenter{position:absolute;inset:0;display:flex;flex-direction:column;
  align-items:center;justify-content:center;pointer-events:none}
.donutCenter b{font:600 34px/1 var(--mono);color:var(--text);
  text-shadow:0 0 22px rgba(242,107,15,.35)}
.donutCenter span{font:9px/1 var(--mono);letter-spacing:.22em;
  text-transform:uppercase;color:var(--muted);margin-top:7px}
.donutLabel{margin-top:14px;font:600 11px/1 var(--mono);letter-spacing:.2em;
  text-transform:uppercase;color:var(--dcol,var(--orange2))}
.donutSub{margin-top:7px;font:9px/1.4 var(--mono);color:var(--muted);
  letter-spacing:.1em;text-transform:uppercase}

/* ---------- главная: три блока 01 / 02 / 03 ---------- */
.home{padding:24px 26px 34px;display:flex;flex-direction:column;gap:18px}
.tiles{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));
  gap:18px;align-items:stretch}
.tile{position:relative;display:flex;flex-direction:column;gap:16px;
  height:420px;padding:22px 20px 20px;
  background:
    radial-gradient(130% 90% at 0% 0%,rgba(242,107,15,.15),transparent 62%),
    var(--panel);
  border:1px solid var(--line);
  clip-path:polygon(0 0,calc(100% - 18px) 0,100% 18px,100% 100%,0 100%);
  box-shadow:0 20px 50px rgba(0,0,0,.45)}
.tile::after{content:"";position:absolute;left:0;top:0;width:74px;height:2px;
  background:var(--orange);box-shadow:0 0 18px rgba(242,107,15,.7)}
.tileTop{display:flex;align-items:flex-start;justify-content:space-between;gap:12px}
.tileNum{font:700 52px/0.85 var(--sans);letter-spacing:-.03em;color:var(--orange);
  text-shadow:0 0 34px rgba(242,107,15,.38)}
.tileTag{font:9px/1 var(--mono);letter-spacing:.3em;color:var(--muted);
  text-transform:uppercase;padding-top:8px}
.tile .cardHead{margin-bottom:0}
.tile form{flex:1;display:flex;flex-direction:column;gap:10px}
.tile .dropzone{flex:1;min-height:112px}
.tile button.primary{margin-top:auto}
.tile .recentList{flex:1;max-height:none;min-height:0;overflow:auto;
  -webkit-mask-image:linear-gradient(to bottom,#000 calc(100% - 22px),transparent);
  mask-image:linear-gradient(to bottom,#000 calc(100% - 22px),transparent)}
.recentList::-webkit-scrollbar{width:6px}
.recentList::-webkit-scrollbar-track{background:transparent}
.recentList::-webkit-scrollbar-thumb{background:rgba(242,107,15,.35);
  border-radius:3px}
.mapPanel{position:relative;height:56vh;min-height:400px;border:1px solid var(--line);
  overflow:hidden;
  clip-path:polygon(0 0,calc(100% - 18px) 0,100% 18px,100% 100%,0 100%)}
.mapPanel #map{height:100%}
.mapTag{position:absolute;top:12px;right:12px;z-index:600;display:flex;align-items:center;
  gap:8px;padding:7px 12px;background:rgba(10,7,5,.82);border:1px solid var(--line);
  font:9px/1 var(--mono);letter-spacing:.18em;text-transform:uppercase;color:var(--muted);
  pointer-events:none;backdrop-filter:blur(6px)}
.mapTag b{color:var(--orange2);font-weight:600}
.mapTag .dot{width:6px;height:6px;background:var(--orange);transform:rotate(45deg);
  box-shadow:0 0 8px var(--orange)}

/* ---------- переход между страницами: треугольники ---------- */
.tv-transition{position:fixed;inset:0;z-index:9000;pointer-events:none;
  overflow:hidden;background:transparent}
.tv-cells{position:absolute;inset:0;display:grid;
  grid-template-columns:repeat(6,1fr);grid-template-rows:repeat(5,1fr)}
.tv-transition .cell{position:relative;overflow:hidden}
.tv-transition .cell::before,
.tv-transition .cell::after{content:"";position:absolute;
  left:0;top:0;width:100%;height:100%;
  background:linear-gradient(135deg,#ff8a1f,#e66005 60%,#c94b00);
  transition:transform .6s cubic-bezier(.76,0,.24,1);
  transition-delay:var(--d,0ms);will-change:transform}
.tv-transition .cell::before{clip-path:polygon(0 0,100% 0,100% 100%);
  transform:translate3d(104%,-104%,0)}
.tv-transition .cell::after{clip-path:polygon(0 0,0 100%,100% 100%);
  background:linear-gradient(135deg,#ef6c05,#c94b00 70%,#9c3800);
  transform:translate3d(-104%,104%,0)}
.tv-transition .cell.alt::before{
  background:linear-gradient(135deg,#ef6c05,#c94b00 70%,#9c3800)}
.tv-transition .cell.alt::after{
  background:linear-gradient(135deg,#ff8a1f,#e66005 60%,#c94b00)}
.tv-transition:not(.done) .cell::before,
.tv-transition:not(.done) .cell::after{transform:translate3d(0,0,0)}
.tv-transition.idle{visibility:hidden}
.tv-mark{position:absolute;inset:0;display:flex;flex-direction:column;
  align-items:center;justify-content:center;gap:16px;opacity:0;transform:scale(.94);
  transition:opacity .3s .2s,transform .45s .2s}
.tv-transition:not(.done) .tv-mark{opacity:1;transform:none}
.tv-mark svg{width:88px;height:88px;color:#1a0d05;
  filter:drop-shadow(0 6px 18px rgba(0,0,0,.25))}
.tv-mark .tv-tri{transform-box:fill-box;transform-origin:center;
  transition:transform .45s cubic-bezier(.2,.7,.2,1)}
.tv-transition:not(.done) .tv-mark .tv-tri{
  animation:tvTriSpin 1s linear infinite}
@keyframes tvTriSpin{
  0%{transform:rotate(0deg) scale(.9)}
  50%{transform:rotate(180deg) scale(1.12)}
  100%{transform:rotate(360deg) scale(.9)}
}
.tv-mark b{font:700 12px/1 var(--sans);letter-spacing:.42em;color:#1a0d05;
  text-transform:uppercase;padding-left:.42em}

@media (max-width:900px){
  .donutRow{grid-template-columns:1fr}
  .chartBox{height:240px}
  .home{padding:16px}
  .tiles{grid-template-columns:1fr}
  .tile{height:auto;min-height:0}
  .mapPanel{height:52vh;min-height:320px}
}
@media (max-width:900px){
  .layout{grid-template-columns:1fr;height:auto}
  .sidebar{order:2;border-right:none;border-top:1px solid var(--line)}
  #map{order:1;height:52vh}
  .nav{display:none}
  .stats{gap:12px}
}
</style>
<noscript><style>.tv-transition{display:none}</style></noscript>
</head>
<body>

<header class="topbar">
  <div class="brand">
    <div class="logo">
      <svg viewBox="0 0 100 100" role="img" aria-label="TrashVision">
        <g fill="none" stroke="currentColor" stroke-width="7">
          <circle cx="50" cy="50" r="35"/>
          <path d="M50 2v13M50 85v13M2 50h13M85 50h13"/>
        </g>
        <path d="M50 24 78 73H22z" fill="currentColor"/>
        <ellipse cx="50" cy="56" rx="15" ry="9" fill="#0f0b08"/>
        <circle cx="50" cy="56" r="5" fill="currentColor"/>
      </svg>
    </div>
    <div>
      <h1>TRASHVISION</h1>
      <small>детектор незаконных свалок</small>
    </div>
  </div>
  <nav class="nav">
    <a href="/" class="{% if request.path == '/' %}active{% endif %}">
      <span class="dot"></span> Карта
    </a>
    <a href="/logs" class="{% if request.path == '/logs' %}active{% endif %}">
      <span class="dot"></span> Логи
    </a>
    <a href="/reports" class="{% if request.path == '/reports' %}active{% endif %}">
      <span class="dot"></span> Отчёты
    </a>
  </nav>
  <div class="stats">
    <div class="stat"><b>{{ nav_total }}</b><span>всего</span></div>
    <div class="stat"><b>{{ nav_images }}</b><span>фото</span></div>
    <div class="stat"><b>{{ nav_videos }}</b><span>видео</span></div>
  </div>
</header>

<div class="flashArea" id="flashArea">
  {% with msgs = get_flashed_messages(with_categories=true) %}
    {% for cat, m in msgs %}
      <div class="toast {{ cat }}">{{ m }}</div>
    {% endfor %}
  {% endwith %}
</div>

<div class="tv-transition" id="tvTransition" aria-hidden="true">
  <div class="tv-cells"></div>
  <div class="tv-mark">
    <svg viewBox="0 0 100 100">
      <g fill="none" stroke="currentColor" stroke-width="7">
        <circle cx="50" cy="50" r="35"/>
        <path d="M50 2v13M50 85v13M2 50h13M85 50h13"/>
      </g>
      <g class="tv-tri">
        <path d="M50 24 78 73H22z" fill="currentColor"/>
        <ellipse cx="50" cy="56" rx="15" ry="9" fill="#f26b0f"/>
        <circle cx="50" cy="56" r="5" fill="currentColor"/>
      </g>
    </svg>
    <b>TrashVision</b>
  </div>
</div>

<script>
(function () {
  const ov = document.getElementById("tvTransition");
  const cells = ov.querySelector(".tv-cells");
  const COLS = 6, ROWS = 5;

  for (let r = 0; r < ROWS; r++) {
    for (let c = 0; c < COLS; c++) {
      const el = document.createElement("i");
      el.className = "cell" + ((r + c) % 2 ? " alt" : "");
      const k = (r + c) / (ROWS + COLS - 2);
      el.style.setProperty("--d", Math.round(k * 240 + Math.random() * 60) + "ms");
      cells.appendChild(el);
    }
  }

  const reduce = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
  let timer = null;

  function cover() {
    clearTimeout(timer);
    ov.classList.remove("idle", "done");
    void ov.offsetWidth;
  }

  function reveal() {
    clearTimeout(timer);
    ov.classList.remove("idle");
    requestAnimationFrame(function () {
      requestAnimationFrame(function () { ov.classList.add("done"); });
    });
    setTimeout(function () { ov.classList.add("done"); }, 90);
    timer = setTimeout(function () { ov.classList.add("idle"); }, 1300);
  }

  if (reduce) {
    ov.classList.add("done", "idle");
  } else {
    reveal();
  }

  window.addEventListener("pageshow", function (e) {
    if (!e.persisted || reduce) return;
    cover();
    setTimeout(reveal, 420);
  });

  if (!reduce) {
    let navigating = false;
    document.addEventListener("click", function (e) {
      if (navigating || e.defaultPrevented || e.button !== 0) return;
      if (e.metaKey || e.ctrlKey || e.shiftKey || e.altKey) return;
      const a = e.target.closest ? e.target.closest("a") : null;
      if (!a) return;
      if (a.target && a.target !== "_self") return;
      if (a.hasAttribute("download") || a.classList.contains("exportBtn")) return;
      let url;
      try { url = new URL(a.href, location.href); } catch (err) { return; }
      if (url.origin !== location.origin) return;
      if (url.pathname === location.pathname && url.search === location.search) return;
      e.preventDefault();
      navigating = true;
      cover();
      setTimeout(function () { location.href = url.href; }, 700);
      setTimeout(function () { navigating = false; reveal(); }, 2600);
    });
  }
})();
</script>

<main>{% block content %}{% endblock %}</main>

<script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
<script>
document.addEventListener("DOMContentLoaded", () => {
  const mapEl = document.getElementById("map");
  if (mapEl && window.L) {
    const map = L.map(mapEl).setView([55.751244, 37.618423], 11);
    L.tileLayer(
      "https://tile.openstreetmap.org/{z}/{x}/{y}.png",
      { maxZoom: 19, attribution: "© OpenStreetMap" }
    ).addTo(map);

    const points = window.__POINTS__ || [];
    points.forEach((p, i) => {
      if (!p.lat || !p.lng) return;
      const isVideo = p.type === "video";
      const icon = L.divIcon({
        className: "",
        html: '<div class="markerDot ' + (isVideo ? "video" : "") + '"></div>',
        iconSize: [22, 22], iconAnchor: [11, 11],
      });
      const marker = L.marker([p.lat, p.lng], { icon }).addTo(map);
      marker.bindPopup(
        "<b>#" + p.id + "</b> · " + p.created_at + "<br>" +
        '<span style="color:#94a3b8">' + p.location + "</span><br>" +
        '<span style="font-size:11px">' +
        (isVideo ? "видео" : "фото") + "</span><br>" +
        (p.result ? '<a href="' + p.result + '" target="_blank">Открыть результат</a>' : "")
      );
      if (i === 0) marker.openPopup();
    });
  }

  document.querySelectorAll(".dropzone").forEach((zone) => {
    const input = zone.querySelector("input[type=file]");
    const label = zone.querySelector("span");
    zone.addEventListener("dragover", (e) => {
      e.preventDefault(); zone.classList.add("hover");
    });
    zone.addEventListener("dragleave", () => zone.classList.remove("hover"));
    zone.addEventListener("drop", (e) => {
      e.preventDefault(); zone.classList.remove("hover");
      if (!e.dataTransfer.files.length) return;
      const dt = new DataTransfer();
      dt.items.add(e.dataTransfer.files[0]);
      input.files = dt.files;
      if (label) label.textContent = e.dataTransfer.files[0].name;
    });
  });

  document.querySelectorAll(".toast").forEach((t) => {
    setTimeout(() => {
      t.style.transition = "opacity .4s, transform .4s";
      t.style.opacity = "0";
      t.style.transform = "translateX(30px)";
      setTimeout(() => t.remove(), 400);
    }, 5000);
  });
});
</script>
</body>
</html>
"""

INDEX_HTML = """
{% extends "BASE" %}
{% block title %}Карта · TrashVision{% endblock %}
{% block content %}
<div class="home">
  <div class="tiles">

    <section class="tile">
      <div class="tileTop">
        <span class="tileNum">01</span>
        <span class="tileTag">Фото</span>
      </div>
      <div class="cardHead">
        <div class="icon photo"><svg viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="1.4"><rect x="1.5" y="3.4" width="13" height="10"/><circle cx="8" cy="8.4" r="2.6"/><path d="M4.6 3.4V1.9h6.8v1.5"/></svg></div>
        <div>
          <h2>Загрузить фото</h2>
          <small>YOLO найдёт мусор на снимке</small>
        </div>
      </div>
      <form action="/upload_image" method="post" enctype="multipart/form-data">
        <label class="dropzone">
          <input type="file" name="file" accept="image/*" required
                 onchange="this.parentNode.querySelector('span').textContent = this.files[0]?.name || 'Выберите файл'">
          <div class="dropIcon"><svg viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="1.5"><path d="M8 11.5V3M4.5 6.5L8 3l3.5 3.5"/><path d="M2.5 13.8h11"/></svg></div>
          <span>Перетащите или выберите фото</span>
        </label>
        <input name="location" placeholder="Место (адрес, описание)">
        <div class="row2">
          <input name="lat" placeholder="Широта" value="55.75">
          <input name="lng" placeholder="Долгота" value="37.61">
        </div>
        <button class="primary" type="submit">Обработать фото</button>
      </form>
    </section>

    <section class="tile">
      <div class="tileTop">
        <span class="tileNum">02</span>
        <span class="tileTag">Видео</span>
      </div>
      <div class="cardHead">
        <div class="icon video"><svg viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="1.4"><rect x="1.5" y="4" width="9" height="8"/><path d="M10.5 7l3.5-2v6l-3.5-2z"/></svg></div>
        <div>
          <h2>Загрузить видео</h2>
          <small>раз в 30 кадров, может занять время</small>
        </div>
      </div>
      <form action="/upload_video" method="post" enctype="multipart/form-data">
        <label class="dropzone">
          <input type="file" name="file" accept="video/*" required
                 onchange="this.parentNode.querySelector('span').textContent = this.files[0]?.name || 'Выберите файл'">
          <div class="dropIcon"><svg viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="1.5"><path d="M8 11.5V3M4.5 6.5L8 3l3.5 3.5"/><path d="M2.5 13.8h11"/></svg></div>
          <span>Перетащите или выберите видео</span>
        </label>
        <input name="location" placeholder="Место">
        <div class="row2">
          <input name="lat" placeholder="Широта" value="55.75">
          <input name="lng" placeholder="Долгота" value="37.61">
        </div>
        <button class="primary" type="submit">Обработать видео</button>
      </form>
    </section>

    <section class="tile">
      <div class="tileTop">
        <span class="tileNum">03</span>
        <span class="tileTag">Последние</span>
      </div>
      <div class="cardHead">
        <div class="icon recent"><svg viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="1.4"><circle cx="8" cy="8" r="6.2"/><path d="M8 4.5V8l2.6 1.8"/></svg></div>
        <div>
          <h2>Последние</h2>
          <small>10 записей</small>
        </div>
      </div>
      <div class="recentList">
        {% for p in points[:10] %}
        <div class="recentItem {{ p.type }}">
          <div class="recentTop">
            <b>#{{ p.id }}</b>
            <span class="badge {{ p.type }}">{{ p.type }}</span>
          </div>
          <div class="recentLoc">{{ p.location }}</div>
          <div class="recentMeta">{{ p.created_at }}</div>
          <a class="recentLink" href="{{ p.result }}" target="_blank">Открыть результат &rarr;</a>
        </div>
        {% else %}
        <div class="empty">Пока ничего не обработано</div>
        {% endfor %}
      </div>
    </section>

  </div>

  <div class="mapPanel">
    <div class="mapTag"><span class="dot"></span> Карта обнаружений · <b>{{ nav_total }}</b></div>
    <div id="map"></div>
  </div>
</div>

<script>
  window.__POINTS__ = {{ points|tojson }};
</script>
{% endblock %}
"""

LOGS_HTML = """
{% extends "BASE" %}
{% block title %}Логи · TrashVision{% endblock %}
{% block content %}
<div class="page">
  <h2 class="pageTitle">Журнал обнаружений</h2>
  <p class="pageSub">Все записи по дням, от новых к старым</p>
  {% if not files %}
    <div class="empty">Логов пока нет — обработайте первое фото или видео</div>
  {% endif %}
  {% for f in files %}
    <section class="logFile">
      <div class="logFileHead">
        <span class="logDot"></span>
        <b>{{ f.name }}</b>
        <span class="logSize">{{ f.content|length }} симв.</span>
      </div>
      <pre>{{ f.content }}</pre>
    </section>
  {% endfor %}
</div>
{% endblock %}
"""

REPORTS_HTML = """
{% extends "BASE" %}
{% block title %}Отчёты · TrashVision{% endblock %}
{% block content %}
<div class="page">
  <div class="pageHead">
    <div>
      <h2 class="pageTitle">Сводный отчёт</h2>
      <p class="pageSub">Сформирован {{ generated }}</p>
    </div>
    <a class="exportBtn" href="/reports/export">Скачать CSV</a>
  </div>

  <section class="dashCard">
    <div class="dashHead">
      <h3>Динамика по месяцам · {{ chart.year }}</h3>
      <div class="legend">
        <span class="r"><i></i>Отчёты</span>
        <span class="o"><i></i>Объекты</span>
      </div>
    </div>
    <div class="chartBox"><canvas id="trend"></canvas></div>
  </section>

  <div class="donutRow">
    <section class="donutCard" style="--tint:rgba(242,107,15,.14);--dcol:#ff8a1f">
      <div class="donutWrap">
        <canvas id="dTotal"></canvas>
        <div class="donutCenter"><b>{{ chart.total }}</b><span>записей</span></div>
      </div>
      <div class="donutLabel">Всего записей</div>
      <div class="donutSub">фото {{ chart.photos }} · видео {{ chart.videos }}</div>
    </section>

    <section class="donutCard" style="--tint:rgba(255,196,107,.12);--dcol:#ffc46b">
      <div class="donutWrap">
        <canvas id="dObjects"></canvas>
        <div class="donutCenter"><b>{{ chart.total_objects }}</b><span>объектов</span></div>
      </div>
      <div class="donutLabel">Найдено объектов</div>
      <div class="donutSub">фото {{ chart.photo_objects }} · видео {{ chart.video_objects }}</div>
    </section>

    <section class="donutCard" style="--tint:rgba(255,61,0,.10);--dcol:#ff6a33">
      <div class="donutWrap">
        <canvas id="dFound"></canvas>
        <div class="donutCenter"><b>{{ chart.found_pct }}%</b><span>с находками</span></div>
      </div>
      <div class="donutLabel">Записи с объектами</div>
      <div class="donutSub">{{ chart.with_objects }} из {{ chart.total }}</div>
    </section>
  </div>

  <div class="tableWrap">
    <table>
      <thead>
        <tr><th>ID</th><th>Дата</th><th>Место</th><th>Тип</th><th>Обнаружения</th></tr>
      </thead>
      <tbody>
        {% for r in rows %}
          <tr>
            <td><b>#{{ r[0] }}</b></td>
            <td>{{ r[1] }}</td>
            <td>{{ r[2] }}</td>
            <td><span class="badge {{ r[3] }}">{{ r[3] }}</span></td>
            <td><code>{{ r[4] }}</code></td>
          </tr>
        {% else %}
          <tr><td colspan="5" class="empty">Нет данных</td></tr>
        {% endfor %}
      </tbody>
    </table>
  </div>
</div>

<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.7/dist/chart.umd.min.js"></script>
<script>
const C = {{ chart|tojson }};
const MONO = "'JetBrains Mono', Consolas, monospace";
const GRID = "rgba(242,107,15,.12)";
const TICK = "#9c8570";
Chart.defaults.font.family = MONO;
Chart.defaults.font.size = 10;
Chart.defaults.color = TICK;

const tctx = document.getElementById("trend").getContext("2d");
const grad = tctx.createLinearGradient(0, 0, 0, 300);
grad.addColorStop(0, "rgba(242,107,15,.55)");
grad.addColorStop(1, "rgba(242,107,15,.02)");

new Chart(tctx, {
  type: "line",
  data: {
    labels: C.labels,
    datasets: [
      {
        label: "Отчёты", data: C.months, yAxisID: "y",
        borderColor: "#f26b0f", backgroundColor: grad, fill: true,
        tension: .45, borderWidth: 2, pointRadius: 0, pointHoverRadius: 4,
        pointBackgroundColor: "#ff8a1f"
      },
      {
        label: "Объекты", data: C.objects, yAxisID: "y1",
        borderColor: "#ffc46b", backgroundColor: "transparent", fill: false,
        tension: .45, borderWidth: 2, pointRadius: 0, pointHoverRadius: 4,
        pointBackgroundColor: "#ffc46b"
      }
    ]
  },
  options: {
    responsive: true, maintainAspectRatio: false,
    interaction: { mode: "index", intersect: false },
    plugins: {
      legend: { display: false },
      tooltip: {
        backgroundColor: "#150e09", borderColor: "rgba(242,107,15,.45)",
        borderWidth: 1, titleColor: "#f0e4d4", bodyColor: "#cbb9a5",
        titleFont: { family: MONO, size: 10 }, bodyFont: { family: MONO, size: 10 }
      }
    },
    scales: {
      x: { grid: { color: GRID }, ticks: { color: TICK } },
      y: {
        beginAtZero: true, grid: { color: GRID },
        ticks: { color: TICK, precision: 0 },
        title: { display: true, text: "ОТЧЁТЫ", color: TICK, font: { family: MONO, size: 9 } }
      },
      y1: {
        beginAtZero: true, position: "right", grid: { drawOnChartArea: false },
        ticks: { color: TICK, precision: 0 },
        title: { display: true, text: "ОБЪЕКТЫ", color: TICK, font: { family: MONO, size: 9 } }
      }
    }
  }
});

function donut(id, data, colors) {
  new Chart(document.getElementById(id), {
    type: "doughnut",
    data: {
      labels: ["a", "b"],
      datasets: [{
        data: data, backgroundColor: colors, borderWidth: 0, hoverOffset: 3
      }]
    },
    options: {
      responsive: true, maintainAspectRatio: false, cutout: "72%",
      plugins: { legend: { display: false }, tooltip: { enabled: false } }
    }
  });
}

donut("dTotal", [C.photos, C.videos], ["#f26b0f", "#3a2414"]);
donut("dObjects", [C.photo_objects, C.video_objects], ["#ffc46b", "#5a3517"]);
donut("dFound", [C.with_objects, Math.max(C.total - C.with_objects, 0)], ["#ff6a33", "#2a1a10"]);
</script>
{% endblock %}
"""


# ---------- рендер с учётом наследования ----------
def render_page(name: str, **ctx):
    base = BASE_HTML
    page = {"index": INDEX_HTML, "logs": LOGS_HTML, "reports": REPORTS_HTML}[name]
    body = page.replace('{% extends "BASE" %}', "")
    # извлекаем содержимое content-блока
    if "{% block content %}" in body:
        start = body.index("{% block content %}") + len("{% block content %}")
        end = body.rindex("{% endblock %}")
        body = body[start:end]
    title = "TrashVision"
    if "{% block title %}" in page:
        ts = page.index("{% block title %}") + len("{% block title %}")
        te = page.index("{% endblock %}", ts)
        title = page[ts:te].strip()
    base = base.replace("{% block title %}TrashVision{% endblock %}", title)
    base = base.replace("{% block content %}{% endblock %}", body)
    return render_template_string(base, **ctx)


# ---------- маршруты ----------
@app.context_processor
def inject_globals():
    con = sqlite3.connect(DB)
    total = con.execute("SELECT COUNT(*) FROM reports").fetchone()[0]
    imgs = con.execute(
        "SELECT COUNT(*) FROM reports WHERE source_type='image'"
    ).fetchone()[0]
    vids = con.execute(
        "SELECT COUNT(*) FROM reports WHERE source_type='video'"
    ).fetchone()[0]
    con.close()
    return {
        "nav_total": total,
        "nav_images": imgs,
        "nav_videos": vids,
        "now": datetime.now(),
    }


def _row_to_dict(r):
    return {
        "id": r[0], "created_at": r[1], "location": r[2],
        "lat": r[3], "lng": r[4], "type": r[5],
        "file": r[6], "result": r[7], "detections": r[8],
        "log_url": r[9] if len(r) > 9 else None,
    }


@app.route("/")
def index():
    con = sqlite3.connect(DB)
    rows = con.execute(
        "SELECT id, created_at, location, lat, lng, source_type, "
        "file_url, result_url, detections, log_url "
        "FROM reports ORDER BY id DESC LIMIT 50"
    ).fetchall()
    con.close()
    points = [_row_to_dict(r) for r in rows]
    return render_page("index", points=points)


@app.route("/upload_image", methods=["POST"])
def upload_image():
    f = request.files.get("file")
    location = request.form.get("location", "").strip() or "Не указано"
    try:
        lat = float(request.form.get("lat") or 55.751244)
        lng = float(request.form.get("lng") or 37.618423)
    except ValueError:
        lat, lng = 55.751244, 37.618423

    if not f or not f.filename:
        flash("Файл не выбран", "error")
        return redirect(url_for("index"))

    name = f"{uuid.uuid4().hex[:8]}_{f.filename}"
    path = os.path.join(UPLOAD_DIR, name)
    f.save(path)

    detections, result_url = detect_image(path)
    log_url = write_log(detections, name, location)

    con = sqlite3.connect(DB)
    con.execute(
        "INSERT INTO reports (created_at, location, lat, lng, source_type, "
        "file_url, result_url, detections, log_url, status) "
        "VALUES (?,?,?,?,?,?,?,?,?,'done')",
        (datetime.now().strftime("%Y-%m-%d %H:%M"), location, lat, lng,
         "image", f"/{path}", result_url, str(detections), log_url),
    )
    con.commit()
    con.close()

    flash(f"Фото обработано. Объектов: {len(detections)}", "success")
    return redirect(url_for("index"))


@app.route("/upload_video", methods=["POST"])
def upload_video():
    f = request.files.get("file")
    location = request.form.get("location", "").strip() or "Не указано"
    try:
        lat = float(request.form.get("lat") or 55.751244)
        lng = float(request.form.get("lng") or 37.618423)
    except ValueError:
        lat, lng = 55.751244, 37.618423

    if not f or not f.filename:
        flash("Файл не выбран", "error")
        return redirect(url_for("index"))

    name = f"{uuid.uuid4().hex[:8]}_{f.filename}"
    path = os.path.join(UPLOAD_DIR, name)
    f.save(path)

    summary = detect_video(path)
    dets = summary["detections"]
    log_url = write_log(dets, name, location)

    con = sqlite3.connect(DB)
    con.execute(
        "INSERT INTO reports (created_at, location, lat, lng, source_type, "
        "file_url, result_url, detections, log_url, status) "
        "VALUES (?,?,?,?,?,?,?,?,?,'done')",
        (datetime.now().strftime("%Y-%m-%d %H:%M"), location, lat, lng,
         "video", f"/{path}", summary["result_video"], str(dets), log_url),
    )
    con.commit()
    con.close()

    flash(f"Видео обработано. Вхождений: {len(dets)}", "success")
    return redirect(url_for("index"))


@app.route("/api/points")
def api_points():
    con = sqlite3.connect(DB)
    rows = con.execute(
        "SELECT id, created_at, location, lat, lng, source_type, "
        "result_url, detections FROM reports WHERE lat IS NOT NULL"
    ).fetchall()
    con.close()
    return jsonify([{
        "id": r[0], "date": r[1], "location": r[2],
        "lat": r[3], "lng": r[4], "type": r[5],
        "result": r[6], "count": r[7].count("'class'") if r[7] else 0,
    } for r in rows])


@app.route("/logs")
def logs():
    files = []
    if os.path.exists(LOG_DIR):
        for fn in sorted(os.listdir(LOG_DIR), reverse=True):
            p = os.path.join(LOG_DIR, fn)
            with open(p, encoding="utf-8") as fh:
                files.append({"name": fn, "content": fh.read()})
    return render_page("logs", files=files)


@app.route("/reports")
def reports():
    con = sqlite3.connect(DB)
    rows = con.execute(
        "SELECT id, created_at, location, source_type, detections "
        "FROM reports ORDER BY id DESC"
    ).fetchall()
    con.close()

    total = len(rows)
    by_type = {}
    year = datetime.now().year
    months = [0] * 12
    obj_months = [0] * 12
    photos = videos = 0
    photo_objects = video_objects = 0
    with_objects = 0
    total_objects = 0

    for r in rows:
        det = r[4] or ""
        n = det.count("'class'")
        total_objects += n
        if n:
            with_objects += 1
        st = r[3]
        by_type[st] = by_type.get(st, 0) + 1
        if st == "image":
            photos += 1
            photo_objects += n
        elif st == "video":
            videos += 1
            video_objects += n
        try:
            y = int(str(r[1])[:4])
            m = int(str(r[1])[5:7]) - 1
        except (TypeError, ValueError):
            continue
        if y == year and 0 <= m < 12:
            months[m] += 1
            obj_months[m] += n

    chart = {
        "year": year,
        "labels": ["Январь", "Февраль", "Март", "Апрель", "Май", "Июнь",
                   "Июль", "Август", "Сентябрь", "Октябрь", "Ноябрь", "Декабрь"],
        "months": months,
        "objects": obj_months,
        "total": total,
        "photos": photos,
        "videos": videos,
        "total_objects": total_objects,
        "photo_objects": photo_objects,
        "video_objects": video_objects,
        "with_objects": with_objects,
        "found_pct": round(with_objects * 100 / total) if total else 0,
    }

    return render_page("reports", rows=rows, total=total,
                       by_type=by_type, chart=chart,
                       generated=datetime.now().strftime("%d.%m.%Y %H:%M"))


@app.route("/reports/export")
def export_reports():
    con = sqlite3.connect(DB)
    rows = con.execute(
        "SELECT id, created_at, location, lat, lng, source_type, detections "
        "FROM reports"
    ).fetchall()
    con.close()
    buf = io.StringIO()
    w = csv.writer(buf, delimiter=";")
    w.writerow(["ID", "Дата", "Место", "Широта", "Долгота", "Тип",
                "Обнаружения"])
    for r in rows:
        w.writerow(r)
    mem = io.BytesIO(buf.getvalue().encode("utf-8-sig"))
    return send_file(
        mem, mimetype="text/csv", as_attachment=True,
        download_name=f"report_{datetime.now():%Y%m%d}.csv",
    )


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)