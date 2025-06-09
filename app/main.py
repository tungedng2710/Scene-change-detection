from fastapi import FastAPI, UploadFile, File
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from pathlib import Path
import shutil, subprocess, uuid, glob, os
import time
from pipeline import run_change_detection
import os

BASE_DIR = Path(__file__).resolve().parent
TMP_DIR  = BASE_DIR / "tmp"
TMP_DIR.mkdir(exist_ok=True)

app = FastAPI(title="Scene-Change API")

app.mount("/static", StaticFiles(directory=BASE_DIR / "static"), name="static")
app.mount("/outputs", StaticFiles(directory="outputs"), name="outputs")

@app.get("/", response_class=HTMLResponse)
async def root():
    """Serve the SPA."""
    return (BASE_DIR / "static" / "index.html").read_text(encoding="utf-8")

@app.post("/detect")
async def detect(
    ref_img: UploadFile = File(...),
    test_img: UploadFile = File(...)):
    # unique subfolder per request
    uid = str(uuid.uuid4())
    uid_path = TMP_DIR / uid
    uid_path.mkdir()

    ref_path  = uid_path / "ref.png"
    test_path = uid_path / "test.png"

    # stream files
    for up, dest in [(ref_img, ref_path), (test_img, test_path)]:
        with dest.open("wb") as f:
            shutil.copyfileobj(up.file, f)
            
    current_time = time.time()
    mask_path = f"outputs/{current_time}_mask_overlay.png"
    logs, mask_path, percent = run_change_detection(ref_path, test_path, mask_path, True)

    return {
        "logs": logs,
        "percent": percent,
        # "mask_url": f"/mask/{uid_path.name}" if mask_path else None
        "mask_url": mask_path
    }

@app.get("/mask/{job_id}")
async def get_mask(job_id: str):
    mask = TMP_DIR / job_id / "change_mask_overlay.png"
    if mask.exists():
        return FileResponse(mask, media_type="image/png")
    return {"detail": "mask not ready"}, 404