from __future__ import annotations

import os
import shutil
import subprocess
import sys
import time
import zipfile
from pathlib import Path

DRIVE_FOLDER_ID = "1Sq3RQy5X0e3JBsv_s3SXTNPckHKrFs_x"
TEMP_DIR = Path("D:/HCMAIC/event-retrieval-system/data/_tmp_fix_s01")

TARGET_ZIPS = [f"S01_V{i:03d}.zip" for i in range(1, 13)]

def log(msg: str):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)

def run_cmd(cmd: list[str]) -> bool:
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        log(f"ERROR: Command failed: {' '.join(cmd)}\n{res.stderr}")
        return False
    return True

def fix_single_zip(zip_name: str) -> bool:
    video_id = Path(zip_name).stem  # e.g. S01_V001
    log(f"=== Processing {zip_name} ===")
    
    zip_download_dir = TEMP_DIR / "download"
    zip_download_dir.mkdir(parents=True, exist_ok=True)
    downloaded_zip = zip_download_dir / zip_name
    
    # 1. Download from Google Drive
    log(f"Downloading {zip_name} from Drive...")
    dl_ok = run_cmd([
        "rclone", "copy", f"gdrive:{zip_name}", str(zip_download_dir),
        "--drive-root-folder-id", DRIVE_FOLDER_ID,
        "--drive-chunk-size", "64M"
    ])
    if not dl_ok or not downloaded_zip.exists():
        log(f"Failed to download {zip_name}")
        return False

    orig_size_mb = downloaded_zip.stat().st_size / (1024 * 1024)
    log(f"Downloaded {zip_name} ({orig_size_mb:.1f} MB)")

    # 2. Re-pack zip with normalized structure: video_id/filename.jpg
    fixed_zip_dir = TEMP_DIR / "fixed"
    fixed_zip_dir.mkdir(parents=True, exist_ok=True)
    fixed_zip_path = fixed_zip_dir / zip_name
    
    img_count = 0
    with zipfile.ZipFile(downloaded_zip, "r") as zf_in:
        infolist = zf_in.infolist()
        img_entries = [
            info for info in infolist
            if not info.is_dir() and info.filename.lower().endswith((".jpg", ".jpeg", ".webp", ".png"))
        ]
        img_entries.sort(key=lambda x: Path(x.filename).name)
        img_count = len(img_entries)
        
        if img_count == 0:
            log(f"Warning: No image files found in {zip_name}!")
            return False
            
        with zipfile.ZipFile(fixed_zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf_out:
            for info in img_entries:
                file_name = Path(info.filename).name
                arcname = f"{video_id}/{file_name}"
                data = zf_in.read(info)
                zf_out.writestr(arcname, data)
                
    new_size_mb = fixed_zip_path.stat().st_size / (1024 * 1024)
    log(f"Restructured {zip_name}: {img_count} images -> {new_size_mb:.1f} MB (inside: {video_id}/*.jpg)")

    # 3. Upload back to Google Drive (overwrite)
    log(f"Uploading fixed {zip_name} to Drive...")
    up_ok = run_cmd([
        "rclone", "copy", str(fixed_zip_path), "gdrive:",
        "--drive-root-folder-id", DRIVE_FOLDER_ID,
        "--drive-chunk-size", "64M"
    ])
    if not up_ok:
        log(f"Failed to upload fixed {zip_name}")
        return False

    # 4. Cleanup local temp files
    downloaded_zip.unlink(missing_ok=True)
    fixed_zip_path.unlink(missing_ok=True)
    log(f"Done {zip_name} successfully!")
    return True

def main():
    start_time = time.perf_counter()
    TEMP_DIR.mkdir(parents=True, exist_ok=True)
    success = 0
    failed = 0
    
    for zip_name in TARGET_ZIPS:
        ok = fix_single_zip(zip_name)
        if ok:
            success += 1
        else:
            failed += 1
            log(f"Failed processing {zip_name}, continuing to next...")
            
    # Remove temp dir completely
    shutil.rmtree(TEMP_DIR, ignore_errors=True)
    
    elapsed = time.perf_counter() - start_time
    log("=" * 60)
    log(f"ALL DONE: {success} succeeded, {failed} failed in {elapsed:.1f}s ({elapsed/60:.1f} mins)")
    log("=" * 60)

if __name__ == "__main__":
    main()
