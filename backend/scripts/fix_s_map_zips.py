from __future__ import annotations

import os
import shutil
import subprocess
import sys
import time
import zipfile
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

# Thư mục Google Drive lưu map-keyframes
DRIVE_MAP_FOLDER_ID = "1oeptkdkvB5CVP4QEUFtIoZFBJmZtCgCo"
TEMP_DIR = Path("D:/HCMAIC/event-retrieval-system/data/_tmp_fix_s_maps")

# Nhóm S: S01_V001.zip -> S01_V012.zip
TARGET_ZIPS = [f"S01_V{i:03d}.zip" for i in range(1, 13)]

def log(msg: str):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)

def run_cmd(cmd: list[str]) -> bool:
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        log(f"ERROR: Command failed: {' '.join(cmd)}\n{res.stderr}")
        return False
    return True

def fix_single_s_map_zip(zip_name: str) -> bool:
    log(f"=== Xử lý nhóm {zip_name} ===")
    
    zip_download_dir = TEMP_DIR / "download"
    zip_download_dir.mkdir(parents=True, exist_ok=True)
    downloaded_zip = zip_download_dir / zip_name
    
    # 1. Download từ Google Drive
    log(f"Tải {zip_name} từ Drive...")
    dl_ok = run_cmd([
        "rclone", "copy", f"gdrive:{zip_name}", str(zip_download_dir),
        "--drive-root-folder-id", DRIVE_MAP_FOLDER_ID,
        "--drive-chunk-size", "64M"
    ])
    if not dl_ok or not downloaded_zip.exists():
        log(f"Lỗi: Không tải được {zip_name} từ Drive")
        return False

    orig_size_kb = downloaded_zip.stat().st_size / 1024
    log(f"Đã tải {zip_name} ({orig_size_kb:.1f} KB)")

    # 2. Bóc tách và đưa file .csv ra root của zip (loại bỏ folder S01/ ở giữa)
    fixed_zip_dir = TEMP_DIR / "fixed"
    fixed_zip_dir.mkdir(parents=True, exist_ok=True)
    fixed_zip_path = fixed_zip_dir / zip_name
    
    csv_count = 0
    with zipfile.ZipFile(downloaded_zip, "r") as zf_in:
        infolist = zf_in.infolist()
        csv_entries = [
            info for info in infolist
            if not info.is_dir() and info.filename.lower().endswith(".csv")
        ]
        csv_entries.sort(key=lambda x: Path(x.filename).name)
        csv_count = len(csv_entries)
        
        if csv_count == 0:
            log(f"Cảnh báo: Không tìm thấy file .csv nào trong {zip_name}!")
            return False
            
        with zipfile.ZipFile(fixed_zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf_out:
            for info in csv_entries:
                flat_file_name = Path(info.filename).name
                data = zf_in.read(info)
                zf_out.writestr(flat_file_name, data)
                
    new_size_kb = fixed_zip_path.stat().st_size / 1024
    log(f"Chuẩn hóa {zip_name}: {csv_count} files CSV -> {new_size_kb:.1f} KB (Flat CSV: {flat_file_name})")

    # 3. Upload ghi đè lên Google Drive
    log(f"Upload {zip_name} mới lên Drive...")
    up_ok = run_cmd([
        "rclone", "copy", str(fixed_zip_path), "gdrive:",
        "--drive-root-folder-id", DRIVE_MAP_FOLDER_ID,
        "--drive-chunk-size", "64M"
    ])
    if not up_ok:
        log(f"Lỗi: Không upload được {zip_name}")
        return False

    # 4. Dọn file tạm local
    downloaded_zip.unlink(missing_ok=True)
    fixed_zip_path.unlink(missing_ok=True)
    log(f"✅ Hoàn tất {zip_name} thành công!\n")
    return True

def main():
    start_time = time.perf_counter()
    TEMP_DIR.mkdir(parents=True, exist_ok=True)
    success = 0
    failed = 0
    
    log("BẮT ĐẦU CHUẨN HÓA CÁC FILE MAP ZIP NHÓM S (S01_V001 -> S01_V012)")
    log("=" * 60)
    
    for zip_name in TARGET_ZIPS:
        ok = fix_single_s_map_zip(zip_name)
        if ok:
            success += 1
        else:
            failed += 1
            log(f"Thất bại khi xử lý {zip_name}, tiếp tục file tiếp theo...")
            
    # Xóa sạch thư mục tạm
    shutil.rmtree(TEMP_DIR, ignore_errors=True)
    
    elapsed = time.perf_counter() - start_time
    log("=" * 60)
    log(f"HOÀN TẤT TOÀN BỘ: {success} file thành công, {failed} file thất bại trong {elapsed:.1f}s")
    log("=" * 60)

if __name__ == "__main__":
    main()
