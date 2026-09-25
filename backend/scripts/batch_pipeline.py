"""
batch_pipeline.py — Tự động hóa toàn bộ luồng:
  rclone download zip → unzip → distribution keyframe → rclone upload → cleanup

Chạy:
    conda activate hcmaic
    cd D:\\HCMAIC\\event-retrieval-system\\backend
    python scripts/batch_pipeline.py --start 31 --end 40 --input-folder-id <FOLDER_ID>

Flags:
    --start            N0xx bắt đầu (mặc định 1)
    --end              N0xx kết thúc (mặc định 50)
    --input-folder-id  Google Drive folder ID chứa các file zip N0xx.zip
    --output-folder-id Google Drive folder ID để upload keyframes (mặc định đã set)
    --workers          Số video xử lý song song trong mỗi N0xx (mặc định 3)
    --output-dir       Thư mục lưu keyframes local (mặc định data/keyframes)
    --temp-dir         Thư mục tạm chứa zip + video raw (mặc định data/_tmp)
    --keep-keyframes   Không xóa keyframes local sau khi upload
    --dry-run          In ra lệnh sẽ chạy nhưng không thực thi
    --resume           Bỏ qua N0xx đã upload thành công (dựa vào file .done)
"""
from __future__ import annotations

import argparse
import logging
import multiprocessing
import os
import shutil
import subprocess
import sys
import time
import zipfile
from pathlib import Path

# ───────────────────────────── config defaults ──────────────────────────────

BACKEND_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT    = BACKEND_ROOT.parent

DEFAULT_OUTPUT_FOLDER_ID = "1Sq3RQy5X0e3JBsv_s3SXTNPckHKrFs_x"
# Distribution script tự thêm "keyframes/" vào root, nên root phải là data/
# → keyframes sẽ nằm ở:  data/keyframes/{nxx}/{video_stem}/*.webp
DEFAULT_DATA_DIR         = REPO_ROOT / "data"
DEFAULT_TEMP_DIR         = REPO_ROOT / "data" / "_tmp"
DISTRIBUTION_SCRIPT      = "scripts.keyframe_extraction.run_two_distribution_hsv_motion_v2"

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("batch_pipeline")


# ─────────────────────────────── helpers ────────────────────────────────────

def run(cmd: list[str], dry_run: bool = False) -> int:
    log.info("CMD: %s", " ".join(cmd))
    if dry_run:
        return 0
    result = subprocess.run(cmd, capture_output=False)
    return result.returncode


def nxx_name(n: int) -> str:
    """1 → 'N001', 31 → 'N031'"""
    return f"N{n:03d}"


def find_videos(folder: Path) -> list[Path]:
    """Tìm tất cả video trong folder, tự xử lý nested N0xx/N0xx/."""
    exts = {".mp4", ".mkv", ".avi", ".mov", ".webm"}
    videos = sorted(p for p in folder.rglob("*") if p.suffix.lower() in exts)
    return videos


def unzip_nxx(zip_path: Path, raw_dir: Path, nxx: str) -> Path:
    """
    Giải nén zip_path vào raw_dir/nxx/.
    Xử lý cả trường hợp nested: raw_dir/nxx/nxx/videos → trả về lớp trong.
    """
    target = raw_dir / nxx
    target.mkdir(parents=True, exist_ok=True)

    log.info("[%s] Unzipping %s ...", nxx, zip_path.name)
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(target)

    # Xử lý nested: nếu bên trong chỉ có 1 thư mục con trùng tên nxx
    children = [c for c in target.iterdir() if c.is_dir()]
    if len(children) == 1 and children[0].name.upper() == nxx.upper():
        log.info("[%s] Detected nested folder, using inner: %s", nxx, children[0])
        return children[0]

    return target


def done_flag(temp_dir: Path, nxx: str) -> Path:
    return temp_dir / "done" / f"{nxx}.done"


def mark_done(temp_dir: Path, nxx: str) -> None:
    flag = done_flag(temp_dir, nxx)
    flag.parent.mkdir(parents=True, exist_ok=True)
    flag.touch()


def is_done(temp_dir: Path, nxx: str) -> bool:
    return done_flag(temp_dir, nxx).exists()


# ────────────────────────── process 1 video ─────────────────────────────────

def process_video(args_tuple) -> tuple[str, bool, float]:
    """
    Chạy distribution pipeline cho 1 video.
    Trả về (video_name, success, elapsed_sec).
    Dùng trong multiprocessing — không dùng shared state.
    """
    video_path, output_dir, dry_run = args_tuple
    nxx = video_path.parent.name  # e.g. N031
    # Nếu nested thêm 1 lớp thì parent.parent.name mới là N031
    # → video stem là N031-V001 nên lấy prefix 4 ký tự đầu
    nxx = video_path.stem[:4].upper()  # N031, N032, ...

    log.info("[%s] Processing: %s", nxx, video_path.name)
    started = time.perf_counter()

    cmd = [
        sys.executable, "-m", DISTRIBUTION_SCRIPT,
        "--video", str(video_path),
        "--output-dir", str(output_dir),
        "--overwrite",
    ]

    if dry_run:
        log.info("[DRY RUN] Would run: %s", " ".join(cmd))
        return video_path.name, True, 0.0

    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        cwd=str(BACKEND_ROOT),
        env={**os.environ, "PYTHONPATH": str(BACKEND_ROOT)},
    )
    elapsed = time.perf_counter() - started

    if result.returncode != 0:
        log.error("[%s] FAILED %s (%.1fs):\n%s", nxx, video_path.name, elapsed, result.stderr[-1000:])
        return video_path.name, False, elapsed

    log.info("[%s] Done %s in %.1fs", nxx, video_path.name, elapsed)
    return video_path.name, True, elapsed


# ────────────────────────── process 1 N0xx ──────────────────────────────────

def process_nxx(
    n: int,
    input_folder_id: str,
    output_folder_id: str,
    output_dir: Path,
    temp_dir: Path,
    workers: int,
    keep_keyframes: bool,
    dry_run: bool,
    resume: bool,
    local_zip_dir: Path | None = None,
) -> bool:
    nxx = nxx_name(n)
    log.info("=" * 60)
    log.info("START %s", nxx)
    log.info("=" * 60)

    # Resume: bỏ qua nếu đã upload xong
    if resume and is_done(temp_dir, nxx):
        log.info("[%s] Already done, skipping.", nxx)
        return True

    zip_name  = f"{nxx}.zip"
    raw_dir   = temp_dir / "raw"

    # ── 1. Lấy zip — local hoặc download ──
    if local_zip_dir is not None:
        # Dùng zip có sẵn ở local, KHÔNG copy sang temp (tránh tốn disk)
        zip_local = local_zip_dir / zip_name
        if not dry_run and not zip_local.exists():
            log.error("[%s] Local zip not found: %s", nxx, zip_local)
            return False
        log.info("[%s] Using local zip: %s", nxx, zip_local)
    else:
        # Download từ Drive
        zip_local = temp_dir / zip_name
        log.info("[%s] Downloading %s from Drive ...", nxx, zip_name)
        rc = run([
            "rclone", "copy",
            f"gdrive:{zip_name}",
            str(temp_dir),
            "--drive-root-folder-id", input_folder_id,
            "--progress",
            "--transfers", "4",
        ], dry_run)

        if not dry_run and not zip_local.exists():
            log.error("[%s] Download failed: %s not found", nxx, zip_local)
            return False

    # ── 2. Unzip ──
    if not dry_run:
        video_root = unzip_nxx(zip_local, raw_dir, nxx)
        videos = find_videos(video_root)
    else:
        video_root = raw_dir / nxx
        videos = [video_root / f"{nxx}-V001.mov"]  # placeholder cho dry-run

    if not videos:
        log.error("[%s] No videos found in %s", nxx, video_root)
        return False

    log.info("[%s] Found %d video(s): %s", nxx, len(videos),
             [v.name for v in videos])

    # ── 3. Process videos (parallel) ──
    # output_dir = data/  → distribution script tạo: data/keyframes/{nxx}/{stem}/*.webp
    job_args = [(v, output_dir, dry_run) for v in videos]

    if workers > 1 and len(videos) > 1:
        log.info("[%s] Processing %d videos with %d workers ...", nxx, len(videos), workers)
        with multiprocessing.Pool(processes=min(workers, len(videos))) as pool:
            results = pool.map(process_video, job_args)
    else:
        results = [process_video(a) for a in job_args]

    # Kiểm tra kết quả
    failed = [name for name, ok, _ in results if not ok]
    if failed:
        log.error("[%s] %d video(s) failed: %s", nxx, len(failed), failed)
        return False

    total_time = sum(t for _, _, t in results)
    log.info("[%s] All %d videos processed in %.1fs total", nxx, len(videos), total_time)

    # ── 4. Zip keyframes ──
    # Structure bên trong zip: N031/N031-V001/*.webp, N031/N031-V002/*.webp, ...
    nxx_keyframe_dir = output_dir / "keyframes" / nxx
    zip_output = temp_dir / f"{nxx}.zip"

    if not dry_run and (not nxx_keyframe_dir.exists() or not any(nxx_keyframe_dir.iterdir())):
        log.error("[%s] No keyframes found at %s", nxx, nxx_keyframe_dir)
        return False

    log.info("[%s] Zipping %s → %s ...", nxx, nxx_keyframe_dir, zip_output.name)
    if not dry_run:
        if zip_output.exists():
            zip_output.unlink()
        all_files = sorted(nxx_keyframe_dir.rglob("*"))
        image_files = [f for f in all_files if f.is_file()]
        with zipfile.ZipFile(zip_output, "w", zipfile.ZIP_STORED) as zf:
            for f in image_files:
                # arcname đúng: N001/N001-V001/000000.webp
                # f.relative_to(nxx_keyframe_dir) = N001-V001/000000.webp
                # Path(nxx) / ... = N001/N001-V001/000000.webp
                arcname = Path(nxx) / f.relative_to(nxx_keyframe_dir)
                zf.write(f, arcname)
        zip_size_mb = zip_output.stat().st_size / 1024 / 1024
        log.info("[%s] Zip created: %s (%.1f MB, %d files)",
                 nxx, zip_output.name, zip_size_mb, len(image_files))
    else:
        log.info("[DRY RUN] Would zip %s → %s", nxx_keyframe_dir, zip_output)

    # ── 5. Upload zip ──
    log.info("[%s] Uploading %s → Drive ...", nxx, zip_output.name)
    rc = run([
        "rclone", "copy",
        str(zip_output),
        "gdrive:",
        "--drive-root-folder-id", output_folder_id,
        "--progress",
        "--transfers", "4",
    ], dry_run)

    if rc != 0:
        log.error("[%s] Upload failed!", nxx)
        return False

    log.info("[%s] Upload complete.", nxx)

    # ── 6. Cleanup ──
    if not dry_run:
        log.info("[%s] Cleaning up temp files ...", nxx)
        # Xóa zip input (video gốc)
        if zip_local.exists():
            zip_local.unlink()
        # Xóa raw video
        raw_nxx = raw_dir / nxx
        if raw_nxx.exists():
            shutil.rmtree(raw_nxx)
        # Xóa zip keyframes đã upload
        if zip_output.exists():
            zip_output.unlink()
        # Xóa keyframes local nếu không giữ
        if not keep_keyframes:
            if nxx_keyframe_dir.exists():
                shutil.rmtree(nxx_keyframe_dir)
                log.info("[%s] Removed local keyframes.", nxx)

    mark_done(temp_dir, nxx)
    log.info("[%s] DONE.", nxx)
    return True


# ──────────────────────────────── main ──────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Batch keyframe extraction pipeline N001-N050")
    p.add_argument("--start", type=int, default=1, help="N0xx start (default 1)")
    p.add_argument("--end",   type=int, default=50, help="N0xx end inclusive (default 50)")
    p.add_argument("--input-folder-id", default=None,
                   help="Google Drive folder ID chứa NXxx.zip (bỏ qua nếu dùng --local-zip-dir)")
    p.add_argument("--local-zip-dir", type=Path, default=None,
                   help="Thư mục local chứa sẵn N0xx.zip — bỏ qua bước rclone download")
    p.add_argument("--output-folder-id", default=DEFAULT_OUTPUT_FOLDER_ID,
                   help=f"Drive folder ID upload keyframes (default: {DEFAULT_OUTPUT_FOLDER_ID})")
    p.add_argument("--workers", type=int, default=3,
                   help="Số video xử lý song song trong mỗi N0xx (default 3)")
    p.add_argument("--output-dir", type=Path, default=DEFAULT_DATA_DIR,
                   help="Root data dir — keyframes lưu tại output-dir/keyframes/N0xx/")
    p.add_argument("--temp-dir", type=Path, default=DEFAULT_TEMP_DIR)
    p.add_argument("--keep-keyframes", action="store_true",
                   help="Giữ keyframes local sau khi upload")
    p.add_argument("--dry-run", action="store_true",
                   help="In lệnh sẽ chạy, không thực thi")
    p.add_argument("--resume", action="store_true",
                   help="Bỏ qua N0xx đã có file .done (đã upload)")
    p.add_argument("--only", type=int, nargs="+",
                   help="Chỉ chạy N0xx cụ thể, VD: --only 1 2 3")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    if args.local_zip_dir is None and args.input_folder_id is None:
        raise SystemExit("Cần --local-zip-dir hoặc --input-folder-id")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.temp_dir.mkdir(parents=True, exist_ok=True)
    (args.temp_dir / "raw").mkdir(parents=True, exist_ok=True)

    if args.only:
        targets = sorted(args.only)
    else:
        targets = list(range(args.start, args.end + 1))

    log.info("Pipeline start: %s → %s (%d folders)",
             nxx_name(targets[0]), nxx_name(targets[-1]), len(targets))
    if args.local_zip_dir:
        log.info("Local zip dir      : %s", args.local_zip_dir)
    else:
        log.info("Input Drive folder : %s", args.input_folder_id)
    log.info("Output Drive folder: %s", args.output_folder_id)
    log.info("Workers per folder : %d", args.workers)
    log.info("Resume mode        : %s", args.resume)

    pipeline_start = time.perf_counter()
    success_count = fail_count = 0

    for n in targets:
        ok = process_nxx(
            n=n,
            input_folder_id=args.input_folder_id or "",
            output_folder_id=args.output_folder_id,
            output_dir=args.output_dir,
            temp_dir=args.temp_dir,
            workers=args.workers,
            keep_keyframes=args.keep_keyframes,
            dry_run=args.dry_run,
            resume=args.resume,
            local_zip_dir=args.local_zip_dir,
        )
        if ok:
            success_count += 1
        else:
            fail_count += 1
            log.warning("N%03d failed, continuing to next...", n)

    elapsed = time.perf_counter() - pipeline_start
    log.info("=" * 60)
    log.info("PIPELINE COMPLETE: %d success, %d failed, total %.1fs (%.1f min)",
             success_count, fail_count, elapsed, elapsed / 60)
    log.info("=" * 60)


if __name__ == "__main__":
    main()
