"""
Luong 2 (domain vocab, threshold-based) cua post-OCR correction -- xem
ocr_post_correction_plan.md muc 3.4.

Xu ly: ten rieng, dia danh, ten to chuc/chuong trinh, banner su kien --
KHONG co trong dictionary chuan (luong 1) nhung van dung.

THAY vi list tay 11 entry (overfit theo data da thay, quyet dinh cu bi
danh gia sai), module nay BUILD TU DONG tu document-frequency tren toan bo
JSON OCR output da co (sau khi da qua repetition_guard.py).

Vi sao dung DOCUMENT FREQUENCY (so VIDEO/FRAME KHAC NHAU chua tu do) thay
vi TERM FREQUENCY (tong so lan xuat hien): neu 1 video bi loi lap (Nhom 2
-- "thu thu thu...") thi 1 tu rac co the xuat hien hang chuc lan trong
CHINH video do -> term frequency cao gia tao. Dem theo so video distinct
tranh duoc bay nay vi tu domain that lap lai o NHIEU video khac nhau, con
nhieu OCR ngau nhien thi khong (xem plan muc 3.4).

Rui ro con lai (da ghi nhan, CHUA co giai phap triet de): loi OCR mang
tinh he thong (cung font/cung nguon lap lai nhat quan qua nhieu video) van
co the vuot nguong document-frequency va bi coi nham la dung. Giam thieu
mot phan nho viec luong 2 luon dung CUOI CUNG trong priority cua resolver
(post_ocr_correction.py), chi la catch-all.

CAP NHAT 2026-07-22 (gop build_domain_vocab.py vao day): file CLI wrapper
rieng bi bo (giam so luong file trong pipeline/, nguoi dung phan hoi co
qua nhieu file nho gay roi). CLI gio nam trong khoi `if __name__ ==
"__main__"` cuoi file nay -- chay `python domain_vocab.py --selftest` de
test nhu cu (data gia lap), hoac `python domain_vocab.py --ocr-root ...
--output ...` (hoac `--config configs/ocr_extraction.yaml`) de build that.

CAP NHAT 2026-07-22 (bang chung thuc nghiem tu 18 video that L26-L30):
chay thu voi floor=2 phat hien DUNG rui ro tren -- cac bien the loi OCR
he thong cua "VIETNAM" (VICANAM, VISENAM, VETUAM...) lap nhat quan trong
cung 1 series video, vuot floor=2 va lot vao domain_vocab. Da tang
DEFAULT_MIN_DOC_FREQ_FLOOR len 3 de giam rui ro nay -- CHUA co du lieu de
biet 3 la du hay van con thieu, se can danh gia lai khi co nhieu video
hon (xem ocr_post_correction_plan_addendum.md muc 3). Luu y: voi
ratio=0.0005, floor van la yeu to quyet dinh min_doc_freq cho toi khi
dataset dat ~6.000 video (0.0005 x 6000 = 3) -- tuc la floor gan nhu
LUON la nguong thuc te ap dung o quy mo dataset hien tai cua cuoc thi.

Don vi "document" o day = 1 FILE JSON = 1 VIDEO (dung quy uoc cua
extract_ocr.py: <output_root>/<dataset>/<video_id>.json), KHONG phai tung
frame/keyframe rieng le -- 1 video co nhieu frame lap lai cung 1 banner
van chi tinh la 1 document cho tu do.
"""

from __future__ import annotations

import json
import math
import unicodedata
from collections import Counter
from pathlib import Path

# Nguong toi thieu tuyet doi, bat ke scale dataset nho the nao -- tranh 1
# tu chi xuat hien dung 1 video duy nhat van lot qua threshold ti le khi
# tong so video con it (early stage cua du an).
DEFAULT_MIN_DOC_FREQ_FLOOR = 3
# Ti le tren tong so video da scan -- xem plan muc 3.4:
#   min_doc_freq = max(floor, ceil(ratio * tong_so_video))
DEFAULT_MIN_DOC_FREQ_RATIO = 0.0005


def _normalize_token(token: str) -> str:
    """NFC normalize + lowercase, DUNG CHUNG voi buoc tien xu ly Buoc 0 cua
    toan pipeline post-correction (xem plan muc 3.2) -- tranh 1 tu bi dem
    thanh 2 token khac nhau chi vi lech encode NFC/NFD."""
    return unicodedata.normalize("NFC", token).lower()


def _iter_texts_in_json(json_path: Path):
    """Yield tung text (chua tokenize) trong 1 file JSON OCR cua 1 video.
    Chap nhan CA 2 schema (2 hoac 3 phan tu / frame), giong dung logic
    indexing_pipeline.py de khong bi lech khi save_confidences bat/tat."""
    with json_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    for keyframe_id, values in data.items():
        if len(values) == 2:
            _, texts = values
        elif len(values) == 3:
            _, texts, _ = values
        else:
            raise ValueError(
                f"OCR JSON schema khong dung: keyframe '{keyframe_id}' "
                f"trong '{json_path}' co {len(values)} phan tu (chi chap "
                f"nhan 2 hoac 3)."
            )
        yield from texts


def compute_document_frequency(ocr_json_paths: list[Path]) -> tuple[Counter, int]:
    """Dem document-frequency cho tung token: so file JSON (= so video)
    KHAC NHAU co chua token do it nhat 1 lan (bat ke lap lai bao nhieu lan
    trong CHINH video do).

    Tra ve (doc_freq_counter, tong_so_video_da_scan).
    """
    doc_freq: Counter[str] = Counter()
    total_videos = 0

    for json_path in ocr_json_paths:
        total_videos += 1
        tokens_seen_in_this_video: set[str] = set()

        for text in _iter_texts_in_json(json_path):
            if not text:
                continue
            for raw_token in text.split():
                tokens_seen_in_this_video.add(_normalize_token(raw_token))

        # Cong doc_freq +1 CHO MOI token, chi 1 LAN moi video (dung set o
        # tren de khu trung trong pham vi 1 video truoc khi cong don).
        doc_freq.update(tokens_seen_in_this_video)

    return doc_freq, total_videos


def compute_min_doc_freq(
    total_videos: int,
    ratio: float = DEFAULT_MIN_DOC_FREQ_RATIO,
    floor: int = DEFAULT_MIN_DOC_FREQ_FLOOR,
) -> int:
    """min_doc_freq = max(floor, ceil(ratio * tong_so_video)) -- tuong doi
    theo scale dataset, KHONG hard-code so co dinh (xem plan muc 3.4)."""
    return max(floor, math.ceil(ratio * total_videos))


def build_domain_vocab(
    ocr_json_root: str | Path,
    ratio: float = DEFAULT_MIN_DOC_FREQ_RATIO,
    floor: int = DEFAULT_MIN_DOC_FREQ_FLOOR,
) -> dict:
    """Quet toan bo <ocr_json_root>/<dataset>/<video_id>.json, tinh
    document-frequency, loc theo threshold, tra ve dict:

        {
          "min_doc_freq": <int>,       # threshold thuc te da dung
          "total_videos": <int>,       # tong so video da scan
          "vocab": {token: doc_freq, ...},  # CHI cac token vuot threshold
        }

    LUU Y: ham nay KHONG tu doc yaml config -- xem `_run_cli()` cuoi file
    nay (chay qua `python domain_vocab.py --config ...`) de doc config tu
    ocr_extraction.yaml. Tach rieng de co the goi truc tiep tu code/test
    khong can qua file config.
    """
    ocr_json_root = Path(ocr_json_root)
    json_paths = sorted(ocr_json_root.rglob("*.json"))

    if not json_paths:
        raise FileNotFoundError(
            f"Khong tim thay file JSON OCR nao duoi {ocr_json_root} -- "
            f"kiem tra lai dataset.root trong config."
        )

    doc_freq, total_videos = compute_document_frequency(json_paths)
    min_doc_freq = compute_min_doc_freq(total_videos, ratio=ratio, floor=floor)

    vocab = {
        token: count
        for token, count in doc_freq.items()
        if count >= min_doc_freq
    }

    return {
        "min_doc_freq": min_doc_freq,
        "total_videos": total_videos,
        "vocab": vocab,
    }


def save_domain_vocab(result: dict, output_path: str | Path) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2, sort_keys=True)


def load_domain_vocab(path: str | Path) -> set[str]:
    """Doc file domain_vocab da build, tra ve SET cac token (interface don
    gian nhat de post_ocr_correction.py dung cho exact-match check --
    khong can giu nguyen doc_freq luc lookup, chi can biet co/khong)."""
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    return set(data["vocab"].keys())


def is_domain_word(token: str, vocab: set[str]) -> bool:
    """Exact match trong domain_vocab (da normalize). Dung boi resolver
    (post_ocr_correction.py) o buoc 3 (exact match) va buoc 4 (threshold
    check cho token da co doc_freq nhung chua vuot -- xem module do)."""
    return _normalize_token(token) in vocab


def _run_selftest() -> None:
    import tempfile

    # Self-test: dung du lieu gia lap 3 "video" (file JSON) de kiem tra
    # dung logic document-frequency (khong bi anh huong boi term-frequency
    # gia tao do 1 video lap rac nhieu lan).
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp) / "L21"
        root.mkdir(parents=True)

        # Video 1: tu "sunfest" (domain that) xuat hien 1 lan + rac lap
        # "thu" x10 (gia du CHUA qua repetition_guard de test cach ly).
        (root / "video1.json").write_text(json.dumps({
            "kf1": [[[0, 0, 1, 1]], ["Sunfest 2026 " + "thu " * 10]],
        }), encoding="utf-8")

        # Video 2 & 3: "sunfest" xuat hien lai (domain that, lap qua NHIEU
        # video khac nhau) -- "thu" rac KHONG lap lai o day.
        (root / "video2.json").write_text(json.dumps({
            "kf1": [[[0, 0, 1, 1]], ["Chao mung Sunfest 2026"]],
        }), encoding="utf-8")
        (root / "video3.json").write_text(json.dumps({
            "kf1": [[[0, 0, 1, 1]], ["Sunfest ve roi"]],
        }), encoding="utf-8")

        result = build_domain_vocab(Path(tmp), ratio=0.0005, floor=2)
        print("=== Domain vocab self-test ===")
        print(f"total_videos={result['total_videos']} min_doc_freq={result['min_doc_freq']}")
        print(f"vocab={result['vocab']}")

        assert "sunfest" in result["vocab"], "FAIL: 'sunfest' (domain that, 3 video) phai vuot threshold"
        assert "thu" not in result["vocab"], "FAIL: 'thu' (rac lap trong 1 video) KHONG duoc vuot threshold"
        print("PASS: document-frequency phan biet dung domain-term that vs rac lap-trong-1-video")


def _load_settings_from_yaml(config_path: Path) -> dict:
    """Doc dataset.root + extraction.postprocess.domain_vocab tu
    ocr_extraction.yaml (muc domain_vocab CHUA co san trong yaml hien tai
    -- can them khi tich hop that, dung .get() voi default an toan de
    script van chay duoc khi config chua co muc nay)."""
    import yaml

    with config_path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    dv_cfg = cfg.get("extraction", {}).get("postprocess", {}).get("domain_vocab", {}) or {}

    return {
        "ocr_root": dv_cfg.get("ocr_root", cfg.get("dataset", {}).get("root", "data/processed/ocr")),
        "output": dv_cfg.get("output", "data/processed/domain_vocab.json"),
        "ratio": float(dv_cfg.get("min_doc_freq_ratio", DEFAULT_MIN_DOC_FREQ_RATIO)),
        "floor": int(dv_cfg.get("min_doc_freq_floor", DEFAULT_MIN_DOC_FREQ_FLOOR)),
    }


def _run_cli() -> None:
    """CLI build that -- xem docstring dau file de biet cach chay. Chay 1
    lan (batch, I/O-bound) moi khi co du lieu OCR moi, KHONG phai chay
    per-request luc inference (xem plan muc 6)."""
    import argparse
    import logging

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    logger = logging.getLogger(__name__)

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=str, default=None, help="Duong dan ocr_extraction.yaml")
    parser.add_argument("--ocr-root", type=str, default=None, help="Override root chua JSON OCR (<root>/<dataset>/<video_id>.json)")
    parser.add_argument("--output", type=str, default=None, help="Override duong dan file domain_vocab output")
    parser.add_argument("--ratio", type=float, default=None, help="Override min_doc_freq_ratio")
    parser.add_argument("--floor", type=int, default=None, help="Override min_doc_freq_floor")
    parser.add_argument("--selftest", action="store_true", help="Chay self-test voi data gia lap thay vi build that")
    args = parser.parse_args()

    if args.selftest:
        _run_selftest()
        return

    if args.config:
        settings = _load_settings_from_yaml(Path(args.config))
    else:
        settings = {
            "ocr_root": "data/processed/ocr",
            "output": "data/processed/domain_vocab.json",
            "ratio": DEFAULT_MIN_DOC_FREQ_RATIO,
            "floor": DEFAULT_MIN_DOC_FREQ_FLOOR,
        }

    if args.ocr_root is not None:
        settings["ocr_root"] = args.ocr_root
    if args.output is not None:
        settings["output"] = args.output
    if args.ratio is not None:
        settings["ratio"] = args.ratio
    if args.floor is not None:
        settings["floor"] = args.floor

    logger.info(
        "Building domain_vocab: ocr_root=%s ratio=%s floor=%s",
        settings["ocr_root"], settings["ratio"], settings["floor"],
    )

    result = build_domain_vocab(settings["ocr_root"], ratio=settings["ratio"], floor=settings["floor"])
    save_domain_vocab(result, settings["output"])

    logger.info(
        "Done: %d video da scan, min_doc_freq=%d, %d token vuot threshold -> %s",
        result["total_videos"], result["min_doc_freq"], len(result["vocab"]), settings["output"],
    )


if __name__ == "__main__":
    _run_cli()
