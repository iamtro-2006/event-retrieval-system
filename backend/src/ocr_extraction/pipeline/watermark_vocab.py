"""
Watermark/logo vocab -- nhan dien token OCR lap lai QUA GAN NHU MOI VIDEO
(watermark dai, banner "...ONLINE" co dinh...) de LOC O TANG RETRIEVAL, KHONG
phai o tang extraction (giu nguyen text OCR goc trong JSON/index -- chi giam
trong so luc tinh diem tim kiem, xem repository.py::search_multi_field()).

VI SAO KHONG XU LY O TANG EXTRACTION (theo yeu cau nguoi dung 2026-07-17,
"khả năng là nên xử lí loại bỏ ra lúc retrieval, chứ để vậy thì loãng lắm"):
xoa thang khoi text luc OCR se MAT VINH VIEN thong tin (khong the khoi phuc
neu sau nay can, vd audit/debug), va watermark token VAN co the la tin hieu
that neu nguoi dung CHU DICH tim kenh (vd query "HTV"). Loc o tang retrieval
(giam trong so, khong xoa) giu duoc ca 2: khong loang ket qua tim kiem
thuong, van tim duoc neu nguoi dung thuc su can.

KHAC domain_vocab.py the nao (dung CHUNG co che compute_document_frequency(),
khac tieu chi loc):
    - domain_vocab.py: nguong THAP (floor=3 hoac ratio 0.0005) -- muc dich
      la BAT duoc ten rieng/domain-term HIEM nhung lap lai o vai video khac
      nhau (khong co trong tu dien nhung van dung).
    - watermark_vocab.py (file nay): nguong CAO (mac dinh 0.3 = xuat hien o
      >=30% SO VIDEO) -- muc dich NGUOC LAI, tim token GAN NHU O KHAP MOI
      NOI (dac trung watermark/banner co dinh, khong phai domain-term dac
      thu 1 chu de/su kien).
    - CA HAI co the trung nhau mot phan (vd "htv" co the vuot ca 2 nguong)
      -- khong sao, 2 file phuc vu 2 muc dich khac nhau (domain_vocab dung
      o Luong 2 cua post-correction, watermark_vocab dung o retrieval).

Loc BO tu dien chuan (dictionary.exists()) truoc khi xep hang theo ty le
document-frequency -- neu khong loc, cac stopword tieng Viet thuong ("và",
"của", "có"...) se chiem het top ranking chi vi la tu thuong gap trong BAT
KY van ban tieng Viet nao, KHONG phai vi la watermark that (da xac nhan qua
du lieu that 30 video 2026-07-24: top-40 khong loc toan stopword, loc xong
moi hien ra dung "htv"/"online" va cac bien the loi OCR cua chung).
"""

from __future__ import annotations

import json
from pathlib import Path

from src.ocr_extraction.pipeline.domain_vocab import compute_document_frequency

DEFAULT_MIN_DOC_FREQ_RATIO = 0.3
# San can toi thieu tuyet doi -- tranh dataset qua nho (vd <10 video) khien
# 1 token xuat hien o 3-4 video da vuot 30% ma thuc ra chua du bang chung.
DEFAULT_MIN_VIDEOS_ABSOLUTE = 5

# (2026-07-24) LOC TAY sau khi chay thu tren 30 video that: xep hang thuan
# theo document-frequency + loc tu dien VAN con lot mot so tu THAT (khong
# phai watermark) chi vi tinh co lap lai o nhieu video CUNG chu de (vd du
# lich/thoi su dia phuong) -- giong dung tinh chat "de nham" da gap voi
# ACRONYM_WHITELIST truoc day (xem TIEN_DO_OCR.md 2026-07-22). Vi du that
# bi loai o day sau khi kiem tra thu cong:
#   - "mekong": DIA DANH THAT (song Mekong/DBSCL) -- video du lich/thoi su
#     mien Tay nhac lai nhieu lan, KHONG phai watermark, se lam mat kha
#     nang tim kiem that neu nguoi dung go "mekong".
#   - "thuy": nhieu kha nang la 1 PHAN TEN RIENG (vd "Thúy" mat dau) --
#     rui ro tuong tu, khong du bang chung la watermark.
#   - "quayphim": cum "quay phim" (credit lam phim) -- noi dung THAT co
#     the nguoi dung muon tim (vd ten quay phim), khong phai logo/banner
#     co dinh vo nghia.
#   - "mat", "19", "hit": qua ngan hoac qua chung chung de ket luan chac
#     chan la watermark (co the la mot phan cua tu/so that trong ngu canh
#     khac), thieu bang chung ro rang bang cac bien the ho HTV/ONLINE.
# GIU LAI: 2 ho bien the loi OCR ro rang (da xac nhan qua nhieu video khac
# nhau, cung 1 nguon watermark that): "htv"/"htm"/"huv"/"hov"/"hty" (doc
# nham watermark dai HTV) va "online"/"onlines"/"onlin"/"unline"/"onl"/
# "onli"/"on"/"line" (doc nham/cat vun banner "...ONLINE" co dinh -- "on"/
# "line"/"ng"/"th" la cac manh vun do box-detection cat roi banner nay).
MANUAL_EXCLUDE: set[str] = {"mekong", "thuy", "quayphim", "mat", "19", "hit"}


def build_watermark_vocab(
    ocr_json_root: str | Path,
    dictionary,
    ratio: float = DEFAULT_MIN_DOC_FREQ_RATIO,
    min_videos_absolute: int = DEFAULT_MIN_VIDEOS_ABSOLUTE,
) -> dict:
    """Quet toan bo <ocr_json_root>/<dataset>/<video_id>.json (dung lai
    compute_document_frequency() cua domain_vocab.py), loc token KHONG co
    trong tu dien chuan (`dictionary.exists()`) VA co document-frequency
    ratio >= `ratio` (VA >= `min_videos_absolute` video tuyet doi).

    `dictionary`: bat ky object co .exists(token) -> bool (Protocol
    SpellDictionary, xem post_ocr_correction.py) -- truyen vao ngoai (khong
    tu import LexiconDictionary o day) de test duoc voi dictionary gia,
    khong bat buoc phai co vi_wordlist.txt that.
    """
    ocr_json_root = Path(ocr_json_root)
    json_paths = sorted(ocr_json_root.rglob("*.json"))
    if not json_paths:
        raise FileNotFoundError(
            f"Khong tim thay file JSON OCR nao duoi {ocr_json_root}."
        )

    doc_freq, total_videos = compute_document_frequency(json_paths)
    min_count = max(min_videos_absolute, round(ratio * total_videos))

    watermark_tokens = {
        token: count
        for token, count in doc_freq.items()
        if count >= min_count
        and len(token) >= 2
        and not dictionary.exists(token)
        and token not in MANUAL_EXCLUDE
    }

    return {
        "min_doc_freq_ratio": ratio,
        "min_videos_absolute": min_videos_absolute,
        "min_count_applied": min_count,
        "total_videos": total_videos,
        "watermark_tokens": watermark_tokens,
    }


def save_watermark_vocab(result: dict, output_path: str | Path) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2, sort_keys=True)


def load_watermark_tokens(path: str | Path) -> set[str]:
    """Doc file da build, tra ve SET token (interface don gian nhat cho
    repository.py -- chi can biet co/khong, khong can giu doc_freq)."""
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    return set(data["watermark_tokens"].keys())


def _run_cli() -> None:
    import argparse
    import logging

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    logger = logging.getLogger(__name__)

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ocr-root", type=str, required=True, help="Root chua JSON OCR (<root>/<dataset>/<video_id>.json)")
    parser.add_argument("--output", type=str, default="data/processed/watermark_vocab.json")
    parser.add_argument("--ratio", type=float, default=DEFAULT_MIN_DOC_FREQ_RATIO)
    parser.add_argument("--min-videos", type=int, default=DEFAULT_MIN_VIDEOS_ABSOLUTE)
    args = parser.parse_args()

    from src.ocr_extraction.pipeline.viet_dictionary import LexiconDictionary

    dictionary = LexiconDictionary()
    result = build_watermark_vocab(
        args.ocr_root, dictionary, ratio=args.ratio, min_videos_absolute=args.min_videos,
    )
    save_watermark_vocab(result, args.output)

    logger.info(
        "Done: %d video da scan, min_count=%d, %d token watermark -> %s",
        result["total_videos"], result["min_count_applied"],
        len(result["watermark_tokens"]), args.output,
    )


if __name__ == "__main__":
    _run_cli()
