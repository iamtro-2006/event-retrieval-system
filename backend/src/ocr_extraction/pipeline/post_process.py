"""
post_process.py -- (MOI, 2026-08-12) Pipeline hau-OCR RIENG, tach hoan toan
khoi extract_ocr.py theo yeu cau leader: "tao pipeline rieng cho ocr chu kh
gop chung mot lan chay nua" -- chay qua:

    python -m scripts.ocr.post.run

KHAC voi ExtractOCRPipeline._postprocess() (van con trong extract_ocr.py,
nhung tu 2026-08-12 mac dinh TAT ca 3 co postprocess trong
configs/ocr_extraction.yaml, xem comment o do): pipeline nay:
  1. DOC lai JSON da extract xong (input_root/<dataset>/<video_id>.json,
     dung schema [boxes, texts] hoac [boxes, texts, confidences] -- CHI doc
     boxes+texts, LUON BO confidences neu co, khong dung o buoc nao ca) --
     KHONG chay lai model OCR, thuan CPU, chay lai duoc bao nhieu lan tuy
     thich ma khong ton GPU.
  2. Them 3 buoc MOI dac thu VLM (Qwen) TRUOC repetition_guard/post_correction
     cu -- xem vlm_text_normalize.py:
       2a. dedupe_exact()        -- loai dong lap y het trong CUNG 1 frame.
       2b. split_glued_phrases() -- tach cum bi dinh boi ':'/'-'.
       2c. normalize_math_notation() -- unicode sup/subscript -> ASCII.
  3. repetition_guard (Nhom 2, lap trong 1 chuoi) -- TAI SU DUNG nguyen
     repetition_guard.py, LUON goi voi confidence=None (VLM khong co
     confidence that -- xem is_repetition_garbage(): confidence=None tu
     dong fallback ve hanh vi nghiem ngat cu, KHONG can code rieng).
  4. post_correction (Nhom 3, sua chinh ta: dict/regex/domain/bigram/
     diacritic) -- TAI SU DUNG nguyen post_ocr_correction.py + cach wiring
     dictionary/domain_vocab/bigram_table/diacritic_base_index y het
     ExtractOCRPipeline.__init__ (xem extract_ocr.py).

CO Y THUC KHONG lam (theo dung yeu cau leader, "kh dung confidence thi
chuyen giao... kh can loai dau boi vi kieu gi con nay cung halu"):
  - KHONG co buoc confidence_reject (loai text co confidence thap) -- buoc
    nay TON TAI trong extract_ocr.py._postprocess() nhung KHONG duoc mang
    sang day, vi (a) VLM khong co confidence that de loc, (b) leader da
    quyet dinh KHONG can loc hallucination cua VLM o buoc nay.
  - KHONG doc/ghi truong `confidences` du input co hay khong -- output
    LUON LA schema 2 phan tu [boxes, texts] (khong bao gio 3 phan tu), vi
    pipeline nay khong tao ra confidence moi nao.

Input layout : <input_root>/<dataset>/<video_id>.json
Output layout: <output_root>/<dataset>/<video_id>.json (CUNG schema
    [keyframe_id] -> [boxes, texts] nhu extract_ocr.py, de tuong thich
    nguoc voi indexing_pipeline.py/schemas.py phia sau -- KHONG doi schema).
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Callable

from tqdm import tqdm

from src.ocr_extraction.pipeline.repetition_guard import is_repetition_garbage
from src.ocr_extraction.pipeline.post_ocr_correction import correct_text
from src.ocr_extraction.pipeline.domain_vocab import load_domain_vocab
from src.ocr_extraction.pipeline.viet_dictionary import (
    LexiconDictionary,
    HunspellDictionary,
    CombinedDictionary,
    load_wordlist,
)
from src.ocr_extraction.pipeline import bigram_rerank
from src.ocr_extraction.pipeline import diacritic_restoration
from src.ocr_extraction.pipeline.vlm_text_normalize import (
    split_glued_phrases,
    normalize_math_notation,
)


# ---------------------------------------------------------------------
# (MOI, 2026-08-12 -- refactor kien truc, KHONG doi hanh vi) 4 "bucket"
# tong quat cho cac buoc hau-VLM (2a/2b/2c) + repetition_guard, thay vi
# goi ten tung ham cu the tuan tu trong _normalize_frame(). Ly do: 5 buoc
# cu (dedupe/split/math/repetition_guard/post_correction) doc rieng le
# trong nhin GIONG 5 fix cu the, nhung thuc chat moi buoc thuoc 1 trong 4
# "shape" thao tac ben duoi -- phan loai theo SHAPE (khong phai theo NOI
# DUNG loi sua) moi la truc tong quat dung, vi noi dung loi (vd them
# separator moi ngoai ':'/' - ') PHAI cho data that (xem thao luan thiet
# ke 2026-08-12, KHONG doan mo o day):
#
#   - frame filter    : (boxes, texts) cua 1 frame -> (boxes, texts) it
#                        hon -- loai theo dieu kien tren CA list (vd trung
#                        lap voi phan tu khac). dedupe_exact thuoc day.
#   - line segmenter   : 1 dong -> N dong (>=1) -- box goc duoc nhan ban
#                        cho N dong con. split_glued_phrases thuoc day.
#   - line rewriter    : 1 dong -> 1 dong (doi NOI DUNG, KHONG doi so
#                        luong/box). normalize_math_notation thuoc day.
#   - line filter      : 1 dong -> giu/bo (predicate, KHONG doi noi dung
#                        dong con lai). repetition_guard (is_repetition_
#                        garbage) thuoc day.
#
# Token-resolver (post_correction/correct_text) KHONG nam trong 4 bucket
# tren -- no da tu tong quat hoa san tu truoc (resolver chain uu tien
# rieng, SpellDictionary la 1 Protocol pluggable, xem post_ocr_correction.
# py), nen giu nguyen la 1 buoc rieng, khong ep vao khung nay.
#
# Loi ich: 1 buoc moi sau nay CHI can tra loi "no thuoc bucket nao" roi
# append vao dung list trong __init__ (xem duoi) -- KHONG can sua them gi
# trong _normalize_frame()/cac ham _apply_*_* (thu tu chay bucket van co
# dependency that: filter frame -> segment -> rewrite -> filter dong ->
# resolve token, xem docstring _normalize_frame()).
# ---------------------------------------------------------------------

def _dedupe_exact_with_boxes(
    boxes: list[list[float]], texts: list[str]
) -> tuple[list[list[float]], list[str]]:
    """Bien the CUA dedupe_exact() (xem vlm_text_normalize.py) dong bo theo
    boxes -- KHONG goi thang dedupe_exact() vi ham do chi nhan/tra
    list[str] don thuan, khong biet box nao di kem tung text. Logic giong
    HET dedupe_exact(): giu lan xuat hien DAU TIEN cua moi gia tri (sau
    strip), chuoi rong/toan whitespace khong bi dedupe."""
    seen: set[str] = set()
    kept_boxes: list[list[float]] = []
    kept_texts: list[str] = []
    for box, text in zip(boxes, texts):
        key = text.strip()
        if key and key in seen:
            continue
        if key:
            seen.add(key)
        kept_boxes.append(box)
        kept_texts.append(text)
    return kept_boxes, kept_texts


class PostOCRPipeline:
    """Xem docstring dau file. Doc config tu cfg["post_process"] (KHAC
    cfg["extraction"] cua ExtractOCRPipeline -- 2 section doc lap, khong
    dung chung field nao, de 2 pipeline that su tach roi nhau ca ve code
    lan cau hinh)."""

    def __init__(self, cfg: dict, logger: logging.Logger | None = None):
        self.cfg = cfg
        self.logger = logger or logging.getLogger(__name__)

        pp_cfg = cfg["post_process"]

        self.input_root = Path(pp_cfg["input_root"])
        self.output_root = Path(pp_cfg["output_root"])
        self.skip_existing = bool(pp_cfg.get("skip_existing", False))

        # --- Buoc 2a/2b/2c (MOI, dac thu VLM) -- bat/tat rieng tung buoc,
        # mac dinh true ca 3 (xem vlm_text_normalize.py).
        self.enable_dedupe = bool(pp_cfg.get("dedupe", True))
        self.enable_split_glued_phrases = bool(pp_cfg.get("split_glued_phrases", True))
        self.enable_normalize_math = bool(pp_cfg.get("normalize_math_notation", True))

        # --- Buoc 3: repetition_guard (tai su dung nguyen tham so y het
        # extraction.postprocess.repetition_guard_params, xem extract_ocr.py).
        self.enable_repetition_guard = bool(pp_cfg.get("repetition_guard", True))
        rg_params_cfg = pp_cfg.get("repetition_guard_params", {}) or {}
        self.repetition_guard_params = {
            "min_repeat_hard": int(rg_params_cfg.get("min_repeat_hard", 5)),
            "min_repeat_soft": int(rg_params_cfg.get("min_repeat_soft", 3)),
            "soft_confidence_threshold": float(rg_params_cfg.get("soft_confidence_threshold", 0.5)),
            "exempt_numeric": bool(rg_params_cfg.get("exempt_numeric", True)),
        }

        # --- 4 bucket tong quat (xem ghi chu kien truc dau file) -- moi
        # bucket la list[(ten, ham)], rong neu co lien quan bi tat qua
        # config. Thu tu bucket CO Y NGHIA (dependency that, xem docstring
        # _normalize_frame()); thu tu CAC HAM TRONG CUNG 1 bucket cung
        # chay tuan tu neu sau nay co >1 ham cung bucket.
        self.frame_filters: list[tuple[str, Callable]] = []
        if self.enable_dedupe:
            self.frame_filters.append(("dedupe_exact", _dedupe_exact_with_boxes))

        self.line_segmenters: list[tuple[str, Callable]] = []
        if self.enable_split_glued_phrases:
            self.line_segmenters.append(("split_glued_phrases", split_glued_phrases))

        self.line_rewriters: list[tuple[str, Callable]] = []
        if self.enable_normalize_math:
            self.line_rewriters.append(("normalize_math_notation", normalize_math_notation))

        self.line_filters: list[tuple[str, Callable]] = []
        if self.enable_repetition_guard:
            rg_params = self.repetition_guard_params  # capture, xem ham duoi

            def _repetition_guard_keep(text: str, _params: dict = rg_params) -> bool:
                # confidence LUON None (VLM khong co confidence that -- xem
                # docstring dau file va _apply_line_filters()).
                return not is_repetition_garbage(text, confidence=None, **_params)

            self.line_filters.append(("repetition_guard", _repetition_guard_keep))

        # --- Buoc 4: post_correction (tai su dung nguyen cach wiring cua
        # ExtractOCRPipeline.__init__ -- xem extract_ocr.py, KHONG doi logic,
        # chi copy sang de 2 pipeline doc lap nhau khong phu thuoc chay
        # truoc/sau nhau).
        self.enable_post_correction = bool(pp_cfg.get("post_correction", True))
        pc_cfg = pp_cfg.get("post_correction_config", {}) or {}

        self.dictionary = None
        self.domain_vocab: set[str] = set()
        self.correction_cache: dict = {}
        self.bigram_near_words_cache: dict = {}

        if self.enable_post_correction:
            dictionary_kind = str(pc_cfg.get("dictionary", "combined")).lower()
            if dictionary_kind == "hunspell":
                self.dictionary = HunspellDictionary()
            elif dictionary_kind in ("lexicon", "merged"):
                self.dictionary = LexiconDictionary()
            elif dictionary_kind == "combined":
                self.dictionary = CombinedDictionary([LexiconDictionary(), HunspellDictionary()])
            else:
                raise ValueError(
                    f"post_correction_config.dictionary khong hop le: "
                    f"{dictionary_kind!r} (chi nhan 'lexicon', 'merged', "
                    f"'hunspell', hoac 'combined')."
                )

            domain_vocab_path = pc_cfg.get("domain_vocab_path")
            if domain_vocab_path and Path(domain_vocab_path).exists():
                self.domain_vocab = load_domain_vocab(domain_vocab_path)
                self.logger.info(
                    "post_process: da nap domain_vocab (%d tu) tu %s",
                    len(self.domain_vocab), domain_vocab_path,
                )
            else:
                self.logger.warning(
                    "post_process: KHONG tim thay domain_vocab (path=%s) -- "
                    "luong 2 tam thoi tat (dung set rong).",
                    domain_vocab_path,
                )

            bigram_path = pc_cfg.get("bigram_table_path")
            self.bigram_table = bigram_rerank.load_bigram_table(
                bigram_path if bigram_path else bigram_rerank.DEFAULT_BIGRAM_PATH
            )
            self.bigram_alphabet = (
                bigram_rerank.alphabet_from_bigram_table(self.bigram_table)
                if self.bigram_table else ""
            )
            if self.bigram_table:
                self.logger.info(
                    "post_process: da nap bigram_table (%d cap tu).",
                    len(self.bigram_table),
                )
            else:
                self.logger.warning("post_process: KHONG tim thay vi_bigram_freq.tsv.")

            wordlist = load_wordlist()
            self.diacritic_base_index = diacritic_restoration.build_base_index(wordlist)
            self.logger.info(
                "post_process: da xay diacritic_base_index (%d skeleton tu %d tu).",
                len(self.diacritic_base_index), len(wordlist),
            )
        else:
            self.bigram_table = {}
            self.bigram_alphabet = ""
            self.diacritic_base_index = {}

    # ------------------------------------------------------------------
    # Discovery -- CUNG layout voi ExtractOCRPipeline.output_json_for(),
    # vi input_root o day thuong CHINH LA output_root cua buoc extract.
    # ------------------------------------------------------------------

    def scan_input_files(self) -> list[Path]:
        if not self.input_root.exists():
            raise FileNotFoundError(f"post_process input_root not found: {self.input_root}")

        files = []
        for dataset_dir in sorted(self.input_root.iterdir()):
            if not dataset_dir.is_dir():
                continue
            for f in sorted(dataset_dir.glob("*.json")):
                files.append(f)
        return files

    def output_json_for(self, input_file: Path) -> Path:
        dataset = input_file.parent.name
        return self.output_root / dataset / input_file.name

    # ------------------------------------------------------------------
    # Core processing -- 1 frame (1 keyframe_id trong 1 file JSON)
    # ------------------------------------------------------------------

    def _normalize_frame(
        self, boxes: list[list[float]], texts: list[str]
    ) -> tuple[list[list[float]], list[str]]:
        """Chay 4 bucket tong quat (frame-filter -> line-segmenter ->
        line-rewriter -> line-filter, xem ghi chu kien truc dau file) +
        token-resolver rieng, cho 1 frame. Thu tu bucket la dependency
        THAT (khong doi tuy y):
          1. frame-filter TRUOC segmenter -- dedupe tren dong GOC (VLM lap
             ca dong y het), khong phai tren tung cum da tach nho.
          2. segmenter TRUOC rewriter -- rewriter (vd math notation) can
             chay tren TUNG cum ngu nghia da tach, khong phai ca cum dinh
             lien (an toan hon, tranh sua nham qua ranh gioi cum).
          3. rewriter TRUOC line-filter -- repetition_guard can xet dang
             text DA chuan hoa (vd sau khi doi ky hieu toan hoc), khong
             phai dang tho.
          4. line-filter TRUOC token-resolver -- khong ton cong resolve
             chinh ta cho dong da bi loai la rac.

        giu boxes/texts dong bo index xuyen suot (loai/tach 1 text thi box
        tuong ung cung phai loai/nhan doi theo, dung quy uoc cu cua
        extract_ocr.py._postprocess()).

        LUU Y ve box khi line-segmenter tach 1 dong thanh N dong: box GOC
        (kha nang la placeholder [0,0,0,0] cho engine qwen, hoac toa do
        that cho paddle_vietocr) duoc DUNG LAI (copy) cho ca N dong con --
        day la XAP XI (dong con khong co toa do rieng, vi VLM khong tra
        bbox theo tung cum ngu nghia trong 1 dong), chap nhan duoc vi muc
        dich chinh la ho tro tim kiem theo token, khong phai hien thi box
        chinh xac tung cum.
        """
        boxes, texts = self._apply_frame_filters(boxes, texts)
        boxes, texts = self._apply_line_segmenters(boxes, texts)
        boxes, texts = self._apply_line_rewriters(boxes, texts)
        boxes, texts = self._apply_line_filters(boxes, texts)
        texts = self._apply_token_resolver(texts)
        return boxes, texts

    # ------------------------------------------------------------------
    # Bucket runner -- 1 ham rieng cho moi "shape" thao tac (xem ghi chu
    # kien truc dau file). Them 1 buoc moi sau nay KHONG can sua ham nao
    # trong nhom nay -- chi can append (ten, ham) vao dung list bucket
    # trong __init__.
    # ------------------------------------------------------------------

    def _apply_frame_filters(
        self, boxes: list[list[float]], texts: list[str]
    ) -> tuple[list[list[float]], list[str]]:
        """(boxes, texts) -> (boxes, texts) it hon -- loai theo dieu kien
        tren CA list (vd trung lap voi phan tu khac trong cung frame)."""
        for _name, fn in self.frame_filters:
            boxes, texts = fn(boxes, texts)
        return boxes, texts

    def _apply_line_segmenters(
        self, boxes: list[list[float]], texts: list[str]
    ) -> tuple[list[list[float]], list[str]]:
        """1 dong -> N dong (>=1) -- box goc nhan ban cho tat ca dong con.
        Neu co >1 segmenter dang ky, chay TUAN TU (segmenter sau nhan dau
        vao la KET QUA cua segmenter truoc, khong phai dong goc) -- hien
        tai chi co 1 (split_glued_phrases) nen chua co case nhieu tang
        thuc te, nhung viet san cho dung tong quat."""
        if not self.line_segmenters:
            return boxes, texts
        out_boxes: list[list[float]] = []
        out_texts: list[str] = []
        for box, text in zip(boxes, texts):
            pieces = [text]
            for _name, fn in self.line_segmenters:
                next_pieces: list[str] = []
                for p in pieces:
                    next_pieces.extend(fn(p))
                pieces = next_pieces
            for p in pieces:
                out_boxes.append(box)
                out_texts.append(p)
        return out_boxes, out_texts

    def _apply_line_rewriters(
        self, boxes: list[list[float]], texts: list[str]
    ) -> tuple[list[list[float]], list[str]]:
        """1 dong -> 1 dong (doi NOI DUNG, khong doi so luong/box). Nhieu
        rewriter (neu co sau nay) chay tuan tu tren KET QUA cua rewriter
        truoc do (giong tinh than line-segmenter o tren)."""
        if not self.line_rewriters:
            return boxes, texts
        out_texts: list[str] = []
        for text in texts:
            for _name, fn in self.line_rewriters:
                text = fn(text)
            out_texts.append(text)
        return boxes, out_texts

    def _apply_line_filters(
        self, boxes: list[list[float]], texts: list[str]
    ) -> tuple[list[list[float]], list[str]]:
        """1 dong -> giu/bo (predicate, khong doi noi dung dong con lai).
        1 dong bi bat ky filter nao tra False la bi loai (AND logic, giong
        het hanh vi repetition_guard don le truoc day)."""
        if not self.line_filters:
            return boxes, texts
        kept_boxes: list[list[float]] = []
        kept_texts: list[str] = []
        for box, text in zip(boxes, texts):
            if all(fn(text) for _name, fn in self.line_filters):
                kept_boxes.append(box)
                kept_texts.append(text)
        return kept_boxes, kept_texts

    def _apply_token_resolver(self, texts: list[str]) -> list[str]:
        """post_correction (Nhom 3) -- tai su dung nguyen resolver cua
        post_ocr_correction.py. KHONG nam trong 4 bucket tren (xem ghi chu
        kien truc dau file: day da la 1 khung tong quat rieng, chain uu
        tien voi SpellDictionary pluggable, khong can ep vao khung filter/
        segment/rewrite)."""
        if not (self.enable_post_correction and self.dictionary is not None):
            return texts
        corrected_texts = []
        for text in texts:
            if not text:
                corrected_texts.append(text)
                continue
            corrected, _results = correct_text(
                text, self.dictionary, self.domain_vocab,
                cache=self.correction_cache,
                bigram_table=self.bigram_table,
                bigram_alphabet=self.bigram_alphabet,
                bigram_near_words_cache=self.bigram_near_words_cache,
                diacritic_base_index=self.diacritic_base_index,
            )
            corrected_texts.append(corrected)
        return corrected_texts

    def process_file(self, input_file: Path) -> int:
        output_file = self.output_json_for(input_file)

        if output_file.exists() and self.skip_existing:
            self.logger.info("Skip existing: %s", output_file)
            return 0

        with input_file.open("r", encoding="utf-8") as f:
            raw = json.load(f)

        data: dict[str, list] = {}
        for keyframe_id, entry in raw.items():
            # entry co the la [boxes, texts] hoac [boxes, texts, confidences]
            # -- buoc nay LUON bo confidences neu co (xem docstring dau file).
            boxes, texts = entry[0], entry[1]
            boxes, texts = self._normalize_frame(boxes, texts)
            if texts:
                data[keyframe_id] = [boxes, texts]

        output_file.parent.mkdir(parents=True, exist_ok=True)
        with output_file.open("w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

        self.logger.info("Saved %d keyframes -> %s", len(data), output_file)
        return len(data)

    def run(self) -> None:
        input_files = self.scan_input_files()
        self.logger.info("Found %d OCR JSON files to post-process", len(input_files))

        total_saved = 0
        for input_file in tqdm(input_files, desc="post-ocr"):
            try:
                total_saved += self.process_file(input_file)
            except Exception as e:
                self.logger.error("Failed %s: %s", input_file, e)

        self.logger.info("Total keyframes post-processed: %d", total_saved)
