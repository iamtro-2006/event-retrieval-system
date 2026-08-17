"""
Khoi phuc dau tieng Viet (diacritic restoration) -- xu ly 1 phan cua real-
word error (nhom "sai dau/ky tu phu", xem TIEN_DO_OCR.md entry 2026-07-30
"chot: doc lai toan bo 90 note"): token OCR DA la 1 tu that trong tu dien
(qua duoc Buoc 1 dict_exact cua post_ocr_correction.py) nhung SAI DAU so voi
ngu canh (vd "MẤT" dung nhung ngu canh can "MẮT").

KHAC bigram_rerank.py (module da co) o 2 diem:
  1. KHONG GIAN CANDIDATE: bigram_rerank._near_words() sinh candidate qua
     edit-distance-1 tren TOAN BO tu vung (hang nghin candidate/token, qua
     nhieu nhieu -- day la ly do bigram_rerank khi thu lam AUTO-CORRECT
     (khong chi flag) co precision chi 6.7%, xem TIEN_DO_OCR.md 07-30). O
     day, candidate CHI la cac tu CUNG SKELETON (khung phu am/nguyen am sau
     khi bo dau, xem strip_diacritics()) -- thu hep dung theo BAN CHAT loi
     da xac dinh duoc qua doc thu cong 90 note ground truth.
  2. HANH VI KHI TRIGGER: bigram_rerank CHI gan nhan "uncertain" (khong tu
     sua). O day, khi vuot NGUONG (giong het MIN_CANDIDATE_COUNT/MIN_RATIO
     cua bigram_rerank -- TAI SU DUNG nguyen ngưỡng đã calibrate, khong tu
     bia nguong moi), THAT SU tra ve candidate de caller GHI DE corrected --
     xem ket qua kiem chung 2026-07-30 (chot): precision-khi-trigger 90.0%
     (9/10), FP tren nhom correct chi 1.8% (1/55 dong) -- du tin cay de tu
     dong sua, KHAC bigram_rerank (chua du tin cay, van chi flag).

QUAN TRONG -- PHAM VI: module nay CHI xu ly dung nhom da kiem chung (sai
dau/ky tu phu tren tu DA hop le trong tu dien). KHONG dung cho nhom "sai ky
tu that" (khac skeleton) hay nhom "dong am" (khong co khac biet chinh ta) --
2 nhom do van CHUA co giai phap du tin cay (xem TIEN_DO_OCR.md).
"""

from __future__ import annotations

import unicodedata

from src.ocr_extraction.pipeline import bigram_rerank
from src.ocr_extraction.pipeline.viet_dictionary import LexiconDictionary, load_wordlist

# (2026-07-31, MOI) Nguong RIENG cho module nay -- KHONG con tai su dung
# nguyen ngưỡng cua bigram_rerank nua (30, 80.0). Ly do: nguong do calibrate
# cho bai toan KHAC (_near_words(), hang nghin candidate/token tren toan bo
# tu vung), trong khi module nay candidate da bi thu hep san con 2-3 lua
# chon CUNG SKELETON (qua base_index) -- khong gian tim kiem nho hon RAT
# NHIEU nen chiu duoc nguong long hon ma FP khong tang tuong ung.
#
# Grid-search tren 42 cap nhom 1 (recall) + 55 dong "correct" (FP), xem
# dev/ocr_eval/tune_diacritic_threshold.py + TIEN_DO_OCR.md entry 2026-07-31
# "Huong 1: do lai nguong":
#   (30, 80.0) cu:  recall 23.8%, precision@trigger 90.0%, FP dong 1.8% (1/55)
#   (10, 30.0):     recall 38.1%, precision@trigger 93.8%, FP dong 1.8% (1/55)
#                    -- CAI THIEN THUAN TUY, FP KHONG DOI, khong danh doi gi.
#   (5, 10.0) CHON: recall 59.5%, precision@trigger 84.0%, FP dong 3.6% (2/55)
#                    -- them dung 1 FP MOI ("nam"->"năm" trong cau ran watermark
#                    nhieu, ground truth nhan la correct nhung cau qua nhieu
#                    de tin chac 100%) so voi nguong (10,30.0); FP con lai
#                    ("vồ"->"vỏ" trong ten rieng "Cá Vồ") la FP DA BIET tu
#                    truoc (07-30), khong phai moi.
# Duoi 10.0 (vd 9.0), FP nhay len tier tiep theo (7.3%, gap doi) trong khi
# recall chi tang them ~2 diem % -- KHONG dang danh doi, dung o day.
MIN_CANDIDATE_COUNT = 5
MIN_RATIO = 10.0

_STRIP_MAP = {
    "à": "a", "á": "a", "ả": "a", "ã": "a", "ạ": "a",
    "ă": "a", "ằ": "a", "ắ": "a", "ẳ": "a", "ẵ": "a", "ặ": "a",
    "â": "a", "ầ": "a", "ấ": "a", "ẩ": "a", "ẫ": "a", "ậ": "a",
    "è": "e", "é": "e", "ẻ": "e", "ẽ": "e", "ẹ": "e",
    "ê": "e", "ề": "e", "ế": "e", "ể": "e", "ễ": "e", "ệ": "e",
    "ì": "i", "í": "i", "ỉ": "i", "ĩ": "i", "ị": "i",
    "ò": "o", "ó": "o", "ỏ": "o", "õ": "o", "ọ": "o",
    "ô": "o", "ồ": "o", "ố": "o", "ổ": "o", "ỗ": "o", "ộ": "o",
    "ơ": "o", "ờ": "o", "ớ": "o", "ở": "o", "ỡ": "o", "ợ": "o",
    "ù": "u", "ú": "u", "ủ": "u", "ũ": "u", "ụ": "u",
    "ư": "u", "ừ": "u", "ứ": "u", "ử": "u", "ữ": "u", "ự": "u",
    "ỳ": "y", "ý": "y", "ỷ": "y", "ỹ": "y", "ỵ": "y",
    "đ": "d",
}


def _normalize(token: str) -> str:
    return unicodedata.normalize("NFC", token).lower()


def strip_diacritics(word: str) -> str:
    """Bo dau thanh + ky tu phu (ă/â/ê/ơ/ư/đ) ve khung co ban -- CHI dung de
    xac dinh 2 tu co "cung skeleton" hay khong (khong dung de hien thi/luu
    ket qua, luon giu nguyen dang co dau that su cua wordlist)."""
    return "".join(_STRIP_MAP.get(ch, ch) for ch in _normalize(word))


def build_base_index(words: "LexiconDictionary | set[str] | list[str]") -> dict[str, list[str]]:
    """Dao nguoc 1 tap tu: skeleton-khong-dau -> list cac tu THAT cung khung
    nhung khac dau (vd "ma" -> ["ma","má","mà","mả","mã","mạ"], tuy wordlist
    thuc te co du het cac bien the hay khong).

    `words` co the la: (a) 1 LexiconDictionary (lay `_single_token_words` --
    TUONG THICH NGUOC voi cach goi ban dau), hoac (b) truc tiep 1 set/list
    tu don (KHUYEN DUNG -- doc lap voi SpellDictionary nao dang duoc cau
    hinh, xem ghi chu duoi). Neu caller dang dung HunspellDictionary hoac
    CombinedDictionary (KHONG co attribute `_single_token_words`), truyen
    truc tiep `viet_dictionary.load_wordlist()` (single-token subset) thay vi
    dictionary instance -- xem extract_ocr.py.

    Goi 1 LAN duy nhat luc khoi tao pipeline (giong cach load_bigram_table()
    duoc goi 1 lan trong ExtractOCRPipeline.__init__), KHONG goi lai moi
    token/frame."""
    word_iter = words._single_token_words if hasattr(words, "_single_token_words") else words
    index: dict[str, list[str]] = {}
    for w in word_iter:
        if " " in w:
            # Bo qua cum-tu nhieu-token (vd "nhi khoa") -- module nay chi
            # xu ly TUNG TOKEN don, giong pham vi dictionary.exists() cho
            # tu don trong viet_dictionary.py (_single_token_words).
            continue
        base = strip_diacritics(w)
        index.setdefault(base, []).append(w)
    return index


def suggest_diacritic_autocorrect(
    token: str,
    prev_token: str | None,
    next_token: str | None,
    base_index: dict[str, list[str]],
    bigram_table: dict[tuple[str, str], int],
    min_candidate_count: int = MIN_CANDIDATE_COUNT,
    min_ratio: float = MIN_RATIO,
) -> str | None:
    """Tra ve candidate KHAC DAU (cung skeleton) neu co bang chung ngu canh
    ĐỦ MẠNH (giong het nguong bigram_rerank.suggest_real_word_flag(), TAI SU
    DUNG nguyen cong thuc kiem tra -- xem module docstring), hoac None neu
    khong co gi vuot nguong. KHAC bigram_rerank.suggest_real_word_flag(): ham
    nay CHI tim trong base_index (cung skeleton), KHONG goi _near_words()
    (edit-distance-1 toan bo tu vung).

    Caller (post_ocr_correction.correct_text()) chi nen goi ham nay cho token
    da "pass" o Buoc 1 (dict_exact, tuc la DA la tu that) -- giong dieu kien
    ap dung cua bigram_rerank, VA nen goi TRUOC bigram_rerank (neu ham nay
    tra ve candidate, GHI DE corrected luon, khong can bigram_rerank flag
    lai token do nua)."""
    if not bigram_table:
        return None
    if prev_token is None and next_token is None:
        return None

    normalized = _normalize(token)
    candidates = [v for v in base_index.get(strip_diacritics(normalized), []) if v != normalized]
    if not candidates:
        return None

    original_evidence = bigram_rerank._context_evidence(bigram_table, prev_token, normalized, next_token)

    best_candidate = None
    best_evidence = 0
    for cand in candidates:
        evidence = bigram_rerank._context_evidence(bigram_table, prev_token, cand, next_token)
        if evidence > best_evidence:
            best_evidence = evidence
            best_candidate = cand

    if best_candidate is None:
        return None
    if best_evidence < min_candidate_count:
        return None
    if best_evidence < (original_evidence + 1) * min_ratio:
        return None

    return best_candidate


if __name__ == "__main__":
    print("=== diacritic_restoration self-test ===")
    dictionary = LexiconDictionary()
    base_index = build_base_index(dictionary)
    print(f"Da xay base_index: {len(base_index)} skeleton tu {len(dictionary._single_token_words)} tu")

    bigram_table = bigram_rerank.load_bigram_table(bigram_rerank.DEFAULT_BIGRAM_PATH)
    if not bigram_table:
        print("(Bo qua test can bigram_table -- khong tim thay vi_bigram_freq.tsv)")
    else:
        # Vi du that tu TIEN_DO_OCR.md entry 07-30 (chot): "Phát" trong ngu
        # canh "giáo Hòa Hảo" nen duoc sua thanh "phật".
        cases = [
            ("phát", "chinh", "giáo", "phật"),  # ky vong sua dung
            ("mất", "đôi", None, "mắt"),         # ky vong sua dung (show "Doi Mat Mekong")
        ]
        for token, prev, nxt, expected in cases:
            result = suggest_diacritic_autocorrect(token, prev, nxt, base_index, bigram_table)
            status = "PASS" if result == expected else "CHECK"
            print(f"[{status}] suggest_diacritic_autocorrect({token!r}, prev={prev!r}, next={nxt!r}) "
                  f"-> {result!r} (ky vong {expected!r})")

        # (BIET LA FP HIEM, KHONG PHAI BUG) "vồ" trong ten rieng "Cá Vồ" bi
        # sua nham thanh "vỏ" -- day CHINH LA 1/55 dong FP da do duoc trong
        # TIEN_DO_OCR.md entry 07-30 (chot). Ghi lai o day de test nay KHONG
        # BAO GIO "PASS gia" (vd neu sau nay ai do vo tinh sua nguong lam FP
        # nay bien mat ma khong biet ly do, hoac nguoc lai lam no xuat hien
        # them o cho khac ma khong ai kiem tra) -- day la gioi han DA BIET,
        # chap nhan duoc (1.8% FP tren mau 55 dong), KHONG phai muc tieu
        # "sua cho het" o day.
        known_fp = suggest_diacritic_autocorrect("vồ", "cá", "cờ", base_index, bigram_table)
        print(f"[GHI NHAN] suggest_diacritic_autocorrect('vồ', prev='cá', next='cờ') -> {known_fp!r} "
              f"(FP DA BIET tu ten rieng 'Cá Vồ', xem TIEN_DO_OCR.md -- KHONG phai loi code)")
