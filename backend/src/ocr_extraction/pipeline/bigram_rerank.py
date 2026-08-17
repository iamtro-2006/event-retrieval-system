"""
Bigram context re-rank -- xu ly real-word error (tu OCR doc SAI nhung ra
1 tu tieng Viet KHAC van hop le, vd "Bình" -> "thể", "Mắt" -> "Mát") ma
dictionary don-tu (Luong 1, exists()/suggest()) KHONG bao gio phat hien
duoc, vi token sai VAN pass Buoc 1 (dict_exact) -- ca token dung va token
sai deu la tu that, chi khac nhau o co HOP NGU CANH hay khong.

CO CHE (xem giai thich day du da trinh bay voi nguoi dung 2026-07-24):
    - Dung bang tan suat cap-tu (vi_bigram_freq.tsv, build tu 3.2 trieu
      cau binhvq/news-corpus, xem build_bigram.py/TIEN_DO_OCR.md) de biet
      1 cap (w1, w2) co "tu nhien" trong tieng Viet hay khong.
    - Voi 1 token DA PASS Buoc 1 (dict_exact -- tuc la 1 tu that), sinh
      cac candidate GAN no (edit-distance 1-2, dung CHUNG co che voi
      LexiconDictionary._edits1/_edits2 nhung o day sinh doc lap, chi can
      dictionary.exists() de loc candidate hop le -- KHONG phu thuoc
      LexiconDictionary cu the, tuong thich voi bat ky SpellDictionary nao).
    - So sanh "bang chung ngu canh": dem bigram(prev, token) + bigram(token,
      next) so voi bigram(prev, candidate) + bigram(candidate, next) cho
      TUNG candidate. Neu token GOC gan nhu KHONG BAO GIO xuat hien canh
      prev/next (dem ~0) MA co 1 candidate xuat hien RAT NHIEU (vuot ca
      nguong tuyet doi lan ti le so voi token goc) -> nghi ngo real-word
      error, GAN CO flag "uncertain" (KHONG tu dong sua).

QUYET DINH THIET KE QUAN TRONG (theo yeu cau nguoi dung 2026-07-24, pham
vi tich hop LAN DAU): CHI GAN NHAN NGHI VAN, KHONG tu dong ghi de
`corrected` -- rui ro sua sai token DANG DUNG cao hon loi ich neu threshold
chua duoc kiem chung dau du (moi co ~35 vi du real-word-error trong ground
truth, chua du de tin tuong nguong tu dong sua). Caller (extract_ocr.py)
VAN GIU corrected=token GOC, chi source/action doi de danh dau cho buoc
review/log rieng -- xem CorrectionResult.suggested trong post_ocr_correction.py.

KHAC VOI cache token don trong post_ocr_correction.py: buoc nay PHU THUOC
NGU CANH (prev/next token), nen KHONG dung correction_cache dung chung
(cache do gia dinh ket qua CHI phu thuoc token, khong phu thuoc frame/cau
nao). Buoc nay chay THEM (overlay) SAU khi correct_text() da tinh xong
ket qua context-free binh thuong qua cache -- xem correct_text() trong
post_ocr_correction.py.
"""

from __future__ import annotations

import unicodedata
from pathlib import Path

DEFAULT_BIGRAM_PATH = Path(__file__).parent / "vi_bigram_freq.tsv"

# Nguong bao thu (lan dau tich hop, xem docstring dau file ve ly do CHI
# gan nhan "uncertain" chu KHONG tu sua):
#   - MIN_CANDIDATE_COUNT: candidate phai xuat hien it nhat tung nay lan
#     CANH prev/next trong corpus moi duoc coi la "co bang chung manh".
#     3.2 trieu cau la mau con nho (~16.5% full corpus) nen KHONG dat qua
#     cao -- 30 la muc "xuat hien vai chuc lan trong 3.2M cau", du de
#     loai tinh co nhung khong qua khat khe toi mat het candidate that.
#   - MIN_RATIO: candidate phai co tong bigram-evidence GAP IT NHAT tung
#     nay lan token goc (token goc + 1 de tranh chia cho 0). Ti le lon
#     (80x) de uu tien precision hon recall trong lan dau nay -- thu FN
#     (bo sot canh bao that) con hon FP (bao dong gia lam nhieu output).
MIN_CANDIDATE_COUNT = 30
MIN_RATIO = 80.0


def _normalize(token: str) -> str:
    return unicodedata.normalize("NFC", token).lower()


def load_bigram_table(path: str | Path = DEFAULT_BIGRAM_PATH) -> dict[tuple[str, str], int]:
    """Doc file vi_bigram_freq.tsv (word1\\tword2\\tcount, xem build_bigram.py).
    Tra ve dict RONG neu file khong ton tai -- TUY CHON giong vi_word_freq_full.tsv,
    KHONG raise, de pipeline van chay duoc (chi tat buoc bigram re-rank) khi
    chua co file nay."""
    path = Path(path)
    if not path.exists():
        return {}
    table: dict[tuple[str, str], int] = {}
    with path.open("r", encoding="utf-8") as f:
        header_skipped = False
        for line in f:
            line = line.rstrip("\n")
            if not line:
                continue
            if not header_skipped:
                header_skipped = True
                if line.startswith("word1\tword2\tcount"):
                    continue
            parts = line.split("\t")
            if len(parts) != 3:
                continue
            w1, w2, count_str = parts
            try:
                count = int(count_str)
            except ValueError:
                continue
            table[(_normalize(w1), _normalize(w2))] = count
    return table


def alphabet_from_bigram_table(table: dict[tuple[str, str], int]) -> str:
    """Sinh bang chu cai TU CHINH cac tu xuat hien trong bang bigram (giong
    tinh than _alphabet_from_wordlist trong viet_dictionary.py) -- dung cho
    _near_words() de sinh candidate edit-distance-1, dam bao cover dung
    het ky tu tieng Viet co dau THUC SU xuat hien trong corpus, khong
    can import cheo sang viet_dictionary.py (giu 2 module doc lap)."""
    chars: set[str] = set()
    for w1, w2 in table:
        chars.update(w1)
        chars.update(w2)
    chars.discard(" ")
    return "".join(sorted(chars))


def _bigram_count(table: dict[tuple[str, str], int], w1: str | None, w2: str | None) -> int:
    if w1 is None or w2 is None:
        return 0
    return table.get((_normalize(w1), _normalize(w2)), 0)


def _context_evidence(
    table: dict[tuple[str, str], int],
    prev_token: str | None,
    word: str,
    next_token: str | None,
) -> int:
    """Tong bang chung ngu canh cho 1 tu (trai + phai). Token o dau/cuoi cau
    (prev/next = None) chi tinh 1 phia con lai."""
    return _bigram_count(table, prev_token, word) + _bigram_count(table, word, next_token)


def _near_words(
    token: str,
    dictionary,
    alphabet: str,
    near_words_cache: dict[str, set[str]] | None = None,
) -> set[str]:
    """Sinh candidate tu that (dictionary.exists() == True) cach `token`
    dung 1 phep sua Levenshtein (them/xoa/doi 1 ky tu) -- dung CHUNG co che
    voi LexiconDictionary._edits1() nhung KHONG phu thuoc class do, chi can
    dictionary.exists() (tuong thich Protocol SpellDictionary bat ky).
    KHAC voi dictionary.suggest(): suggest() chi chay khi token KHONG ton
    tai (tra ve [] neu token da la tu that) -- ham nay CHU DICH sinh
    candidate ke ca khi token DA la tu that (vi day chinh la truong hop can
    xu ly: real-word error, token goc VAN hop le).

    `near_words_cache` (TUY CHON): KET QUA HAM NAY CHI PHU THUOC
    (token, dictionary, alphabet) -- KHONG phu thuoc ngu canh (prev/next),
    KHAC voi suggest_real_word_flag() noi no duoc goi (ham do MOI phu thuoc
    ngu canh). Vi vay an toan dung 1 cache DUNG CHUNG qua nhieu lan goi/
    nhieu token lap lai (watermark, logo kenh...) -- giong tinh than
    correction_cache trong post_ocr_correction.py, tranh tinh lai edit-
    distance-1 (O(len(token) * len(alphabet)) candidate, moi candidate 1
    lan dictionary.exists()) moi lan gap lai CUNG 1 token."""
    if near_words_cache is not None and token in near_words_cache:
        return near_words_cache[token]

    word = _normalize(token)
    splits = [(word[:i], word[i:]) for i in range(len(word) + 1)]
    deletes = [a + b[1:] for a, b in splits if b]
    replaces = [a + c + b[1:] for a, b in splits if b for c in alphabet]
    inserts = [a + c + b for a, b in splits for c in alphabet]
    candidates = set(deletes + replaces + inserts)
    candidates.discard(word)
    result = {c for c in candidates if dictionary.exists(c)}

    if near_words_cache is not None:
        near_words_cache[token] = result
    return result


def suggest_real_word_flag(
    token: str,
    prev_token: str | None,
    next_token: str | None,
    dictionary,
    bigram_table: dict[tuple[str, str], int],
    alphabet: str,
    min_candidate_count: int = MIN_CANDIDATE_COUNT,
    min_ratio: float = MIN_RATIO,
    near_words_cache: dict[str, set[str]] | None = None,
) -> str | None:
    """Tra ve candidate NGHI VAN (str) neu co bang chung ngu canh manh cho
    thay `token` (da la tu that, qua Buoc 1) co the la real-word error, hoac
    None neu khong co gi bat thuong / khong du bang chung. KHONG tu sua --
    xem docstring dau file.

    `near_words_cache`: xem docstring _near_words() -- truyen thang xuong,
    KHONG lien quan/KHONG dung chung voi correction_cache (cache do phu
    thuoc token DON, cache nay chi cho phan sinh candidate cung tu do,
    van tinh lai bang chung ngu canh moi lan vi do MOI phu thuoc prev/next)."""
    if not bigram_table:
        return None
    if prev_token is None and next_token is None:
        # Khong co ngu canh nao ca (cau 1 tu) -- khong du du lieu de danh gia.
        return None

    original_evidence = _context_evidence(bigram_table, prev_token, token, next_token)

    best_candidate = None
    best_evidence = 0
    for cand in _near_words(token, dictionary, alphabet, near_words_cache=near_words_cache):
        cand_evidence = _context_evidence(bigram_table, prev_token, cand, next_token)
        if cand_evidence > best_evidence:
            best_evidence = cand_evidence
            best_candidate = cand

    if best_candidate is None:
        return None
    if best_evidence < min_candidate_count:
        return None
    if best_evidence < (original_evidence + 1) * min_ratio:
        return None

    return best_candidate
