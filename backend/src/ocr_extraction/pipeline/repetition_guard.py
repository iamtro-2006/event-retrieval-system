"""
Repetition guard - phat hien chuoi text OCR bi "ket lap" (1 token lap lien
tiep nhieu lan, dau hieu dac trung cua seq2seq recognition bi loi tren crop
qua dai/day dac chu -- thuong o vung ban do).

KHAC voi post_ocr_correction.py: day la PATTERN-DETECTION (phat hien & loai
bo), khong phai SIMILARITY-MATCHING (sua thanh tu dung). Khong dung chung 1
ham vi ban chat 2 loai loi khac nhau -- xem proposal_post_ocr_model.md muc 0b.

Dung de loc truoc khi ghi JSON, hoac loc ngay trong extract_ocr.py sau buoc
recognition (tuong tu cach goi post_ocr_correction.correct_texts()).

--- CAP NHAT 2026-07-18 (review + fix 3 lo hong phat hien qua test that) ---

1. NFC normalize truoc khi so sanh token: 2 chu hien thi giong het nhau
   (vd "Phu") nhung OCR ra 2 kieu encode Unicode khac nhau (NFC vs NFD)
   truoc day se KHONG duoc coi la trung nhau (so sanh `==` that bai) ->
   bo lot lap that. Da chot dung NFC cho toan he thong (xem
   ocr_post_correction_plan.md muc 3.2) -- file nay truoc day chua ap dung,
   gio da dong bo.

2. Ngoai le so/chu so (`exempt_numeric`): cum lap la so dien thoai/ma hien
   thi (vd doc so "khong khong khong chin") se KHONG bi coi la rac, vi day
   la noi dung that co the xuat hien tren man hinh (banner/bien hieu), khac
   voi loi decode that.

3. Ket hop SO LAN LAP + CONFIDENCE (thay vi chi 1 nguong nhi phan): rac that
   (Nhom 2, seq2seq bi ket) thuong lap RAT NHIEU lan (vd 15 lan trong data
   that da thay) VA thuong di kem confidence thap hon. Banner/quang cao that
   nhan manh bang cach lap chu (vd "Giam gia Giam gia Giam gia") thuong chi
   lap 3-4 lan VA confidence cao (chu ro, de doc). Vi vay:
   - lap >= min_repeat_hard (mac dinh 5) -> luon coi la rac, bat ke confidence.
   - lap >= min_repeat_soft (mac dinh 3) nhung < min_repeat_hard -> CHI coi
     la rac neu confidence < soft_confidence_threshold (mac dinh 0.5).
     Neu khong co confidence truyen vao (vd goi ham cu, khong sua doi caller),
     GIU HANH VI CU (coi la rac) de tuong thich nguoc.

--- CAP NHAT 2026-07-19 (va lo hong thu 3, lo hong cuoi cung con lai) ---

4. False negative do dau cau lech giua cac lan lap: truoc day so sanh gram
   dung TOKEN GOC (con giu dau cau) -- vd "thu, thu. thu! thu" (rac that)
   se ra 4 token KHAC NHAU ve mat chuoi ky tu ("thu,", "thu.", "thu!",
   "thu") du doc giong het nhau, nen khong bao gio duoc coi la lap ->
   bo lot.
   Fix: tao them 1 danh sach RIENG `compare_tokens` (tung token da
   `.strip(_PUNCT_STRIP)`) CHI dung de SO SANH 2 gram co giong nhau khong.
   Token GOC (con nguyen dau cau) van duoc giu lai song song va truyen cho
   `_is_numeric_gram()` nhu cu -- khong doi hanh vi ngoai le so (fix 2),
   khong doi cach tra ve text goc.
   Edge case moi phai chan: sau khi strip dau cau, 1 token co the thanh
   chuoi RONG (vd token goc chi la "..." hoac "!!!"). Neu ca gram deu la
   chuoi rong sau strip thi KHONG duoc tinh la "lap that" -- day khong
   phai loi seq2seq ket lap ve tu ngu nghia, tinh nham se tao false
   positive moi (vd 1 cau co nhieu dau "..." lien tiep do ngat cau that).

Tat ca nguong o day co the truyen qua tham so ham (extract_ocr.py doc tu
yaml -> configs/ocr_extraction.yaml, muc extraction.postprocess.
repetition_guard_params) thay vi hard-code, de tinh chinh duoc ma khong can
sua code.
"""

from __future__ import annotations

import re
import unicodedata

# --- Nguong mac dinh (co the override qua yaml, xem docstring tren) -------
DEFAULT_MIN_REPEAT_HARD = 5          # lap >= so nay -> luon la rac, bat ke confidence
DEFAULT_MIN_REPEAT_SOFT = 3          # lap >= so nay (nhung < hard) -> can check confidence
DEFAULT_SOFT_CONFIDENCE_THRESHOLD = 0.5  # duoi nguong nay (trong vung soft) moi coi la rac
DEFAULT_EXEMPT_NUMERIC = True        # cum toan so/chu so -> khong bao gio coi la rac

# Tuong thich nguoc: 1 so noi (vd sanity_check_one_video.py cu) co the con
# import DEFAULT_MIN_REPEAT truc tiep -- giu lai alias tro ve gia tri soft.
DEFAULT_MIN_REPEAT = DEFAULT_MIN_REPEAT_SOFT

_VIETNAMESE_NUMBER_WORDS = {
    "không", "một", "hai", "ba", "bốn", "năm", "sáu", "bảy", "tám", "chín",
    "mười", "trăm", "nghìn", "ngàn", "triệu", "tỷ",
}
_PUNCT_STRIP = ".,!?;:()[]{}\"'“”‘’…"


def _is_numeric_gram(gram: list[str]) -> bool:
    """True neu MOI token trong gram la so (digit) hoac chu so tieng Viet.
    Dung de loai tru case so dien thoai/ma hien thi (vd "khong khong khong
    chin") khoi bi coi la rac lap."""
    for tok in gram:
        stripped = tok.strip(_PUNCT_STRIP)
        if not stripped:
            return False
        if stripped.isdigit():
            continue
        if stripped in _VIETNAMESE_NUMBER_WORDS:
            continue
        return False
    return True


def is_repetition_garbage(
    text: str,
    confidence: float | None = None,
    min_repeat_hard: int = DEFAULT_MIN_REPEAT_HARD,
    min_repeat_soft: int = DEFAULT_MIN_REPEAT_SOFT,
    soft_confidence_threshold: float = DEFAULT_SOFT_CONFIDENCE_THRESHOLD,
    exempt_numeric: bool = DEFAULT_EXEMPT_NUMERIC,
) -> bool:
    """True neu `text` chua 1 token HOAC 1 cum 2-3 token lap lien tiep, theo
    logic ket hop so-lan-lap + confidence (xem docstring dau file).

    `confidence`: confidence score (0.0-1.0) cua CHINH box/text nay (tu
    VietOCR). Truyen None neu khong co (vd goi tu code cu, hoac
    recognizer="paddleocr" khong co confidence) -- se fallback ve hanh vi
    nghiem ngat cu (coi moi lap >= min_repeat_soft la rac).

    Viec so sanh "2 token co lap nhau khong" bo qua dau cau bao quanh (xem
    fix 4, 2026-07-19) -- "thu," va "thu." va "thu!" va "thu" deu duoc coi
    la CUNG 1 token khi xet lap, dung nguyen ban co dau cau khi tra ve/dua
    vao _is_numeric_gram.

    Vi du:
        is_repetition_garbage("thu thu thu thu thu", confidence=0.3) -> True
            (lap 5 lan >= min_repeat_hard, rac chac chan bat ke confidence)
        is_repetition_garbage("Giảm giá Giảm giá Giảm giá", confidence=0.9) -> False
            (lap 3 lan, nhung confidence cao -> nhieu kha nang la banner that)
        is_repetition_garbage("không không không chín", confidence=0.9) -> False
            (cum toan so -> exempt_numeric, khong bao gio coi la rac)
        is_repetition_garbage("thu, thu. thu! thu", confidence=0.3) -> True
            (4 lan lap cung 1 tu, chi khac dau cau bao quanh -- fix 4)
    """
    if not text:
        return False

    # Fix 1: NFC normalize truoc khi so sanh, tranh bo lot do lech encoding.
    normalized = unicodedata.normalize("NFC", text)
    tokens = normalized.lower().split()
    if len(tokens) < min_repeat_soft:
        return False

    # Fix 4 (2026-07-19): danh sach rieng chi de SO SANH gram, da bo dau
    # cau bao quanh -- KHONG dung danh sach nay de tra ve text hay de dua
    # vao _is_numeric_gram (van dung `tokens` goc cho viec do).
    compare_tokens = [t.strip(_PUNCT_STRIP) for t in tokens]

    for n in (1, 2, 3):
        if len(tokens) < n * min_repeat_soft:
            continue
        i = 0
        while i + n <= len(tokens):
            gram = tokens[i:i + n]
            compare_gram = compare_tokens[i:i + n]

            # Fix 4 edge case: gram toan rong sau khi strip dau cau (vd
            # cum "..." "..." lien tiep) -- khong phai loi ket lap tu ngu
            # nghia, bo qua de tranh false positive moi.
            if all(not tok for tok in compare_gram):
                i += 1
                continue

            count = 1
            j = i + n
            while j + n <= len(tokens) and compare_tokens[j:j + n] == compare_gram:
                count += 1
                j += n

            if count >= min_repeat_soft:
                # Fix 2: ngoai le so/chu so -- khong bao gio coi la rac.
                if exempt_numeric and _is_numeric_gram(gram):
                    i += 1
                    continue

                # Fix 3: ket hop so-lan-lap + confidence.
                if count >= min_repeat_hard:
                    return True  # lap qua nhieu -> chac chan rac, bat ke confidence
                if confidence is None:
                    return True  # khong co confidence de tham chieu -> giu hanh vi cu
                if confidence < soft_confidence_threshold:
                    return True  # lap it hon nhung confidence thap -> van coi la rac
                # Nguoc lai (lap it, confidence cao) -> nhieu kha nang la banner
                # that nhan manh, KHONG coi la rac -- tiep tuc quet vi tri khac.

            i += 1

    return False


def filter_repetition_garbage(
    texts: list[str],
    confidences: list[float | None] | None = None,
    min_repeat_hard: int = DEFAULT_MIN_REPEAT_HARD,
    min_repeat_soft: int = DEFAULT_MIN_REPEAT_SOFT,
    soft_confidence_threshold: float = DEFAULT_SOFT_CONFIDENCE_THRESHOLD,
    exempt_numeric: bool = DEFAULT_EXEMPT_NUMERIC,
) -> list[str]:
    """Loai bo hoan toan cac text bi coi la repetition garbage khoi list.

    LUU Y: ham nay KHONG duoc extract_ocr.py._postprocess() goi truc tiep
    (pipeline that tu goi is_repetition_garbage() tung box de giu dong bo
    voi boxes/confidences theo dung index) -- ham nay chi de dung doc lap
    (script test, hoac noi nao chi co list text don thuan, khong can giu
    dong bo voi boxes).
    """
    if confidences is None:
        confidences = [None] * len(texts)
    return [
        t for t, c in zip(texts, confidences)
        if not is_repetition_garbage(
            t, confidence=c,
            min_repeat_hard=min_repeat_hard,
            min_repeat_soft=min_repeat_soft,
            soft_confidence_threshold=soft_confidence_threshold,
            exempt_numeric=exempt_numeric,
        )
    ]


if __name__ == "__main__":
    # Test bang ca case that tu sanity-check L28_V001 lan case rui ro moi
    # phat hien (xem CAP NHAT 2026-07-18 / 2026-07-19 o dau file).
    samples_no_conf = [
        "Việt cho thu thu thu thu thu thu thu thu thu thu thu thu thu thu thu",
        "Nguyên đân đân đân chi minh minh minh minh minh minh minh minh minh",
        "Vĩnh Lê Chánh Lưng Hầu Phú Loàn Hai Liện Thu Nhi Hou Thu Nhi Hou Hoành Long Hau Phú Loài Phú Loài Phú Loài",
        "gia dỡ chà với gia đình mình",
        "Ho Chi Minh City",
        "TRƯƠNG CHÍ HUNG",
    ]
    print("=== Case goc (khong co confidence -- fallback hanh vi cu) ===")
    for s in samples_no_conf:
        flag = is_repetition_garbage(s)
        print(f"{'GARBAGE ' if flag else 'OK      '} | {s[:70]}")

    print("\n=== Case rui ro moi (co confidence) ===")
    cases = [
        ("Giảm giá Giảm giá Giảm giá", 0.9, False, "banner that, lap 3 lan, confidence cao -> GIU"),
        ("Giảm giá Giảm giá Giảm giá", 0.3, True, "lap 3 lan nhung confidence thap -> XOA"),
        ("không không không chín", 0.9, False, "so dien thoai -> exempt_numeric -> GIU"),
        ("thu thu thu thu thu thu thu thu thu thu thu thu thu thu thu", 0.9, True, "lap 15 lan (>= hard) -> XOA bat ke confidence"),
    ]
    for text, conf, expected, note in cases:
        flag = is_repetition_garbage(text, confidence=conf)
        status = "PASS" if flag == expected else "FAIL"
        print(f"[{status}] {'GARBAGE' if flag else 'OK     '} | conf={conf} | {text[:40]!r} | {note}")

    print("\n=== Fix NFC/NFD (dung confidence thap de co lap, tranh bi fix 3 che mat) ===")
    nfc = unicodedata.normalize("NFC", "Phú")
    nfd = unicodedata.normalize("NFD", "Phú")
    mixed = f"{nfc} Loài {nfd} Loài {nfc} Loài"
    flag = is_repetition_garbage(mixed, confidence=0.2)
    print(f"{'GARBAGE ' if flag else 'OK      '} (ky vong GARBAGE) | {mixed!r}")

    print("\n=== Fix 4 moi (2026-07-19): dau cau lech giua cac lan lap ===")
    punct_cases = [
        ("thu, thu. thu! thu", 0.3, True, "4 lan lap cung 1 tu, khac dau cau bao quanh -> XOA (fix 4)"),
        ("thu, thu. thu! thu", 0.9, False, "cung lap dau cau lech, nhung confidence cao + chi 4 lan (< hard) -> GIU"),
        ("Việt Nam, Việt Nam. Việt Nam!", 0.2, True, "cum 2 token lap, lech dau cau -> XOA"),
        ("... ... ...", 0.2, False, "toan dau cau, rong sau khi strip -> KHONG tinh la lap (tranh false positive moi)"),
        ("không, không. không! chín", 0.9, False, "van la cum so (dau cau lech) -> exempt_numeric van ap dung dung -> GIU"),
    ]
    for text, conf, expected, note in punct_cases:
        flag = is_repetition_garbage(text, confidence=conf)
        status = "PASS" if flag == expected else "FAIL"
        print(f"[{status}] {'GARBAGE' if flag else 'OK     '} | conf={conf} | {text[:40]!r} | {note}")
