"""
vlm_text_normalize.py -- (MOI, 2026-08-12) 3 buoc chuan hoa rieng cho text
do VLM (Qwen3-VL) sinh ra, tach khoi post_ocr_correction.py (luong 1/2/3
sua chinh ta) vi day la 3 loai loi KHAC NHAU, dac trung cua VLM full-frame
OCR chu khong phai loi detect+recognize kieu Paddle/VietOCR:

    1. dedupe_exact()          -- VLM doi khi "ket lap" CA 1 DONG (khac
                                   repetition_guard.py: cai do bat lap TRONG
                                   1 chuoi, con day la lap CA DONG trong
                                   list texts cua 1 frame). Vi du that do
                                   tu L23_V003 000036: "TON DONG A" xuat
                                   hien 99 LAN rieng biet trong list texts
                                   CUNG 1 frame.
    2. split_glued_phrases()   -- tach cac cum bi dinh boi ':' hoac ' - '
                                   khi day la 2 cum ngu nghia khac nhau (vd
                                   "Chieu cao: 1m72" -> ["Chieu cao", "1m72"]),
                                   NHUNG giu nguyen khi ':' nam giua 2 chu so
                                   (timestamp/ty le, vd "00:53:22") -- xem
                                   vi du that duoi day, lay tu doc that
                                   data/processed/L23/*.json (918 dong co
                                   ':', ĐA kiem tra thu cong: TOAN BO cac
                                   truong hop ':' giua 2 chu so la timestamp,
                                   16 truong hop con lai la dung "nhan: gia
                                   tri" nhu "Nam sinh: 1997", "DT: 090.000.000").
    3. normalize_math_notation() -- unicode superscript/subscript (vd "x²",
                                   "H₂O") -> dang go-ban-phim-duoc: superscript
                                   -> "^" + chu so (x² -> x^2, dung quy uoc
                                   so mu toan hoc), subscript -> chu so thuong
                                   khong dau ^ (H₂O -> H2O, dung quy uoc chi
                                   so hoa hoc). CANH BAO (2026-08-12): CHUA
                                   tim thay vi du that nao trong data/processed/
                                   L23 (dataset the thao, khong co cong thuc
                                   toan/hoa) -- ham nay viet dua tren MO TA
                                   cua leader (vi du "C2H5O12", "x mu 2"), CHUA
                                   duoc kiem chung tren du lieu that co cong
                                   thuc toan/hoa. Neu VLM tao ra superscript/
                                   subscript theo cach KHAC (vd xuong dong,
                                   khoang trang giua so va chu, khong phai
                                   unicode superscript/subscript that) thi ham
                                   nay se KHONG bat duoc -- can data that de
                                   mo rong, xem TIEN_DO_OCR.md.

Ca 3 ham deu THUAN (pure function, khong phu thuoc confidence/frame/video
nao) -- dung duoc doc lap trong post_process.py (buoc goi ca 3 truoc khi
chay repetition_guard + post_ocr_correction, xem docstring PostOCRPipeline).
"""

from __future__ import annotations

import re

# ---------------------------------------------------------------------
# 1. Dedupe (list-level, KHAC repetition_guard.is_repetition_garbage --
#    xem docstring dau file)
# ---------------------------------------------------------------------

def dedupe_exact(texts: list[str]) -> list[str]:
    """Loai cac dong TRUNG KHOP CHINH XAC (sau strip whitespace) voi 1 dong
    da xuat hien truoc do trong CUNG list `texts` (1 frame). Giu lai dong
    XUAT HIEN DAU TIEN, giu nguyen thu tu. Chuoi rong/toan whitespace KHONG
    bi dedupe (giu nguyen so luong, de caller/repetition_guard xu ly rieng).

    LUU Y cho caller can giu boxes dong bo index: dung ham nay CHI de test
    doc lap tren list text don thuan -- post_process.py tu lam vong lap
    rieng giu boxes/texts dong bo (xem PostOCRPipeline._normalize_frame),
    khong goi thang ham nay tren pipeline that.
    """
    seen: set[str] = set()
    out: list[str] = []
    for t in texts:
        key = t.strip()
        if key and key in seen:
            continue
        if key:
            seen.add(key)
        out.append(t)
    return out


# ---------------------------------------------------------------------
# 2. Split glued phrases
# ---------------------------------------------------------------------

# ':' KHONG duoc coi la diem tach neu 1 trong 2 ky tu lien ke truoc/sau la
# chu so (bao ve timestamp "00:53:22", ty le "19:31:17", KHONG bao ve
# "Nam sinh: 1997" vi ben trai la chu cai). ' - ' (co khoang trang 2 ben)
# duoc coi la diem tach -- KHONG tach dau '-' dinh lien vao chu/so (vd
# "re-tweet", "20-24") de tranh pha vo tu ghep/khoang gia tri.
_SPLIT_PATTERN = re.compile(r"\s*(?<!\d):(?!\d)\s*|\s+-\s+")


def split_glued_phrases(text: str) -> list[str]:
    """Tach 1 dong text thanh nhieu dong con neu no dang dinh 2+ cum ngu
    nghia khac nhau boi ':' hoac ' - ' (xem _SPLIT_PATTERN + vi du that o
    docstring dau file). Tra ve [text] KHONG DOI neu khong co diem tach nao
    (hoac tach xong chi con <=1 phan khac rong).

    Vi du that (tu data/processed/L23):
        "Chieu cao: 1m72"        -> ["Chieu cao", "1m72"]
        "HTV - TON DONG A"       -> ["HTV", "TON DONG A"]
        "00:53:22"               -> ["00:53:22"] (KHONG tach, ':' giua 2 so)
        "Tong thanh tich: 19:31:17" -> ["Tong thanh tich", "19:31:17"]
                                       (chi tach ':' dau, ':' trong "19:31:17"
                                       van giua 2 so nen giu nguyen)
    """
    if not text or not text.strip():
        return [text]

    parts = [p.strip() for p in _SPLIT_PATTERN.split(text)]
    parts = [p for p in parts if p]
    return parts if parts else [text]


# ---------------------------------------------------------------------
# 3. Math/chemistry notation normalize
# ---------------------------------------------------------------------

_SUPERSCRIPT_MAP = {
    "⁰": "0", "¹": "1", "²": "2", "³": "3",
    "⁴": "4", "⁵": "5", "⁶": "6", "⁷": "7",
    "⁸": "8", "⁹": "9", "⁺": "+", "⁻": "-",
    "⁼": "=", "⁽": "(", "⁾": ")", "ⁿ": "n",
}

_SUBSCRIPT_MAP = {
    "₀": "0", "₁": "1", "₂": "2", "₃": "3",
    "₄": "4", "₅": "5", "₆": "6", "₇": "7",
    "₈": "8", "₉": "9", "₊": "+", "₋": "-",
    "₌": "=", "₍": "(", "₎": ")",
}


def normalize_math_notation(text: str) -> str:
    """Chuan hoa unicode superscript/subscript thanh dang go-ban-phim-duoc,
    de nguoi dung tim kiem go "X2"/"X^2" van khop:
        superscript (so mu, vd "x²")  -> "^" + chu so thuong (x² -> x^2)
        subscript   (chi so, vd "H₂O") -> chu so thuong, KHONG co "^"
                                          (H₂O -> H2O, dung quy uoc hoa hoc)
    Cac chuoi superscript/subscript LIEN TIEP duoc gop thanh 1 nhom (vd
    "x²³" -> "x^23", khong phai "x^2^3"). Ky tu khong thuoc 2 bang tren giu
    nguyen khong doi.
    """
    if not text:
        return text

    out: list[str] = []
    i = 0
    n = len(text)
    while i < n:
        ch = text[i]
        if ch in _SUPERSCRIPT_MAP:
            run = []
            while i < n and text[i] in _SUPERSCRIPT_MAP:
                run.append(_SUPERSCRIPT_MAP[text[i]])
                i += 1
            out.append("^" + "".join(run))
            continue
        if ch in _SUBSCRIPT_MAP:
            run = []
            while i < n and text[i] in _SUBSCRIPT_MAP:
                run.append(_SUBSCRIPT_MAP[text[i]])
                i += 1
            out.append("".join(run))
            continue
        out.append(ch)
        i += 1
    return "".join(out)


if __name__ == "__main__":
    print("=== dedupe_exact (vi du that L23_V003 000036: 'TON DONG A' x99) ===")
    sample = ["HTV"] + ["TON DONG A"] * 5 + ["THE THAO"]
    result = dedupe_exact(sample)
    status = "PASS" if result == ["HTV", "TON DONG A", "THE THAO"] else "FAIL"
    print(f"[{status}] {len(sample)} dong -> {len(result)} dong: {result}")

    print("\n=== split_glued_phrases (vi du that tu L23) ===")
    split_cases = [
        ("Chiều cao: 1m72", ["Chiều cao", "1m72"]),
        ("Cân nặng: 62kg", ["Cân nặng", "62kg"]),
        ("ĐT: 090.000.000", ["ĐT", "090.000.000"]),
        ("00:53:22", ["00:53:22"]),
        ("Tổng thành tích: 19:31:17", ["Tổng thành tích", "19:31:17"]),
        ("HTV - TÔN ĐỒNG A", ["HTV", "TÔN ĐỒNG A"]),
        ("Chặng 4 - 124 km", ["Chặng 4", "124 km"]),
        ("Quảng Ninh: Hôm nay cái gì đó", ["Quảng Ninh", "Hôm nay cái gì đó"]),
        ("02:46:45 / 00:21", ["02:46:45 / 00:21"]),
    ]
    for text, expected in split_cases:
        result = split_glued_phrases(text)
        status = "PASS" if result == expected else "FAIL"
        print(f"[{status}] {text!r} -> {result}")

    print("\n=== normalize_math_notation (CHUA co vi du that -- xem canh bao docstring) ===")
    math_cases = [
        ("x²", "x^2"),
        ("H₂O", "H2O"),
        ("C₂H₅OH", "C2H5OH"),
        ("x² + y² = z²", "x^2 + y^2 = z^2"),
        ("bình thường", "bình thường"),  # khong co gi de doi
    ]
    for text, expected in math_cases:
        result = normalize_math_notation(text)
        status = "PASS" if result == expected else "FAIL"
        print(f"[{status}] {text!r} -> {result!r}")
