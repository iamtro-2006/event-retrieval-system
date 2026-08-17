"""
Luong 3 (regex/rule-based) cua post-OCR correction -- xem ocr_post_correction_plan.md
muc 3.5.

Xu ly cac token KHONG phai "tu" theo nghia ngon ngu -- dictionary (luong 1)
va domain_vocab tan suat (luong 2) khong ap dung duoc cho loai nay:

    - So + don vi do luong (ml, l, g, kg, muong, thia...)
    - Ngay gio, tien te, phan tram, do
    - So dien thoai / ma so
    - Acronym viet hoa (VTV, TP.HCM, UBND...) -- ghep VOI WHITELIST NHO, khong
      dung heuristic "toan chu hoa" don thuan (precision thap, de nham voi
      loi OCR case that -- xem plan muc 3.5 va Ratinov & Roth 2009 ve
      gazetteer-as-feature).

KHONG xu ly o day (theo plan):
    - Am thuc/cong thuc nau an (chi phan chu, khong phai so+don vi) --
      thuoc luong 1, la tu tieng Viet thuong.
    - Thuong hieu/ten rieng nuoc ngoai (Samsung, FIFA...) -- gop xu ly o
      luong 2 (domain_vocab), khong tach luong rieng.

Moi ham public o day tra ve None neu KHONG match pattern nao (=> khong phai
viec cua luong 3, de resolver thu luong khac), hoac 1 RegexMatch neu match
(text co the da duoc sua, hoac giu nguyen neu pattern hop le san).
"""

from __future__ import annotations

import re
from dataclasses import dataclass

# (2026-08-13, MOI) py3langid: model nhan dien ngon ngu nhe (dong goi san
# trong package pip, KHONG can tai model rieng qua mang -- khac fastText/
# transformer). Tuy chon -- neu chua cai, is_non_vietnamese_line() tu dong
# bo qua phan langid, chi dung heuristic stopword cu (an toan tuong thich
# nguoc). Xem TIEN_DO_OCR.md entry "langid" de biet ly do chon + do luong.
try:
    import py3langid as _langid
except ImportError:
    _langid = None

# strip_diacritics() dung chung voi diacritic_restoration.py (KHONG copy lai
# bang bo dau rieng o day -- tranh lech neu ban goc doi sau nay). Khong lo
# vong lap import: diacritic_restoration.py chi import bigram_rerank/
# viet_dictionary, khong import regex_rules.
from src.ocr_extraction.pipeline import diacritic_restoration

# ---------------------------------------------------------------------
# Whitelist acronym -- CHUA CHOT DANH SACH CUOI (xem plan muc 7, viec con
# treo "Xac nhan whitelist acronym ban dau"). Day la tap khoi diem, mo rong
# them khi gap acronym that trong data nhung KHONG duoc tu dong suy ra
# acronym tu heuristic "toan chu hoa" (plan da canh bao ro ly do -- da
# kiem chung: quet 1501 chuoi toan-hoa tren data that, da so la tu thuong
# viet hoa kieu tieu de, KHONG phai acronym that).
#
# CAP NHAT 2026-07-22:
#   - ATM, TP them tu ket qua quet data that (L26-L30), xac nhan la
#     acronym that (khong phai OCR noise) qua kiem tra thu cong.
#   - 21 acronym con lai lay tu chinh vi-DauCu.dic/vi-DauMoi.dic (da qua
#     kiem chung boi nguoi duy tri hunspell-vi) -- nguon dang tin hon tu
#     dat tay ca nhan, xem hunspell-vi tren github 1ec5/hunspell-vi.
#   - Da tim nguon mo co san danh sach acronym/to chuc Viet Nam (vd NER
#     dataset VLSP) nhung KHONG co dataset flat list nao dung duoc ngay --
#     cac nguon NER (undertheseanlp/ner, vietner...) la training data cho
#     model nhan dien thuc the, khong phai danh sach acronym don gian,
#     tich hop se ton nhieu cong hon loi ich luc nay.
#   - Luu y rieng: search cho thay "HTV" bi dung CHUNG boi nhieu dai khac
#     nhau (TP.HCM, Ha Noi, Ha Tinh) -- khong anh huong muc dich o day
#     (chi can biet la acronym that, khong can biet dai nao).
# ---------------------------------------------------------------------
ACRONYM_WHITELIST: set[str] = {
    "VTV", "VTV1", "VTV2", "VTV3", "VTC", "HTV",
    "THVL", "ANTV", "QPVN", "VOV", "SCTV", "VTVCAB",
    "TP.HCM", "TP.HN", "TPHCM", "TP",
    "UBND", "HĐND", "CSGT", "CAND",
    "FIFA", "UEFA", "AFF", "SEA", "SEAGAMES",
    "WHO", "UNESCO", "NASA", "UNICEF",
    "USD", "VND", "EUR", "ATM",
    # 21 acronym tu vi-DauCu.dic / vi-DauMoi.dic (hunspell-vi that):
    "ABC", "ASCII", "GIF", "HCM", "HK", "HTML", "JPEG", "LHQ",
    "PDF", "PNG", "RAM", "TCVN", "TV", "TW", "URL", "Unicode",
    "VIQR", "VISCII", "VN", "VNI",
    # (2026-07-22 toi, MOI) tap con da loc tu danh sach dong doi tong hop
    # (NN/phap luat/doanh nghiep/giao duc/CNTT/internet/tai chinh/y te/to
    # chuc QT/don vi do/thoi gian/common tieng Anh) -- xem TIEN_DO_OCR.md
    # entry 2026-07-22 (toi) de biet ly do cac muc 1-2-3 chu (GD, HS, SV,
    # TS, GS, DN, CP...) bi LOAI vi de trung rac OCR ngan viet hoa. Don vi
    # do khong them vao day vi da co match_measurement() rieng.
    "BHXH", "BHYT", "NHNN", "THCS", "THPT", "ĐH",
    "GDP", "CPI", "GBP", "JPY",
    "COVID", "COVID-19",
    "UN", "WTO", "EU", "NATO", "OECD", "OPEC",
    "GPS", "PIN", "OTP", "QR", "USB", "AI", "CEO",
    # (2026-07-23) "HIV"/"AIDS" DA BI LOAI khoi day -- kiem chung tren 30
    # video that (L26-L30) cho thay 355 lan "HIV" xuat hien deu la loi OCR
    # doc nham watermark dai "HTV" (cung ho bien the HTM/HIM/HUV/HUM, vs
    # 4848 lan doc DUNG "HTV"), khong phai noi dung y te that trong domain
    # du lieu nay (nau an/du lich/thoi su dia phuong/game show/tu thien).
    # Neu domain du lieu doi (co noi dung y te that), can them lai va kiem
    # chung rieng, KHONG chi phuc hoi may moc theo whitelist cu.

    # (2026-07-25, MOI) tap con da loc tu file "Acronym whitelist.docx"
    # dong doi tong hop, gui qua Cowork -- ap dung CUNG tieu chi loc nhu
    # 07-22 (uu tien acronym DAI/PHAN BIET RO, LOAI cac ma 1-2-3 chu de
    # trung syllable/rac OCR thuong). Da LOAI HOAN TOAN:
    #   - Nhom phap luat/ke toan chuyen sau (BLDS, BLHS, HDXX, QSDD, TNDS,
    #     TNHS, VADS, VAHS, DN, CP, MTV, GD, PGD, TGD, KTT...) -- ly do cu
    #     (07-22) van dung: kho xuat hien tren video thi that (tin tuc/du
    #     lich/am thuc), rui ro trung syllable cao (vd "GD"/"CP"/"HD" rat
    #     ngan).
    #   - Nhom giao duc ngan (GS, PGS, TS, ThS, CD, GV, HS, SV) -- giu
    #     nguyen quyet dinh 07-22 "canh giac voi ma ngan", KHONG doi y du
    #     co trong file moi.
    #   - Nhom thoi gian (AD, BC, CE, AM, PM...) -- qua ngan/qua chung
    #     chung, rui ro cao, gia tri thap cho domain nay.
    #   - Don vi do (mm, km, kg...) -- da co match_measurement() rieng,
    #     KHONG trung lap o day (giong quyet dinh 07-22).
    #   - "WB" (2 chu, rui ro) -- loai, giu UNESCO/UNICEF/IMF/ASEAN/APEC.
    #   - "P/E", "EPS", "ROA", "ROE", "NAV", "FX" -- ngan/mo ho, gia tri
    #     thap cho domain nay, loai.
    #   - "IP", "JS", "ML", "DL", "CV" (2 chu) -- loai vi qua ngan.
    #   - "OKR", "SOP", "ETA", "FYI", "TBD", "DIY", "CT" -- loai (ngan/gia
    #     tri thap/de trung tu thuong).
    "TAND", "VKSND", "CQĐT", "PCCC", "BHTN", "BTC", "BCA", "BQP", "BNV",
    "BTP", "BKHĐT", "BNNPTNT", "BLĐTBXH",
    "TNHH", "CTCP", "HĐQT", "GTGT", "TNCN", "TNDN", "MST", "DNNN",
    "API", "SDK", "GUI", "CLI", "CPU", "GPU", "SSD", "HDD", "SQL", "JSON",
    "XML", "CSV", "YAML", "LLM",
    "HTTP", "HTTPS", "FTP", "TCP", "UDP", "DNS", "URI", "CSS", "JWT",
    "SSH", "SSL", "TLS", "SMTP", "IMAP", "REST", "RPC",
    "PPI", "PMI", "IPO", "EBITDA", "ETF",
    "BMI", "ICU", "PCR", "MRI", "ECG", "EEG", "COPD", "SARS",
    "IMF", "ASEAN", "APEC",
    "CFO", "COO", "CTO", "CIO", "CMO", "KPI", "FAQ", "ASAP", "RFID", "SIM",
}

# ---------------------------------------------------------------------
# (2026-07-30, MOI) Whitelist tu muon tieng Anh thong dung trong caption
# truyen hinh/giai tri/du lich VN -- xem TIEN_DO_OCR.md entry 2026-07-30
# "Lead phan hoi test module + phat hien bug moi (loanword bi 'sua' sai)".
#
# BOI CANH: LexiconDictionary (Viet74K/underthesea, thuan tu tieng Viet)
# KHONG chua tu muon tieng Anh -- resolver Buoc 4 (catch-all suggest())
# coi BAT KY token khong khop tu dien nao cung la "loi chinh ta can sua",
# nen "TOUR" (dung, tu muon rat pho bien trong tin tuc du lich) bi
# LexiconDictionary.suggest() "sua" thanh "tout" (tinh co la 1 tu hiem
# trong wordlist, gan edit-distance-1 voi "tour") -- VO NGHIA. Da kiem
# chung THAT bang LexiconDictionary that (khong phai doan mo): "show"->
# "shop", "live"->"lie", "hot"->"hoạt", "check"->"chức", "note"->"nghe",
# "gold"->"golf", "style"->"etylen", "fan"->"an", "clip"->"elip",
# "trend"->"treng" -- deu bi sua sai tuong tu. Day CHINH LA co che dung
# sau phan hoi lead 07-30 "post_ocr luc dung hon luc sai" -- khong phai
# loi ngau nhien, la 1 LOP LOI he thong co the liet ke/du doan duoc.
#
# CO CHE (giong match_acronym o duoi, nhung KHAC 2 diem quan trong):
#   1. So sanh KHONG phan biet hoa/thuong (giong acronym), nhung GIU
#      NGUYEN CASE GOC cua token OCR (KHONG chuan hoa ve 1 dang nhu
#      acronym lam) -- vi loanword xuat hien ca hoa/thuong/hoa dau trong
#      thuc te (vd "Tour", "TOUR", "tour" deu hop le), khong co 1 "dang
#      chuan" duy nhat nhu acronym (luon viet hoa).
#   2. Danh sach nay la TAP KHOI DIEM (giong ACRONYM_WHITELIST luc dau),
#      CHUA CHOT DANH SACH CUOI -- can mo rong khi gap tu muon moi trong
#      data that (2026), theo dung tinh than "khong tu dong suy ra tu
#      heuristic, chi mo rong tu bang chung that" da ap dung cho acronym.
# ---------------------------------------------------------------------
ENGLISH_LOANWORD_WHITELIST: set[str] = {
    # Du lich/giai tri (domain chinh cua dataset hien tai theo TIEN_DO_OCR).
    "tour", "show", "live", "trailer", "video", "online", "hot", "trend",
    "hit", "fashion", "beauty", "mc", "dj", "step", "dance", "cover",
    "remix", "vlog", "blogger", "streamer", "gameshow", "talkshow",
    "format", "casting", "comeback", "livestream",
    # Cong nghe/thiet bi thuong gap trong caption tin tuc/quang cao.
    "check", "note", "gold", "style", "fan", "clip", "wifi", "camera",
    "laptop", "smartphone", "app", "website", "email",
    # Thuong mai/quang cao.
    "sale", "voucher", "combo", "menu", "spa", "gym",
    # (2026-08-13, MOI) Tu review 14 token CAN XEM con sot cua dict_suggest
    # tren L23 (xem TIEN_DO_OCR.md entry cung ngay) -- da doi chieu ANH GOC
    # THAT xac nhan day la ten/brand that, khong phai loi chinh ta:
    # "comple"/"veston" (tu muon Phap = "bo vest", bien hieu tiem may that),
    # "doido" (ten shop/brand tren bien hieu), "vina" (tien to brand tieng
    # Viet cuc pho bien, xuat hien don le tren banner bi cat canh khung).
    "comple", "veston", "doido", "vina",
}


# (2026-07-25) Danh sach dia danh (tinh/thanh + mapping cu->moi sau sap
# nhap) va thuong hieu quoc te tu file "Acronym whitelist.docx" (dong doi
# tong hop, xem TIEN_DO_OCR.md entry 2026-07-25) -- CO CHU DICH KHONG dua
# thang vao ACRONYM_WHITELIST o tren, vi 2 nhom nay khac ban chat: day la
# TU/CUM TU DAY DU (khong phai ma viet tat ngan), va can GIU NGUYEN HOA-
# THUONG dung (vd "An Giang", khong phai "AN GIANG") -- co che
# match_acronym() hien tai CHUAN HOA VE VIET HOA (dung cho OCR "vtv"
# thuong van nhan ra VTV), ap dung sai cho ten rieng se lam sai dang viet
# hoa/thuong that. Can 1 co che rieng (vd domain_vocab/whitelist ten rieng
# moi, match theo CUM TU nhieu token, khong phai tung token don le) --
# CHUA code, de day lam du lieu tham khao cho buoc thiet ke sau. Xem
# PLACENAME_MAP / BRAND_WHITELIST duoi day.
PLACENAME_MAP: dict[str, str] = {
    # cu (truoc sap nhap) -> moi (2025+), theo file goc. Ten tinh/thanh
    # hien hanh (khong doi) anh xa ve chinh no.
    "Bà Rịa - Vũng Tàu": "Thành phố Hồ Chí Minh",
    "Bình Dương": "Thành phố Hồ Chí Minh",
    "Bắc Giang": "Bắc Ninh",
    "Hải Dương": "Hải Phòng",
    "Thái Bình": "Hưng Yên",
    "Nam Định": "Ninh Bình",
    "Hà Nam": "Ninh Bình",
    "Quảng Bình": "Quảng Trị",
    "Kon Tum": "Quảng Ngãi",
    "Đắk Nông": "Lâm Đồng",
    "Bình Thuận": "Lâm Đồng",
    "Phú Yên": "Đắk Lắk",
    "Ninh Thuận": "Khánh Hòa",
    "Bến Tre": "Vĩnh Long",
    "Trà Vinh": "Vĩnh Long",
    "Tiền Giang": "Đồng Tháp",
    "Long An": "Tây Ninh",
    "Sóc Trăng": "Cần Thơ",
    "Hậu Giang": "Cần Thơ",
    "Bạc Liêu": "Cà Mau",
    "Kiên Giang": "An Giang",
    "Vĩnh Phúc": "Phú Thọ",
    "Hòa Bình": "Phú Thọ",
    "Yên Bái": "Lào Cai",
    "Hà Giang": "Tuyên Quang",
    "Bắc Kạn": "Thái Nguyên",
}

# Thuong hieu quoc te pho bien tai Viet Nam (tu file goc, muc 14) -- danh
# sach tham khao, CHUA wiring vao pipeline (xem ghi chu tren).
BRAND_WHITELIST: set[str] = {
    "AJINOMOTO", "Samsung", "Apple", "Sony", "LG", "Panasonic", "Sharp",
    "Toshiba", "Hitachi", "Philips", "Canon", "Nikon", "Casio", "Brother",
    "Epson", "Dell", "HP", "Lenovo", "Asus", "Acer", "MSI", "Intel", "AMD",
    "NVIDIA", "Qualcomm", "MediaTek", "Microsoft", "Google", "Android",
    "Windows", "Chrome", "YouTube", "Netflix", "Spotify", "TikTok", "Meta",
    "Facebook", "Instagram", "WhatsApp", "OpenAI", "ChatGPT", "Gemini",
    "Claude", "Qwen", "Llama", "Xiaomi", "Huawei", "OPPO", "Vivo",
    "Realme", "OnePlus", "Nokia", "Motorola", "Honor", "POCO", "Redmi",
    "Leica", "Zeiss", "Dyson", "Bosch", "Siemens", "Electrolux",
    "Whirlpool", "Midea", "Haier", "Hisense", "Daikin", "Carrier",
    "Kawasaki", "Honda", "Toyota", "Lexus", "Mazda", "Subaru", "Suzuki",
    "Mitsubishi", "Nissan",
}

# (2026-08-13) Dia danh/su kien KHONG phai ten tinh/thanh (nen khong thuoc
# PLACENAME_MAP) -- phat hien tu chay sanity-test that tren bo
# ocr_test_diverse_tuned (5 dataset, 30 video): cac ten nay bi dict_suggest
# "sua" thanh tu vo nghia vi khong nam trong wordlist tieng Viet. Them tay
# tung ten (khong dung nguon tong hop ngoai vi sandbox chan
# raw.githubusercontent.com/api.github.com -- xem TIEN_DO_OCR.md). "Camau"
# la dang dinh lien khong dau gap trong OCR that, khac voi "Ca Mau" (2 tu,
# da co san trong PLACENAME_MAP qua entry "Bac Lieu" -> "Ca Mau").
LANDMARK_WHITELIST: set[str] = {
    "Bà Nà", "Tịnh Biên", "Vĩnh Gia", "Sông Đốc", "Camau",
}

# ---------------------------------------------------------------------
# (2026-07-25, MOI) Whitelist ten rieng (dia danh + thuong hieu) -- CO CHE
# rieng, KHONG dung chung match_acronym() (xem ghi chu tren PLACENAME_MAP/
# BRAND_WHITELIST ve ly do: can match CUM TU nhieu token + GIU NGUYEN hoa-
# thuong that, khong chuan hoa nhu acronym).
#
# QUYET DINH THIET KE (theo yeu cau nguoi dung 2026-07-25, huong AN TOAN):
# CHI dung 2 danh sach nay de LOAI token/cum token ra khoi cac luong sua
# loi khac (dictionary/domain_vocab/bigram re-rank co the vo tinh "sua"
# nham ten rieng it gap, vd "Ajinomoto" khong phai tu tieng Viet nen
# LexiconDictionary.suggest() co the de xuat sai; "Binh Duong" ca 2 tu
# rieng le deu la tu that nen khong bi Luong 1 dong vao, nhung van co the
# bi bigram re-rank nghi oan vi hiem trong corpus tin tuc). KHONG tu dong
# doi ten TINH CU sang TINH MOI trong text -- viec do se lam SAI LECH noi
# dung goc cua video (vd video quay tai "Binh Duong" nam ghi hinh, tu dong
# doi thanh "Thanh pho Ho Chi Minh" se gay hieu lam ve dia diem that su
# trong tu lieu) -- day KHONG phai viec cua tang post-correction (chi sua
# loi OCR, khong doi noi dung).
# ---------------------------------------------------------------------


def _tokenize_phrase(phrase: str) -> tuple[str, ...]:
    return tuple(w.lower() for w in phrase.split())


def _build_proper_noun_phrases() -> frozenset[tuple[str, ...]]:
    phrases: set[tuple[str, ...]] = set()
    for old, new in PLACENAME_MAP.items():
        phrases.add(_tokenize_phrase(old))
        phrases.add(_tokenize_phrase(new))
    for brand in BRAND_WHITELIST:
        phrases.add(_tokenize_phrase(brand))
    for landmark in LANDMARK_WHITELIST:
        phrases.add(_tokenize_phrase(landmark))
    return frozenset(phrases)


# Cache 1 lan luc import module (PLACENAME_MAP/BRAND_WHITELIST la hang so
# module-level, khong doi luc runtime).
_PROPER_NOUN_PHRASES: frozenset[tuple[str, ...]] = _build_proper_noun_phrases()
_MAX_PHRASE_LEN: int = max((len(p) for p in _PROPER_NOUN_PHRASES), default=1)


# (2026-08-13, MOI) Guard cap-DONG cho van ban khong phai tieng Viet -- xem
# TIEN_DO_OCR.md entry "do ty le dong khong phai tieng Viet": data that lan
# ca dong tieng Anh/chu Han/Han-Han nguyen cau (bien hieu song ngu, phu de
# nuoc ngoai) -- vi cac dong nay khong khop tu dien Viet nen truot het qua
# Buoc 1-3, roi bi Buoc 4 (dict_suggest catch-all) "sua" thanh tu tieng Viet
# vo nghia (do tren 4193 luot dict_suggest hien tai: 947 luot, 22.6%, nam
# tren dong dang nghi loai nay). Khac cac guard token-le (allcaps/code) --
# day la guard o CAP DONG, ap dung TRUOC ca proper_noun_spans.
_RE_CJK_OR_OTHER_SCRIPT = re.compile(
    "[一-鿿぀-ヿ가-퟿Ѐ-ӿ]"
)
_VIET_DIACRITIC_CHARS = (
    "ăâêôơưđĂÂÊÔƠƯĐ"
    "àằầèềìòồờùừỳáắấéếíóốớúứýạặậẹệịọộợụựỵãẵẫẽễĩõỗỡũữỹảẳẩẻểỉỏổởủửỷ"
)
_RE_VIET_DIACRITIC = re.compile(f"[{_VIET_DIACRITIC_CHARS}]")
_RE_WORD = re.compile(r"[A-Za-zÀ-ỹ]+")
_ENGLISH_STOPWORDS = {
    "the", "and", "is", "are", "to", "of", "in", "for", "this", "that",
    "you", "we", "with", "on", "at", "by", "from", "as", "it", "be",
    "was", "were", "have", "has", "will", "can", "not", "your", "our",
    "please", "visitors", "advised", "strictly", "should", "must",
}
_ENGLISH_LINE_MIN_STOPWORDS = 2
_ENGLISH_LINE_MIN_WORDS = 4

# (2026-08-13, MOI) Nguong do dai toi thieu de TIN ket qua py3langid --
# do that tren 455 video: KHONG co nguong nay, langid.classify() doan sai
# tren dong NGAN (dac thu OCR video: watermark/ten mon/ten rieng 1-2 tu)
# nghiem trong (68.24% TOAN BO dong bi bao "khong phai tieng Viet", 5.539
# dong CO dau tieng Viet that bi doan nham thanh Séc/Nhat/Kurd/Slovak...).
# Voi nguong >=8 tu: false positive giam ve 18/12668 dong xet (~0.14%), va
# sau khi uu tien dau tieng Viet that (xem duoi) con lai 0/136 dong moi bat
# duoc nho langid. Duoi nguong nay, KHONG goi langid, chi dung heuristic
# stopword cu (van an toan cho dong ngan hon).
_LANGID_MIN_WORDS = 8


def is_non_vietnamese_line(text: str) -> bool:
    """True neu CA DONG trong nghi khong phai tieng Viet. Thu tu kiem tra
    (uu tien AN TOAN -- dau tieng Viet THAT luon thang moi tin hieu khac):
        1. Co dau tieng Viet that trong dong -> LUON coi la tieng Viet,
           bo qua moi kiem tra con lai (xem bai hoc bug Cyrillic duoi).
        2. Co ky tu CJK/script khac (Han/Han/Cyrillic) -> khong phai Viet.
        3. Dong dai (>= 8 tu) VA py3langid (neu co cai) phan loai KHAC "vi"
           -> khong phai Viet. Chi ap dung cho dong DAI vi do that cho thay
           langid doan sai nhieu tren dong ngan (xem _LANGID_MIN_WORDS).
        4. Dong co >= 4 tu VA >= 2 stopword tieng Anh thong dung -> nghi
           tieng Anh (heuristic cu, van dung cho dong ngan hon nguong 3).

    (2026-08-13, DA SUA bug thuc te) KIEM TRA DAU TIENG VIET TRUOC ca CJK/
    script khac -- ban dau lam nguoc lai (CJK check truoc), gay FALSE
    POSITIVE that: do tren 455 video phat hien 25/1637 dong bi flag oan la
    tin tuc tieng Viet THAT (vd "Sau rieng dong lanh Dak Lak..."), chi vi
    OCR lan 1 ky tu Cyrillic-lookalike (K/y/o/p tieng Nga nhin giong het
    Latin K/y/o/p, loi font/OCR rat pho bien) khien CJK-check kich hoat
    TRUOC KHI kip thay dau tieng Viet ro rang trong cau. Nguyen tac nay ap
    dung CHUNG cho ca langid (buoc 3) -- neu co dau tieng Viet, khong bao
    gio goi langid ca, tranh lap lai kieu bug tuong tu.
    """
    if _RE_VIET_DIACRITIC.search(text):
        return False
    if _RE_CJK_OR_OTHER_SCRIPT.search(text):
        return True
    words = _RE_WORD.findall(text)
    if _langid is not None and len(words) >= _LANGID_MIN_WORDS:
        lang, _score = _langid.classify(text)
        if lang != "vi":
            return True
    if len(words) < _ENGLISH_LINE_MIN_WORDS:
        return False
    eng_hits = sum(1 for w in words if w.lower() in _ENGLISH_STOPWORDS)
    return eng_hits >= _ENGLISH_LINE_MIN_STOPWORDS


_RE_PURE_DIGITS = re.compile(r"^\d+$")
_RE_BIB_CODE = re.compile(r"^[A-Z]{2,4}$")


def find_bib_code_spans(tokens: list[str]) -> set[int]:
    """Quet danh sach token tim ma bib dua xe (vd "037 GNT R.Maikin", "051
    DPG") -- vi du that tu L23, xem verify_dict_suggest_L23.py + thao luan
    lop 3 voi Hao 2026-08-12: cac ma nay (2-4 chu hoa, dung ngay sau 1 so)
    bi dict_suggest doan sai thanh tu vo nghia (vd "TLT"->"tot", "GNT"
    ->"gat", "DPG"->"dan") vi khong co trong tu dien va khong du lap lai
    qua nhieu video de vao domain_vocab.

    Dieu kien nhan dien (2 token LIEN TIEP, KHONG can token thu 3 ten
    nguoi -- xem ly do trong thao luan thiet ke: yeu cau chi 2 token da du
    dac trung, doi hoi them token thu 3 se bo sot cac dong nhu "051 DPG"
    khong co ten kem theo):
        token[i]   : toan chu so (vd "037", "051")
        token[i+1] : 2-4 chu cai HOA lien tuc (vd "GNT", "DPG")
    Tra ve INDEX cua token[i+1] (ma bib) -- KHONG gom token[i] (so) vi so
    khong bi dict_suggest doan sai (bi guard ty-le-chu-cai trong
    LexiconDictionary.suggest() chan tu truoc, xem viet_dictionary.py).

    Cung ap dung tuong tu proper_noun_spans: dat o Buoc 0.5 cua
    correct_text(), PASS THANG token khop, khong qua resolver chinh.
    """
    protected: set[int] = set()
    n = len(tokens)
    for i in range(n - 1):
        if _RE_PURE_DIGITS.match(tokens[i]) and _RE_BIB_CODE.match(tokens[i + 1]):
            protected.add(i + 1)
    return protected


def find_proper_noun_spans(tokens: list[str]) -> set[int]:
    """Quet danh sach token (DA qua tokenize()+normalize_token(), CHUA qua
    resolver nao) tim cum token khop VOI PLACENAME_MAP/BRAND_WHITELIST
    (so sanh khong phan biet hoa/thuong, GIU NGUYEN token goc -- ham nay
    CHI tra ve VI TRI, khong sua text). Uu tien khop cum DAI truoc (vd uu
    tien "Ba Ria - Vung Tau" hon la khop rieng le "Ba"/"Ria").

    Tra ve set cac INDEX (theo vi tri trong `tokens`) thuoc 1 cum ten rieng
    da nhan dien -- caller (post_ocr_correction.correct_text()) dung de
    BO QUA hoan toan resolver (dictionary/domain_vocab/bigram re-rank) cho
    cac vi tri nay, giu nguyen token goc 100%.
    """
    n = len(tokens)
    lowered = [t.lower() for t in tokens]
    protected: set[int] = set()
    i = 0
    while i < n:
        matched_len = 0
        max_try = min(_MAX_PHRASE_LEN, n - i)
        for length in range(max_try, 0, -1):
            candidate = tuple(lowered[i : i + length])
            if candidate in _PROPER_NOUN_PHRASES:
                matched_len = length
                break
        if matched_len:
            protected.update(range(i, i + matched_len))
            i += matched_len
        else:
            i += 1
    return protected


# (2026-08-13, MOI) Khung-xuong-khong-dau (>=2 tu) -> chinh ta dung goc, cho
# CUNG nguon du lieu voi _PROPER_NOUN_PHRASES (PLACENAME_MAP + LANDMARK_
# WHITELIST) -- xu ly truong hop OCR that SAI DAU ngay tu dau nen khong khop
# CHINH XAC duoc voi find_proper_noun_spans() (vd OCR ra "TỊNH BIỂN" trong
# khi ten that la "Tịnh Biên", khac dau o tu thu 2 -- xem TIEN_DO_OCR.md
# entry 2026-08-13 "vá lần 2", phat hien qua sanity-test that tren bo
# ocr_test_diverse_tuned). CHI ap dung tu >=2 tu (so voi tim_proper_noun_
# spans khop ca 1 tu): tu don bo dau de trung ngau nhien voi tu pho bien
# khac (vd "biên"/"biển" cung 1 nghia neu bo dau rieng le), con khop LIEN
# TIEP 2-3 tu thi dac trung hon nhieu, it rui ro hon. BRAND_WHITELIST
# (thuong hieu quoc te) KHONG dua vao day -- da la chu La-tinh khong dau,
# khong co van de sai-dau-tieng-Viet.
def _build_landmark_skeletons() -> dict[tuple[str, ...], str]:
    skeletons: dict[tuple[str, ...], str] = {}
    source_phrases = list(PLACENAME_MAP.keys()) + list(PLACENAME_MAP.values()) + list(LANDMARK_WHITELIST)
    for phrase in source_phrases:
        words = phrase.split()
        if len(words) < 2:
            continue
        skeleton = tuple(diacritic_restoration.strip_diacritics(w).lower() for w in words)
        skeletons[skeleton] = phrase
    return skeletons


_LANDMARK_SKELETONS: dict[tuple[str, ...], str] = _build_landmark_skeletons()
_MAX_SKELETON_LEN: int = max((len(k) for k in _LANDMARK_SKELETONS), default=2)

# (2026-08-13) Ho nguoi Viet pho bien nhat (nguon: thong ke ho pho bien VN,
# ~90% dan so roi vao nhom nay) -- dung de CHAN false positive: cum "Quang
# Trí" (ten nguoi, vd "Ngô Quang Trí") va "Quảng Trị" (ten tinh) TRUNG khung
# xuong sau khi bo dau ("quang tri") -- phat hien qua sanity-test that
# (L28_a/L28_V004.json, dong "NGÔ QUANG TRÍ" bi sua nham thanh "NGÔ Quảng
# Trị"). Neu token NGAY TRUOC 1 cum khop khung-xuong nam trong danh sach nay,
# coi la ten nguoi (Ho + Ten), BO QUA sua -- uu tien an toan (thà bo sot 1
# dia danh that hiem gap ngay sau ho, con hon pha ten nguoi that gap thuong
# xuyen hon nhieu trong OCR credit/chu ky).
VIETNAMESE_SURNAMES: set[str] = {
    "nguyễn", "trần", "lê", "phạm", "hoàng", "huỳnh", "phan", "vũ", "võ",
    "đặng", "bùi", "đỗ", "hồ", "ngô", "dương", "lý",
}


def find_landmark_skeleton_corrections(tokens: list[str]) -> dict[int, str]:
    """Quet tim cum token TRUNG KHUNG-XUONG (bo dau, khong phan biet hoa-
    thuong) voi 1 dia danh da biet trong _LANDMARK_SKELETONS. KHAC
    find_proper_noun_spans(): ham do chi GIU NGUYEN token khop CHINH XAC,
    ham nay SUA VE chinh ta dung khi chi khop khung-xuong (chap nhan token
    goc sai dau). Caller (post_ocr_correction.correct_text()) nen goi ham
    nay SAU proper_noun_spans/bib_code_spans va CHI dung ket qua cho vi tri
    chua duoc 2 tap kia bat (uu tien khop chinh xac truoc).

    BO QUA cum khop neu token ngay truoc no la 1 ho trong VIETNAMESE_SURNAMES
    (nghi la ten nguoi, xem ghi chu tren VIETNAMESE_SURNAMES).

    Tra ve {index: tu_dung_tai_vi_tri_do} (da flatten san theo tung token,
    khong phai theo cum, de caller dung truc tiep nhu 1 dict tra cuu).
    """
    n = len(tokens)
    result: dict[int, str] = {}
    i = 0
    while i < n:
        matched_len = 0
        matched_phrase = ""
        max_try = min(_MAX_SKELETON_LEN, n - i)
        for length in range(max_try, 1, -1):
            skeleton = tuple(
                diacritic_restoration.strip_diacritics(tokens[i + k]).lower()
                for k in range(length)
            )
            if skeleton in _LANDMARK_SKELETONS:
                matched_len = length
                matched_phrase = _LANDMARK_SKELETONS[skeleton]
                break
        if matched_len and i > 0 and tokens[i - 1].lower() in VIETNAMESE_SURNAMES:
            matched_len = 0  # nghi ten nguoi (Ho + cum vua khop), bo qua
        if matched_len:
            for k, word in enumerate(matched_phrase.split()):
                result[i + k] = word
            i += matched_len
        else:
            i += 1
    return result


# So + don vi do luong / tien te / thoi gian / nhiet do thuong gap.
_UNITS = (
    r"ml|l|lít|g|kg|mg|tấn|tạ|"
    r"muỗng|thìa|"
    r"m|cm|mm|km|"
    r"độ|°C|°F|"
    r"h|giờ|phút|s|giây|"
    r"%|"
    r"đ|₫|vnđ|usd|\$"
)

# So thap phan/nguyen: cho phep dau "," hoac "." lam phan cach thap phan
# (OCR tieng Viet hay lan lon 2 kieu).
_NUMBER = r"\d+(?:[.,]\d+)?"

# Ky tu de bi OCR nham lan VOI chu so (dung de xay dung 1 phien ban "long"
# hon cua _NUMBER, cho phep cac ky tu nay xen vao phan SO -- KHONG dung
# cho phan don vi/chu, tranh sua nham "ml" -> "m1" (xem _fix_number_prefix).
_DIGIT_CONFUSABLES = {
    "O": "0", "o": "0",
    "l": "1", "I": "1", "i": "1",
    "S": "5",
    "B": "8",
}
_CONFUSABLE_CHARS = "".join(_DIGIT_CONFUSABLES)
_NUMBER_LOOSE = rf"[0-9{_CONFUSABLE_CHARS}]+(?:[.,][0-9{_CONFUSABLE_CHARS}]+)?"

_RE_MEASUREMENT = re.compile(
    rf"^{_NUMBER}\s*(?:{_UNITS})$", re.IGNORECASE
)
# Ban "long" -- group 1 la phan so (co the lan ky tu confusable), group 2
# la phan don vi. Chi dung de PHAT HIEN can sua, khong dung de PASS truc
# tiep (phai fix group 1 roi match lai bang _RE_MEASUREMENT).
_RE_MEASUREMENT_LOOSE = re.compile(
    rf"^({_NUMBER_LOOSE})\s*({_UNITS})$", re.IGNORECASE
)

# Ngay/thang/nam: dd/mm/yyyy, dd-mm-yyyy, dd.mm.yyyy (chap nhan nam 2 hoac
# 4 chu so). Gio: hh:mm hoac hh:mm:ss.
_RE_DATE = re.compile(r"^\d{1,2}[/\-.]\d{1,2}(?:[/\-.]\d{2,4})?$")
_RE_TIME = re.compile(r"^\d{1,2}:\d{2}(?::\d{2})?$")
_RE_DATE_LOOSE = re.compile(rf"^({_NUMBER_LOOSE})[/\-.]({_NUMBER_LOOSE})(?:[/\-.]({_NUMBER_LOOSE}))?$")
_RE_TIME_LOOSE = re.compile(rf"^({_NUMBER_LOOSE}):({_NUMBER_LOOSE})(?::({_NUMBER_LOOSE}))?$")

# So dien thoai VN (10-11 so, co the co dau cach/gach ngang da bi tokenize
# dinh lien vao 1 token) hoac ma so chung (>= 4 chu so lien tuc).
_RE_PHONE = re.compile(r"^(0|\+84)\d{9,10}$")
_RE_CODE = re.compile(r"^[A-Z0-9]{4,}$")

# Phan tram / tien te dinh lien so (vd "50%", "20.000đ", "100.000VNĐ").
_RE_PERCENT = re.compile(rf"^{_NUMBER}%$")
_RE_PERCENT_LOOSE = re.compile(rf"^({_NUMBER_LOOSE})%$")
_RE_CURRENCY = re.compile(rf"^{_NUMBER}\s*(?:đ|₫|vnđ|VND|USD|\$)$", re.IGNORECASE)
_RE_CURRENCY_LOOSE = re.compile(rf"^({_NUMBER_LOOSE})\s*(đ|₫|vnđ|VND|USD|\$)$", re.IGNORECASE)


@dataclass
class RegexMatch:
    category: str          # "measurement" | "date" | "time" | "phone" | "code"
                            # | "percent" | "currency" | "acronym"
    corrected: str          # token sau khi sua (co the giong token goc neu
                             # khong can sua gi, chi can PASS)
    action: str             # "pass" (giu nguyen) | "correct" (da sua ky tu)


def _fix_number_group(number_str: str) -> str:
    """Sua nham lan O/0, l/1/I, ... CHI trong 1 chuoi da duoc regex khoanh
    vung la phan SO (group rieng, tach khoi phan chu/don vi ben canh) --
    tranh loi cu: sua ca chuoi lam hong luon phan don vi (vd 'ml' -> 'm1'
    neu sua mu quang ca token 'S00ml')."""
    return "".join(_DIGIT_CONFUSABLES.get(ch, ch) for ch in number_str)


def match_measurement(token: str) -> RegexMatch | None:
    """So + don vi do luong (vd '500ml', '2 kg', '30 độ')."""
    if _RE_MEASUREMENT.match(token):
        return RegexMatch("measurement", token, "pass")
    # Thu sua digit confusable CHI trong phan so (group 1), giu nguyen
    # phan don vi (group 2) -- vd 'S00ml' -> so='S00'->'500', don vi='ml'.
    loose = _RE_MEASUREMENT_LOOSE.match(token)
    if loose:
        number, unit = loose.group(1), loose.group(2)
        fixed_number = _fix_number_group(number)
        candidate = f"{fixed_number}{unit}"
        if _RE_MEASUREMENT.match(candidate):
            return RegexMatch("measurement", candidate, "correct")
    return None


def match_datetime(token: str) -> RegexMatch | None:
    """Ngay/thang/nam hoac gio:phut(:giay)."""
    if _RE_DATE.match(token):
        return RegexMatch("date", token, "pass")
    if _RE_TIME.match(token):
        return RegexMatch("time", token, "pass")

    date_loose = _RE_DATE_LOOSE.match(token)
    if date_loose:
        parts = [_fix_number_group(g) for g in date_loose.groups() if g is not None]
        sep = "/" if "/" in token else ("-" if "-" in token else ".")
        candidate = sep.join(parts)
        if _RE_DATE.match(candidate):
            return RegexMatch("date", candidate, "correct")

    time_loose = _RE_TIME_LOOSE.match(token)
    if time_loose:
        parts = [_fix_number_group(g) for g in time_loose.groups() if g is not None]
        candidate = ":".join(parts)
        if _RE_TIME.match(candidate):
            return RegexMatch("time", candidate, "correct")
    return None


def match_percent_currency(token: str) -> RegexMatch | None:
    """Phan tram hoac tien te dinh lien so (vd '50%', '20.000đ')."""
    if _RE_PERCENT.match(token):
        return RegexMatch("percent", token, "pass")
    if _RE_CURRENCY.match(token):
        return RegexMatch("currency", token, "pass")

    pct_loose = _RE_PERCENT_LOOSE.match(token)
    if pct_loose:
        candidate = f"{_fix_number_group(pct_loose.group(1))}%"
        if _RE_PERCENT.match(candidate):
            return RegexMatch("percent", candidate, "correct")

    cur_loose = _RE_CURRENCY_LOOSE.match(token)
    if cur_loose:
        number, unit = cur_loose.group(1), cur_loose.group(2)
        candidate = f"{_fix_number_group(number)}{unit}"
        if _RE_CURRENCY.match(candidate):
            return RegexMatch("currency", candidate, "correct")
    return None


def match_phone_or_code(token: str) -> RegexMatch | None:
    """So dien thoai VN hoac ma so chung (>=4 ky tu chu+so lien tuc, vd ma
    san pham/su kien)."""
    stripped = token.replace(".", "").replace("-", "").replace(" ", "")
    if _RE_PHONE.match(stripped):
        return RegexMatch("phone", stripped, "pass")
    if _RE_CODE.match(token) and any(ch.isdigit() for ch in token):
        return RegexMatch("code", token, "pass")
    return None


def match_acronym(token: str, whitelist: set[str] | None = None) -> RegexMatch | None:
    """Acronym khop VOI WHITELIST -- KHONG dung heuristic 'toan chu hoa'
    (xem canh bao trong plan muc 3.5: precision thap, de nham voi loi OCR
    case, vd 'THU' co the la loi OCR cua 'Thu' chu khong phai acronym)."""
    wl = whitelist if whitelist is not None else ACRONYM_WHITELIST
    # So sanh khong phan biet hoa/thuong cho token nhung whitelist entry giu
    # nguyen form chuan de tra ve (vd luon tra "VTV" du token OCR ra "vtv").
    upper = token.upper()
    for entry in wl:
        if upper == entry.upper():
            return RegexMatch("acronym", entry, "pass")
    return None


def match_loanword(token: str, whitelist: set[str] | None = None) -> RegexMatch | None:
    """Tu muon tieng Anh thong dung (xem ENGLISH_LOANWORD_WHITELIST) --
    KHAC match_acronym(): so sanh khong phan biet hoa/thuong nhung GIU
    NGUYEN token goc (khong chuan hoa ve 1 dang) vi loanword khong co
    "dang chuan hoa" duy nhat nhu acronym."""
    wl = whitelist if whitelist is not None else ENGLISH_LOANWORD_WHITELIST
    if token.lower() in wl:
        return RegexMatch("loanword", token, "pass")
    return None


def classify(token: str, acronym_whitelist: set[str] | None = None) -> RegexMatch | None:
    """Thu tat ca cac pattern luong 3 theo thu tu tu "chat" nhat (it nham
    lan nhat) den "long" hon. Tra ve RegexMatch dau tien khop, hoac None
    neu khong pattern nao ap dung (=> khong phai viec cua luong 3, de
    resolver roi qua luong 1/2).
    """
    for matcher in (
        match_datetime,
        match_percent_currency,
        match_measurement,
        match_phone_or_code,
    ):
        result = matcher(token)
        if result is not None:
            return result

    # Loanword check truoc acronym -- ca 2 deu la whitelist "chac chan",
    # thu tu giua chung khong quan trong (khong the vua la acronym vua la
    # loanword cung luc), dat truoc de gan voi nhom "tu/cum tu co nghia"
    # hon la nhom ky hieu vet-tat.
    loanword_result = match_loanword(token)
    if loanword_result is not None:
        return loanword_result

    # Acronym check sau cung trong luong 3 (van la match "chac chan" theo
    # whitelist, nhung dat sau de uu tien cac pattern so/ngay-gio truoc --
    # tranh 1 chuoi vua giong ma so vua trung whitelist gay nham lan thu tu).
    return match_acronym(token, acronym_whitelist)


if __name__ == "__main__":
    samples = [
        ("500ml", "measurement"),
        ("2kg", "measurement"),
        ("S00ml", "measurement"),   # digit confusable
        ("20/07/2026", "date"),
        ("14:30", "time"),
        ("50%", "percent"),
        ("20.000đ", "currency"),
        ("0987654321", "phone"),
        ("L28V001", "code"),
        ("VTV", "acronym"),
        ("vtv1", "acronym"),
        ("Việt", None),             # tu thuong, khong thuoc luong 3
        ("THU", None),              # KHONG duoc coi la acronym (khong whitelist)
    ]
    print("=== Luong 3 (regex/rule-based) self-test ===")
    for token, expected in samples:
        result = classify(token)
        got = result.category if result else None
        status = "PASS" if got == expected else "FAIL"
        detail = f"-> {result.category} ({result.action}): {result.corrected!r}" if result else "-> no match"
        print(f"[{status}] {token!r:20} {detail}")
