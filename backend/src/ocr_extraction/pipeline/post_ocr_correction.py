"""
post_ocr_correction.py -- ban dung lai (2026-07-20) theo kien truc moi trong
ocr_post_correction_plan.md. File cu (list tay domain_vocab 11 entry, import
path sai) da bi xoa 2026-07-17.

Buoc 0 (bat buoc, chay truoc khi re nhanh): tokenize + NFC normalize -- xem
plan muc 3.2. Sau do 3 luong chay:

    Luong 1 (dictionary)   -- xem SpellDictionary / NullDictionary duoi day.
                               CAP NHAT 2026-07-21: da co implementation
                               THAT dau tien -- `viet_dictionary.LexiconDictionary`,
                               dung wordlist bundle trong package
                               `underthesea` (72,547 entry, Apache-2.0) VI
                               Viet74K/hunspell-vi/binhvq that KHONG tai
                               duoc trong moi truong dev hien tai (network
                               bi chan boi allowlist, khong co quyen root
                               de apt install hunspell-vi -- xem canh bao
                               chi tiet dau file viet_dictionary.py). Suggest
                               rank bang Norvig edit-distance (PROXY tam,
                               CHUA co frequency corpus that). NullDictionary
                               van giu lai lam stub/fallback cho test don vi
                               khong can load wordlist that.
    Luong 2 (domain vocab)  -- domain_vocab.py (da code rieng).
    Luong 3 (regex/rule)    -- regex_rules.py (da code rieng).

Resolver (muc 3.6), thu tu uu tien da chot:
    Buoc 1: Luong 3 (regex, tru acronym da co whitelist rieng nhung van
            thuoc luong 3) -> match -> xu ly xong, dung.
    Buoc 2: Luong 1 EXACT match (token co san trong tu dien, KHONG phai
            suggest) -> co -> PASS, dung.
    Buoc 3: domain_vocab EXACT match (token da vuot threshold luc build) ->
            co -> PASS, dung.
    Buoc 4: catch-all cuoi -- chay song song Hunspell.suggest() (luong 1)
            + domain_vocab threshold check (luong 2) -> uu tien domain_vocab
            neu match (xem GHI CHU duoi day ve viec buoc nay co the trung
            lap voi buoc 3), nguoc lai dung candidate tot nhat tu suggest;
            neu khong co candidate nao -> GIU NGUYEN token goc (khong du
            bang chung de sua, tranh sua sai con hon khong sua).

    GHI CHU THAT THA (design gap chua giai quyet trong plan goc): vi
    domain_vocab dung O DAY la CUNG 1 set da threshold-filter luc build
    (domain_vocab.load_domain_vocab), "domain_vocab exact match" (buoc 3)
    va "domain_vocab vuot threshold" (buoc 4) VE MAT LOGIC la CUNG 1 dieu
    kien -- neu buoc 3 da False thi buoc 4 kiem tra lai se LUON False,
    khong bao gio "cuu" duoc token o day. Code van giu ca 2 buoc de KHOP
    THU TU voi plan (va de tuong thich neu sau nay domain_vocab expose
    them 1 threshold "long" hon rieng cho buoc 4), nhung trong thuc te
    hien tai, buoc 4 chi con tac dung that qua nhanh Hunspell.suggest().
    Neu muon buoc 4 THAT SU khac buoc 3, can domain_vocab.py tra ve 2
    muc threshold khac nhau (vd 1 nguong "chac chan" cho buoc 3, 1 nguong
    "co the" long hon cho buoc 4) -- CHUA LAM, ghi lai o day de khong
    ai tuong nham la da xong.

CAP NHAT 2026-07-22 (fix hieu nang, xem HANDOFF/TIEN_DO_OCR.md): them
tham so `cache` tuy chon cho correct_token()/correct_text() -- xem docstring
correct_token() ben duoi. Ly do: chay full dataset thuc te phat hien
Buoc 4 (dictionary.suggest(), dac biet HunspellDictionary/spylls) qua cham
khi khong co cache, vi token rac lap lai (watermark, logo kenh...) bi tinh
lai tu dau moi lan gap.

CAP NHAT 2026-08-12 (lop 1/4 cho van de dict_suggest doan sai tren data
VLM/L23, xem verify_dict_suggest_L23.py + thao luan voi Hao): them Buoc
1.5 (dict_exact tren token da bo dau cau trang tri o ria, vd "(Hà" ->
"hà") giua Buoc 1 va Buoc 2 -- xem _strip_edge_punct()/CorrectionResult.
source == "dict_exact_stripped". Day la lop AN TOAN NHAT trong 4 lop de
xuat giam sai so cua dict_suggest (Buoc 4): xu ly 37/153 token OCR that
tu L23 dang bi day xuong Buoc 4 doan edit-distance oan trong khi ban chat
DA la tu dung trong tu dien, gom ca 2 case Buoc 4 dang doan SAI ("(Hu)"
-> "khu" thay vi "hu", "(Hà" -> "nhà" thay vi "hà"). 3 lop con lai (domain
_vocab build lai tren L23, regex pattern ma bib dua xe, guard chan token
khong giong tieng Viet truoc Buoc 4) CHUA lam, xem TIEN_DO_OCR.md.
"""

from __future__ import annotations

import unicodedata
from dataclasses import dataclass
from typing import Protocol

from src.ocr_extraction.pipeline import regex_rules
from src.ocr_extraction.pipeline.domain_vocab import is_domain_word
from src.ocr_extraction.pipeline import bigram_rerank
from src.ocr_extraction.pipeline import diacritic_restoration


# ---------------------------------------------------------------------
# Buoc 0 -- tien xu ly dung chung (xem plan muc 3.2)
# ---------------------------------------------------------------------

def tokenize(text: str) -> list[str]:
    """Tach cau thanh token cach nhau boi khoang trang (moi token ~ 1 am
    tiet tieng Viet, theo cach tokenize trong paper #9 -- xem plan)."""
    return text.split()


def normalize_token(token: str) -> str:
    """NFC normalize -- CHI la chuan hoa encoding, KHONG xoa dau thanh (xem
    canh bao trong plan muc 3.2: day la diem KHAC voi cac paper goc, data
    cua ta la OCR tho nen bat buoc phai co buoc nay)."""
    return unicodedata.normalize("NFC", token)


# ---------------------------------------------------------------------
# Luong 1 -- dictionary interface (pluggable, CHUA co implementation that)
# ---------------------------------------------------------------------

class SpellDictionary(Protocol):
    """Interface luong 1 -- xem plan muc 3.3. Implementation that (Viet74K
    + hunspell-vi union, rank suggest bang binhvq/Leipzig) CHUA duoc code
    (plan muc 7). Viet resolver dua vao Protocol nay de co the thay
    NullDictionary bang implementation that SAU nay ma KHONG phai sua
    resolver."""

    def exists(self, token: str) -> bool:
        """True neu token khop CHINH XAC voi 1 entry trong tu dien (khong
        phai suggest/gan dung)."""
        ...

    def suggest(self, token: str) -> list[str]:
        """Danh sach candidate (da rank theo tan suat, tot nhat truoc), rong
        neu khong co goi y nao (vd Hunspell tra ve rong)."""
        ...


class NullDictionary:
    """STUB -- luon bao 'khong co trong tu dien, khong co goi y'. Dung de
    resolver chay duoc va test duoc TRUOC KHI Viet74K/hunspell-vi duoc
    tich hop that (xem plan muc 7). KHONG dung ban nay trong production --
    voi NullDictionary, buoc 2 (luong 1 exact) va nua dau buoc 4 (Hunspell
    suggest) LUON mien, moi token se roi thang xuong domain_vocab hoac giu
    nguyen."""

    def exists(self, token: str) -> bool:
        return False

    def suggest(self, token: str) -> list[str]:
        return []


# ---------------------------------------------------------------------
# Resolver
# ---------------------------------------------------------------------

# (MOI, 2026-08-12) Dau cau trang tri hay dinh o RIA token do tokenize()
# chi tach theo khoang trang -- vd "(Hà" tu cau "Đông Anh (Hà Nội)" (dau
# ngoac dinh lien chu, khong co khoang trang o giua). Buoc 1 dict_exact
# khong khop vi con dau cau, token bi day xuong Buoc 4 doan edit-distance
# oan trong khi ban chat DA la tu dung. Vi du that tu L23: "(Hu)" dang bi
# dict_suggest doan sai thanh "khu" (dung ra la "hu"), "(Hà" doan sai
# thanh "nhà" (dung ra la "hà") -- xem thao luan verify dict_suggest voi
# Hao 2026-08-12. CHI gom dau cau da co bang chung that trong L23 (ngoac
# don, phay, nhay) -- KHONG gom "." hoac ":" vi 2 ky tu do co the la 1
# phan cua so thap phan/timestamp, du split_glued_phrases da xu ly rieng
# ":" o cap do ca dong (khong phai token).
_EDGE_PUNCT = "(),\"'"


def _strip_edge_punct(token: str) -> str:
    return token.strip(_EDGE_PUNCT)


# (MOI, 2026-08-12; SUA 2026-08-12 sau khi do tren 455 video) Lop 4 -- guard
# cho token NGHI KHONG PHAI tieng Viet (brand/ten rieng/tu muon chua tung
# gap, khong the liet ke truoc trong whitelist nhu lop 3 -- xem thao luan
# thiet ke voi Hao). Dieu kien nghi ngo: token (sau khi bo dau cau ria)
# toan chu cai VIET HOA, dai >=2 ky tu.
#
# BAN DAU (chan HAN TRUOC khi goi suggest()): do tren 75 token con lai cua
# L23 SAU lop 1+2+3 -- bat 35/37 token da xac nhan SAI (94.6% recall), chan
# oan 1/38 token SAI+OK da biet ("ĐƯỜC"->"được"). Nhung do rong ra toan bo
# 455 video (9 dataset, xem verify_dict_suggest_full.py) phat hien ty le
# chan oan THAT SU cao hon nhieu tren mau random 40 token (~2-4/40, uoc
# luong 5-10%) -- vi nhieu token toan hoa KHONG phai brand/ten rieng ma la
# TU TIENG VIET THAT chi sai/mat dau thanh (vd "VĂ"->"và", "RØI"->"rồi",
# dict_suggest se doan DUNG neu khong bi chan) -- xem TIEN_DO_OCR.md.
#
# SUA: KHONG con chan truoc khi goi suggest() nua. Thay vao do, GOI
# suggest() TRUOC, roi chi chan khi candidate top-1 KHAC khung phu am
# (skeleton, dung chung ham strip_diacritics() cua diacritic_restoration.py)
# so voi token -- tuc la CHO QUA neu candidate chi khac dau thanh (giong
# "VĂ"->"và"), CHAN neu candidate la 1 tu khac han (giong "KARINA"
# ->"koruna", khac khung "karina" vs "koruna" o vi tri o/a).
#
# RUI RO DA BIET (chua co giai phap, xem thao luan thiet ke): cach nay
# KHONG co bang chung ngu canh (bigram) nhu diacritic_restoration.py dang
# dung (precision 90%) -- token cung khung nhung SAI dau thanh (vd khung
# "mat" co the la "mắt"/"mát"/"mật", candidate top-1 chi theo tan suat
# chung co the chon SAI nghia dinh) van co the bi ap dung nham. Chap nhan
# rui ro nay TAM THOI, CAN do lai tren data that sau khi code xong (xem
# TIEN_DO_OCR.md) truoc khi coi la xong.
_ALLCAPS_GUARD_MIN_LEN = 2


def _looks_non_vietnamese_allcaps(token: str) -> bool:
    core = _strip_edge_punct(token)
    return (
        len(core) >= _ALLCAPS_GUARD_MIN_LEN
        and core.isalpha()
        and core.isupper()
    )


# (2026-08-13, MOI) Guard rieng cho token dang MA/KY HIEU -- xem
# TIEN_DO_OCR.md entry "review 14 token CAN XEM (L23)": phat hien 5/14 token
# dict_suggest sua sai (A3, H1, TV9, T&T, X.P) la MA DUONG/MODEL DIEN THOAI/
# TEN QUAN/LOGO KENH THAT, khong phai loi chinh ta -- diem chung la co CHU
# SO XEN CHU CAI hoac ky tu '&'/'.' o giua token, dieu KHONG BAO GIO xay ra
# trong 1 tu tieng Viet that. Khac _looks_non_vietnamese_allcaps() (chi bat
# toan chu hoa), guard nay ap dung BAT KE hoa/thuong (vd "T&T", "X.P" khong
# phai toan-hoa).
_CODE_SPECIAL_CHARS = "&."


def _looks_like_code_token(token: str) -> bool:
    core = _strip_edge_punct(token)
    if not core:
        return False
    has_letter = any(ch.isalpha() for ch in core)
    has_digit = any(ch.isdigit() for ch in core)
    if has_letter and has_digit:
        return True
    if has_letter and any(ch in _CODE_SPECIAL_CHARS for ch in core):
        return True
    return False


def _same_skeleton(token: str, candidate: str) -> bool:
    """True neu `token` va `candidate` CHI khac nhau o dau thanh/ky tu phu
    (khung phu am giong het, xem diacritic_restoration.strip_diacritics())
    -- dung de phan biet 'candidate chi sua dau' (an toan hon) voi
    'candidate la tu khac han' (rui ro cao, kieu brand->tu vo nghia)."""
    core = _strip_edge_punct(token)
    return (
        diacritic_restoration.strip_diacritics(core).lower()
        == diacritic_restoration.strip_diacritics(candidate).lower()
    )


@dataclass
class CorrectionResult:
    original: str
    corrected: str
    action: str      # "pass" | "correct" | "uncertain"
    source: str       # "regex" | "dict_exact" | "dict_exact_stripped"
                       # | "domain_exact" | "domain_threshold" | "dict_suggest"
                       # | "unresolved" | "bigram_flagged"
                       # | "dict_exact_stripped" (MOI 2026-08-12, xem
                       #   _strip_edge_punct() -- token chi khop tu dien SAU
                       #   khi bo dau cau trang tri o ria, vd "(Hà" -> "hà")
                       # | "proper_noun_whitelist" (MOI 2026-07-25, xem
                       #   regex_rules.find_proper_noun_spans())
                       # | "bib_code_whitelist" (MOI 2026-08-12, xem
                       #   regex_rules.find_bib_code_spans() -- ma bib dua
                       #   xe dung ngay sau 1 token toan so, vd "037 GNT")
                       # | "non_vietnamese_line" (MOI 2026-08-13, xem
                       #   regex_rules.is_non_vietnamese_line() -- CA DONG
                       #   nghi khong phai tieng Viet, pass THANG toan bo
                       #   token trong dong, khong qua Buoc 1-4 nao)
                       # | "diacritic_autocorrect" (MOI 2026-07-30, xem
                       #   diacritic_restoration.py -- KHAC bigram_flagged:
                       #   day la truong hop DUY NHAT trong toan bo resolver
                       #   THAT SU ghi de corrected cho 1 tu da "pass" Buoc 1
                       #   dict_exact, dua tren nguong da kiem chung rieng
                       #   (precision 90.0%, FP 1.8% -- xem TIEN_DO_OCR.md
                       #   entry 07-30 "chot")
                       # | "landmark_skeleton_match" (MOI 2026-08-13,
                       #   xem regex_rules.find_landmark_skeleton_
                       #   corrections()) -- dia danh PLACENAME_MAP/
                       #   LANDMARK_WHITELIST khop theo khung-xuong bo
                       #   dau (OCR sai dau tu dau), THAT SU sua ve
                       #   chinh ta dung (khac proper_noun_whitelist
                       #   chi giu nguyen token goc)
    # (MOI, 2026-07-24) Chi duoc dien khi source == "bigram_flagged" -- tu
    # NGHI VAN (real-word error) ma bigram_rerank.py de xuat, xem docstring
    # module do. QUAN TRONG: `corrected` VAN giu nguyen token GOC trong
    # truong hop nay (KHONG tu dong ghi de) -- `suggested` chi de LOG/REVIEW,
    # khong duoc dung de thay the corrected o buoc nay (xem ly do trong
    # bigram_rerank.py docstring: chua du du lieu de tin tuong nguong tu
    # dong sua).
    suggested: str | None = None


def correct_token(
    token: str,
    dictionary: SpellDictionary,
    domain_vocab: set[str],
    acronym_whitelist: set[str] | None = None,
    cache: dict[str, CorrectionResult] | None = None,
) -> CorrectionResult:
    """Xu ly 1 token theo dung 4 buoc resolver (plan muc 3.6). Token dau
    vao PHAI da qua tokenize()+normalize_token() (Buoc 0) truoc khi goi ham
    nay -- ham nay khong tu lam lai buoc 0.

    `cache` (MOI, 2026-07-22): dict tuy chon, key = token (da normalize),
    value = CorrectionResult da tinh truoc do. Neu duoc truyen vao (khac
    None), ham se:
      1. Tra ve ngay ket qua trong cache neu token da gap truoc do (KHONG
         goi lai regex_rules/dictionary/domain_vocab).
      2. Neu chua co, tinh binh thuong nhu cu, roi LUU vao cache truoc khi
         return.

    Ly do them: token rac lap lai rat nhieu qua nhieu frame/video
    (watermark, logo kenh, bien hieu mo...) -- khong co cache thi Buoc 4
    (dictionary.suggest(), dac biet cham voi HunspellDictionary/spylls -- co
    the mat 1-2+ giay/tu do phan tich hinh thai hoc) bi tinh lai TU DAU moi
    lan gap lai CUNG 1 token, gay cham nghiem trong khi chay full dataset
    (hang nghin token/video). Mac dinh None = KHONG cache (giu nguyen hanh
    vi cu, an toan cho unit test doc lap, xem __main__ ben duoi van goi
    khong truyen cache).

    AN TOAN VE MAT LOGIC: ket qua correct_token() CHI phu thuoc
    (token, dictionary, domain_vocab, acronym_whitelist) -- KHONG phu thuoc
    frame/video nao dang duoc xu ly. Vi vay dung 1 cache DUNG CHUNG cho ca
    lan chay (toan bo dataset, nhieu video) la dung/an toan -- KHONG can (va
    KHONG nen) tao cache rieng moi video, vi se mat het loi ich cache cho
    token lap lai GIUA cac video (vd watermark/logo kenh xuat hien o hau
    het video).
    """
    if cache is not None and token in cache:
        return cache[token]

    original = token

    # (2026-07-23, MOI -- DAO THU TU so voi ban goc) Buoc 1: luong 1 EXACT
    # match (tu dien chuan) chay TRUOC regex/acronym, khong phai sau nhu
    # ban goc. LY DO (bug that phat hien qua data thuc te 30 video, xem
    # TIEN_DO_OCR.md 2026-07-23): match_acronym() trong regex_rules.py so
    # sanh KHONG phan biet hoa/thuong roi tra ve dang CHUAN HOA VIET HOA
    # cua whitelist -- day la thiet ke CO CHU DICH (de OCR doc "vtv" thuong
    # van nhan ra VTV), nhung neu 1 entry whitelist TRUNG voi 1 tu tieng
    # Viet/tu muon THAT (vd da tung co "AI" trung dai tu "ai", "PIN" trung
    # tu "pin"/cuc pin), no se ep tu dung thanh sai (vd "ai cung" -> "AI
    # cung"). Dao thu tu -- tra tu dien THAT truoc -- giai quyet TAN GOC
    # cho CA acronym hien tai LAN acronym them sau nay, khong can go tung
    # entry rieng le moi lan phat hien trung: neu token da la tu that
    # (dictionary.exists(), khong phan biet hoa/thuong khi so, nhung TRA
    # VE token GOC nguyen case), coi nhu xong, KHONG dua qua acronym nua.
    # Token that su la acronym viet hoa dung tu OCR (vd "AI" hoa san) van
    # duoc dictionary.exists() khop (vi ham nay tu lowercase de so) va tra
    # ve CHINH NO khong doi -- khong mat kha nang nhan acronym dung case.
    # Chi mat dung 1 truong hop hiem: OCR doc acronym SAI thanh chu thuong
    # ma chu thuong do TRUNG tu that (vd nguon that la "AI" nhung OCR doc
    # ra "ai") -- se KHONG duoc chuan hoa lai thanh hoa, giu nguyen "ai".
    # Danh doi nay chap nhan duoc: giong loai gioi han "real-syllable
    # error" da biet (xem HUONG_DAN muc 6), con hon la ACTIVE sua tu dung
    # thanh sai o MOI lan gap (nhu bug cu).
    #
    # QUAN TRONG: KHONG dao domain_vocab (Buoc 3 ben duoi) len truoc buoc
    # nay -- domain_vocab hoc TU CHINH output OCR tho, dang chua san dang
    # chu thuong cua watermark (vd "htv" da co trong domain_vocab.json that,
    # xem TIEN_DO_OCR.md). Neu dao domain_vocab len truoc acronym thi se
    # VO HIEU HOA viec chuan hoa "htv"->"HTV" -- dung dictionary CHUAN
    # (khong phai domain_vocab) la an toan vi no chi chua tu tieng Viet
    # chuan, khong chua rac OCR.
    if dictionary.exists(token):
        result = CorrectionResult(original, token, "pass", "dict_exact")
        if cache is not None:
            cache[token] = result
        return result

    # Buoc 1.5 (MOI, 2026-08-12): dict_exact tren token da bo dau cau
    # trang tri o ria (xem _strip_edge_punct() o tren). Van la "tra tu
    # dien that", chi khac o cho thu lai sau khi don dau cau -- vi vay dat
    # NGAY SAU Buoc 1, TRUOC Buoc 2 (regex/acronym), giu dung nguyen tac da
    # chot o Buoc 1: tu dien that luon uu tien hon regex/acronym.
    stripped = _strip_edge_punct(token)
    if stripped and stripped != token and dictionary.exists(stripped):
        result = CorrectionResult(original, stripped, "correct", "dict_exact_stripped")
        if cache is not None:
            cache[token] = result
        return result

    # Buoc 2: luong 3 (regex/rule) -- chay SAU tu dien chuan (xem ghi chu
    # dao thu tu o Buoc 1 ben tren).
    regex_match = regex_rules.classify(token, acronym_whitelist)
    if regex_match is not None:
        result = CorrectionResult(
            original=original,
            corrected=regex_match.corrected,
            action=regex_match.action,
            source="regex",
        )
        if cache is not None:
            cache[token] = result
        return result

    # Buoc 3: domain_vocab EXACT match (da threshold-filter luc build).
    if is_domain_word(token, domain_vocab):
        result = CorrectionResult(original, token, "pass", "domain_exact")
        if cache is not None:
            cache[token] = result
        return result

    # Buoc 4: catch-all -- chay "song song" Hunspell suggest + domain
    # threshold check (xem GHI CHU THAT THA o docstring dau file ve viec
    # nhanh domain_vocab o day thuc te se luon False neu buoc 3 da False,
    # cho toi khi domain_vocab.py expose 2 muc threshold rieng biet).
    if is_domain_word(token, domain_vocab):
        result = CorrectionResult(original, token, "pass", "domain_threshold")
        if cache is not None:
            cache[token] = result
        return result

    candidates = dictionary.suggest(token)
    if candidates:
        top_candidate = candidates[0]
        # (MOI, 2026-08-12; SUA sau khi do tren 455 video) Lop 4 -- chi chan
        # khi token NGHI khong phai tieng Viet (toan hoa, xem
        # _looks_non_vietnamese_allcaps()) VA candidate top-1 KHAC khung phu
        # am voi token (xem _same_skeleton()) -- tuc la candidate sua CA
        # chu chu khong chi dau thanh, rui ro cao (kieu brand->tu vo
        # nghia). Neu candidate CUNG khung (chi khac dau, kieu "VĂ"->"và"),
        # VAN cho qua nhu dict_suggest binh thuong -- xem thao luan thiet ke
        # + rui ro con lai (chua co bang chung ngu canh) trong ghi chu tren
        # _looks_non_vietnamese_allcaps().
        # (2026-08-13, MOI) Them dieu kien _looks_like_code_token() -- bat
        # them token dang ma/model/ten quan co CHU SO xen chu hoac '&'/'.'
        # o giua (vd "A3", "H1", "TV9", "T&T", "X.P"), khong phu thuoc
        # hoa/thuong (khac allcaps guard). Xem TIEN_DO_OCR.md entry cung ngay.
        looks_risky = _looks_non_vietnamese_allcaps(token) or _looks_like_code_token(token)
        if looks_risky and not _same_skeleton(token, top_candidate):
            result = CorrectionResult(original, token, "uncertain", "unresolved")
            if cache is not None:
                cache[token] = result
            return result

        result = CorrectionResult(original, top_candidate, "correct", "dict_suggest")
        if cache is not None:
            cache[token] = result
        return result

    # Khong co bang chung nao de sua -- GIU NGUYEN, danh dau "uncertain"
    # (khac voi "pass" -- "pass" nghia la CO bang chung token dung, con
    # "uncertain" nghia la khong tim duoc gi, thu tot hon la khong sua
    # sai them). Caller (vd extract_ocr.py sau nay) co the log/dem so
    # luong "uncertain" de danh gia coverage cua 3 luong.
    result = CorrectionResult(original, token, "uncertain", "unresolved")
    if cache is not None:
        cache[token] = result
    return result


def correct_text(
    text: str,
    dictionary: SpellDictionary,
    domain_vocab: set[str],
    acronym_whitelist: set[str] | None = None,
    cache: dict[str, CorrectionResult] | None = None,
    bigram_table: dict[tuple[str, str], int] | None = None,
    bigram_alphabet: str = "",
    bigram_near_words_cache: dict[str, set[str]] | None = None,
    diacritic_base_index: dict[str, list[str]] | None = None,
) -> tuple[str, list[CorrectionResult]]:
    """Chay full pipeline (Buoc 0 + resolver) tren 1 cau/text, tra ve
    (text_da_sua, list_ket_qua_tung_token) de caller co the log/debug chi
    tiet tung quyet dinh neu can.

    `cache`: xem docstring correct_token() -- truyen thang xuong tung
    token. Caller nen tao 1 dict RONG duy nhat cho ca lan chay toan bo
    dataset (vd `self.correction_cache = {}` trong __init__ cua
    ExtractOCRPipeline) va truyen lai dict DO (khong tao moi) o moi lan
    goi correct_text(), de cache tich luy duoc qua nhieu video.

    `bigram_table`/`bigram_alphabet` (MOI, 2026-07-24): TUY CHON -- neu co
    (bigram_table khac None/rong), chay THEM 1 buoc overlay SAU resolver
    chinh (xem bigram_rerank.py): voi MOI token da "pass" o Buoc 1
    (dict_exact -- tuc la tu that, KHONG phai token da bi sua o cac buoc
    khac), kiem tra bang chung ngu canh (bigram trai/phai). Neu nghi ngo
    real-word error, GHI DE action="uncertain", source="bigram_flagged",
    suggested=<candidate> vao ket qua cua token DO trong `results` -- text
    tra ve (join tu corrected_tokens) KHONG DOI (van dung token GOC, xem
    ly do bao thu trong bigram_rerank.py docstring). Neu bigram_table=None
    (mac dinh), buoc nay bi bo qua hoan toan, hanh vi giong y het truoc
    khi co tinh nang nay -- an toan tuong thich nguoc cho caller chua
    truyen tham so moi.

    `diacritic_base_index` (MOI, 2026-07-30, xem diacritic_restoration.py):
    TUY CHON -- neu co (khac None), chay THEM 1 buoc TRUOC bigram_flagged:
    voi MOI token da "pass" Buoc 1 (dict_exact), thu tim candidate CUNG
    SKELETON (khac dau) co bang chung ngu canh du MANH (dung chung nguong
    voi bigram_rerank, xem diacritic_restoration.MIN_CANDIDATE_COUNT/
    MIN_RATIO). KHAC bigram_flagged: neu tim duoc, GHI DE corrected THAT SU
    (action="correct", source="diacritic_autocorrect") -- day la buoc DUY
    NHAT trong resolver THAT SU tu dong sua 1 tu da hop le trong tu dien,
    dua tren ket qua kiem chung rieng (precision 90.0%, FP 1.8% tren nhom
    "correct" -- xem TIEN_DO_OCR.md entry 07-30 "chot"). Token DA duoc sua
    o buoc nay se BI BO QUA khoi vong bigram_flagged phia duoi (da giai
    quyet xong, khong can flag lai). Neu diacritic_base_index=None (mac
    dinh), buoc nay bi bo qua hoan toan -- an toan tuong thich nguoc.
    """
    normalized_tokens = [normalize_token(t) for t in tokenize(text)]

    # (2026-08-13, MOI) Buoc -1 -- ca DONG nghi khong phai tieng Viet (xem
    # regex_rules.is_non_vietnamese_line() + TIEN_DO_OCR.md entry cung ngay).
    # Chay TRUOC ca proper_noun_spans: neu khop, PASS THANG toan bo token
    # trong dong, KHONG qua Buoc 1-4 nao ca -- an toan vi Buoc 1-3 von chi
    # khop CHINH XAC (khong lam gi voi tu tieng Anh/chu Han von khong co
    # trong tu dien Viet), chi Buoc 4 (dict_suggest catch-all) moi thuc su
    # "sua" nham chung thanh tu tieng Viet vo nghia.
    is_foreign_line = regex_rules.is_non_vietnamese_line(text)

    # (2026-07-25, MOI) Buoc 0.5 -- whitelist ten rieng (dia danh/thuong
    # hieu, xem regex_rules.PLACENAME_MAP/BRAND_WHITELIST). Chay TRUOC ca
    # resolver chinh: cac vi tri khop se duoc PASS THANG, giu nguyen token
    # goc 100%, KHONG qua dictionary/domain_vocab/regex/bigram re-rank --
    # tranh tinh trang cac luong do "sua nham" ten rieng it gap (vd
    # "Ajinomoto" khong phai tu tieng Viet nen co the bi Luong 1 goi y sai;
    # "Binh Duong"/"Vam"... hiem trong corpus tin tuc nen de bi bigram
    # re-rank nghi oan). Xem quyet dinh thiet ke trong regex_rules.py.
    proper_noun_spans = regex_rules.find_proper_noun_spans(normalized_tokens)

    # (MOI, 2026-08-12) Buoc 0.5b -- ma bib dua xe (xem
    # regex_rules.find_bib_code_spans() + thao luan lop 3 voi Hao). Cung
    # nguyen tac voi proper_noun_spans o tren: PASS THANG, khong qua
    # resolver -- 2 tap span nay KHONG the giao nhau (proper_noun_spans
    # khop cum ten rieng/dia danh nhieu token co san trong PLACENAME_MAP/
    # BRAND_WHITELIST, bib_code_spans khop rieng 1 token chu-hoa dung sau
    # 1 token toan-so, 2 dieu kien khac ban chat), nhung van uu tien
    # proper_noun_spans truoc cho ro rang thu tu neu sau nay co sua doi.
    bib_code_spans = regex_rules.find_bib_code_spans(normalized_tokens)

    # (2026-08-13, MOI) Buoc 0.5c -- khung-xuong dia danh (xem regex_rules.
    # find_landmark_skeleton_corrections()). KHAC proper_noun_spans o tren:
    # ap dung khi OCR SAI DAU ngay tu dau nen khong khop CHINH XAC duoc (vd
    # "TỊNH BIỂN" that trong OCR ung voi "Tịnh Biên" that) -- buoc nay SUA
    # VE chinh ta dung thay vi chi giu nguyen. Tinh SAU proper_noun_spans/
    # bib_code_spans, nhung uu tien 2 tap do khi trung vi tri (xem elif o
    # duoi -- chi ap dung cho vi tri CHUA bi 2 tap kia bat).
    landmark_corrections = regex_rules.find_landmark_skeleton_corrections(normalized_tokens)

    results = []
    corrected_tokens = []
    for i, token in enumerate(normalized_tokens):
        if is_foreign_line:
            result = CorrectionResult(token, token, "pass", "non_vietnamese_line")
        elif i in proper_noun_spans:
            result = CorrectionResult(token, token, "pass", "proper_noun_whitelist")
        elif i in bib_code_spans:
            result = CorrectionResult(token, token, "pass", "bib_code_whitelist")
        elif i in landmark_corrections:
            result = CorrectionResult(token, landmark_corrections[i], "correct", "landmark_skeleton_match")
        else:
            result = correct_token(token, dictionary, domain_vocab, acronym_whitelist, cache=cache)
        results.append(result)
        corrected_tokens.append(result.corrected)

    # (MOI, 2026-07-30) Buoc diacritic-autocorrect -- CHAY TRUOC bigram_flagged
    # o duoi. Xem diacritic_restoration.py + docstring tham so o tren de biet
    # ly do/nguong. CHI xet token da PASS Buoc 1 (dict_exact) -- giong dieu
    # kien ap dung cua bigram_flagged. Token da duoc sua o day se co
    # source="diacritic_autocorrect" (khac "dict_exact"), nen vong lap
    # bigram_flagged phia duoi (dieu kien "source != dict_exact: continue")
    # se TU DONG bo qua, khong can code them gi de "loai tru" ca.
    if diacritic_base_index and bigram_table:
        for i, result in enumerate(results):
            if result.source != "dict_exact":
                continue
            prev_token = normalized_tokens[i - 1] if i > 0 else None
            next_token = normalized_tokens[i + 1] if i + 1 < len(normalized_tokens) else None
            suggestion = diacritic_restoration.suggest_diacritic_autocorrect(
                normalized_tokens[i], prev_token, next_token,
                diacritic_base_index, bigram_table,
            )
            if suggestion is not None:
                results[i] = CorrectionResult(
                    original=result.original,
                    corrected=suggestion,
                    action="correct",
                    source="diacritic_autocorrect",
                )
                corrected_tokens[i] = suggestion

    if bigram_table:
        for i, result in enumerate(results):
            if result.source != "dict_exact":
                # Chi xet token da PASS Buoc 1 (tu that, chua bi sua boi
                # buoc nao khac) -- xem docstring bigram_rerank.py ve ly do
                # (day chinh la truong hop resolver hien tai KHONG the tu
                # nghi ngo duoc, vi ca token dung lan sai deu la tu hop le).
                continue
            # (2026-07-25) DA THU va DA BO: loai token da khop domain_vocab
            # khoi bigram re-rank (y tuong: token lap lai >=3 video la
            # "dang tin cay"). KET QUA THUC TE: recall roi tu 68.9% xuong
            # con 32.2% (danh doi qua te cho muc giam FP tu 28.2%->10.3%)
            # -- LY DO: domain_vocab (threshold tan suat >=3 video, KHONG
            # phan biet dung/sai) da bi "nhiem" chinh cac loi OCR he thong
            # LAP LAI qua nhieu video (vd "mất" xuat hien 7 lan vi la loi
            # OCR on dinh cua show "ĐÔI MẮT MEKONG", trong khi "mắt" DUNG
            # lai KHONG co trong domain_vocab; tuong tu "ghiên"(sai, 6 lan)
            # vs "ghiền"(dung, 0 lan), "khoe"(sai, 8 lan) vs "khỏe"(dung, 0
            # lan) -- xem TIEN_DO_OCR.md entry 2026-07-25). Loai domain_vocab
            # khoi bigram vo tinh BAO VE chinh cac loi he thong lap lai nay
            # khoi bi bigram re-rank bat -- day la nhung case bigram re-rank
            # dang lam TOT NHAT (loi lap lai co bang chung ro rang). Da
            # QUYET DINH KHONG ap dung mitigation nay -- giu nguyen hanh vi
            # cu (chi loai proper_noun_whitelist, xem tren). Ghi lai o day
            # de KHONG ai thu lai y tuong nay ma khong doc canh bao truoc.
            prev_token = normalized_tokens[i - 1] if i > 0 else None
            next_token = normalized_tokens[i + 1] if i + 1 < len(normalized_tokens) else None
            flagged = bigram_rerank.suggest_real_word_flag(
                normalized_tokens[i], prev_token, next_token,
                dictionary, bigram_table, bigram_alphabet,
                near_words_cache=bigram_near_words_cache,
            )
            if flagged is not None:
                results[i] = CorrectionResult(
                    original=result.original,
                    corrected=result.corrected,  # KHONG DOI -- xem docstring
                    action="uncertain",
                    source="bigram_flagged",
                    suggested=flagged,
                )

    return " ".join(corrected_tokens), results


if __name__ == "__main__":
    # Self-test: dictionary gia lap nho (thay Viet74K/hunspell-vi that,
    # CHUA co - xem plan muc 7) + domain_vocab gia lap, kiem tra dung thu
    # tu uu tien buoc 1-4. Khong truyen cache o day (mac dinh None) -- moi
    # test doc lap voi nhau, dung y do goc.
    class FakeDictionary:
        _EXACT = {"học", "sinh", "chào", "mừng", "và"}
        _SUGGEST = {"hoc": ["học"], "kuquan": []}

        def exists(self, token: str) -> bool:
            return token.lower() in self._EXACT

        def suggest(self, token: str) -> list[str]:
            return self._SUGGEST.get(token.lower(), [])

    fake_dict = FakeDictionary()
    fake_domain_vocab = {"sunfest", "vieon"}

    samples = [
        ("500ml", "regex", "500ml"),          # buoc 1
        ("học", "dict_exact", "học"),          # buoc 2
        ("sunfest", "domain_exact", "sunfest"),  # buoc 3
        ("hoc", "dict_suggest", "học"),        # buoc 4 (suggest thanh cong)
        ("kuquan", "unresolved", "kuquan"),    # buoc 4 (khong co goi y -> giu nguyen)
    ]

    print("=== post_ocr_correction resolver self-test ===")
    for token, expected_source, expected_corrected in samples:
        result = correct_token(token, fake_dict, fake_domain_vocab)
        status = "PASS" if (result.source == expected_source and result.corrected == expected_corrected) else "FAIL"
        print(f"[{status}] {token!r:12} -> {result.corrected!r:10} (source={result.source}, action={result.action})")

    print("\n=== correct_text() tren ca cau (dictionary gia) ===")
    text = "Chào mừng sunfest 500ml hoc sinh"
    corrected, results = correct_text(text, fake_dict, fake_domain_vocab)
    print(f"input : {text}")
    print(f"output: {corrected}")

    # Test rieng cho cache (MOI, 2026-07-22): goi 2 lan cung 1 token, xac
    # nhan lan 2 tra ve object CorrectionResult y het lan 1 (tu cache, khong
    # tinh lai) va cache co dung entry.
    print("\n=== Test cache (MOI) ===")
    shared_cache: dict[str, CorrectionResult] = {}
    r1 = correct_token("hoc", fake_dict, fake_domain_vocab, cache=shared_cache)
    r2 = correct_token("hoc", fake_dict, fake_domain_vocab, cache=shared_cache)
    status = "PASS" if (r1 is shared_cache.get("hoc") and r2 is r1) else "FAIL"
    print(f"[{status}] goi 'hoc' 2 lan qua cache -> cung 1 object, cache co {len(shared_cache)} entry: {list(shared_cache.keys())}")

    # Integration test voi dictionary THAT (viet_dictionary.LexiconDictionary)
    # -- kiem tra toan bo pipeline chay dung voi nguon du lieu that (du la
    # nguon tam, xem canh bao trong viet_dictionary.py), khong chi voi
    # FakeDictionary gia lap o tren.
    try:
        from src.ocr_extraction.pipeline.viet_dictionary import LexiconDictionary
        real_dict = LexiconDictionary()
        print("\n=== correct_text() tren ca cau (LexiconDictionary THAT) ===")
        text2 = "Chao mung hoc sinh sunfest 500ml"
        corrected2, results2 = correct_text(text2, real_dict, fake_domain_vocab)
        print(f"input : {text2}")
        print(f"output: {corrected2}")
        for r in results2:
            print(f"  {r.original!r:12} -> {r.corrected!r:12} (source={r.source})")
    except FileNotFoundError as e:
        print(f"\n(Bo qua integration test voi LexiconDictionary that: {e})")
