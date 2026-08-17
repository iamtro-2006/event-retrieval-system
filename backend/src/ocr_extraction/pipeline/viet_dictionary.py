"""
Luong 1 (dictionary-based) -- implementation THAT dau tien cho SpellDictionary
Protocol dinh nghia trong post_ocr_correction.py, thay the NullDictionary
stub.

!!! QUAN TRONG -- DOC KY TRUOC KHI DUNG TRONG PRODUCTION !!!

Plan (ocr_post_correction_plan.md muc 3.3) chi dinh nguon la Viet74K +
hunspell-vi (union), rank suggest bang tan suat binhvq/news-corpus hoac
Leipzig frequency list. Trong moi truong code nay (sandbox, khong co
quyen root de apt install hunspell-vi, va network bi chan boi allowlist
-- da thu curl raw.githubusercontent.com va bi tra ve 403 "blocked-by-
allowlist"), KHONG tai duoc ca Viet74K lan hunspell-vi lan binhvq/Leipzig
that.

Thay the tam thoi dang dung O DAY:
    - Nguon tu dien: wordlist 72,547 entry (tu don + cum tu) duoc BUNDLE
      SAN trong package pip `underthesea` (license Apache-2.0 -- da nam
      trong danh sach nguon duoc plan phe duyet muc 4). Day la dictionary
      noi bo underthesea dung cho word-segmentation (CRF), KHONG PHAI
      chinh xac Viet74K, nhung cung phuc vu dung muc dich: 1 tap word/
      cum-tu tieng Viet hop le de tra cuu ton tai. Da export ra file
      text tinh (vi_wordlist.txt) thay vi phu thuoc truc tiep vao duong
      dan noi bo cua underthesea (dictionary.bin nam trong thu muc model
      CRF, KHONG phai public API, co the doi giua cac ban underthesea).
    - Rank suggest: CAP NHAT 2026-07-21 -- DA CO tan suat that, tinh tu
      binhvq/news-corpus (qua mirror Hugging Face vietgpt/binhvq_news_vi,
      cung license MIT) bang build_frequency_table.py, luu trong
      vi_word_freq.tsv. Candidate cung muc edit-distance gio duoc rank
      theo tan suat GIAM DAN (tu pho bien hon xep truoc) thay vi alphabet
      -- sua dung loi da phat hien qua test that ("hoc" -> "hec" thay vi
      "hoc" -> "hoc" do "hec" xep truoc "hoc" theo alphabet du gan nhu
      khong ai dung tu "hec").
      2026-07-22: DA CHAY FULL 19,365,593 cau that (khong con la ban test
      100K nua) -- 1,548,715 tu duy nhat, luu trong vi_word_freq_full.tsv.
      Neu khong tim thay file nay, class tu dong fallback ve alphabet
      (hanh vi cu, khong crash).

CAP NHAT 2026-07-22 -- Viet74K + hunspell-vi THAT:
    - Lead da duyet dung ca 2 nguon (GPL-2.0 khong bi cuoc thi gioi han).
    - Viet74K.txt: kiem tra thi PHAT HIEN trung 100% voi wordlist dang dung
      (underthesea) -- dictionary noi bo underthesea von duoc build tu
      chinh Viet74K, nen KHONG can gop them gi, LexiconDictionary coi nhu
      da dung dung Viet74K tu dau roi.
    - hunspell-vi: nguoi dung upload ca 2 bien the "vi-DauCu" (1 kieu dat
      dau thanh) va "vi-DauMoi" (kieu con lai) -- 2 file .dic/.aff nay
      KHONG dong y voi nhau ve cach dat dau cho 1 so tu (vd "hoà" chi
      DauMoi chap nhan, "hòa" chi DauCu chap nhan -- da kiem chung bang
      test that). Vi data OCR video co the lan ca 2 kieu, class
      HunspellDictionary duoi day dung UNION ca 2 (exists() dung neu 1
      trong 2 chap nhan) thay vi chon 1 kieu -- tranh bao loi oan cho tu
      dung nhung khac kieu dat dau.
    - Dung thu vien `spylls` (pure Python, khong can bien dich hunspell C)
      de doc file .dic/.aff that -- pip install spylls.
    - suggest() cua hunspell-vi (qua affix rules that) UNION voi suggest
      cua LexiconDictionary (edit-distance tren Viet74K/underthesea), roi
      rank chung bang tan suat that (giong LexiconDictionary._rank()).

CAP NHAT 2026-07-22 -- doi chieu voi 5 paper tham khao (xem plan muc 5 +
addendum): Mei et al. 2016 (arxiv 1611.06950) la nguon anh huong ro nhat
o day.
    - BO "transpose" (hoan vi 2 ky tu lien ke) khoi _edits1(). Mei et al.
      giai thich: transpose la dac trung cua loi GO PHIM CON NGUOI (vd
      "teh" thay vi "the"), KHONG phai loi OCR -- ho dung Levenshtein
      thuan (khong Damerau-Levenshtein) vi ly do nay. _edits1() truoc day
      ke thua nguyen si thuat toan Norvig (thiet ke cho loi go phim), gio
      bo transpose de giam candidate rac khong phan anh dung ban chat loi
      OCR. CHUA co data loi OCR that de kiem chung thuc nghiem tren tieng
      Viet -- day la ap dung finding cua paper (domain sach scan tieng
      Anh 1907) sang domain khac (video OCR tieng Viet), can theo doi khi
      co nhieu du lieu that hon.
    - KHONG ap dung "feature-weighted regression ranking" (6 feature +
      AdaBoost, Section 4.3-4.4 cua paper) o thoi diem nay: ho CAN data co
      nhan (candidate dung/sai) de train regressor, ta KHONG co (khong co
      tap loi OCR tieng Viet co ground-truth). Tu bia trong so (weight)
      tay ma khong co cach kiem chung se la "rigor gia" -- de danh cho
      toi khi co du lieu that gan nhan, hoac dung tam thoi phuong an don
      gian hon (chi tan suat, dang lam). Xem addendum plan de biet chi
      tiet cac feature ho dung (edit-distance score, LCS similarity,
      exact/relaxed-context popularity) de tham khao khi lam bigram
      re-rank (muc 3.7).

VIEC CON LAI (muc 7):
    - hunspell-vi that co the phat hien mot so loi ngu phap/compound tot
      hon list tinh, nhung day van CHUA phai buoc bigram re-rank (muc 3.7,
      van de danh "lam sau").
"""

from __future__ import annotations

import unicodedata
from pathlib import Path

_DEFAULT_WORDLIST_PATH = Path(__file__).parent / "vi_wordlist.txt"
_DEFAULT_FREQ_PATH = Path(__file__).parent / "vi_word_freq_full.tsv"
# LICH SU: ban dau doc "vi_word_freq.tsv" (chi 100K cau, dung de test nhanh
# xem co logic co chay dung khong). 2026-07-22: da co ban FULL 19.3 trieu
# cau that (1,548,715 tu), luu rieng ten khac ("vi_word_freq_full.tsv") vi
# outputs folder khong cho ghi de/xoa file cu -- "vi_word_freq.tsv" (ban
# 100K cau) van con ton tai nhung KHONG CON DUNG NUA, coi nhu file cu, bo
# qua neu thay no.

# Norvig-style edit generation can bung no ve so luong candidate neu bang
# chu cai qua lon -- gioi han o day chi bao gom ky tu THUC SU xuat hien
# trong wordlist (build dong luc load, xem load_wordlist()), tranh hard-code
# thieu/thua ky tu co dau tieng Viet.


def _normalize(token: str) -> str:
    return unicodedata.normalize("NFC", token).lower()


def load_wordlist(path: str | Path = _DEFAULT_WORDLIST_PATH) -> set[str]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Khong tim thay wordlist tai {path}. Xem docstring dau file -- "
            f"day la file da export san (vi_wordlist.txt), khong tu sinh ra."
        )
    with path.open("r", encoding="utf-8") as f:
        return {_normalize(line.strip()) for line in f if line.strip()}


def load_frequency_table(path: str | Path = _DEFAULT_FREQ_PATH) -> dict[str, int]:
    """Doc file tan suat (word\\tcount, xem build_frequency_table.py). Tra
    ve dict RONG neu file khong ton tai -- KHONG raise, vi day la du lieu
    TUY CHON (LexiconDictionary van chay duoc, chi fallback ve alphabet khi
    thieu -- xem docstring dau file)."""
    path = Path(path)
    if not path.exists():
        return {}
    freq: dict[str, int] = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line:
                continue
            word, _, count_str = line.partition("\t")
            if not count_str:
                continue
            freq[_normalize(word)] = int(count_str)
    return freq


def _alphabet_from_wordlist(words: set[str]) -> str:
    """Sinh bang chu cai TU CHINH wordlist (thay vi hard-code bang chu cai
    tieng Viet) -- dam bao cover dung het cac ky tu (ca dau thanh) thuc su
    xuat hien trong nguon, khong thieu/thua."""
    chars: set[str] = set()
    for w in words:
        chars.update(w)
    chars.discard(" ")
    return "".join(sorted(chars))


def _generate_deletes(word: str, max_dist: int) -> set[str]:
    """Sinh TAT CA bien the cua `word` bang cach XOA (khong them/doi) toi da
    `max_dist` ky tu (bat ky vi tri/to hop nao), KE CA chinh `word` (0 lan
    xoa). Day la nen tang cua ky thuat SymSpell (Wolf Garbe) -- KHAC han
    Norvig _edits1() cu: chi phi sinh candidate phu thuoc DO DAI tu
    (O(len^2) cho max_dist=2), hoan toan KHONG phu thuoc KICH THUOC BANG
    CHU CAI (truoc day 108 ky tu tieng Viet co dau lam _edits1() sinh
    them/thay ky tu bung no candidate -- xem TIEN_DO_OCR.md entry 07-30).
    """
    results = {word}
    frontier = {word}
    for _ in range(max_dist):
        next_frontier: set[str] = set()
        for w in frontier:
            for i in range(len(w)):
                next_frontier.add(w[:i] + w[i + 1 :])
        results |= next_frontier
        frontier = next_frontier
    return results


def _levenshtein(a: str, b: str, max_dist: int) -> int:
    """Edit-distance chuan (them/xoa/doi 1 ky tu, KHONG transpose -- giu
    dung quyet dinh thiet ke cu, xem docstring _edits1_LEGACY/Mei et al.
    2016 o duoi). Chi dung de VERIFY/RANK candidate DA duoc thu hep boi
    delete-index (tap candidate luc nay rat nho, vai chuc phan tu, nen
    O(len_a * len_b) o day khong dang ke -- khac han truoc day khi phai
    tinh cho hang nghin candidate)."""
    la, lb = len(a), len(b)
    if abs(la - lb) > max_dist:
        return max_dist + 1
    prev = list(range(lb + 1))
    for i in range(1, la + 1):
        curr = [i] + [0] * lb
        ai = a[i - 1]
        for j in range(1, lb + 1):
            cost = 0 if ai == b[j - 1] else 1
            curr[j] = min(prev[j] + 1, curr[j - 1] + 1, prev[j - 1] + cost)
        prev = curr
    return prev[lb]


class LexiconDictionary:
    """Implementation SpellDictionary dua tren 1 set tu/cum-tu tinh (xem
    canh bao o dau file ve nguon du lieu tam thoi). Tuong thich Protocol
    SpellDictionary trong post_ocr_correction.py (exists() + suggest())."""

    def __init__(
        self,
        wordlist_path: str | Path = _DEFAULT_WORDLIST_PATH,
        freq_path: str | Path = _DEFAULT_FREQ_PATH,
        max_edit_distance: int = 2,
    ):
        self._words = load_wordlist(wordlist_path)
        # Chi giu tu DON (khong khoang trang) cho viec tra exists()/suggest()
        # tung token -- cum tu nhieu-token trong wordlist (vd "nhi khoa")
        # danh cho buoc bigram/luong 2 sau nay, khong dung o day.
        self._single_token_words = {w for w in self._words if " " not in w}
        self._alphabet = _alphabet_from_wordlist(self._single_token_words)
        self.max_edit_distance = max_edit_distance
        # Tan suat that (binhvq/news-corpus) -- xem docstring dau file. Neu
        # file khong ton tai, tra ve {} va suggest() se tu dong fallback ve
        # sap xep alphabet (hanh vi cu, van chay duoc khong crash).
        self._freq = load_frequency_table(freq_path)

        # (2026-07-31, MOI) Delete-index kieu SymSpell -- xem
        # TIEN_DO_OCR.md entry 07-30 "Can nhac toi uu LexiconDictionary.suggest()
        # (huong SymSpell)". Tien tinh 1 LAN luc load: voi MOI tu trong
        # wordlist, sinh cac bien the XOA toi da `max_edit_distance` ky tu,
        # dung lam key tra nguoc ve tap tu goc. Query suggest() sau nay chi
        # can sinh deletes cua CHINH token dau vao (re, ~len^2, khong phu
        # thuoc alphabet) roi tra index thay vi sinh toan bo
        # them/xoa/doi qua 108 ky tu nhu _edits1_LEGACY cu.
        self._delete_index: dict[str, set[str]] = {}
        for w in self._single_token_words:
            for variant in _generate_deletes(w, max_edit_distance):
                self._delete_index.setdefault(variant, set()).add(w)

    def exists(self, token: str) -> bool:
        return _normalize(token) in self._single_token_words

    def _edits1_legacy(self, word: str) -> set[str]:
        """GIU LAI CHO THAM KHAO/TEST DOI CHIEU -- ban Norvig-style cu,
        KHONG con duoc suggest() goi nua (xem _generate_deletes() +
        _delete_index o tren, ly do thay the trong TIEN_DO_OCR.md entry
        07-30 "profiling"/07-31 "toi uu SymSpell"). Sinh tat ca bien the
        cach `word` dung 1 phep sua Levenshtein (them/xoa/doi 1 ky tu).
        KHONG sinh transpose (hoan vi 2 ky tu lien ke) -- khac thuat toan
        Norvig goc (thiet ke cho loi go phim con nguoi). Mei et al. 2016
        (arxiv 1611.06950, xem docstring dau file) chi ra transpose la dac
        trung loi go phim, hiem gap trong loi OCR, nen ho dung Levenshtein
        thuan thay vi Damerau-Levenshtein. Bo de giam candidate rac."""
        splits = [(word[:i], word[i:]) for i in range(len(word) + 1)]
        deletes = [a + b[1:] for a, b in splits if b]
        replaces = [a + c + b[1:] for a, b in splits if b for c in self._alphabet]
        inserts = [a + c + b for a, b in splits for c in self._alphabet]
        return set(deletes + replaces + inserts)

    def _known(self, candidates: set[str]) -> set[str]:
        return {c for c in candidates if c in self._single_token_words}

    def _rank(self, candidates: set[str]) -> list[str]:
        """Sap xep candidate cung 1 muc edit-distance. Uu tien tan suat
        THAT (binhvq/news-corpus, giam dan) neu co (self._freq khong rong);
        candidate khong co trong bang tan suat (freq=0, vd tu qua hiem
        hoac bang tan suat moi chi la ban test 100K cau) roi xuong cuoi,
        tie-break bang alphabet de ket qua van deterministic. Neu KHONG co
        bang tan suat nao ca (self._freq rong -- vd chua chay
        build_frequency_table.py), fallback nguyen ve sap xep alphabet
        thuan (hanh vi cu)."""
        if not self._freq:
            return sorted(candidates)
        return sorted(candidates, key=lambda w: (-self._freq.get(w, 0), w))

    def suggest(self, token: str) -> list[str]:
        """Tra ve candidate hop le trong wordlist, uu tien edit-distance
        THAP nhat truoc (1 truoc, 2 sau neu edit-distance-1 khong ra ket
        qua nao). Trong cung 1 muc edit-distance, rank theo tan suat that
        (xem _rank()) neu co vi_word_freq.tsv, nguoc lai fallback alphabet.

        (2026-07-30) GUARD chan token "rac" truoc khi sinh edit-distance --
        xem TIEN_DO_OCR.md entry 2026-07-30 (profiling): token dai/nhieu ky
        tu so-dau cau (vd so dien thoai dinh lien nhieu so) KHONG phai tu
        tieng Viet that (khong syllable nao dai >~15-18 ky tu, khong co ly
        do sua chinh ta cho chuoi toan so/dau cau) -- tra ve [] NGAY, giu
        nguyen token goc, KHONG can sinh edit-distance gi ca. Nguong 20 ky
        tu + ty le chu-cai duoi 50% la uoc luong AN TOAN, CHUA test dinh
        luong ky.

        (2026-07-31) Sinh candidate qua delete-index kieu SymSpell (xem
        _generate_deletes()/_delete_index) thay vi Norvig _edits1_legacy --
        loai bo hoan toan nguyen nhan bung-no-to-hop cu (bang chu cai 108
        ky tu nhan vao them/thay ky tu, roi nhan doi o buoc leo edit-
        distance-2 -- 1 token tung do 23s rieng no, xem TIEN_DO_OCR.md).
        Vi ly do bung-no do KHONG con nua, gioi han cu "chi leo edit-
        distance-2 khi token <=12 ky tu" cung BO -- moi token qua duoc 2
        guard tren (<=20 ky tu, >=50% chu cai) deu duoc thu ca edit1 lan
        edit2, KHONG rieng gi 12 ky tu tro xuong nhu truoc.
        """
        normalized = _normalize(token)
        if not normalized or normalized in self._single_token_words:
            return []

        if len(normalized) > 20:
            return []
        alpha_count = sum(1 for ch in normalized if ch.isalpha())
        if len(normalized) > 0 and alpha_count / len(normalized) < 0.5:
            return []

        # (2026-07-31, MOI) Dung delete-index (SymSpell) thay vi _edits1_legacy
        # -- xem _generate_deletes()/_delete_index o __init__. Sinh deletes
        # cua CHINH token (re, O(len^2), khong phu thuoc alphabet), tra
        # nguoc index de lay tap candidate ma KHONG can sinh
        # them/xoa/doi qua toan bo 108 ky tu. Ket qua CANDIDATE SET tuong
        # duong 100% ban _edits1_legacy cu (ly thuyet SymSpell: 1 tu W nam
        # trong ban kinh edit-distance d cua token Q khi va chi khi ton tai
        # 1 chuoi chung dat duoc bang cach XOA toi da d ky tu tu CA W lan Q
        # -- xem Garbe, "1000x Faster Spelling Correction", 2012), chi khac
        # o CACH SINH re hon nhieu, khong doi ket qua cuoi.
        #
        # Van giu nguyen 2 guard do dai/ty-le-chu-cai o tren (token >20 ky
        # tu hoac <50% chu cai -- vd so dien thoai dinh lien) DU delete-index
        # da loai bo hoan toan nguyen nhan bung-no-to-hop goc (O(alphabet)),
        # vi 2 guard nay van co gia tri rieng: loc token CHAC CHAN khong
        # phai tu tieng Viet (rac OCR/so) truoc khi ton bat ky chi phi nao,
        # giu nguyen hanh vi/ket qua da kiem chung voi test suite hien co.
        query_deletes = _generate_deletes(normalized, self.max_edit_distance)
        candidate_words: set[str] = set()
        for qd in query_deletes:
            candidate_words |= self._delete_index.get(qd, set())
        candidate_words.discard(normalized)  # da loai o dieu kien dau ham, phong ho

        edit1 = {w for w in candidate_words if _levenshtein(normalized, w, 1) == 1}
        if edit1:
            return self._rank(edit1)

        if self.max_edit_distance >= 2:
            edit2 = {w for w in candidate_words if _levenshtein(normalized, w, 2) == 2}
            if edit2:
                return self._rank(edit2)

        return []


_DEFAULT_HUNSPELL_PREFIXES = (
    Path(__file__).parent / "vi-DauCu",
    Path(__file__).parent / "vi-DauMoi",
)


class HunspellDictionary:
    """Implementation SpellDictionary dung hunspell-vi THAT (vi-DauCu +
    vi-DauMoi, UNION ca 2 -- xem canh bao dau file ve ly do khong chon 1
    kieu dat dau) qua thu vien `spylls` (pure Python, khong can bien dich
    hunspell C goc).

    Moi prefix trong `dic_aff_prefixes` phai co ca <prefix>.dic va
    <prefix>.aff ton tai (spylls.hunspell.Dictionary.from_files quy uoc
    nay). Neu 1 trong 2 file thieu, raise FileNotFoundError NGAY luc
    khoi tao (nguyen tac "sai schema/thieu file phai chet som", giong
    indexing_pipeline.py).
    """

    def __init__(
        self,
        dic_aff_prefixes: tuple[str | Path, ...] = _DEFAULT_HUNSPELL_PREFIXES,
        freq_path: str | Path = _DEFAULT_FREQ_PATH,
    ):
        from spylls.hunspell import Dictionary as _SpyllsDictionary

        self._dicts = []
        for prefix in dic_aff_prefixes:
            prefix = Path(prefix)
            dic_path, aff_path = prefix.with_suffix(".dic"), prefix.with_suffix(".aff")
            if not dic_path.exists() or not aff_path.exists():
                raise FileNotFoundError(
                    f"Thieu {dic_path} hoac {aff_path} -- can ca 2 file "
                    f".dic/.aff cho moi bien the hunspell-vi."
                )
            self._dicts.append(_SpyllsDictionary.from_files(str(prefix)))

        self._freq = load_frequency_table(freq_path)

    def exists(self, token: str) -> bool:
        # UNION: tu chi hop le theo 1 trong 2 kieu dat dau van duoc chap
        # nhan -- xem canh bao dau file (test that: "hòa"/"hoà" moi ben
        # chi chap nhan 1 kieu).
        return any(d.lookup(token) for d in self._dicts)

    def suggest(self, token: str) -> list[str]:
        """Gop candidate tu CA 2 dictionary (moi hunspell dict co the goi
        y khac nhau do khac affix rule/kieu dau), rank chung theo tan suat
        that (giong LexiconDictionary._rank()) neu co, nguoc lai giu thu
        tu goc cua spylls (da tu sap theo do "gan" morphologically, KHONG
        phai thuan edit-distance nhu LexiconDictionary)."""
        candidates: set[str] = set()
        for d in self._dicts:
            if d.lookup(token):
                return []  # dung 1 trong 2 dict da cong nhan -- khong can suggest
            candidates.update(d.suggest(token))

        if not candidates:
            return []
        if self._freq:
            return sorted(candidates, key=lambda w: (-self._freq.get(_normalize(w), 0), w))
        return sorted(candidates)


class CombinedDictionary:
    """Gop nhieu SpellDictionary lam MOT -- tra loi cau hoi "sao phai chon 1
    trong 2 nguon". Khong co ly do gi phai chon 1: LexiconDictionary
    (Viet74K/underthesea) va HunspellDictionary (hunspell-vi that) phu
    nhau chu khong thay the nhau -- vd HunspellDictionary hieu bien cach
    tu qua affix rule (nhieu dang chia/ghep) ma wordlist tinh khong co,
    nguoc lai wordlist tinh co the co ten rieng/tu vung HunspellDictionary
    thieu. Gop UNION ca hai se cover RONG HON bat ky nguon don le nao.

    exists(): True neu BAT KY dictionary con nao noi co (KHONG can ca 2
    dong y -- 1 nguon xac nhan la du, giong nguyen tac "exact match thang
    near-match" cua resolver o muc lon hon).

    suggest(): gop candidate tu TAT CA dictionary con, KHU TRUNG (vd cung
    1 tu "học" co the duoc ca LexiconDictionary lan HunspellDictionary de
    xuat -- chi giu 1 lan), roi rank LAI TU DAU bang tan suat that (khong
    dung thu tu rieng cua tung nguon con, vi 2 nguon co the sap xep khac
    nhau -- can 1 tieu chuan CHUNG de so sanh cong bang giua candidate tu
    2 nguon khac nhau).
    """

    def __init__(
        self,
        dictionaries: list,
        freq_path: str | Path = _DEFAULT_FREQ_PATH,
    ):
        if not dictionaries:
            raise ValueError("CombinedDictionary can it nhat 1 dictionary con.")
        self._dicts = list(dictionaries)
        # Tu rank RIENG (khong dua vao self._freq noi bo cua tung dict con)
        # -- dam bao TAT CA candidate (bat ke tu nguon nao) duoc so sanh
        # tren CUNG 1 bang tan suat, cong bang.
        self._freq = load_frequency_table(freq_path)

    def exists(self, token: str) -> bool:
        return any(d.exists(token) for d in self._dicts)

    def suggest(self, token: str) -> list[str]:
        if self.exists(token):
            return []

        candidates: set[str] = set()
        for d in self._dicts:
            candidates.update(_normalize(c) for c in d.suggest(token))

        if not candidates:
            return []
        if self._freq:
            return sorted(candidates, key=lambda w: (-self._freq.get(w, 0), w))
        return sorted(candidates)


if __name__ == "__main__":
    print("=== LexiconDictionary self-test (wordlist that tu underthesea) ===")
    dictionary = LexiconDictionary()
    print(f"Loaded {len(dictionary._single_token_words)} single-token entries, "
          f"alphabet size={len(dictionary._alphabet)}")

    exact_cases = ["học", "sinh", "chào", "mừng", "Việt", "và"]
    for w in exact_cases:
        status = "PASS" if dictionary.exists(w) else "FAIL"
        print(f"[{status}] exists({w!r}) = {dictionary.exists(w)} (ky vong True)")

    print(f"\nFrequency table: {'DA CO ' + str(len(dictionary._freq)) + ' tu' if dictionary._freq else 'KHONG CO (fallback alphabet)'}")

    print()
    # Ban nay kiem tra CA thu hang: khong chi "co trong candidate" (như
    # truoc) ma con phai la CANDIDATE DAU TIEN (index 0) -- day chinh la
    # cai bi sai truoc khi co tan suat that (vd "hoc" -> "hec" thay vi
    # "hoc" -> "hoc" du ca hai deu la candidate hop le).
    suggest_cases = [
        ("hoc", "học"),     # mat dau -> insert 1 ky tu co dau
        ("hoac", "hoặc"),   # tuong tu
        ("mung", "mừng"),   # truoc day bi xep sau "bung" do alphabet
    ]
    for token, expected in suggest_cases:
        candidates = dictionary.suggest(token)
        top1_correct = bool(candidates) and candidates[0] == expected
        in_list = expected in candidates
        status = "PASS" if top1_correct else ("PARTIAL" if in_list else "FAIL")
        print(f"[{status}] suggest({token!r}) -> {candidates[:5]}{'...' if len(candidates) > 5 else ''} "
              f"(ky vong TOP1 la {expected!r}, tong {len(candidates)} candidate)")

    print()
    no_suggest = dictionary.suggest("xyzxyzxyz")
    status = "PASS" if no_suggest == [] else "FAIL"
    print(f"[{status}] suggest('xyzxyzxyz') -> {no_suggest} (ky vong rong, qua xa moi tu that)")

    print("\n=== HunspellDictionary self-test (vi-DauCu + vi-DauMoi that qua spylls) ===")
    try:
        hunspell_dict = HunspellDictionary()
        print(f"Frequency table: {'DA CO ' + str(len(hunspell_dict._freq)) + ' tu' if hunspell_dict._freq else 'KHONG CO'}")

        # exists() phai dung UNION -- tu chi hop le o 1 trong 2 kieu dat
        # dau van phai duoc chap nhan (khong bao loi oan).
        union_cases = [
            ("hòa", True),   # chi vi-DauCu chap nhan
            ("hoà", True),   # chi vi-DauMoi chap nhan
            ("học", True),   # ca 2 chap nhan
            ("xyzxyzxyz", False),
        ]
        for w, expected in union_cases:
            got = hunspell_dict.exists(w)
            status = "PASS" if got == expected else "FAIL"
            print(f"[{status}] exists({w!r}) = {got} (ky vong {expected})")

        print()
        for token, expected in [("hoc", "học"), ("hoac", "hoặc")]:
            candidates = hunspell_dict.suggest(token)
            top1_correct = bool(candidates) and candidates[0] == expected
            status = "PASS" if top1_correct else ("PARTIAL" if expected in candidates else "FAIL")
            print(f"[{status}] suggest({token!r}) -> {candidates[:5]}{'...' if len(candidates) > 5 else ''} "
                  f"(ky vong TOP1 la {expected!r})")
    except FileNotFoundError as e:
        print(f"(Bo qua -- chua co file vi-DauCu/vi-DauMoi: {e})")

    print("\n=== CombinedDictionary self-test (LexiconDictionary + HunspellDictionary gop lai) ===")
    try:
        combined = CombinedDictionary([LexiconDictionary(), HunspellDictionary()])
        # exists() phai dung neu BAT KY nguon con nao xac nhan.
        for w, expected in [("học", True), ("hòa", True), ("hoà", True), ("xyzxyzxyz", False)]:
            got = combined.exists(w)
            status = "PASS" if got == expected else "FAIL"
            print(f"[{status}] exists({w!r}) = {got} (ky vong {expected})")

        print()
        for token, expected in [("hoc", "học"), ("hoac", "hoặc")]:
            candidates = combined.suggest(token)
            top1_correct = bool(candidates) and candidates[0] == expected
            status = "PASS" if top1_correct else ("PARTIAL" if expected in candidates else "FAIL")
            print(f"[{status}] suggest({token!r}) -> {candidates[:5]}{'...' if len(candidates) > 5 else ''} "
                  f"(ky vong TOP1 la {expected!r}, tong {len(candidates)} candidate -- gop tu 2 nguon, khu trung)")
    except FileNotFoundError as e:
        print(f"(Bo qua -- chua co du file cho CombinedDictionary: {e})")
