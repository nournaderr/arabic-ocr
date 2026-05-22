"""Arabic-specific utilities: alphabet, dot mapping, text normalisation, HMDB labels."""
import unicodedata

# ── Alphabet ──────────────────────────────────────────────────────────────────
ARABIC_LETTERS: list[str] = list(
    "ابتثجحخدذرزسشصضطظعغفقكلمنهوي"
)

# ── Dot map ───────────────────────────────────────────────────────────────────
# Maps (body_class, dot_count_above, dot_count_below) → final Unicode character.
# Classifiers are trained on body classes (fewer classes → better accuracy).
# dot_features() provides the dot counts; this map reconstructs the full letter.
DOT_MAP: dict[tuple[str, int, int], str] = {
    # ba-body: ب ت ث (and closely related ن)
    ("ba",   0, 1): "ب",   # 1 below
    ("ba",   2, 0): "ت",   # 2 above
    ("ba",   3, 0): "ث",   # 3 above
    ("nun",  1, 0): "ن",   # 1 above — similar body, open bowl
    ("ya",   0, 2): "ي",   # 2 below — extended tail form

    # haa-body: ح ج خ
    ("haa",  0, 0): "ح",
    ("haa",  0, 1): "ج",   # 1 below
    ("haa",  1, 0): "خ",   # 1 above

    # dal-body: د ذ
    ("dal",  0, 0): "د",
    ("dal",  1, 0): "ذ",   # 1 above

    # ra-body: ر ز
    ("ra",   0, 0): "ر",
    ("ra",   1, 0): "ز",   # 1 above

    # sin-body: س ش
    ("sin",  0, 0): "س",
    ("sin",  3, 0): "ش",   # 3 above

    # sad-body: ص ض
    ("sad",  0, 0): "ص",
    ("sad",  1, 0): "ض",   # 1 above

    # tha-body (ta marbuta base): ط ظ
    ("ta",   0, 0): "ط",
    ("ta",   1, 0): "ظ",   # 1 above

    # ayn-body: ع غ
    ("ayn",  0, 0): "ع",
    ("ayn",  1, 0): "غ",   # 1 above

    # fa/qa-body: ف ق
    ("fa",   1, 0): "ف",   # 1 above
    ("qa",   2, 0): "ق",   # 2 above

    # Letters with no dotted variant (body class = letter)
    ("alef", 0, 0): "ا",
    ("lam",  0, 0): "ل",
    ("mim",  0, 0): "م",
    ("ha",   0, 0): "ه",
    ("waw",  0, 0): "و",
    ("kaf",  0, 0): "ك",
}

# ── HMDB folder name → Unicode ────────────────────────────────────────────────
# Maps LetterName_Position folder names (used in HMDB and similar printed-text
# datasets) to their Unicode representation.
#
# Position suffix conventions:
#   _Isolated : standalone glyph          → plain char
#   _Start    : initial form (word-right) → char + tatweel "ـ"
#   _Middle   : medial form               → tatweel + char + tatweel
#   _End      : final form  (word-left)   → tatweel + char
#
# Tatweel (U+0640 ـ) is used purely as a visual position indicator in labels;
# it is NOT part of the recognised text output.
#
# Letter naming follows common Arabic OCR paper conventions:
#   ح = Haa  (pharyngeal H, no dot)
#   ه = Heh  (glottal H, no dot)   ← different name avoids collision
#   ت = Taa  (regular T, 2 dots)
#   ط = Ttaa (emphatic T, no dot)  ← double-T avoids collision
#   ث = Thaa (Th sound)
#   ذ = Thal
#   ظ = Dhaa
HMDB_TO_UNICODE: dict[str, str] = {
    # ── Alef  ا ──────────────────────────────────────────────────────────────
    "Alef_Isolated": "ا",
    "Alef_Start":    "اـ",
    "Alef_Middle":   "ـاـ",
    "Alef_End":      "ـا",

    # ── Baa  ب ───────────────────────────────────────────────────────────────
    "Baa_Isolated": "ب",
    "Baa_Start":    "بـ",
    "Baa_Middle":   "ـبـ",
    "Baa_End":      "ـب",

    # ── Taa  ت ───────────────────────────────────────────────────────────────
    "Taa_Isolated": "ت",
    "Taa_Start":    "تـ",
    "Taa_Middle":   "ـتـ",
    "Taa_End":      "ـت",

    # ── Thaa  ث ──────────────────────────────────────────────────────────────
    "Thaa_Isolated": "ث",
    "Thaa_Start":    "ثـ",
    "Thaa_Middle":   "ـثـ",
    "Thaa_End":      "ـث",

    # ── Jeem  ج ──────────────────────────────────────────────────────────────
    "Jeem_Isolated": "ج",
    "Jeem_Start":    "جـ",
    "Jeem_Middle":   "ـجـ",
    "Jeem_End":      "ـج",

    # ── Haa  ح (pharyngeal) ───────────────────────────────────────────────────
    "Haa_Isolated": "ح",
    "Haa_Start":    "حـ",
    "Haa_Middle":   "ـحـ",
    "Haa_End":      "ـح",

    # ── Khaa  خ ──────────────────────────────────────────────────────────────
    "Khaa_Isolated": "خ",
    "Khaa_Start":    "خـ",
    "Khaa_Middle":   "ـخـ",
    "Khaa_End":      "ـخ",

    # ── Dal  د ───────────────────────────────────────────────────────────────
    "Dal_Isolated": "د",
    "Dal_Start":    "دـ",
    "Dal_Middle":   "ـدـ",
    "Dal_End":      "ـد",

    # ── Thal  ذ ──────────────────────────────────────────────────────────────
    "Thal_Isolated": "ذ",
    "Thal_Start":    "ذـ",
    "Thal_Middle":   "ـذـ",
    "Thal_End":      "ـذ",

    # ── Raa  ر ───────────────────────────────────────────────────────────────
    "Raa_Isolated": "ر",
    "Raa_Start":    "رـ",
    "Raa_Middle":   "ـرـ",
    "Raa_End":      "ـر",

    # ── Zain  ز ──────────────────────────────────────────────────────────────
    "Zain_Isolated": "ز",
    "Zain_Start":    "زـ",
    "Zain_Middle":   "ـزـ",
    "Zain_End":      "ـز",

    # ── Seen  س ──────────────────────────────────────────────────────────────
    "Seen_Isolated": "س",
    "Seen_Start":    "سـ",
    "Seen_Middle":   "ـسـ",
    "Seen_End":      "ـس",

    # ── Sheen  ش ─────────────────────────────────────────────────────────────
    "Sheen_Isolated": "ش",
    "Sheen_Start":    "شـ",
    "Sheen_Middle":   "ـشـ",
    "Sheen_End":      "ـش",

    # ── Saad  ص ──────────────────────────────────────────────────────────────
    "Saad_Isolated": "ص",
    "Saad_Start":    "صـ",
    "Saad_Middle":   "ـصـ",
    "Saad_End":      "ـص",

    # ── Daad  ض ──────────────────────────────────────────────────────────────
    "Daad_Isolated": "ض",
    "Daad_Start":    "ضـ",
    "Daad_Middle":   "ـضـ",
    "Daad_End":      "ـض",

    # ── Ttaa  ط (emphatic T) ──────────────────────────────────────────────────
    "Ttaa_Isolated": "ط",
    "Ttaa_Start":    "طـ",
    "Ttaa_Middle":   "ـطـ",
    "Ttaa_End":      "ـط",

    # ── Dhaa  ظ ──────────────────────────────────────────────────────────────
    "Dhaa_Isolated": "ظ",
    "Dhaa_Start":    "ظـ",
    "Dhaa_Middle":   "ـظـ",
    "Dhaa_End":      "ـظ",

    # ── Ain  ع ───────────────────────────────────────────────────────────────
    "Ain_Isolated": "ع",
    "Ain_Start":    "عـ",
    "Ain_Middle":   "ـعـ",
    "Ain_End":      "ـع",

    # ── Ghain  غ ─────────────────────────────────────────────────────────────
    "Ghain_Isolated": "غ",
    "Ghain_Start":    "غـ",
    "Ghain_Middle":   "ـغـ",
    "Ghain_End":      "ـغ",

    # ── Faa  ف ───────────────────────────────────────────────────────────────
    "Faa_Isolated": "ف",
    "Faa_Start":    "فـ",
    "Faa_Middle":   "ـفـ",
    "Faa_End":      "ـف",

    # ── Qaaf  ق ──────────────────────────────────────────────────────────────
    "Qaaf_Isolated": "ق",
    "Qaaf_Start":    "قـ",
    "Qaaf_Middle":   "ـقـ",
    "Qaaf_End":      "ـق",

    # ── Kaf  ك ───────────────────────────────────────────────────────────────
    "Kaf_Isolated": "ك",
    "Kaf_Start":    "كـ",
    "Kaf_Middle":   "ـكـ",
    "Kaf_End":      "ـك",

    # ── Lam  ل ───────────────────────────────────────────────────────────────
    "Lam_Isolated": "ل",
    "Lam_Start":    "لـ",
    "Lam_Middle":   "ـلـ",
    "Lam_End":      "ـل",

    # ── Meem  م ──────────────────────────────────────────────────────────────
    "Meem_Isolated": "م",
    "Meem_Start":    "مـ",
    "Meem_Middle":   "ـمـ",
    "Meem_End":      "ـم",

    # ── Noon  ن ──────────────────────────────────────────────────────────────
    "Noon_Isolated": "ن",
    "Noon_Start":    "نـ",
    "Noon_Middle":   "ـنـ",
    "Noon_End":      "ـن",

    # ── Heh  ه (glottal) ─────────────────────────────────────────────────────
    "Heh_Isolated": "ه",
    "Heh_Start":    "هـ",
    "Heh_Middle":   "ـهـ",
    "Heh_End":      "ـه",

    # ── Waw  و ───────────────────────────────────────────────────────────────
    "Waw_Isolated": "و",
    "Waw_Start":    "وـ",
    "Waw_Middle":   "ـوـ",
    "Waw_End":      "ـو",

    # ── Yaa  ي ───────────────────────────────────────────────────────────────
    "Yaa_Isolated": "ي",
    "Yaa_Start":    "يـ",
    "Yaa_Middle":   "ـيـ",
    "Yaa_End":      "ـي",

    # ── Special characters ───────────────────────────────────────────────────
    # Hamza  ء  (only appears isolated)
    "Hamza_Isolated":            "ء",
    "Hamza_Above_Isolated":      "ء",

    # Alef with Hamza above  أ
    "Alef_Hamza_Isolated":       "أ",
    "Alef_Hamza_Above_Isolated": "أ",
    "Alef_Hamza_Above_Start":    "أـ",
    "Alef_Hamza_Above_End":      "ـأ",

    # Alef with Hamza below  إ
    "Alef_Hamza_Below_Isolated": "إ",
    "Alef_Hamza_Below_End":      "ـإ",

    # Alef with Madda  آ
    "Alef_Madda_Isolated":       "آ",
    "Alef_Madda_End":            "ـآ",

    # Waw with Hamza above  ؤ
    "Waw_Hamza_Isolated":        "ؤ",
    "Waw_Hamza_End":             "ـؤ",

    # Yeh with Hamza above  ئ
    "Yeh_Hamza_Isolated":        "ئ",
    "Yeh_Hamza_Start":           "ئـ",
    "Yeh_Hamza_Middle":          "ـئـ",
    "Yeh_Hamza_End":             "ـئ",

    # Teh Marbuta  ة  (word-final only; also some datasets have _Isolated)
    "Teh_Marbuta_Isolated":      "ة",
    "Teh_Marbuta_End":           "ة",
    "Ta_Marbuta_Isolated":       "ة",
    "Ta_Marbuta_End":            "ة",

    # Alef Wasla  ٱ
    "Alef_Wasla_Isolated":       "ٱ",
    "Alef_Wasla_Start":          "ٱـ",

    # Lam-Alef ligature  لا
    "Laa_Isolated":              "لا",
    "Laa_End":                   "لا",

    # Lam-Alef with Madda  لآ
    "Lam_Alf_Mad_Isolated":      "لآ",
    "Lam_Alf_Mad_End":           "لآ",

    # Lam-Alef with Hamza above  لأ
    "Lam_Alf_Hamza_Isolated":    "لأ",
    "Lam_Alf_Hamza_End":         "لأ",

    # ── Arabic-Indic digits ───────────────────────────────────────────────────
    # Digit labels used in HMDB-style datasets (English word → Arabic-Indic digit)
    "Zero":  "٠",  "One":   "١",  "Two":   "٢",  "Three": "٣",  "Four":  "٤",
    "Five":  "٥",  "Six":   "٦",  "Seven": "٧",  "Eight": "٨",  "Nine":  "٩",
    # Also handle numeric string labels (some datasets use "0"-"9")
    "0": "٠", "1": "١", "2": "٢", "3": "٣", "4": "٤",
    "5": "٥", "6": "٦", "7": "٧", "8": "٨", "9": "٩",
}

# ── HMDB letter-name aliases ──────────────────────────────────────────────────
# Different Arabic OCR datasets use slightly different transliterations.
# This table maps alternative spellings to the canonical name used above,
# so HMDB_TO_UNICODE works regardless of which convention the training data used.
_LETTER_ALIASES: dict[str, str] = {
    # ── Standard shorter forms ────────────────────────────────────────────────
    "Ba":    "Baa",    # ب
    "Ta":    "Taa",    # ت
    "Tha":   "Thaa",   # ث
    "Jim":   "Jeem",   # ج
    "Ha":    "Heh",    # ه (glottal H; Ha_* and Haa_* are separate classes in this dataset)
    "Kha":   "Khaa",   # خ
    "Dhal":  "Thal",   # ذ
    "Ra":    "Raa",    # ر
    "Sin":   "Seen",   # س
    "Shin":  "Sheen",  # ش
    "Sad":   "Saad",   # ص
    "Dad":   "Daad",   # ض
    "Tah":   "Ttaa",   # ط
    "Zah":   "Dhaa",   # ظ
    "Fa":    "Faa",    # ف
    "Qaf":   "Qaaf",   # ق
    "Mim":   "Meem",   # م
    "Mem":   "Meem",   # م (variant)
    "Nun":   "Noon",   # ن
    "Ya":    "Yaa",    # ي
    "Ye":    "Yaa",    # ي
    # ── Dataset-specific spellings seen in the wild ───────────────────────────
    "Alf":    "Alef",  # ا — "Alf" used in some Egyptian datasets
    "Zal":    "Thal",  # ذ — "Zal" (Zain+Alef) is a common alt. transliteration
    "Gen":    "Jeem",  # ج — Egyptian pronunciation is "g"; some datasets use "Gen"
    "Geem":   "Jeem",  # ج — same reason
    "Kha":    "Khaa",  # خ — already above, kept for clarity
    "Dha":    "Dhaa",  # ظ
    "Tah":    "Ttaa",  # ط — already above
    "Yeh":    "Yaa",   # ي — "Yeh" used in Unicode standard names
    "Non":    "Noon",  # ن
    "Wow":    "Waw",   # و — informal spelling
    "Kef":    "Kaf",   # ك
    "Laam":   "Lam",   # ل
    "Mim":    "Meem",  # م — already above
    # ── This dataset's specific spellings ────────────────────────────────────
    "Gem":           "Jeem",             # ج — variant of Gen
    "Shen":          "Sheen",            # ش
    "Zin":           "Zain",             # ز
    "Yaa_Dot":       "Yaa",              # ي — explicitly marks dotted form
    "Lam_Alf":       "Laa",              # لا ligature
    "Alf_Hamza_Under":  "Alef_Hamza_Below",  # إ
    "Alf_Hamza_Above":  "Alef_Hamza_Above",  # أ
}

for _alias, _canonical in _LETTER_ALIASES.items():
    for _pos in ("Isolated", "Start", "Middle", "End"):
        _ak = f"{_alias}_{_pos}"
        _ck = f"{_canonical}_{_pos}"
        if _ak not in HMDB_TO_UNICODE and _ck in HMDB_TO_UNICODE:
            HMDB_TO_UNICODE[_ak] = HMDB_TO_UNICODE[_ck]

# ── Dot count map ─────────────────────────────────────────────────────────────
# Maps HMDB letter base-name (label before last "_Position") to
# (dots_above, dots_below). Used by filter_candidates_by_dots().
_LETTER_DOT_COUNTS: dict[str, tuple[int, int]] = {
    # 0 dots
    "Alef": (0, 0), "Ain": (0, 0), "Dal": (0, 0), "Haa": (0, 0),
    "Heh": (0, 0),  "Kaf": (0, 0), "Lam": (0, 0), "Meem": (0, 0),
    "Raa": (0, 0),  "Saad": (0, 0), "Seen": (0, 0), "Ttaa": (0, 0),
    "Waw": (0, 0),  "Hamza": (0, 0), "Alef_Hamza_Above": (0, 0),
    "Alef_Hamza_Below": (0, 0), "Alef_Madda": (0, 0), "Waw_Hamza": (0, 0),
    "Alef_Wasla": (0, 0), "Laa": (0, 0),
    # 1 dot above
    "Daad": (1, 0), "Dhaa": (1, 0), "Faa": (1, 0), "Ghain": (1, 0),
    "Khaa": (1, 0), "Noon": (1, 0), "Thal": (1, 0), "Zain": (1, 0),
    # 2 dots above
    "Qaaf": (2, 0), "Taa": (2, 0), "Teh_Marbuta": (2, 0),
    # 3 dots above
    "Sheen": (3, 0), "Thaa": (3, 0),
    # 1 dot below
    "Baa": (0, 1), "Jeem": (0, 1),
    # 2 dots below
    "Yaa": (0, 2), "Yeh_Hamza": (0, 2),
    # digits: no distinguishing dots
    "Zero": (0, 0), "One": (0, 0), "Two": (0, 0), "Three": (0, 0),
    "Four": (0, 0), "Five": (0, 0), "Six": (0, 0), "Seven": (0, 0),
    "Eight": (0, 0), "Nine": (0, 0),
}
# Add aliases so alternative transliterations also resolve correctly.
_DOT_COUNT_ALIASES: dict[str, str] = {
    "Ha": "Heh", "Kha": "Khaa", "Ba": "Baa", "Ta": "Taa", "Tha": "Thaa",
    "Jim": "Jeem", "Ra": "Raa", "Sin": "Seen", "Shin": "Sheen",
    "Sad": "Saad", "Dad": "Daad", "Tah": "Ttaa", "Zah": "Dhaa",
    "Fa": "Faa", "Qaf": "Qaaf", "Mim": "Meem", "Mem": "Meem",
    "Nun": "Noon", "Ya": "Yaa", "Ye": "Yaa", "Alf": "Alef",
    "Zal": "Thal", "Dhal": "Thal", "Gen": "Jeem", "Geem": "Jeem",
    "Gem": "Jeem", "Shen": "Sheen", "Zin": "Zain", "Yaa_Dot": "Yaa",
    "Yeh": "Yaa", "Non": "Noon", "Wow": "Waw", "Kef": "Kaf",
    "Laam": "Lam", "Dha": "Dhaa",
}
for _a, _c in _DOT_COUNT_ALIASES.items():
    if _a not in _LETTER_DOT_COUNTS and _c in _LETTER_DOT_COUNTS:
        _LETTER_DOT_COUNTS[_a] = _LETTER_DOT_COUNTS[_c]

# ── Position-suffix maps ──────────────────────────────────────────────────────
# Maps CharCrop.position tag → HMDB folder suffix.
# Used by filter_candidates_by_position() to restrict inference to the
# correct positional-form classes when trained on Option-A HMDB labels.
HMDB_POSITION_MAP: dict[str, str] = {
    "isolated": "Isolated",
    "initial":  "Start",
    "medial":   "Middle",
    "final":    "End",
}

# Inverse: HMDB suffix → CharCrop.position tag
HMDB_SUFFIX_TO_POSITION: dict[str, str] = {v: k for k, v in HMDB_POSITION_MAP.items()}


def hmdb_label_to_unicode(folder_name: str) -> str:
    """Convert an HMDB folder name to its Unicode label.

    Returns an empty string for unknown folder names.
    Example: "Ain_Start" → "عـ"
    """
    return HMDB_TO_UNICODE.get(folder_name, "")


def filter_candidates_by_position(
    candidates: list[tuple[str, float]],
    position: str,
) -> list[tuple[str, float]]:
    """Keep only candidates whose HMDB label matches the character's position.

    When the classifier is trained on Option-A HMDB labels (LetterName_Position),
    each prediction string looks like "Ain_Start".  This filter restricts the
    top-K list to entries that end with the correct suffix, giving a free
    accuracy boost at zero cost.

    Falls back to the full candidate list if no matching entries are found
    (e.g., when classifier uses plain Unicode labels instead of HMDB labels).

    Args:
        candidates: list of (label, confidence) from classifier.predict().
        position:   CharCrop.position tag ("isolated" | "initial" | "medial" | "final").

    Returns:
        Filtered and renormalised candidate list (same length ≤ original).
    """
    suffix = HMDB_POSITION_MAP.get(position, "")
    if not suffix:
        return candidates

    filtered = [
        (label, conf) for label, conf in candidates
        if label.endswith(f"_{suffix}")
    ]
    return filtered if filtered else candidates   # fallback: keep all


def filter_candidates_by_dots(
    candidates: list[tuple[str, float]],
    dots_above: int,
    dots_below: int,
) -> list[tuple[str, float]]:
    """Keep candidates whose expected dot count matches the observed count.

    Only applied when dots_above + dots_below > 0 (at least one dot observed).
    Falls back to the full list when no candidate matches (unknown label or
    ambiguous dot detection).

    The letter base-name is extracted by stripping the last _Position suffix,
    e.g. "Khaa_Middle" → base "Khaa" → expected (1, 0).
    """
    if dots_above == 0 and dots_below == 0:
        return candidates

    observed = (dots_above, dots_below)
    filtered = []
    for label, conf in candidates:
        base = label.rsplit("_", 1)[0]
        expected = _LETTER_DOT_COUNTS.get(base)
        if expected is None or expected == observed:
            filtered.append((label, conf))

    return filtered if filtered else candidates


# ── Text utilities ─────────────────────────────────────────────────────────────

def normalize_arabic(text: str) -> str:
    """Strip harakat (diacritics) and normalise Unicode for lexicon matching."""
    stripped = "".join(
        ch for ch in text
        if not (0x064B <= ord(ch) <= 0x065F)
    )
    return unicodedata.normalize("NFC", stripped)


def is_arabic_char(ch: str) -> bool:
    """True if the character falls in the Arabic Unicode block U+0600–U+06FF."""
    return 0x0600 <= ord(ch) <= 0x06FF


def join_text_rtl(words: list[str]) -> str:
    """Join Arabic words with U+200F RIGHT-TO-LEFT MARK for correct bidi display.

    Always wrap the result in arabic-reshaper + python-bidi when rendering
    in PIL or matplotlib — this function handles plain string joining only.
    """
    rtl_mark = "‏"
    return rtl_mark + (" ".join(words)) + rtl_mark
