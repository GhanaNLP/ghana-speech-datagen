"""Default text sources per language.

The ``ghana-tts-36k`` model was trained with a literal language tag prepended to
every transcript (``<|lang:CODE|> the text``).  See the training data builder in
``VoxCPM-2/training/ghana-tts-training/voxcpm_ghana_data.py``:

    tag = f"<|lang:{code}|> "
    ds  = ds.map(lambda r: {"text": tag + (r["text"] or "").strip(), ...})

This module gives every supported language a ready-made pool of text to
synthesise, so users don't have to bring their own sentences.  By default the
text comes from the multilingual :data:`GHANA_SPEECH` dataset (one config per
language).  Extra sources can be added per-language in :data:`_EXTRA_SOURCES`
with a single line — see the Twi example.

Nothing here loads audio: only the transcript column is read.
"""

from __future__ import annotations

from dataclasses import dataclass

# --------------------------------------------------------------------------- #
# Datasets
# --------------------------------------------------------------------------- #
GHANA_SPEECH = "ghananlpcommunity/ghana-speech"

# How many texts to pull per run when the caller doesn't pass --max-samples.
DEFAULT_TEXT_CAP = 5000


@dataclass(frozen=True)
class TextSource:
    """A place to read synthesis text from.

    ``dataset`` is an HF dataset id, ``config`` its config/subset (or ``None``),
    ``text_column`` the transcript column, and ``split`` the split to read.
    """
    dataset: str
    text_column: str = "text"
    config: str | None = None
    split: str = "train"


# --------------------------------------------------------------------------- #
# Language table
# --------------------------------------------------------------------------- #
# (ghana-speech config, lang code == the model's <|lang:CODE|> tag, display name)
#
# The code matches how training derived it (voxcpm_ghana_data.code_for_dir):
# Twi is split by dialect (twi-akuapem / twi-asante); every other language uses
# the trailing ISO code of its config name.
_GHANA_SPEECH_LANGS: list[tuple[str, str, str]] = [
    ("Akuapem_Twi_twi",       "twi-akuapem", "Akuapem Twi"),
    ("Anyin_any",             "any",         "Anyin"),
    ("Asante_Twi_twi",        "twi-asante",  "Asante Twi"),
    ("Avatime_avn",           "avn",         "Avatime"),
    ("Bassar_Ntcham_bud",     "bud",         "Bassar Ntcham"),
    ("Bimoba_bim",            "bim",         "Bimoba"),
    ("Birifor_Southern_biv",  "biv",         "Southern Birifor"),
    ("Bissa_bib",             "bib",         "Bissa"),
    ("Buli_bwu",              "bwu",         "Buli"),
    ("Chumburung_ncu",        "ncu",         "Chumburung"),
    ("Dagaare_dga",           "dga",         "Dagaare"),
    ("Dagbani_dag",           "dag",         "Dagbani"),
    ("Dangme_ada",            "ada",         "Dangme"),
    ("Deg_mzw",               "mzw",         "Deg"),
    ("Ewe_ewe",               "ewe",         "Ewe"),
    ("Fante_fat",             "fat",         "Fante"),
    ("Fulfulde_Maasina_ffm",  "ffm",         "Maasina Fulfulde"),
    ("Gikyode_acd",           "acd",         "Gikyode"),
    ("Gonja_gjn",             "gjn",         "Gonja"),
    ("Hausa_hau",             "hau",         "Hausa"),
    ("Kabiye_kbp",            "kbp",         "Kabiye"),
    ("Kasem_xsm",             "xsm",         "Kasem"),
    ("Konkomba_xon",          "xon",         "Konkomba"),
    ("Konni_kma",             "kma",         "Konni"),
    ("Kusaal_kus",            "kus",         "Kusaal"),
    ("Lelemi_lef",            "lef",         "Lelemi"),
    ("Mampruli_maw",          "maw",         "Mampruli"),
    ("Nawuri_naw",            "naw",         "Nawuri"),
    ("Ninkare_gur",           "gur",         "Ninkare (Frafra)"),
    ("Nkonya_nko",            "nko",         "Nkonya"),
    ("Ntrubo_ntr",            "ntr",         "Ntrubo"),
    ("Nzema_nzi",             "nzi",         "Nzema"),
    ("Paasaal_sig",           "sig",         "Paasaal"),
    ("Sehwi_sfw",             "sfw",         "Sehwi"),
    ("Sekpele_lip",           "lip",         "Sekpele"),
    ("Selee_snw",             "snw",         "Selee"),
    ("Sisaala_Tumulung_sil",  "sil",         "Tumulung Sisaala"),
    ("Siwu_akp",              "akp",         "Siwu"),
    ("Tampulma_tpm",          "tpm",         "Tampulma"),
    ("Tem_kdh",               "kdh",         "Tem"),
    ("Tuwuli_bov",            "bov",         "Tuwuli"),
    ("Vagla_vag",             "vag",         "Vagla"),
]


@dataclass(frozen=True)
class Language:
    code: str
    name: str
    gs_config: str | None  # ghana-speech config for reference audio, if any


LANGUAGES: dict[str, Language] = {
    code: Language(code=code, name=name, gs_config=config)
    for config, code, name in _GHANA_SPEECH_LANGS
}
# English is a valid model tag but lives in a separate dataset (no ghana-speech
# config), so it has no default reference-audio pool of its own.
LANGUAGES["en"] = Language(code="en", name="English", gs_config=None)


# --------------------------------------------------------------------------- #
# Text-source registry:  lang code -> [TextSource, ...]
# --------------------------------------------------------------------------- #
# Built automatically from ghana-speech, then extended with _EXTRA_SOURCES.
SOURCES: dict[str, list[TextSource]] = {
    code: [TextSource(dataset=GHANA_SPEECH, config=config, text_column="text")]
    for config, code, _ in _GHANA_SPEECH_LANGS
}

# ---- Add extra per-language text sources here (one line each) ---------------
# Each entry appends to the language's default ghana-speech source.
_EXTRA_SOURCES: dict[str, list[TextSource]] = {
    # Twi also draws from a 500-hour health-domain ASR corpus.
    "twi-asante": [
        TextSource("ghananlpcommunity/twi-health-asr-gemini-500hrs",
                   text_column="transcription"),
    ],
    "twi-akuapem": [
        TextSource("ghananlpcommunity/twi-health-asr-gemini-500hrs",
                   text_column="transcription"),
    ],
}
# ----------------------------------------------------------------------------

# English default text: reuse the community English ASR corpus.
SOURCES["en"] = [
    TextSource("ghananlpcommunity/ghana-english-asr-2700hrs",
               text_column="corrected_text"),
]

for _code, _srcs in _EXTRA_SOURCES.items():
    SOURCES.setdefault(_code, []).extend(_srcs)


# --------------------------------------------------------------------------- #
# Aliases  (user-typed -> canonical code)
# --------------------------------------------------------------------------- #
_ALIASES: dict[str, str] = {}
for _config, _code, _name in _GHANA_SPEECH_LANGS:
    _ALIASES[_code.lower()] = _code
    _ALIASES[_config.lower()] = _code             # full config name
    _ALIASES[_name.lower()] = _code               # display name
_ALIASES["en"] = "en"
_ALIASES["english"] = "en"
_ALIASES["twi"] = "twi-asante"                     # most common default
_ALIASES["akan"] = "twi-asante"


# --------------------------------------------------------------------------- #
# Public helpers
# --------------------------------------------------------------------------- #
def lang_tag(code: str) -> str:
    """Return the literal tag the model expects, e.g. ``'<|lang:ewe|> '``."""
    return f"<|lang:{code}|> "


def resolve_lang(name: str) -> str:
    """Normalise a user-supplied language to a canonical code.

    Accepts the code, the ghana-speech config name, or the display name
    (case-insensitive). Raises ``ValueError`` for unknown languages.
    """
    key = (name or "").strip().lower()
    if key in _ALIASES:
        return _ALIASES[key]
    raise ValueError(
        f"Unknown language '{name}'. Run with --list-langs to see supported codes."
    )


def sources_for(code: str) -> list[TextSource]:
    """Return the ordered list of text sources for a language code."""
    if code not in SOURCES:
        raise ValueError(f"No text sources registered for language '{code}'.")
    return SOURCES[code]


def load_texts(code: str, max_samples: int | None, token: str | None) -> list[str]:
    """Read raw synthesis texts for a language from all its registered sources.

    Streams each source and pulls a roughly even share up to ``max_samples``
    (or :data:`DEFAULT_TEXT_CAP` when unset).  No length filtering or tag
    prepending happens here — the caller handles that.

    We ``take()`` without a streaming ``shuffle()`` on purpose: shuffling buffers
    ~1000 full rows (audio included) before yielding, which is wasteful when only
    the transcript is needed. The caller down-samples the combined pool randomly.
    """
    from datasets import load_dataset

    srcs = sources_for(code)
    cap = max_samples or DEFAULT_TEXT_CAP
    per_source = max(1, -(-cap // len(srcs)))  # ceil division

    texts: list[str] = []
    for src in srcs:
        try:
            ds = load_dataset(src.dataset, src.config or None, split=src.split,
                              streaming=True, token=token)
            for ex in ds.take(per_source):
                t = (ex.get(src.text_column) or "").strip()
                if t:
                    texts.append(t)
        except Exception as e:
            # A missing/renamed source shouldn't kill the whole run.
            import sys
            print(f"⚠️  skipping text source {src.dataset} "
                  f"({src.config or 'default'}): {e}", file=sys.stderr)
            continue

    return texts


def format_language_table() -> str:
    """Human-readable list of supported languages for ``--list-langs``."""
    lines = [f"{'CODE':<12} LANGUAGE                 TEXT SOURCES"]
    for code in sorted(LANGUAGES):
        lang = LANGUAGES[code]
        n = len(SOURCES.get(code, []))
        extra = "" if n <= 1 else f" (+{n - 1} extra)"
        lines.append(f"{code:<12} {lang.name:<24} {n} source{'s' if n != 1 else ''}{extra}")
    return "\n".join(lines)
