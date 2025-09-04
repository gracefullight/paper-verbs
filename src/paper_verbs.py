from __future__ import annotations

import importlib
import re
import sys
from collections import Counter
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import pandas as pd
from pandas import DataFrame
from spacy.language import Language
from spacy.tokens import Doc, Token

# Project paths
MODULE_DIR = Path(__file__).resolve().parent
ASSET_DIR = MODULE_DIR / "assets"

# Optional heavy deps: import lazily or with fallbacks so import of this
# module doesn't fail in environments without them (e.g., during tests).
try:  # PyMuPDF
    import fitz as _fitz

    fitz: Any | None = _fitz
except Exception:  # pragma: no cover - only for environments without PyMuPDF
    fitz = None

# tqdm: import at module top-level with fallback
try:  # pragma: no cover - optional dep
    from tqdm import tqdm
except Exception:  # pragma: no cover - only for environments without tqdm

    def tqdm(iterable: Iterable[Any], **kwargs: Any) -> Iterable[Any]:
        return iterable


"""spaCy: English (lazy loader without auto-download)."""
spacy: Any | None
try:
    import spacy as _spacy

    spacy = _spacy
except Exception:  # pragma: no cover - optional dep
    spacy = None

_NLP_EN: Language | None = None


def get_nlp_en() -> Language | None:
    """Return a cached spaCy English pipeline.
    Tries package import (en_core_web_sm.load()) first, then spacy.load('en_core_web_sm').
    Returns None if unavailable. Emits failure reasons to stderr for debugging.
    """
    global _NLP_EN
    if _NLP_EN is not None:
        return _NLP_EN
    if spacy is None:
        return None
    # 1) Prefer direct package import to avoid env/path issues
    try:
        pkg = importlib.import_module("en_core_web_sm")
        _NLP_EN = pkg.load()
        return _NLP_EN
    except Exception as e1:
        sys.stderr.write(f"[WARN] Failed en_core_web_sm.load(): {e1}\n")
    # 2) Fallback to spacy.load
    try:
        _NLP_EN = spacy.load("en_core_web_sm")
        return _NLP_EN
    except Exception as e2:
        sys.stderr.write(f"[WARN] Failed spacy.load('en_core_web_sm'): {e2}\n")
        return None
    return None


SENT_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")
EN_POS_KEEP: set[str] = {"VERB", "AUX"}  # include AUX by default
LEMMA_EXCLUDE: set[str] = {"preprint"}


# ---------------------------
# PDF → Text (+ cut references)
# ---------------------------
def extract_text_from_pdf(pdf_path: Path) -> str:
    """Extract text from PDF using PyMuPDF and remove trailing References-like sections."""
    try:
        if fitz is None:
            raise ImportError("PyMuPDF (fitz) is not installed")
        doc = fitz.open(pdf_path)
        texts: list[str] = []
        for page in doc:
            texts.append(page.get_text("text"))
        full_text = "\n".join(texts)
        return cut_references_tail(full_text)
    except Exception as e:
        sys.stderr.write(f"[WARN] Failed to read {pdf_path}: {e}\n")
        return ""


def cut_references_tail(text: str) -> str:
    """Heuristically remove trailing 'References/Bibliography/Acknowledgements' section.
    Only cut if the trigger appears in the latter half of document to avoid false positives.
    """
    lowered = text.lower()
    triggers = [
        "\nreferences",  # typical
        "\nbibliography",  # alternative
        "\nacknowledg",  # acknowledgements
        "\nreference",  # singular (rare but seen in some camera-ready)
        "\nworks cited",  # humanities style (just in case)
    ]
    cutoff_idx = -1
    half = len(text) * 0.5
    # Prefer the last occurring trigger near the end
    for trig in triggers:
        idx = lowered.rfind(trig)
        if idx != -1 and idx > half:
            cutoff_idx = max(cutoff_idx, idx)
    if cutoff_idx != -1:
        return text[:cutoff_idx]
    return text


def sent_tokenize_simple(text: str) -> list[str]:
    if not text:
        return []
    return re.split(SENT_SPLIT_RE, text.strip())


# ---------------------------
# NLP helpers
# ---------------------------
def is_counted_verb(token: Token, include_aux: bool) -> bool:
    """Return True if token should be counted as a verb for our stats.
    Uses strict morphological tags to reduce false positives.
    - VERB: only VB* (VB, VBD, VBG, VBN, VBP, VBZ)
    - AUX: included only when include_aux=True (VB* or MD for modals)
    """
    pos = token.pos_
    tag = token.tag_
    if pos == "VERB":
        return tag.startswith("VB")
    if pos == "AUX":
        if not include_aux:
            return False
        return tag.startswith("VB") or tag == "MD"
    return False


def verbs_from_english(text: str, include_aux: bool = True) -> Iterable[str]:
    """Extract verb lemmas (and AUX if include_aux=True) from English text using spaCy."""
    nlp = get_nlp_en()
    if nlp is None:
        sys.stderr.write(
            "[ERROR] spaCy English model not available. Install with one of:\n"
            "  - uv sync   (pyproject includes en-core-web-sm)\n"
            "  - uv pip install https://github.com/explosion/spacy-models/releases/download/"
            "en_core_web_sm-3.7.1/en_core_web_sm-3.7.1-py3-none-any.whl\n"
        )
        return
    doc = nlp(text)
    for token in doc:
        if is_counted_verb(token, include_aux=include_aux):
            lemma = token.lemma_.lower()
            if not lemma.isalpha():
                continue
            if len(lemma) <= 2:
                continue
            if lemma in LEMMA_EXCLUDE:
                continue
            yield lemma


def build_verb_phrase(token: Token) -> list[str]:
    """Build a simple verb phrase around a verb/AUX token using dependency relations.
    We gather:
      - auxiliaries (aux, auxpass), negation (neg), adverbs (advmod), particles (prt),
      - prepositions attached to the verb (prep), and a light direct object head.
    Returned as a list of token texts ordered by document position.
    """
    parts: set[Token] = {token}

    # Collect immediate children that form the core phrase
    for child in token.children:
        if child.dep_ in {"aux", "auxpass", "neg", "advmod", "prt"}:
            parts.add(child)
        if child.dep_ in {"prep"}:
            parts.add(child)
        # Optionally include a light direct object headword (not full subtree)
        if child.dep_ in {"dobj", "obj", "attr", "acomp"}:
            parts.add(child)

    # Also include left-edge auxiliaries like "will", "have", "been" that might be on the left chain
    # (walk up to include governing AUX if current token is main VERB)
    head = token.head
    if (
        head is not None
        and head != token
        and head.pos_ == "AUX"
        and head.dep_ in {"aux", "auxpass"}
    ):
        parts.add(head)

    # Order by token.i to preserve original order
    ordered = sorted(parts, key=lambda t: t.i)
    return [t.text for t in ordered]


def infer_tense(token: Token) -> str:
    """Infer coarse tense/aspect from token.tag_ and auxiliaries around the verb.
    Labels: present, past, progressive, perfect, future, other
    """
    tag = token.tag_
    # Progressive
    if tag == "VBG":
        return "progressive"
    # Past-ish
    if tag in {"VBD", "VBN"}:
        # Distinguish perfect participle when 'have' AUX exists
        if any(c.dep_ == "aux" and c.lemma_.lower() == "have" for c in token.children) or (
            token.head.pos_ == "AUX" and token.head.lemma_.lower() == "have"
        ):
            return "perfect"
        return "past"
    # Future (aux 'will' or 'shall' nearby)
    if any(c.dep_ == "aux" and c.lemma_.lower() in {"will", "shall"} for c in token.children) or (
        token.head.pos_ == "AUX" and token.head.lemma_.lower() in {"will", "shall"}
    ):
        return "future"
    # Present / base / 3sg present
    if tag in {"VBZ", "VBP", "VB"}:
        return "present"
    return "other"


def infer_voice(token: Token) -> str:
    """Infer voice: passive if auxpass or nsubjpass present, or VBN with 'be' auxiliary.
    Otherwise active.
    """
    # Direct indicators
    if token.dep_ == "auxpass":
        return "passive"
    if any(c.dep_ == "auxpass" for c in token.children):
        return "passive"
    if any(c.dep_ == "nsubjpass" for c in token.children):
        return "passive"
    # be + VBN pattern (is/was/were/been/being + VBN)
    if token.tag_ == "VBN":
        if any(
            (c.dep_ in {"aux", "auxpass"}) and c.lemma_.lower() == "be" for c in token.children
        ) or (token.head.pos_ == "AUX" and token.head.lemma_.lower() == "be"):
            return "passive"
    return "active"


def analyze_doc(
    doc: Doc, include_aux: bool
) -> tuple[list[str], list[str], Counter[str], Counter[str]]:
    """From a spaCy doc, extract:
    - verb lemmas,
    - verb phrases,
    - tense distribution,
    - voice distribution
    """
    lemmas: list[str] = []
    phrases: list[str] = []
    tense_counts: Counter[str] = Counter()
    voice_counts: Counter[str] = Counter()

    for token in doc:
        if is_counted_verb(token, include_aux=include_aux):
            # lemma
            lemma = token.lemma_.lower()
            if lemma.isalpha() and len(lemma) > 2 and lemma not in LEMMA_EXCLUDE:
                lemmas.append(lemma)
            # phrase
            phrase_tokens = build_verb_phrase(token)
            # Compact consecutive spaces
            phrase_text = re.sub(r"\s+", " ", " ".join(phrase_tokens)).strip()
            if phrase_text:
                phrases.append(phrase_text)
            # tense / voice
            tense_counts[infer_tense(token)] += 1
            voice_counts[infer_voice(token)] += 1

    return lemmas, phrases, tense_counts, voice_counts


# ---------------------------
# Main pipeline
# ---------------------------
def process_pdfs(
    input_dir: Path,
    include_aux_en: bool = True,
) -> tuple[
    Counter[str],
    Counter[str],
    Counter[str],
    Counter[str],
]:
    """Scan PDFs and aggregate:
      - verb lemma counts,
      - verb phrase counts,
      - tense distribution,
      - voice distribution
    Returns: verb_counts, phrase_counts, tense_counts, voice_counts
    """
    verb_counts: Counter[str] = Counter()
    phrase_counts: Counter[str] = Counter()
    tense_counts_total: Counter[str] = Counter()
    voice_counts_total: Counter[str] = Counter()

    # Find PDFs case-insensitively (handles .pdf, .PDF, mixed-case)
    pdfs: list[Path] = sorted(
        [p for p in input_dir.rglob("**/*") if p.is_file() and p.suffix.lower() == ".pdf"]
    )
    if not pdfs:
        print(f"[INFO] No PDFs found under {input_dir.resolve()}")
        return (verb_counts, phrase_counts, tense_counts_total, voice_counts_total)

    nlp = get_nlp_en()
    if nlp is None:
        sys.stderr.write(
            "[ERROR] spaCy English model not available. Install with one of:\n"
            "  - uv sync   (pyproject includes en-core-web-sm)\n"
            "  - uv pip install https://github.com/explosion/spacy-models/releases/download/"
            "en_core_web_sm-3.7.1/en_core_web_sm-3.7.1-py3-none-any.whl\n"
        )
        return (verb_counts, phrase_counts, tense_counts_total, voice_counts_total)

    for pdf in tqdm(pdfs, desc="PDFs"):
        text = extract_text_from_pdf(pdf)
        if not text.strip():
            continue

        # Parse once per file
        try:
            doc = nlp(text)
        except Exception as e:
            sys.stderr.write(f"[WARN] NLP processing failed for {pdf}: {e}\n")
            continue

        lemmas, phrases, tense_c, voice_c = analyze_doc(doc, include_aux=include_aux_en)

        # Aggregate counts
        verb_counts.update(lemmas)
        phrase_counts.update(phrases)
        tense_counts_total.update(tense_c)
        voice_counts_total.update(voice_c)

    return (verb_counts, phrase_counts, tense_counts_total, voice_counts_total)


# ---------------------------
# Save utilities
# ---------------------------
def save_verb_csv(verb_counts: Counter[str], out_csv: Path) -> DataFrame:
    rows: list[dict[str, object]] = []
    for rank, (lemma, cnt) in enumerate(verb_counts.most_common(), start=1):
        rows.append({"rank": rank, "verb": lemma, "count": cnt})
    df = pd.DataFrame(rows, columns=["rank", "verb", "count"])
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False, encoding="utf-8")
    return df


def save_phrase_csv(phrase_counts: Counter[str], out_csv: Path) -> DataFrame:
    rows: list[dict[str, object]] = []
    for rank, (phrase, cnt) in enumerate(phrase_counts.most_common(), start=1):
        rows.append({"rank": rank, "verb_phrase": phrase, "count": cnt})
    df = pd.DataFrame(rows, columns=["rank", "verb_phrase", "count"])
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False, encoding="utf-8")
    return df


def pct(n: int, total: int) -> float:
    return (100.0 * n / total) if total > 0 else 0.0


def print_distributions(tense_counts: Counter[str], voice_counts: Counter[str]) -> None:
    t_total = sum(tense_counts.values())
    v_total = sum(voice_counts.values())

    print("\n=== Tense distribution ===")
    for k in ["present", "past", "perfect", "progressive", "future", "other"]:
        print(
            f"{k:11s}: {tense_counts.get(k, 0):6d}  ({pct(tense_counts.get(k, 0), t_total):5.1f}%)"
        )

    print("\n=== Voice distribution ===")
    for k in ["active", "passive"]:
        print(
            f"{k:11s}: {voice_counts.get(k, 0):6d}  ({pct(voice_counts.get(k, 0), v_total):5.1f}%)"
        )


def main() -> None:
    """Run the analyzer with built-in defaults (no CLI options).

    Defaults:
    - input_dir: src/assets
    - verbs_csv: src/verbs.csv
    - phrases_csv: src/phrases.csv
    - include_aux_en: False (exclude auxiliary/modal verbs)
    """
    input_dir = ASSET_DIR
    verbs_csv = MODULE_DIR / "verbs.csv"
    phrases_csv = MODULE_DIR / "phrases.csv"

    (
        verb_counts,
        phrase_counts,
        tense_counts,
        voice_counts,
    ) = process_pdfs(
        input_dir=input_dir,
        include_aux_en=False,
    )

    if not verb_counts and not phrase_counts:
        print("[INFO] No verbs/phrases extracted. Check PDFs or spaCy install.")
        return

    df_verbs = save_verb_csv(verb_counts, verbs_csv)
    df_phrases = save_phrase_csv(phrase_counts, phrases_csv)

    print_distributions(tense_counts, voice_counts)

    print("\n=== Top 100 verb lemmas ===")
    for i, (w, c) in enumerate(verb_counts.most_common(100), 1):
        print(f"{i:>3}. {w:<20} {c}")

    print(
        f"\n[OK] Saved:\n- {verbs_csv.resolve()} ({len(df_verbs)} rows)\n- {phrases_csv.resolve()} ({len(df_phrases)} rows)"
    )


if __name__ == "__main__":
    main()
