#!/usr/bin/env python3
# ruff: noqa: T201, BLE001

from __future__ import annotations

import argparse
import re
import sys
from collections import Counter, defaultdict
from collections.abc import Iterable
from pathlib import Path

# Optional heavy deps: import lazily or with fallbacks so import of this
# module doesn't fail in environments without them (e.g., during tests).
try:  # PyMuPDF
    import fitz  # type: ignore
except Exception:  # pragma: no cover - only for environments without PyMuPDF
    fitz = None  # type: ignore[assignment]

try:  # pandas
    import pandas as pd  # type: ignore
except Exception:  # pragma: no cover - only for environments without pandas
    pd = None  # type: ignore[assignment]

try:  # tqdm
    from tqdm import tqdm  # type: ignore
except Exception:  # pragma: no cover - only for environments without tqdm
    def tqdm(iterable, **kwargs):  # type: ignore[no-redef]
        return iterable

# ===== spaCy: English =====
try:
    import spacy

    _NLP_EN = spacy.load("en_core_web_sm")
except Exception:
    _NLP_EN = None

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from spacy.tokens import Doc, Token


SENT_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")
EN_POS_KEEP: set[str] = {"VERB", "AUX"}  # include AUX by default


# ---------------------------
# PDF → Text (+ cut references)
# ---------------------------
def extract_text_from_pdf(pdf_path: Path) -> str:
    """Extract text from PDF using PyMuPDF and remove trailing References-like sections."""
    try:
        if fitz is None:  # type: ignore[truthy-function]
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
def verbs_from_english(text: str, include_aux: bool = True) -> Iterable[str]:
    """Extract verb lemmas (and AUX if include_aux=True) from English text using spaCy.
    No lemma exclusion list is applied.
    """
    if _NLP_EN is None:
        raise RuntimeError("spaCy(en_core_web_sm)가 설치/다운로드되지 않았습니다.")
    doc = _NLP_EN(text)
    for token in doc:
        if token.pos_ in EN_POS_KEEP:
            if (not include_aux) and token.pos_ == "AUX":
                continue
            lemma = token.lemma_.lower()
            if not lemma.isalpha():
                continue
            if len(lemma) <= 2:
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
        if token.pos_ in EN_POS_KEEP:
            if (not include_aux) and token.pos_ == "AUX":
                continue
            # lemma
            lemma = token.lemma_.lower()
            if lemma.isalpha() and len(lemma) > 2:
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
    examples_top_k: int = 50,
    examples_per_item: int = 2,
) -> tuple[
    Counter[str],
    dict[str, list[str]],
    Counter[str],
    dict[str, list[str]],
    Counter[str],
    Counter[str],
]:
    """Scan PDFs, aggregate:
      - verb lemma counts (+ examples),
      - verb phrase counts (+ examples),
      - tense distribution,
      - voice distribution
    Returns:
      verb_counts, verb_examples, phrase_counts, phrase_examples, tense_counts, voice_counts
    """
    verb_counts: Counter[str] = Counter()
    phrase_counts: Counter[str] = Counter()
    tense_counts_total: Counter[str] = Counter()
    voice_counts_total: Counter[str] = Counter()
    verb_examples: dict[str, list[str]] = defaultdict(list)
    phrase_examples: dict[str, list[str]] = defaultdict(list)

    pdfs: list[Path] = sorted(list(input_dir.rglob("*.pdf")))
    if not pdfs:
        print(f"[INFO] No PDFs found under {input_dir.resolve()}")
        return (
            verb_counts,
            verb_examples,
            phrase_counts,
            phrase_examples,
            tense_counts_total,
            voice_counts_total,
        )

    if _NLP_EN is None:
        raise RuntimeError("spaCy(en_core_web_sm)가 설치/다운로드되지 않았습니다.")

    for pdf in tqdm(pdfs, desc="PDFs"):
        text = extract_text_from_pdf(pdf)
        if not text.strip():
            continue

        # Parse once per file
        try:
            doc = _NLP_EN(text)
        except Exception as e:
            sys.stderr.write(f"[WARN] NLP processing failed for {pdf}: {e}\n")
            continue

        lemmas, phrases, tense_c, voice_c = analyze_doc(doc, include_aux=include_aux_en)

        # Aggregate counts
        verb_counts.update(lemmas)
        phrase_counts.update(phrases)
        tense_counts_total.update(tense_c)
        voice_counts_total.update(voice_c)

        # Collect examples (sentence-level)
        if examples_top_k > 0:
            top_lemmas = {w for w, _ in verb_counts.most_common(examples_top_k * 3)}
            top_phrases = {p for p, _ in phrase_counts.most_common(examples_top_k * 3)}
            per_item_limit = examples_per_item

            sents = sent_tokenize_simple(text)
            for s in sents:
                s_norm = s.strip()
                if not s_norm or len(s_norm) > 500:
                    continue
                # Find lemmas in sentence
                try:
                    sdoc = _NLP_EN(s_norm)
                except Exception:
                    continue

                # Lemma examples
                s_lemmas = set()
                for t in sdoc:
                    if t.pos_ in EN_POS_KEEP:
                        if (not include_aux_en) and t.pos_ == "AUX":
                            continue
                        l = t.lemma_.lower()
                        if l in top_lemmas and len(verb_examples[l]) < per_item_limit:
                            s_lemmas.add(l)
                for l in s_lemmas:
                    if len(verb_examples[l]) < per_item_limit:
                        verb_examples[l].append(s_norm)

                # Phrase examples (exact text match after build)
                s_phrases_found = set()
                for t in sdoc:
                    if t.pos_ in EN_POS_KEEP:
                        if (not include_aux_en) and t.pos_ == "AUX":
                            continue
                        p_tokens = build_verb_phrase(t)
                        p_text = re.sub(r"\s+", " ", " ".join(p_tokens)).strip()
                        if p_text in top_phrases and len(phrase_examples[p_text]) < per_item_limit:
                            s_phrases_found.add(p_text)
                for p_text in s_phrases_found:
                    if len(phrase_examples[p_text]) < per_item_limit:
                        phrase_examples[p_text].append(s_norm)

    # Keep examples only for final top_k
    if examples_top_k > 0:
        top_lemmas_final = {w for w, _ in verb_counts.most_common(examples_top_k)}
        verb_examples = {k: verb_examples[k] for k in top_lemmas_final if k in verb_examples}

        top_phrases_final = {p for p, _ in phrase_counts.most_common(examples_top_k)}
        phrase_examples = {k: phrase_examples[k] for k in top_phrases_final if k in phrase_examples}

    return (
        verb_counts,
        verb_examples,
        phrase_counts,
        phrase_examples,
        tense_counts_total,
        voice_counts_total,
    )


# ---------------------------
# Save utilities
# ---------------------------
def save_verb_csv(
    verb_counts: Counter[str], verb_examples: dict[str, list[str]], out_csv: Path
) -> pd.DataFrame:
    if pd is None:
        raise RuntimeError("pandas가 설치되어 있지 않습니다.")
    rows: list[dict[str, object]] = []
    for rank, (lemma, cnt) in enumerate(verb_counts.most_common(), start=1):
        rows.append(
            {
                "rank": rank,
                "verb": lemma,
                "count": cnt,
                "examples": " || ".join(verb_examples.get(lemma, [])),
            }
        )
    df = pd.DataFrame(rows, columns=["rank", "verb", "count", "examples"])
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False, encoding="utf-8")
    return df


def save_phrase_csv(
    phrase_counts: Counter[str], phrase_examples: dict[str, list[str]], out_csv: Path
) -> pd.DataFrame:
    if pd is None:
        raise RuntimeError("pandas가 설치되어 있지 않습니다.")
    rows: list[dict[str, object]] = []
    for rank, (phrase, cnt) in enumerate(phrase_counts.most_common(), start=1):
        rows.append(
            {
                "rank": rank,
                "verb_phrase": phrase,
                "count": cnt,
                "examples": " || ".join(phrase_examples.get(phrase, [])),
            }
        )
    df = pd.DataFrame(rows, columns=["rank", "verb_phrase", "count", "examples"])
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


# ---------------------------
# CLI
# ---------------------------
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyze English verbs/verb phrases, tense & voice from PDFs (References auto-removed)."
    )
    parser.add_argument("--input_dir", type=str, default="assets", help="PDF folder (recursive)")
    parser.add_argument(
        "--verbs_csv", type=str, default="verbs.csv", help="Output CSV for verb lemmas"
    )
    parser.add_argument(
        "--phrases_csv", type=str, default="phrases.csv", help="Output CSV for verb phrases"
    )
    parser.add_argument(
        "--exclude_aux", action="store_true", help="Exclude AUX from counting/phrases"
    )
    parser.add_argument(
        "--examples_top_k", type=int, default=50, help="Keep examples for top-K items"
    )
    parser.add_argument(
        "--examples_per_item", type=int, default=2, help="Examples per item to keep"
    )
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    verbs_csv = Path(args.verbs_csv)
    phrases_csv = Path(args.phrases_csv)

    (
        verb_counts,
        verb_examples,
        phrase_counts,
        phrase_examples,
        tense_counts,
        voice_counts,
    ) = process_pdfs(
        input_dir=input_dir,
        include_aux_en=not args.exclude_aux,
        examples_top_k=args.examples_top_k,
        examples_per_item=args.examples_per_item,
    )

    if not verb_counts and not phrase_counts:
        print("[INFO] No verbs/phrases extracted. Check PDFs or spaCy install.")
        return

    df_verbs = save_verb_csv(verb_counts, verb_examples, verbs_csv)
    df_phrases = save_phrase_csv(phrase_counts, phrase_examples, phrases_csv)

    print_distributions(tense_counts, voice_counts)

    print("\n=== Top 30 verb lemmas ===")
    for i, (w, c) in enumerate(verb_counts.most_common(30), 1):
        print(f"{i:>2}. {w:<20} {c}")

    print("\n=== Top 30 verb phrases ===")
    for i, (p, c) in enumerate(phrase_counts.most_common(30), 1):
        print(f"{i:>2}. {p:<40} {c}")

    print(
        f"\n[OK] Saved:\n- {verbs_csv.resolve()} ({len(df_verbs)} rows)\n- {phrases_csv.resolve()} ({len(df_phrases)} rows)"
    )


if __name__ == "__main__":
    main()
