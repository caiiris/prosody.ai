"""
Feature extraction for the poetry era classifier webapp.
Computes all 7 stylistic features from raw poem text.
"""

import re
import numpy as np

# ── Constants ──────────────────────────────────────────────────────────────────

PUNCT_CHARS = set(".!?;:,")

ARCHAIC_WORDS = {
    'thee', 'thou', 'thy', 'thine', 'thyself', 'ye',
    'hath', 'doth', 'dost', 'hast', 'wilt', 'shalt', 'canst',
    'wouldst', 'shouldst', 'couldst', 'wert', 'didst',
    "'tis", "'twas", "'twere", "'gainst", "o'er", "e'er", "ne'er",
    'hence', 'thence', 'whence', 'hither', 'thither', 'whither',
    'wherefore', 'forsooth', 'perchance', 'mayhap', 'methinks',
    'betwixt', 'nigh', 'ere', 'lest',
    'aught', 'naught', 'morrow', 'prithee',
    'hark', 'lo', 'fie', 'alas',
    'unto', 'thereof', 'whereof', 'wherein', 'herein', 'therein',
    'amongst', 'whilst', 'oft',
}
MODERN_EST = {
    'best', 'rest', 'test', 'west', 'nest', 'chest', 'quest',
    'guest', 'fest', 'jest', 'zest', 'pest', 'vest', 'crest',
    'forest', 'interest', 'protest', 'request', 'suggest',
    'harvest', 'modest', 'honest', 'latest', 'greatest',
    'largest', 'smallest', 'highest', 'lowest', 'nearest',
    'oldest', 'newest', 'fastest', 'slowest', 'biggest',
    'earnest', 'tempest', 'purest', 'truest', 'surest',
    'dearest', 'nearest', 'fairest', 'darkest', 'deepest', 'sweetest',
}
MODERN_ETH = {
    'seth', 'beth', 'meth', 'death', 'beneath', 'breath',
    'teeth', 'shibboleth', 'macbeth', 'elizabeth', 'nazareth',
}

_STRIP_RE   = re.compile(r"[^\w\s]")
_ARCHAIC_RE = re.compile(r"[^a-z'\s]")


# ── Text helpers ───────────────────────────────────────────────────────────────

def get_lines(text: str) -> list[str]:
    return [l.strip() for l in text.split("\n") if l.strip()]


def tokenize(text: str) -> list[str]:
    t = _STRIP_RE.sub(" ", text.lower())
    return [w for w in t.split() if w.isalpha() and len(w) > 1]


# ── Feature 1: cap_rate ────────────────────────────────────────────────────────

def cap_rate(lines: list[str]) -> float:
    if not lines:
        return float("nan")
    return sum(1 for l in lines if l and l[0].isupper()) / len(lines)


# ── Feature 2: punct_rate ──────────────────────────────────────────────────────

def punct_rate(lines: list[str]) -> float:
    if not lines:
        return float("nan")
    return sum(1 for l in lines if l and l[-1] in PUNCT_CHARS) / len(lines)


# ── Feature 3: colon_density ───────────────────────────────────────────────────

def colon_density(text: str) -> float:
    words = text.split()
    if not words:
        return 0.0
    return sum(1 for ch in text if ch in (";", ":")) / len(words) * 100


# ── Feature 4: concrete_abstract_ratio ────────────────────────────────────────

def concrete_abstract_ratio(text: str, lex: dict[str, float]) -> float:
    words = tokenize(text)
    if not words:
        return float("nan")
    scored = [lex[w] for w in words if w in lex]
    if not scored:
        return float("nan")
    concrete = sum(1 for s in scored if s < 3.0)
    abstract = sum(1 for s in scored if s >= 3.0)
    return concrete / (abstract + 1)


# ── Feature 5: adv_verb_ratio ──────────────────────────────────────────────────

def adv_verb_ratio(text: str, pos_tag_fn) -> float:
    words = tokenize(text)
    if not words:
        return float("nan")
    tagged = pos_tag_fn(words)
    adverbs = sum(1 for _, tag in tagged if tag.startswith("RB"))
    verbs   = sum(1 for _, tag in tagged if tag.startswith("VB"))
    return adverbs / (verbs + 1)


# ── Feature 6: rhyme_rate ──────────────────────────────────────────────────────

def _last_phones(word: str, cmu: dict) -> list | None:
    entries = cmu.get(word.lower())
    if not entries:
        return None
    phones = entries[0]
    vowels = [i for i, p in enumerate(phones) if p[-1].isdigit()]
    if not vowels:
        return None
    return phones[vowels[-1]:]


def rhyme_rate(lines: list[str], cmu: dict) -> float:
    endings = []
    for l in lines:
        words = re.sub(r"[^\w\s]", "", l).split()
        if words:
            ph = _last_phones(words[-1], cmu)
            endings.append(ph)
        else:
            endings.append(None)
    pairs = [(endings[i], endings[i + 1])
             for i in range(len(endings) - 1)
             if endings[i] and endings[i + 1]]
    if not pairs:
        return 0.0
    return sum(1 for a, b in pairs if a == b) / len(pairs)


# ── Feature 7: archaic_density ────────────────────────────────────────────────

def archaic_density(text: str) -> float:
    t = _ARCHAIC_RE.sub(" ", text.lower())
    words = t.split()
    if not words:
        return 0.0
    count = sum(1 for w in words if w in ARCHAIC_WORDS)
    for w in words:
        if w.endswith("eth") and len(w) > 4 and w not in MODERN_ETH and w not in ARCHAIC_WORDS:
            count += 1
        if w.endswith("est") and len(w) > 4 and w not in MODERN_EST and w not in ARCHAIC_WORDS:
            count += 1
    return count / len(words) * 100


# ── Master compute function ────────────────────────────────────────────────────

def compute_features(text: str, lex: dict, cmu: dict, pos_tag_fn) -> dict:
    """
    Compute all 7 features from raw poem text.
    Returns a dict with float values (may contain NaN for some features
    if the poem is too short).
    """
    lines = get_lines(text)
    return {
        "cap_rate":               cap_rate(lines),
        "punct_rate":             punct_rate(lines),
        "colon_density":          colon_density(text),
        "concrete_abstract_ratio": concrete_abstract_ratio(text, lex),
        "adv_verb_ratio":         adv_verb_ratio(text, pos_tag_fn),
        "rhyme_rate":             rhyme_rate(lines, cmu),
        "archaic_density":        archaic_density(text),
    }
