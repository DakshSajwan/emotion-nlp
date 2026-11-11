"""Text preprocessing module for Emotion Detection in Text.

Provides two distinct cleaning pipelines:
- minimal_clean: Preserves casing, punctuation, and grammar structure for transformer models (DistilBERT).
- heavy_clean: Lowercases, expands contractions, removes punctuation, with optional lemmatization
  and stopword filtering (preserving emotion-critical negations) for classical models (TF-IDF).
"""

from typing import Optional, Set
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# ---------------------------------------------------------------------------
# Contractions & Negations
# ---------------------------------------------------------------------------
CONTRACTIONS: dict[str, str] = {
    "can't": "can not",
    "won't": "will not",
    "n't": " not",
    "'re": " are",
    "'s": " is",
    "'d": " would",
    "'ll": " will",
    "'t": " not",
    "'ve": " have",
    "'m": " am",
    "i'm": "i am",
    "it's": "it is",
    "don't": "do not",
    "doesn't": "does not",
    "didn't": "did not",
    "haven't": "have not",
    "hasn't": "has not",
    "hadn't": "had not",
    "wouldn't": "would not",
    "shouldn't": "should not",
    "couldn't": "could not",
    "aren't": "are not",
    "isn't": "is not",
    "wasn't": "was not",
    "weren't": "were not",
}

# Critical negation words that must never be removed, even when stopword removal is enabled
NEGATION_WORDS: Set[str] = {"not", "no", "never", "n't", "nor"}

# ---------------------------------------------------------------------------
# Lazy NLTK Initialization
# ---------------------------------------------------------------------------
_STOPWORDS: Optional[Set[str]] = None
_LEMMATIZER: Optional[WordNetLemmatizer] = None


def _ensure_nltk_data() -> None:
    """Ensure required NLTK datasets are downloaded automatically at first use."""
    required_packages = [
        ("corpora/stopwords", "stopwords"),
        ("corpora/wordnet", "wordnet"),
        ("corpora/omw-1.4", "omw-1.4"),
    ]
    for path, package_id in required_packages:
        try:
            nltk.data.find(path)
        except LookupError:
            try:
                nltk.download(package_id, quiet=True)
            except Exception:
                pass


# Automatic download at first module import
try:
    _ensure_nltk_data()
except Exception:
    pass


def ensure_nltk_resources() -> None:
    """Public helper to verify and download required NLTK corpora."""
    _ensure_nltk_data()


def _get_stopwords() -> Set[str]:
    """Retrieve NLTK English stopwords minus critical negation words."""
    global _STOPWORDS
    if _STOPWORDS is None:
        _ensure_nltk_data()
        _STOPWORDS = set(stopwords.words("english")) - NEGATION_WORDS
    return _STOPWORDS


def _get_lemmatizer() -> WordNetLemmatizer:
    """Retrieve WordNetLemmatizer instance with guaranteed data availability."""
    global _LEMMATIZER
    if _LEMMATIZER is None:
        _ensure_nltk_data()
        _LEMMATIZER = WordNetLemmatizer()
    return _LEMMATIZER


# ---------------------------------------------------------------------------
# Preprocessing Pipelines
# ---------------------------------------------------------------------------
def minimal_clean(text: Optional[str]) -> str:
    """Minimal text cleaning suitable for Transformer models (e.g., DistilBERT).

    Strips whitespace, collapses multi-spaces, and strips URLs and @mentions while
    preserving casing, punctuation, and sentence syntax.

    Args:
        text: Raw text string or None.

    Returns:
        Cleaned text string. Returns empty string if input is None, empty, or non-string.
    """
    if text is None or not isinstance(text, str):
        return ""

    # Remove URLs (http, https, www)
    text = re.sub(r"https?://\S+|www\.\S+", "", text)

    # Remove @mentions
    text = re.sub(r"@\w+", "", text)

    # Collapse multiple whitespaces and strip
    text = re.sub(r"\s+", " ", text).strip()

    return text


def heavy_clean(
    text: Optional[str],
    remove_stopwords: bool = False,
    lemmatize: bool = True,
) -> str:
    """Aggressive text cleaning suitable for classical models (TF-IDF + Logistic Regression).

    Lowercases, expands contractions, removes URLs, mentions, and extra punctuation,
    with optional stopword removal (preserving negations) and lemmatization.

    Args:
        text: Raw text string or None.
        remove_stopwords: Whether to remove non-negation English stopwords.
        lemmatize: Whether to lemmatize tokens using WordNetLemmatizer.

    Returns:
        Cleaned text string. Returns empty string if input is None, empty, or non-string.
    """
    if text is None or not isinstance(text, str):
        return ""

    # Lowercase
    text = text.lower()

    # Standardize curly apostrophes
    text = text.replace("’", "'").replace("‘", "'")

    # Remove URLs and @mentions
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)
    text = re.sub(r"@\w+", " ", text)

    # Expand contractions
    for contraction, expansion in CONTRACTIONS.items():
        pattern = r"\b" + re.escape(contraction) + r"\b"
        text = re.sub(pattern, expansion, text)
        # Also handle attached n't (e.g. wasn't -> was not)
        if contraction == "n't":
            text = text.replace("n't", " not")

    # Remove non-alphanumeric characters (keep words and spaces)
    text = re.sub(r"[^a-z0-9\s]", " ", text)

    # Tokenize by whitespace
    tokens = text.split()

    # Optional stopword filtering (preserving negations)
    if remove_stopwords:
        stop_words = _get_stopwords()
        tokens = [t for t in tokens if t in NEGATION_WORDS or t not in stop_words]

    # Optional lemmatization
    if lemmatize:
        lemmatizer = _get_lemmatizer()
        protected = {"was", "has", "as", "us", "this", "yes"}
        tokens = [t if t in protected else lemmatizer.lemmatize(t) for t in tokens]

    return " ".join(tokens).strip()
