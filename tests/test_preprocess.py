"""Unit tests for text preprocessing pipelines."""

import pytest
from src.preprocess import minimal_clean, heavy_clean, NEGATION_WORDS


def test_empty_and_none_input() -> None:
    """Verify both functions handle None, empty strings, and whitespace without crashing."""
    assert minimal_clean(None) == ""
    assert minimal_clean("") == ""
    assert minimal_clean("     \n\t   ") == ""

    assert heavy_clean(None) == ""
    assert heavy_clean("") == ""
    assert heavy_clean("     \n\t   ") == ""


def test_minimal_clean_preserves_structure() -> None:
    """Verify minimal_clean removes URLs/mentions while preserving casing, punctuation, and contractions."""
    sample = "  Hey @alice! Check out https://example.com/test now... I'm so thrilled!!   "
    expected = "Hey ! Check out now... I'm so thrilled!!"
    assert minimal_clean(sample) == expected


def test_heavy_clean_contractions_and_lemmatization() -> None:
    """Verify heavy_clean lowercases, expands contractions, removes punctuation, and lemmatizes."""
    sample = "I shouldn't have eaten those 3 apples! They're awful. @chef https://food.org"
    result = heavy_clean(sample, remove_stopwords=False, lemmatize=True)
    # Contractions expanded: shouldn't -> should not, they're -> they are
    # Punctuation removed, apples -> apple (lemmatized)
    assert "should not" in result
    assert "they are" in result
    assert "apple" in result
    assert "apples" not in result
    assert "!" not in result
    assert "@" not in result
    assert "https" not in result


def test_negations_preserved_with_stopword_removal() -> None:
    """Verify critical negation words are never removed when stopword removal is enabled."""
    sample = "I am not happy and I will never celebrate, nor do I have no regret!"
    result = heavy_clean(sample, remove_stopwords=True, lemmatize=True)

    # Check negations are strictly preserved
    assert "not" in result
    assert "never" in result
    assert "nor" in result
    assert "no" in result

    # Standard non-negation stopwords (e.g., "am", "and", "will", "do", "i", "have") should be removed
    for word in ["am", "will", "have"]:
        assert word not in result.split()
