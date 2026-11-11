"""Unit tests for EmotionPredictor inference API in src/predict.py."""

import pytest
from src.config import ACTIVE_DISTILBERT, LABELS
from src.predict import EmotionPredictor
from tests.conftest import is_model_present, requires_model


def test_empty_string_raises_value_error() -> None:
    """Verify empty or whitespace strings raise ValueError for both models."""
    for model_type in ["baseline", "distilbert"]:
        predictor = EmotionPredictor(model_type=model_type)
        with pytest.raises(ValueError, match=r"(?i)empty.*whitespace"):
            predictor.predict("")
        with pytest.raises(ValueError, match=r"(?i)empty.*whitespace"):
            predictor.predict("     \n\t   ")
        with pytest.raises(ValueError, match=r"(?i)empty.*whitespace"):
            predictor.predict_batch(["valid text", "   "])


def test_long_string_truncation() -> None:
    """Verify very long input is truncated to MAX_LENGTH tokens and executes without error."""
    long_text = "I feel overjoyed and ecstatic today! " * 80  # ~560 words
    target_models = ["baseline"]
    for m in [ACTIVE_DISTILBERT, "distilbert"]:
        if is_model_present(m) and m not in target_models:
            target_models.append(m)
    for model_type in target_models:
        predictor = EmotionPredictor(model_type=model_type)
        res = predictor.predict(long_text)
        assert res["label"] in LABELS
        assert 0.0 <= res["confidence"] <= 1.0


def test_batch_size_matches_input() -> None:
    """Verify predict_batch output length matches input list length."""
    texts = [
        "I am so sad and heartbroken.",
        "Today is the best day of my life!",
        "I am utterly terrified of this sound.",
        "That made me so angry and furious.",
    ]
    target_models = ["baseline"]
    for m in [ACTIVE_DISTILBERT, "distilbert"]:
        if is_model_present(m) and m not in target_models:
            target_models.append(m)
    for model_type in target_models:
        predictor = EmotionPredictor(model_type=model_type)
        results = predictor.predict_batch(texts)
        assert len(results) == len(texts)
        assert len(results) == 4


def test_predicted_label_is_valid() -> None:
    """Verify predicted label is always a valid class from config.LABELS."""
    text = "What a wonderful and joyous surprise!"
    target_models = ["baseline"]
    for m in [ACTIVE_DISTILBERT, "distilbert"]:
        if is_model_present(m) and m not in target_models:
            target_models.append(m)
    for model_type in target_models:
        predictor = EmotionPredictor(model_type=model_type)
        res = predictor.predict(text)
        assert res["label"] in LABELS


def test_probabilities_sum_to_one() -> None:
    """Verify all class probabilities sum to 1.0 within float tolerance."""
    text = "I am feeling extremely nervous about the upcoming exam."
    target_models = ["baseline"]
    for m in [ACTIVE_DISTILBERT, "distilbert"]:
        if is_model_present(m) and m not in target_models:
            target_models.append(m)
    for model_type in target_models:
        predictor = EmotionPredictor(model_type=model_type)
        res = predictor.predict(text)
        probs_dict = res["all_probs"]
        prob_sum = sum(probs_dict.values())
        assert pytest.approx(prob_sum, abs=1e-3) == 1.0
        for label, prob in probs_dict.items():
            assert label in LABELS
            assert 0.0 <= prob <= 1.0


def test_both_models_give_same_output_format() -> None:
    """Verify baseline and active transformer produce identical output dictionaries and keys."""
    target_model = ACTIVE_DISTILBERT if is_model_present(ACTIVE_DISTILBERT) else "distilbert"
    requires_model(target_model)
    text = "I can't believe I won the championship!"
    p_base = EmotionPredictor(model_type="baseline")
    p_dist = EmotionPredictor(model_type=target_model)

    res_base = p_base.predict(text)
    res_dist = p_dist.predict(text)

    # Root prediction keys
    assert set(res_base.keys()) == {"label", "confidence", "all_probs"}
    expected_dist_keys = {"label", "confidence", "all_probs"}
    if target_model == "distilbert_v2":
        expected_dist_keys.add("is_confident")
    assert set(res_dist.keys()) == expected_dist_keys

    # Identical label probability keys in exact config.LABELS order
    assert list(res_base["all_probs"].keys()) == LABELS
    assert list(res_dist["all_probs"].keys()) == LABELS

    # Value types
    assert isinstance(res_base["label"], str)
    assert isinstance(res_dist["label"], str)
    assert isinstance(res_base["confidence"], float)
    assert isinstance(res_dist["confidence"], float)


def test_invalid_model_type_raises_value_error() -> None:
    """Verify invalid model_type strings raise ValueError."""
    with pytest.raises(ValueError, match=r"Invalid model_type"):
        EmotionPredictor(model_type="nonexistent_model")


@requires_model("distilbert_v2")
def test_distilbert_v2_model_type_initialization() -> None:
    """Verify distilbert_v2 is accepted as a valid model_type."""
    predictor = EmotionPredictor(model_type="distilbert_v2")
    assert predictor.model_type == "distilbert_v2"


@requires_model("distilbert_v2")
def test_distilbert_v2_prediction() -> None:
    """Verify distilbert_v2 prediction runs and returns valid schema."""
    predictor = EmotionPredictor(model_type="distilbert_v2")
    res = predictor.predict("I am feeling great today!")
    assert res["label"] in LABELS
    assert 0.0 <= res["confidence"] <= 1.0
    assert list(res["all_probs"].keys()) == LABELS


@requires_model("calibration")
def test_calibration_file_loads() -> None:
    """Verify calibration.json exists, loads correctly, and contains valid parameters."""
    import json
    from src.config import MODELS_DIR

    cal_path = MODELS_DIR / "distilbert_v2" / "calibration.json"
    assert cal_path.exists(), f"Calibration file not found at {cal_path}"

    with open(cal_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    assert "temperature" in data
    assert "abstain_threshold" in data
    assert isinstance(data["temperature"], (float, int))
    assert isinstance(data["abstain_threshold"], (float, int))
    assert 0.1 <= data["temperature"] <= 3.0
    assert 0.40 <= data["abstain_threshold"] <= 0.95


@requires_model("distilbert_v2")
def test_is_confident_is_bool() -> None:
    """Verify EmotionPredictor for distilbert_v2 outputs is_confident as a boolean."""
    predictor = EmotionPredictor(model_type="distilbert_v2")
    res = predictor.predict("I am feeling so wonderful and delighted!")
    assert "is_confident" in res
    assert isinstance(res["is_confident"], bool)

    res_neutral = predictor.predict("The meeting starts at 3 pm in room 201.")
    assert "is_confident" in res_neutral
    assert isinstance(res_neutral["is_confident"], bool)
