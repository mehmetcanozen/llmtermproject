from __future__ import annotations

import torch

from src.attention_gate import GateSettings, is_ignored_gate_token, score_attention_to_passages


def test_attention_scoring_sums_passage_spans() -> None:
    layer = torch.zeros((1, 2, 1, 8), dtype=torch.float32)
    layer[:, :, :, 2:4] = 0.10
    layer[:, :, :, 5:7] = 0.05
    scores = score_attention_to_passages(
        (layer, layer),
        {"P1": {"token_start": 2, "token_end_exclusive": 4}, "P2": {"token_start": 5, "token_end_exclusive": 7}},
        tail_layers=2,
    )
    assert round(scores["P1"], 3) == 0.2
    assert round(scores["P2"], 3) == 0.1


def test_gate_ignored_token_classes() -> None:
    assert is_ignored_gate_token(".")
    assert is_ignored_gate_token("[")
    assert is_ignored_gate_token("P1")
    assert not is_ignored_gate_token("revenue")


def test_gate_settings_defaults_match_phase_plan() -> None:
    settings = GateSettings()
    assert settings.tail_layers == 4
    assert settings.rolling_window == 4
    assert settings.threshold == 0.10
    assert settings.consecutive_failures == 3
    assert settings.skip_initial_generated_tokens == 8
