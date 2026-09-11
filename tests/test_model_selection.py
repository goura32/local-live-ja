from local_live.llm.ollama import choose_qwen35_9b_model


def test_local_model_selection_uses_observed_tags_not_a_guess():
    tags = [
        {"name": "qwen3.5:27b-q4_K_M", "details": {"parameter_size": "27.8B"}},
        {"name": "qwen3.5:9b-q4_K_M", "details": {"parameter_size": "9.7B"}},
        {"name": "gemma4:12b-it-qat", "details": {"parameter_size": "11.9B"}},
    ]
    assert choose_qwen35_9b_model(tags) == "qwen3.5:9b-q4_K_M"
    assert choose_qwen35_9b_model([{ "name": "gemma4:12b", "details": {} }]) is None
