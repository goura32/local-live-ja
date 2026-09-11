from pathlib import Path

from local_live.config import load_config, load_openrouter_key, nested


def test_credential_loader_does_not_return_missing_or_invalid_file(tmp_path):
    missing = load_openrouter_key(tmp_path / "missing.key")
    assert missing.value is None
    assert missing.reason == "missing"

    key_file = tmp_path / "openrouter.key"
    key_file.write_text("  redacted-test-value\n", encoding="utf-8")
    key_file.chmod(0o600)
    loaded = load_openrouter_key(key_file)
    assert loaded.value == "redacted-test-value"
    assert loaded.reason == "ok"


def test_default_config_uses_measured_vram_aware_gpu_asr_type():
    config = load_config(Path(__file__).parents[1] / "config/default.yaml")
    assert nested(config, "asr", "gpu_default_compute_type") == "int8_float16"
