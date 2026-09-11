from local_live.bench import _provider_pair
from local_live.cli import _make_providers


def test_provider_pair_keeps_local_budget_and_allows_openrouter_budget_override():
    config = {
        "llm": {
            "local_model": "observed-local",
            "max_tokens": 96,
            "openrouter_max_tokens": 256,
        },
        "credentials": {"openrouter_file": "/missing/key"},
    }
    providers = _provider_pair(config)
    assert providers["local"].max_tokens == 96
    assert providers["openrouter"].max_tokens == 256


def test_cli_provider_factory_accepts_openrouter_budget_override():
    config = {
        "llm": {
            "local_model": "observed-local",
            "max_tokens": 96,
            "openrouter_max_tokens": 256,
        },
        "credentials": {"openrouter_file": "/missing/key"},
    }
    providers = _make_providers(config)
    assert providers["local"].max_tokens == 96
    assert providers["openrouter"].max_tokens == 256
