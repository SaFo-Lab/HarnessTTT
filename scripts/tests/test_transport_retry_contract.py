from types import SimpleNamespace

from inner.runtime.harness_runner import LLMEndpoint, _override_llm
from outer.proposing.propose import _configure_transport


def test_zero_sdk_retries_still_means_exactly_one_outer_attempt():
    llm = SimpleNamespace(
        model=None,
        base_url=None,
        api_key=None,
        timeout=None,
        max_retries=2,
        temperature=None,
        top_p=None,
        max_tokens=None,
        extra_params={},
    )
    config = SimpleNamespace(llm_config=llm, retry_attempts=2)
    endpoint = LLMEndpoint(max_retries=0)

    _override_llm(config, endpoint)

    assert config.retry_attempts == 1
    assert config.llm_config.max_retries == 0
    assert config.llm_config.stream_idle_timeout_ms == 600_000


def test_sdk_retry_count_does_not_increase_outer_agent_attempts():
    llm = SimpleNamespace(
        model=None,
        base_url=None,
        api_key=None,
        timeout=None,
        max_retries=0,
        temperature=None,
        top_p=None,
        max_tokens=None,
        extra_params={},
    )
    config = SimpleNamespace(llm_config=llm, retry_attempts=1)

    _override_llm(config, LLMEndpoint(max_retries=2))

    assert config.retry_attempts == 1
    assert config.llm_config.max_retries == 2


def test_proposer_uses_request_retry_without_outer_trajectory_retry():
    llm = SimpleNamespace(
        timeout=None,
        max_retries=2,
        stream=False,
        stream_idle_timeout_ms=None,
    )
    config = SimpleNamespace(llm_config=llm, retry_attempts=2)

    _configure_transport(config, timeout=1800, max_retries=1)

    assert config.retry_attempts == 1
    assert config.llm_config.max_retries == 1
    assert config.llm_config.timeout == 1800
    assert config.llm_config.stream_idle_timeout_ms == 1_800_000
