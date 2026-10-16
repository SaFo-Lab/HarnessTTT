"""Contract of inner.runtime.harness_runner._override_llm: endpoint injection, nested extra_params of released packages,
vLLM-only request fields stripped for hosted APIs.  Pure-Python (no NexAU): the agent config is a stand-in object."""
from __future__ import annotations

import copy
import unittest
from types import SimpleNamespace

from inner.runtime.harness_runner import LLMEndpoint, VLLM_ONLY_EXTRA_BODY, _flatten_extra_params, _override_llm

# llm_config of a released Hugging Face package, as NexAU's LLMConfig(**yaml) stores it: every unknown key lands in
# extra_params, so the YAML's own `extra_params:` mapping appears nested one level down
RELEASED_EXTRA = {"extra_params": {"extra_body": {"top_k": 20, "chat_template_kwargs": {"enable_thinking": False}}}}


def fake_config(extra):
    llm = SimpleNamespace(model="m", base_url="u", api_key="k", timeout=1.0, max_retries=0, stream=False,
                          temperature=0.1, top_p=0.5, max_tokens=10, extra_params=copy.deepcopy(extra))
    return SimpleNamespace(llm_config=llm, retry_attempts=3)


class FlattenTest(unittest.TestCase):
    def test_nested_extra_params_are_hoisted_and_nothing_named_extra_params_survives(self):
        extra = copy.deepcopy(RELEASED_EXTRA)
        _flatten_extra_params(extra)
        self.assertNotIn("extra_params", extra)
        self.assertEqual(extra["extra_body"]["top_k"], 20)
        self.assertEqual(extra["extra_body"]["chat_template_kwargs"], {"enable_thinking": False})

    def test_outer_values_win_over_nested_ones(self):
        extra = {"extra_body": {"top_k": 5}, "extra_params": {"extra_body": {"top_k": 20, "min_p": 0.1}, "logit_bias": {"1": 2}}}
        _flatten_extra_params(extra)
        self.assertEqual(extra["extra_body"], {"top_k": 5, "min_p": 0.1})
        self.assertEqual(extra["logit_bias"], {"1": 2})


class OverrideTest(unittest.TestCase):
    def test_released_package_reaches_the_client_with_a_flat_extra_body(self):
        cfg = fake_config(RELEASED_EXTRA)
        ep = LLMEndpoint(model="qwen", base_url="http://127.0.0.1:8800/v1", top_k=40, enable_thinking=False, seed=7)
        _override_llm(cfg, ep, preserve_sampling=True, top_k_override=None)
        extra = cfg.llm_config.extra_params
        self.assertNotIn("extra_params", extra)
        self.assertEqual(extra["extra_body"]["top_k"], 40)                        # endpoint top_k applies (no package override)
        self.assertEqual(extra["extra_body"]["chat_template_kwargs"], {"enable_thinking": False})
        self.assertEqual(extra["extra_body"]["seed"], 7)
        self.assertEqual(cfg.llm_config.model, "qwen")
        self.assertEqual(cfg.retry_attempts, 1)
        self.assertEqual((cfg.llm_config.temperature, cfg.llm_config.top_p, cfg.llm_config.max_tokens), (0.1, 0.5, 10))

    def test_candidate_top_k_from_meta_json_wins(self):
        cfg = fake_config({})
        _override_llm(cfg, LLMEndpoint(top_k=20), preserve_sampling=True, top_k_override=3)
        self.assertEqual(cfg.llm_config.extra_params["extra_body"]["top_k"], 3)

    def test_hosted_api_gets_no_vllm_only_fields(self):
        cfg = fake_config({"extra_body": {"top_k": 20, "min_p": 0.05, "chat_template_kwargs": {"enable_thinking": True}, "seed": 1},
                           "extra_params": {"extra_body": {"repetition_penalty": 1.1}}})
        ep = LLMEndpoint(model="gpt", base_url="https://api.openai.com/v1", vllm_extras=False, enable_thinking=None, seed=None)
        _override_llm(cfg, ep, preserve_sampling=True)
        body = cfg.llm_config.extra_params.get("extra_body", {})
        for key in VLLM_ONLY_EXTRA_BODY:
            self.assertNotIn(key, body)
        self.assertEqual(body, {"seed": 1, "max_completion_tokens": 10})           # OpenAI understands seed; the cap moves
        self.assertNotIn("extra_params", cfg.llm_config.extra_params)
        self.assertEqual(set(cfg.llm_config.additional_drop_params), {"max_tokens", "temperature", "top_p"})

    def test_hosted_api_keeps_sampling_on_request(self):
        cfg = fake_config({})
        _override_llm(cfg, LLMEndpoint(vllm_extras=False, enable_thinking=None, hosted_sampling=True), preserve_sampling=True)
        self.assertEqual(set(cfg.llm_config.additional_drop_params), {"max_tokens"})

    def test_empty_extra_body_is_not_sent(self):
        cfg = fake_config({})
        _override_llm(cfg, LLMEndpoint(top_k=None, enable_thinking=None, vllm_extras=True), preserve_sampling=False)
        self.assertNotIn("extra_body", cfg.llm_config.extra_params)
        self.assertEqual((cfg.llm_config.temperature, cfg.llm_config.top_p), (0.7, 0.95))

    def test_hosted_api_moves_the_token_cap(self):
        cfg = fake_config({})
        _override_llm(cfg, LLMEndpoint(top_k=None, enable_thinking=None, vllm_extras=False, max_tokens=4096), preserve_sampling=False)
        self.assertEqual(cfg.llm_config.extra_params["extra_body"], {"max_completion_tokens": 4096})
        self.assertIn("max_tokens", cfg.llm_config.additional_drop_params)


if __name__ == "__main__":
    unittest.main()
