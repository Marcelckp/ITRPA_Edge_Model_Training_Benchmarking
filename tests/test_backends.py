import json

import pytest

from slmbench.backends import make_backend
from slmbench.backends.base import BackendError, check_loopback_cmd
from slmbench.backends.ollama import parse_ollama_stream
from slmbench.backends.openai_compat import LlamaCppBackend, parse_sse_stream
from slmbench.config import ExpConfig


class Clock:
    def __init__(self, step=0.1):
        self.t, self.step = 0.0, step

    def __call__(self):
        self.t += self.step
        return self.t


def _sse(obj):
    return f"data: {json.dumps(obj)}"


def test_llama_server_stream_uses_server_timings():
    lines = [": keep-alive", _sse({"choices": [{"delta": {"role": "assistant"}}]}),
             _sse({"choices": [{"delta": {"content": "ye"}}]}), _sse({"choices": [{"delta": {"content": "s"}}]}),
             _sse({"choices": [{"delta": {}}], "timings": {"prompt_n": 50, "prompt_ms": 120.0, "predicted_n": 2,
                                                           "predicted_ms": 80.0}}),
             "data: [DONE]"]
    r = parse_sse_stream(lines, 0.0, Clock())
    assert r.text == "yes" and r.prompt_tokens == 50 and r.output_tokens == 2
    assert r.ttft_s == pytest.approx(0.1)
    assert r.primary_ttft_s == pytest.approx(0.12) and r.primary_tpot_s == pytest.approx(0.04)
    assert r.tokens_per_s == pytest.approx(25.0)


def test_vllm_stream_usage_fallback():
    lines = [_sse({"choices": [{"delta": {"content": "4"}}]}), _sse({"choices": [{"delta": {"content": "2"}}]}),
             _sse({"choices": [], "usage": {"prompt_tokens": 10, "completion_tokens": 2}}), "data: [DONE]"]
    r = parse_sse_stream(lines, 0.0, Clock())
    assert r.text == "42" and r.server_prompt_ms is None and r.primary_ttft_s == r.ttft_s
    assert r.tpot_s == pytest.approx(0.1)


def test_stream_error_raises():
    with pytest.raises(BackendError):
        parse_sse_stream([_sse({"error": {"message": "bad"}})], 0.0, Clock())


def test_ollama_stream_durations():
    lines = [json.dumps({"response": "no", "done": False}),
             json.dumps({"response": "", "done": True, "prompt_eval_count": 30, "prompt_eval_duration": 2e8,
                         "eval_count": 4, "eval_duration": 4e8})]
    r = parse_ollama_stream(lines, 0.0, Clock())
    assert r.text == "no" and r.output_tokens == 4 and r.tpot_s == pytest.approx(0.1)
    assert r.server_prompt_ms == pytest.approx(200.0)


def test_llamacpp_disables_prompt_cache():
    body = LlamaCppBackend({}).request_body([{"role": "user", "content": "x"}], 16, 0.0, 42)
    assert body["cache_prompt"] is False and body["stream"] is True and body["temperature"] == 0.0


@pytest.mark.parametrize("cmd", [["llama-server", "-m", "x"], ["vllm", "serve", "x", "--host", "0.0.0.0"],
                                 ["llama-server", "--host=192.168.1.5"]])
def test_server_must_bind_loopback(cmd):
    with pytest.raises(BackendError):
        check_loopback_cmd(cmd)


def test_loopback_ok():
    check_loopback_cmd(["llama-server", "--host", "127.0.0.1"])


def test_mock_is_deterministic():
    cfg = ExpConfig(model="m", strategy="base", precision="Q4_K_M", engine="llamacpp", tier="edge", device="d",
                    dataset="pubmedqa", context_mode="closed", k=0, seed=42, temperature=0.0, max_new_tokens=16)
    b = make_backend("mock")
    b.load(cfg)
    msgs = [{"role": "user", "content": "Question: q\nAnswer with exactly one word: yes, no, or maybe."}]
    r1, r2 = b.generate(msgs, 16), b.generate(msgs, 16)
    assert r1.text == r2.text and r1.e2e_s == r2.e2e_s and b.is_mock and r1.power_trace
