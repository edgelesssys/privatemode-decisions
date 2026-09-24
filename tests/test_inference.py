"""Reading more options than the server reports in one response.

A fake server stands in for Privatemode: it rejects a ``logprob_token_ids``
longer than 128, as the real deployment does, and reports the same fixed
logprob for an id whichever request asks for it, as measured against the
real one.
"""

import json
import math

import pytest

from decisions import APIError, Choice, SystemOne
from decisions.inference import MAX_LOGPROB_TOKEN_IDS, PREFIX, batches

INDEX_IDS = list(range(1000, 1191))  # 191 single-token indexes, like GLM-5.3-Flash


def logprob(token_id: int) -> float:
    """A raw (unmasked) logprob per index; index 57 is the clear favourite."""
    return -1.0 if token_id == INDEX_IDS[57] else -8.0 - (token_id % 7)


class FakeServer:
    def __init__(self) -> None:
        self.payloads: list[dict] = []

    def post(self, path: str, payload: dict) -> tuple[dict, float]:
        assert path == "/chat/completions"
        self.payloads.append(payload)
        read = payload["logprob_token_ids"]
        if len(read) > MAX_LOGPROB_TOKEN_IDS:
            raise APIError(400, f"Requested logprob_token_ids of length {len(read)}")
        assert set(read) <= set(payload["allowed_token_ids"])
        entries = [{"token": f"token_id:{i}", "logprob": logprob(i)} for i in read]
        return {"choices": [{"logprobs": {"content": [{"top_logprobs": entries}]}}],
                "usage": {"prompt_tokens": 10}}, 0.01

    def close(self) -> None:
        pass


def make_engine(**kwargs) -> tuple[SystemOne, FakeServer]:
    server = FakeServer()
    engine = SystemOne(server, "fake", **kwargs)
    engine.oracle._indexes[PREFIX] = {"ids": INDEX_IDS, "exhausted": True}
    return engine, server


def question(options: int) -> Choice:
    return Choice({f"intent_{i}": None for i in range(options)}, instructions="Which intent?")


def expected(options: int) -> dict[str, float]:
    weights = [math.exp(logprob(i)) for i in INDEX_IDS[:options]]
    return {f"intent_{i}": w / sum(weights) for i, w in enumerate(weights)}


def test_batches_split_at_the_cap():
    assert batches(list(range(151)), 128) == [list(range(128)), list(range(128, 151))]
    assert batches(list(range(5)), 128) == [list(range(5))]
    with pytest.raises(ValueError):
        batches([1], 0)


@pytest.mark.parametrize("mode", ["staged", "parallel", "sequential"])
def test_more_options_than_the_cap_are_read_in_batches(mode):
    engine, server = make_engine()
    result = engine.system_one("I lost my card", {"intent": question(151)}, mode=mode)

    assert len(server.payloads) == 2
    assert all(p["allowed_token_ids"] == INDEX_IDS[:151] for p in server.payloads)
    assert sorted(i for p in server.payloads for i in p["logprob_token_ids"]) == INDEX_IDS[:151]
    answer = result.answers["intent"]
    assert answer.choice == "intent_57"
    assert answer.probabilities == pytest.approx(expected(151))
    assert result.timings["requests"] == 2 and result.usage.output_tokens == 2


def test_up_to_the_cap_is_one_request_per_question():
    engine, server = make_engine()
    result = engine.system_one("state", {"a": question(128), "b": question(3)})

    assert len(server.payloads) == 2
    for payload in server.payloads:
        assert payload["logprob_token_ids"] == payload["allowed_token_ids"]
    assert result.answers["a"].probabilities == pytest.approx(expected(128))


def test_batches_combine_with_permutations():
    engine, server = make_engine()
    result = engine.system_one("state", {"intent": question(151)}, permutations=2)

    assert len(server.payloads) == 4  # two orders, two batches each
    assert sum(result.answers["intent"].probabilities.values()) == pytest.approx(1.0)


def test_a_smaller_cap_is_honoured():
    engine, server = make_engine(max_logprob_ids=50)
    engine.system_one("state", {"intent": question(151)})

    # The batches run in parallel, so they arrive in any order.
    assert sorted(len(p["logprob_token_ids"]) for p in server.payloads) == [1, 50, 50, 50]


def test_a_missing_option_logprob_is_an_error():
    """A backend that ignores logprob_token_ids must not read as probability 0."""
    engine, server = make_engine()
    real_post = server.post

    def drop_one(path, payload):
        body, elapsed = real_post(path, payload)
        body["choices"][0]["logprobs"]["content"][0]["top_logprobs"].pop()
        return body, elapsed

    server.post = drop_one
    with pytest.raises(RuntimeError, match="did not report 1 of 3"):
        engine.system_one("state", {"q": question(3)})


def test_no_top_logprobs_is_an_error():
    engine, server = make_engine()
    server.post = lambda path, payload: ({"choices": [{"logprobs": {"content": [
        {"token": f"token_id:{INDEX_IDS[0]}", "logprob": 0.0, "top_logprobs": []}]}}]}, 0.01)
    with pytest.raises(RuntimeError, match="did not report"):
        engine.system_one("state", {"q": question(3)})


def sent_names(payload: dict) -> list[str]:
    """The option names in the order a request sent them."""
    text = payload["messages"][0]["content"]
    return [o["label"] for o in json.loads(text[text.index("{"):])["options"]]


def test_permutations_follow_the_option_not_the_position():
    engine, server = make_engine()

    def favour(name):
        def post(path, payload):
            server.payloads.append(payload)
            at = sent_names(payload).index(name)
            entries = [{"token": f"token_id:{i}", "logprob": -0.1 if n == at else -5.0}
                       for n, i in enumerate(payload["logprob_token_ids"])]
            return {"choices": [{"logprobs": {"content": [{"top_logprobs": entries}]}}]}, 0.01
        return post

    server.post = favour("intent_2")
    result = engine.system_one("state", {"q": question(3)}, permutations=3)
    assert len(server.payloads) == 3
    assert len({tuple(sent_names(p)) for p in server.payloads}) == 3  # three orders
    answer = result.answers["q"]
    assert answer.choice == "intent_2"
    others = [p for name, p in answer.probabilities.items() if name != "intent_2"]
    assert others[0] == pytest.approx(others[1])


def test_a_position_prior_averages_out():
    engine, server = make_engine()
    server.post = lambda path, payload: ({"choices": [{"logprobs": {"content": [{"top_logprobs": [
        {"token": f"token_id:{i}", "logprob": -0.1 if n == 0 else -5.0}
        for n, i in enumerate(payload["logprob_token_ids"])]}]}}]}, 0.01)
    result = engine.system_one("state", {"q": question(3)}, permutations=3)
    assert list(result.answers["q"].probabilities.values()) == pytest.approx([1 / 3] * 3)
