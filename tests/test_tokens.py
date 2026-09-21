"""The token oracle against a fake ``/completions`` tokenizer."""

import pytest

from decisions.tokens import TokenOracle

PREFIX_IDS = [7, 8]


class FakeTokenizer:
    """``PREFIX`` plus an index below ``single`` adds one token, anything else two."""

    def __init__(self, single: int = 12, offset: int = 100) -> None:
        self.single, self.offset, self.requests = single, offset, 0

    def encode(self, text: str) -> list[int]:
        if text == "answer:":
            return PREFIX_IDS
        index = int(text.removeprefix("answer:"))
        if index < self.single:
            return [*PREFIX_IDS, self.offset + index]
        return [*PREFIX_IDS, 1, 2]

    def post(self, path: str, payload: dict) -> tuple[dict, float]:
        assert path == "/completions" and payload["echo"] and payload["max_tokens"] == 0
        self.requests += 1
        choices = [{"index": n, "logprobs": {"tokens": [f"token_id:{t}" for t in self.encode(p)]}}
                   for n, p in enumerate(payload["prompt"])]
        return {"choices": choices[::-1]}, 0.01  # the server may reorder choices


def test_stops_at_the_first_multi_token_index():
    oracle = TokenOracle(FakeTokenizer(single=12), "fake")
    assert oracle.single_token_indexes("answer:", limit=50) == [100 + i for i in range(12)]


def test_a_shorter_probe_does_not_cap_a_longer_one():
    oracle = TokenOracle(FakeTokenizer(single=12), "fake")
    assert len(oracle.single_token_indexes("answer:", limit=3)) == 3
    assert len(oracle.single_token_indexes("answer:", limit=10)) == 10
    assert len(oracle.single_token_indexes("answer:", limit=50)) == 12


def test_a_probe_is_reused_until_it_expires():
    server = FakeTokenizer()
    oracle = TokenOracle(server, "fake")
    oracle.single_token_indexes("answer:", limit=20)
    requests = server.requests
    oracle.single_token_indexes("answer:", limit=20)
    assert server.requests == requests

    # The alias moved to a model with another tokenizer: after max_age the
    # oracle has to see the new ids, not keep the old ones.
    server.offset = 500
    oracle.max_age = 0
    assert oracle.single_token_indexes("answer:", limit=20)[0] == 500


def test_no_single_token_index_is_an_error():
    with pytest.raises(ValueError):
        TokenOracle(FakeTokenizer(single=0), "fake").single_token_indexes("answer:")
