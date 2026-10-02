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
from decisions.inference import MAX_LOGPROB_TOKEN_IDS, PREAMBLE, PREFIX, batches

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
    # The last line holds the state and the question, in either layout.
    return [o["label"] for o in json.loads(text.rsplit("\n", 1)[-1])["options"]]


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


def test_answers_carry_the_temperature_they_were_reported_at():
    engine, _ = make_engine(temperature=3.0)
    assert engine.system_one("state", {"q": question(3)}).answers["q"].temperature == 3.0
    engine, _ = make_engine()                      # an unmeasured model stays raw
    assert engine.system_one("state", {"q": question(3)}).answers["q"].temperature == 1.0


def test_the_default_temperature_follows_the_model_that_answered():
    from decisions.calibration import default_temperature

    def serving(model):
        server = FakeServer()
        original = server.post

        def post(path, payload):
            body, elapsed = original(path, payload)
            return dict(body, model=model), elapsed
        server.post = post
        engine = SystemOne(server, "glm-flash-latest", temperature=family)
        engine.oracle._indexes[PREFIX] = {"ids": INDEX_IDS, "exhausted": True}
        response = engine.system_one("state", {"q": question(4)})
        assert response.model == model                  # the model that answered, not the alias
        assert response.answers["q"].model == model     # and on each answer, for Calibration
        return response.answers["q"].temperature

    family = None
    flash = default_temperature("glm-5.3-flash", 4)
    assert serving("glm-5.3-flash") == pytest.approx(flash)          # the alias as measured
    assert serving("kimi-k2.6") == pytest.approx(default_temperature("kimi-k2.6", 4))
    assert serving("some-new-model") == 1.0                         # moved to an unmeasured model
    # A task family follows the same rule: the served model's value for it,
    # and raw once the alias has moved to a model without measurements.
    family = "sentiment"
    assert serving("glm-5.3-flash") == default_temperature("glm-5.3-flash", 4, "sentiment")
    assert serving("kimi-k2.6") == default_temperature("kimi-k2.6", 4, "sentiment")
    assert serving("some-new-model") == 1.0


@pytest.mark.parametrize("options", [60, 151])
def test_option_mass_sums_the_options_before_the_mask(options):
    """One read, and two reads under one mask above 128 options: the mass
    is the unmasked probability of all options, across both reads."""
    engine, server = make_engine()
    answer = engine.system_one("state", {"q": question(options)}).answers["q"]
    assert len(server.payloads) == (1 if options <= MAX_LOGPROB_TOKEN_IDS else 2)
    expected = math.fsum(math.exp(logprob(i)) for i in INDEX_IDS[:options])
    assert answer.option_mass == pytest.approx(expected)
    assert 0 < answer.option_mass < 1


def test_the_question_comes_before_the_state_by_default():
    engine, server = make_engine()
    q = question(3)
    engine.system_one({"text": "hi"}, {"q": q})
    whole = json.dumps({"state": {"text": "hi"}, **SystemOne._question(q)}, ensure_ascii=False)
    assert server.payloads[-1]["messages"][0]["content"] == (
        PREAMBLE + SystemOne._question_text(q) + "\n" + whole)
    assert server.payloads[-1]["messages"][1]["content"] == PREFIX
    # With one question, "cost" is the same prompt.
    engine.system_one({"text": "hi"}, {"q": q}, optimize="cost")
    assert server.payloads[-1] == server.payloads[-2]


def test_the_lead_keeps_images_first_and_works_with_history():
    engine, server = make_engine()
    q = question(2)
    engine.system_one("state", {"q": q}, images="data:image/png;base64,AA==")
    content = server.payloads[-1]["messages"][0]["content"]
    assert content[0]["type"] == "image_url"
    assert content[-1]["text"].startswith(PREAMBLE + SystemOne._question_text(q) + "\n")
    engine.system_one("state", {"q": q}, history="earlier decisions")
    before = server.payloads[-1]["messages"][0]["content"][0]["text"]
    assert before.startswith(PREAMBLE + SystemOne._question_text(q) + "\n")
    assert before.endswith("earlier decisions")


@pytest.mark.parametrize("engine_kw, call_kw", [({"optimize": "cost"}, {}), ({}, {"optimize": "cost"})])
def test_cost_leads_every_request_with_all_questions(engine_kw, call_kw):
    engine, server = make_engine(**engine_kw)
    questions = {"a": question(2), "b": Choice({"x": None, "y": None, "z": None}, instructions="Which?")}
    engine.system_one("state", questions, mode="sequential", **call_kw)
    block = "".join(SystemOne._question_text(q) + "\n" for q in questions.values())
    texts = [p["messages"][0]["content"] for p in server.payloads]
    assert len(texts) == 2
    assert all(t.startswith(PREAMBLE + block + '{"state": "state"') for t in texts)
    # Each request still ends with the question it asks.
    assert [sent_names(p) for p in server.payloads] == [list(q.criteria) for q in questions.values()]


def test_accuracy_leads_each_request_with_its_own_question_only():
    engine, server = make_engine()
    questions = {"a": question(2), "b": Choice({"x": None, "y": None}, instructions="Which?")}
    engine.system_one("state", questions, mode="sequential")
    for payload, q in zip(server.payloads, questions.values()):
        assert payload["messages"][0]["content"].startswith(
            PREAMBLE + SystemOne._question_text(q) + '\n{"state"')


def split_prompt(payload: dict) -> tuple[list[dict], dict]:
    """The lead's questions and the asked question of one request."""
    lines = payload["messages"][0]["content"].removeprefix(PREAMBLE).split("\n")
    return [json.loads(line) for line in lines[:-1]], json.loads(lines[-1])


@pytest.mark.parametrize("optimize", ["accuracy", "cost"])
def test_the_lead_numbers_options_as_the_rotated_request_does(optimize):
    """With several option orders, the asked question's lead entry must use
    the order of its request: two maps from number to label in one prompt
    would leave the decode following one of them silently."""
    engine, server = make_engine(optimize=optimize)
    questions = {"a": Choice({f"a{i}": None for i in range(4)}, instructions="First?"),
                 "b": Choice({"x": None, "y": None}, instructions="Second?")}
    engine.system_one("state", questions, permutations=3, mode="sequential")
    assert len(server.payloads) == 3 + 2            # b has two options: two orders
    leads = set()
    for payload in server.payloads:
        lead, asked = split_prompt(payload)
        entry = [q for q in lead if q["question"] == asked["question"]]
        assert len(entry) == 1 and entry[0]["options"] == asked["options"]
        assert len(lead) == (1 if optimize == "accuracy" else 2)
        leads.add(json.dumps(lead))
    if optimize == "cost":
        # Requests in the same order share the block: one per order.
        assert len(leads) == 3


def test_optimize_takes_only_the_two_layouts():
    for bad in (True, False, "own", "all"):
        with pytest.raises(ValueError):
            make_engine(optimize=bad)
    engine, _ = make_engine()
    with pytest.raises(ValueError):
        engine.system_one("state", {"q": question(2)}, optimize="first")


@pytest.mark.parametrize("kwargs, mode", [
    ({}, "parallel"),                                    # only the preamble is shared
    ({"optimize": "cost"}, "staged"),
    ({"images": "data:image/png;base64,AA=="}, "staged"),
    ({"history": "earlier decisions"}, "staged"),
    ({"history": {"a": "earlier decisions"}}, "parallel"),  # a record per request
    ({"mode": "sequential"}, "sequential"),              # as asked
])
def test_staged_only_when_the_requests_share_more_than_the_preamble(kwargs, mode):
    engine, _ = make_engine()
    questions = {"a": question(2), "b": Choice({"x": None, "y": None}, instructions="Which?")}
    assert engine.system_one("state", questions, **kwargs).timings["mode"] == mode


def test_a_history_is_led_by_every_question_so_the_record_is_shared():
    """With the default layout, each request leading with its own question
    would put a different prefix in front of the record for every question."""
    engine, server = make_engine()
    questions = {"a": question(2), "b": Choice({"x": None, "y": None}, instructions="Which?")}
    engine.system_one("state", questions, history="earlier decisions",
                      optimize="accuracy", mode="sequential")
    block = "".join(SystemOne._question_text(q) + "\n" for q in questions.values())
    befores = [p["messages"][0]["content"][0]["text"] for p in server.payloads]
    assert befores[0] == befores[1]
    assert befores[0].startswith(PREAMBLE + block) and befores[0].endswith("earlier decisions")


def test_a_history_per_question_is_led_by_its_own_question():
    """No two requests share a record from a mapping, so the whole block
    would cache nothing more."""
    engine, server = make_engine()
    a, b = question(2), Choice({"x": None, "y": None}, instructions="Which?")
    engine.system_one("state", {"a": a, "b": b}, history={"a": "earlier decisions"},
                      mode="sequential")
    texts = [p["messages"][0]["content"] for p in server.payloads]
    assert texts[0][0]["text"] == (PREAMBLE + SystemOne._question_text(a) + "\n"
                                   + '{"state": "state"}\n\nearlier decisions')
    assert texts[1].startswith(PREAMBLE + SystemOne._question_text(b) + '\n{"state"')
