import pytest

from app.parse import MAX_QUESTIONS, ParseError, parse


def test_single_question_no_context():
    q = parse("Are you AI?\nchoices: yes, no")
    assert q.context == ""
    assert [(x.text, [o.name for o in x.options]) for x in q.questions] == [("Are you AI?", ["yes", "no"])]


def test_context_and_several_questions():
    q = parse("You are a phone agent answering calls.\n\nmore context\n\nWhich team should the call go to?\n"
              "chocies: repair, human, sales\n\nIs this a valid request?\nchoices: yes, no\n")
    assert q.context == "You are a phone agent answering calls.\n\nmore context"
    assert [x.text for x in q.questions] == ["Which team should the call go to?", "Is this a valid request?"]
    assert [x.key for x in q.questions] == ["q1", "q2"]
    assert q.questions[0].line == 5


def test_descriptions_and_separators():
    q = parse("Route it\nchoices: repair = broken devices; sales = new orders | other")
    assert [(o.name, o.description) for o in q.questions[0].options] == [
        ("repair", "broken devices"), ("sales", "new orders"), ("other", None)]


def test_text_between_questions_is_context():
    q = parse("A\nQ1?\nchoices: x, y\nsome note\nQ2?\nchoices: x, y")
    assert q.context == "A\nsome note"


@pytest.mark.parametrize("text,fragment", [
    ("", "nothing to ask"),
    ("choices: a, b", "no question on the line above"),
    ("Q?\nchoices: a, b\nchoices: c, d", "no question on the line above"),
    ("Q?\nchoices: only", "at least two"),
    ("Q?\nchoices: a, a", "appears twice"),
    ("Q?\nchoices: " + ", ".join(str(i) for i in range(40)), "at most 32"),
    ("Q?\nchoices: = nothing, b", "no name"),
])
def test_errors_explain(text, fragment):
    with pytest.raises(ParseError) as exc:
        parse(text)
    assert fragment in str(exc.value)
    assert "choices:" in exc.value.to_dict()["hint"]


def test_no_choices_means_yes_no_on_the_last_line():
    q = parse("Some context here.\n\nIs the sky blue?\n")
    assert q.context == "Some context here."
    assert q.questions[0].text == "Is the sky blue?"
    assert [o.name for o in q.questions[0].options] == ["yes", "no"]
    assert q.questions[0].assumed is True
    assert parse("Are you AI?").to_dict()["questions"][0]["assumed"] is True
    assert parse("Q?\nchoices: a, b").questions[0].assumed is False


def test_an_option_line_in_the_context_is_context():
    q = parse("Option: refund to card\nWhich team?\nchoices: payments, support")
    assert q.context == "Option: refund to card"
    assert [x.text for x in q.questions] == ["Which team?"]


def test_too_many_questions_is_an_error():
    text = "".join(f"Q{i}?\nchoices: a, b\n" for i in range(MAX_QUESTIONS + 1))
    with pytest.raises(ParseError, match=f"at most {MAX_QUESTIONS} questions"):
        parse(text)
    assert len(parse("".join(f"Q{i}?\nchoices: a, b\n" for i in range(MAX_QUESTIONS))).questions) == MAX_QUESTIONS
