"""Turn the text box into an s1 query.

The format is deliberately tiny:

    optional context, any number of lines

    A question, on one line
    choices: option, option, option

    Another question
    choices: a = when a applies, b = when b applies

Every line that is not a question or a choices line is context, and every
question gets the same context. A ``choices:`` line binds to the non-empty
line right above it. An option may carry a description after ``=``.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

MAX_OPTIONS = 32     # s1 answers in one token; the index space it measured on GLM is 32 wide
CHOICES_RE = re.compile(r"^\s*(choices|choice|chocies|choises|choises|options|option)\s*:\s*(.*)$", re.IGNORECASE)

HOW_TO = ("Write a question on one line and put its choices on the next line, like this:\n\n"
          "Are you AI?\nchoices: yes, no\n\n"
          "Without any choices line, the last line is taken as a yes/no question. "
          "Anything above a question is context and is given to the model with every question. "
          "You can ask several questions in one go, and an option can carry a description: "
          "choices: repair = broken devices, sales = new orders.")


class ParseError(ValueError):
    def __init__(self, text: str, line: int | None = None) -> None:
        super().__init__(text)
        self.line = line

    def to_dict(self) -> dict:
        return {"error": str(self), "line": self.line, "hint": HOW_TO}


@dataclass
class Option:
    name: str
    description: str | None = None


@dataclass
class Question:
    key: str
    text: str
    options: list[Option] = field(default_factory=list)
    line: int = 0
    assumed: bool = False    # no choices were given; yes/no was assumed


@dataclass
class Query:
    context: str
    questions: list[Question]

    def to_dict(self) -> dict:
        return {"context": self.context,
                "questions": [{"key": q.key, "question": q.text, "line": q.line, "assumed": q.assumed,
                               "choices": [{"name": o.name, "description": o.description} for o in q.options]}
                              for q in self.questions]}


def parse_options(spec: str, line: int) -> list[Option]:
    parts = [p.strip() for p in re.split(r"[,;|]", spec)]
    parts = [p for p in parts if p]
    options: list[Option] = []
    for part in parts:
        name, _, desc = part.partition("=")
        name, desc = name.strip(), desc.strip()
        if not name:
            raise ParseError(f"line {line}: an option has no name before '='", line)
        options.append(Option(name, desc or None))
    if len(options) < 2:
        raise ParseError(f"line {line}: a question needs at least two choices, separated by commas", line)
    if len(options) > MAX_OPTIONS:
        raise ParseError(f"line {line}: at most {MAX_OPTIONS} choices per question (s1 answers in one token)", line)
    seen: set[str] = set()
    for o in options:
        key = o.name.lower()
        if key in seen:
            raise ParseError(f"line {line}: choice '{o.name}' appears twice", line)
        seen.add(key)
    return options


def parse(text: str) -> Query:
    lines = text.replace("\r\n", "\n").split("\n")
    consumed: set[int] = set()
    questions: list[Question] = []
    for i, raw in enumerate(lines):
        m = CHOICES_RE.match(raw)
        if not m:
            continue
        # the question is the nearest non-empty line above that is not itself a choices line
        j = i - 1
        while j >= 0 and not lines[j].strip():
            j -= 1
        if j < 0 or CHOICES_RE.match(lines[j]) or j in consumed:
            raise ParseError(f"line {i + 1}: this choices line has no question on the line above it", i + 1)
        options = parse_options(m.group(2), i + 1)
        questions.append(Question(key=f"q{len(questions) + 1}", text=lines[j].strip(), options=options, line=j + 1))
        consumed.update((i, j))
    if not questions:
        if not text.strip():
            raise ParseError("nothing to ask yet")
        # No choices anywhere: the last line is the question and the answer is yes or no.
        last = max(k for k, raw in enumerate(lines) if raw.strip())
        questions.append(Question(key="q1", text=lines[last].strip(),
                                  options=[Option("yes"), Option("no")], line=last + 1, assumed=True))
        consumed.add(last)
    context_lines = [raw.rstrip() for k, raw in enumerate(lines) if k not in consumed]
    context = re.sub(r"\n{3,}", "\n\n", "\n".join(context_lines)).strip()
    return Query(context=context, questions=questions)
