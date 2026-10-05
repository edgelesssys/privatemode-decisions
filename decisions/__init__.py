"""Single-token Choice inference over an OpenAI-compatible endpoint.

Privatemode Decisions against a remote vLLM (here: Privatemode's
confidential-computing proxy), so the technique works against a model you
cannot load, download a tokenizer for, or reach with anything but an HTTP
API.
"""

from .calibration import Calibration, calibrate, evaluate
from .client import APIError, OpenAIClient
from .inference import PREFIX, SystemOne
from .tokens import TokenOracle
from .types import Choice, ChoiceAnswer, SystemOneResponse, Usage

__all__ = [
           "PREFIX",
           "APIError",
           "Calibration",
           "Choice",
           "ChoiceAnswer",
           "OpenAIClient",
           "SystemOne",
           "SystemOneResponse",
           "TokenOracle",
           "Usage",
           "calibrate",
           "evaluate",
]
