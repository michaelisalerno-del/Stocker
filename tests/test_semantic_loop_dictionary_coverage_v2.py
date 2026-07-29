"""Root-suite collection shim for the versioned research test module."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

RESEARCH_WORK = (
    Path(__file__).resolve().parents[1]
    / "research"
    / "slrno-v2"
    / "20260714-regime-loop-handoff"
    / "work"
)
RESEARCH_TEST = RESEARCH_WORK / "tests" / "test_semantic_loop_dictionary_coverage_v2.py"


def _load_research_tests() -> ModuleType:
    if str(RESEARCH_WORK) not in sys.path:
        sys.path.insert(0, str(RESEARCH_WORK))
    module_name = "_semantic_loop_dictionary_coverage_v2_research_tests"
    spec = importlib.util.spec_from_file_location(module_name, RESEARCH_TEST)
    if spec is None or spec.loader is None:
        raise RuntimeError("versioned semantic-loop research tests could not be loaded")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


_RESEARCH_TESTS = _load_research_tests()
for _name, _value in vars(_RESEARCH_TESTS).items():
    if _name.startswith("test_"):
        globals()[_name] = _value
