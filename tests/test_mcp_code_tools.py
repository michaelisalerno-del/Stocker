from pathlib import Path

import pytest

from stocker_mcp.security import SecurityError, StockerMCPContext
from stocker_mcp.tools import code


def _context(tmp_path: Path) -> StockerMCPContext:
    repo = tmp_path / "repo"
    home = tmp_path / "StockerLocal"
    repo.mkdir()
    home.mkdir()
    return StockerMCPContext(repo_root=repo, stocker_home=home)


def test_read_code_file_allows_line_ranges(tmp_path: Path) -> None:
    context = _context(tmp_path)
    source = context.repo_root / "example.py"
    source.write_text("one\nSECRET_TOKEN=abc123456789\ndef f():\n    return 1\n", encoding="utf-8")

    result = code.read_code_file("example.py", start_line=3, end_line=4, context=context)

    assert result["path"] == "example.py"
    assert result["start_line"] == 3
    assert result["end_line"] == 4
    assert "def f():" in result["content"]
    assert "SECRET_TOKEN" not in result["content"]


def test_search_code_returns_limited_redacted_matches(tmp_path: Path) -> None:
    context = _context(tmp_path)
    (context.repo_root / "pkg").mkdir()
    (context.repo_root / "pkg" / "alpha.py").write_text(
        "def candidate():\n    return 'edge'\n", encoding="utf-8"
    )
    (context.repo_root / "pkg" / "secret.py").write_text(
        "def candidate():\n    api_key='abc123456789'\n", encoding="utf-8"
    )

    result = code.search_code("candidate", path_glob="pkg/*.py", limit=5, context=context)

    assert result["match_count"] == 2
    assert {match["path"] for match in result["matches"]} == {
        "pkg/alpha.py",
        "pkg/secret.py",
    }


def test_list_files_skips_blocked_secret_paths(tmp_path: Path) -> None:
    context = _context(tmp_path)
    (context.repo_root / "visible.py").write_text("x = 1\n", encoding="utf-8")
    (context.repo_root / ".env").write_text("API_KEY=leak", encoding="utf-8")
    (context.repo_root / ".git").mkdir()
    (context.repo_root / ".git" / "config").write_text("remote = private\n", encoding="utf-8")

    result = code.list_files(limit=20, context=context)

    assert "visible.py" in result["files"]
    assert ".env" not in result["files"]
    assert ".git/config" not in result["files"]


def test_search_code_rejects_traversing_glob(tmp_path: Path) -> None:
    context = _context(tmp_path)

    with pytest.raises(SecurityError):
        code.search_code("anything", path_glob="../*.py", context=context)


def test_git_diff_rejects_option_like_ref(tmp_path: Path) -> None:
    context = _context(tmp_path)

    with pytest.raises(SecurityError):
        code.git_diff(ref="--output=/tmp/leak", context=context)
