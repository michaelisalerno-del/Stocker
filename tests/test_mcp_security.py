from pathlib import Path

import pytest

from stocker_mcp.security import SecurityError, StockerMCPContext, redact_secrets


def test_path_traversal_blocked(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    home = tmp_path / "StockerLocal"
    repo.mkdir()
    home.mkdir()
    outside = tmp_path / "outside.txt"
    outside.write_text("not allowed", encoding="utf-8")
    context = StockerMCPContext(repo_root=repo, stocker_home=home)

    with pytest.raises(SecurityError):
        context.resolve_repo_path("../outside.txt")


def test_reading_env_files_blocked(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    home = tmp_path / "StockerLocal"
    repo.mkdir()
    home.mkdir()
    env_path = repo / ".env"
    env_path.write_text("API_KEY=should-not-leak", encoding="utf-8")
    context = StockerMCPContext(repo_root=repo, stocker_home=home)

    with pytest.raises(SecurityError):
        context.read_text_file(env_path, root=repo)


def test_secret_redaction_masks_sensitive_values() -> None:
    text = "\n".join(
        [
            "api_key=abc123456789",
            "Authorization: Bearer abcdefghijklmnopqrstuvwxyz123456",
            "plain=ok",
        ]
    )

    redacted = redact_secrets(text)

    assert "abc123456789" not in redacted
    assert "abcdefghijklmnopqrstuvwxyz123456" not in redacted
    assert "api_key=[REDACTED]" in redacted
    assert "Authorization: Bearer [REDACTED]" in redacted
    assert "plain=ok" in redacted


def test_secret_redaction_keeps_normal_local_paths() -> None:
    path = "/Users/example/Documents/Codex/2026-06-29-we-are-working-in-my-stocker"

    assert redact_secrets(path) == path
