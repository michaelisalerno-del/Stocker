import os

from stocker_mcp.server import connector_info, tool_metadata


def test_connector_info_excludes_token_value(monkeypatch) -> None:
    monkeypatch.setenv("STOCKER_MCP_TOKEN", "super-secret-token")

    info = connector_info(auth_token_env="STOCKER_MCP_TOKEN")

    assert info["local_url"] == "http://127.0.0.1:8765/mcp"
    assert info["connector_name"] == "Stocker Research"
    assert info["auth"]["header"] == "Authorization: Bearer <STOCKER_MCP_TOKEN>"
    assert "super-secret-token" not in str(info)
    assert "search" in info["tools"]
    assert "fetch" in info["tools"]
    assert info["security_mode"] == "read-only"


def test_connector_info_reports_auth_env_presence(monkeypatch) -> None:
    monkeypatch.delenv("STOCKER_MCP_TOKEN", raising=False)

    missing = connector_info(auth_token_env="STOCKER_MCP_TOKEN")
    monkeypatch.setenv("STOCKER_MCP_TOKEN", "present")
    present = connector_info(auth_token_env="STOCKER_MCP_TOKEN")

    assert missing["auth"]["env_var_set"] is False
    assert present["auth"]["env_var_set"] is True
    assert os.environ["STOCKER_MCP_TOKEN"] not in str(present)


def test_all_tools_have_chatgpt_safe_metadata() -> None:
    metadata = tool_metadata()

    assert metadata
    for item in metadata:
        assert item["name"]
        assert item["title"]
        assert item["description"]
        assert item["annotations"]["readOnlyHint"] is True
        assert item["annotations"]["destructiveHint"] is False
        assert item["annotations"]["openWorldHint"] is False
        lowered = item["name"].lower()
        assert "delete" not in lowered
        assert "order" not in lowered
        assert "broker" not in lowered
