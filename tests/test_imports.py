def test_core_packages_import() -> None:
    import stocker_backtest
    import stocker_core
    import stocker_data
    import stocker_research
    import stocker_runtime

    assert stocker_core.__version__
    assert stocker_data.__version__
    assert stocker_research.__version__
    assert stocker_backtest.__version__
    assert stocker_runtime.__version__


def test_console_launcher_imports() -> None:
    import stocker_launcher

    assert callable(stocker_launcher.main)


def test_console_help_describes_prospective_evaluation_not_execution() -> None:
    from typer.testing import CliRunner

    from stocker_core.cli import app

    result = CliRunner().invoke(app, ["--help"])

    assert result.exit_code == 0
    assert "Stocker research and prospective evaluation utilities." in result.stdout
    assert "research and execution utilities" not in result.stdout
