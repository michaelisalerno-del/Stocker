def test_core_packages_import() -> None:
    import stocker_backtest
    import stocker_core
    import stocker_data
    import stocker_execution
    import stocker_research

    assert stocker_core.__version__
    assert stocker_data.__version__
    assert stocker_research.__version__
    assert stocker_backtest.__version__
    assert stocker_execution.__version__
