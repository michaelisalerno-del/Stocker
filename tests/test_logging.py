from stocker_core.logging import configure_logging


def test_configured_logger_accepts_structured_keyword_fields() -> None:
    logger = configure_logging()

    logger.warning("event_name", component="test")
