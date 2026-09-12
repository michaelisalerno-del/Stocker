"""Read-only PAPER dual-feed diagnostic, with the production numerical profile."""

from stocker_launcher import configure_numeric_runtime


def main() -> None:
    configure_numeric_runtime()
    from stocker_execution.dual_feed_operator import main as diagnostic_main

    diagnostic_main()


if __name__ == "__main__":
    main()
