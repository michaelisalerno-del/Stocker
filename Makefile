SHELL := /bin/bash

.PHONY: bootstrap-mac bootstrap-server check format test

bootstrap-mac:
	bash scripts/bootstrap_mac.sh

bootstrap-server:
	bash scripts/bootstrap_server.sh

check:
	bash scripts/check.sh

format:
	bash scripts/format.sh

test:
	bash scripts/test.sh
