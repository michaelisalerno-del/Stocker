#!/bin/sh
set -eu

config_path="${IBGATEWAY_PROXY_CONFIG:-/etc/ibgateway/loopback-proxy.env}"
systemctl_bin="${IBGATEWAY_SYSTEMCTL:-/usr/bin/systemctl}"
ss_bin="${IBGATEWAY_SS:-/usr/bin/ss}"
sleep_bin="${IBGATEWAY_SLEEP:-/usr/bin/sleep}"
attempts="${IBGATEWAY_READINESS_ATTEMPTS:-60}"
interval_seconds="${IBGATEWAY_READINESS_INTERVAL_SECONDS:-2}"

fail_config() {
    printf '%s\n' \
        "blocked_unsafe_runtime_configuration: ibgateway_daily_readiness:$1" >&2
    exit 78
}

read_upstream_port() {
    [ -r "$config_path" ] || fail_config "config_missing"
    [ "$(wc -l < "$config_path")" -eq 1 ] ||
        fail_config "config_must_have_one_line"

    IFS= read -r config_line < "$config_path" || fail_config "config_unreadable"
    case "$config_line" in
        IBGATEWAY_UPSTREAM_PORT=*) upstream_port="${config_line#*=}" ;;
        *) fail_config "unexpected_config_field" ;;
    esac
    case "$upstream_port" in
        "" | *[!0-9]*) fail_config "upstream_port_must_be_numeric" ;;
        22 | 80 | 443 | 4003) fail_config "upstream_port_reserved" ;;
    esac
    [ "$upstream_port" -ge 1 ] 2>/dev/null ||
        fail_config "upstream_port_out_of_range"
    [ "$upstream_port" -le 65535 ] 2>/dev/null ||
        fail_config "upstream_port_out_of_range"
}

require_positive_integer() {
    case "$1" in
        "" | *[!0-9]* | 0) fail_config "$2" ;;
    esac
}

read_upstream_port
require_positive_integer "$attempts" "attempts_must_be_positive"
require_positive_integer "$interval_seconds" "interval_must_be_positive"
[ -x "$systemctl_bin" ] || fail_config "systemctl_missing"
[ -x "$ss_bin" ] || fail_config "ss_missing"
[ -x "$sleep_bin" ] || fail_config "sleep_missing"

attempt=1
last_reason="gateway_inactive"
while [ "$attempt" -le "$attempts" ]; do
    if "$systemctl_bin" is-active --quiet stocker-ibgateway.service; then
        last_reason="api_port_not_ready"
        if "$ss_bin" -H -ltn "sport = :$upstream_port" 2>/dev/null |
            awk -v endpoint=":$upstream_port" \
                '$1 == "LISTEN" && $4 ~ (endpoint "$") { found = 1 } END { exit !found }'
        then
            printf 'ibgateway_daily_restart:ready:%s\n' "$upstream_port"
            exit 0
        fi
    fi

    if [ "$attempt" -lt "$attempts" ]; then
        "$sleep_bin" "$interval_seconds"
    fi
    attempt=$((attempt + 1))
done

printf 'ibgateway_daily_restart:%s:%s\n' "$last_reason" "$upstream_port" >&2
exit 1
