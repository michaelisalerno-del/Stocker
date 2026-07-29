#!/bin/sh
set -eu

config_path="${IBGATEWAY_PROXY_CONFIG:-/etc/ibgateway/loopback-proxy.env}"
socket_proxyd="${IBGATEWAY_SOCKET_PROXYD:-/usr/lib/systemd/systemd-socket-proxyd}"

fail() {
    printf '%s\n' \
        "blocked_unsafe_runtime_configuration: ibgateway_loopback_proxy:$1" >&2
    exit 78
}

[ -r "$config_path" ] || fail "config_missing"
[ "$(wc -l < "$config_path")" -eq 1 ] || fail "config_must_have_one_line"

IFS= read -r config_line < "$config_path" || fail "config_unreadable"
case "$config_line" in
    IBGATEWAY_UPSTREAM_PORT=*) upstream_port="${config_line#*=}" ;;
    *) fail "unexpected_config_field" ;;
esac
case "$upstream_port" in
    "" | *[!0-9]*) fail "upstream_port_must_be_numeric" ;;
esac
[ "$upstream_port" -ge 1 ] 2>/dev/null || fail "upstream_port_out_of_range"
[ "$upstream_port" -le 65535 ] 2>/dev/null || fail "upstream_port_out_of_range"
case "$upstream_port" in
    22 | 80 | 443 | 4003) fail "upstream_port_reserved" ;;
esac
[ -x "$socket_proxyd" ] || fail "systemd_socket_proxyd_missing"

exec "$socket_proxyd" --connections-max=4 "127.0.0.1:$upstream_port"
