#!/bin/sh
set -eu

config_path="${IBGATEWAY_PROXY_CONFIG:-/etc/ibgateway/loopback-proxy.env}"
nft_bin="${IBGATEWAY_NFT:-/usr/sbin/nft}"

fail() {
    printf '%s\n' \
        "blocked_unsafe_runtime_configuration: ibgateway_boundary_install:$1" >&2
    exit 78
}

[ -r "$config_path" ] || fail "config_missing"
[ "$(wc -l < "$config_path")" -eq 1 ] ||
    fail "config_must_have_one_line"

IFS= read -r config_line < "$config_path" || fail "config_unreadable"
case "$config_line" in
    IBGATEWAY_UPSTREAM_PORT=*) upstream_port="${config_line#*=}" ;;
    *) fail "unexpected_config_field" ;;
esac
case "$upstream_port" in
    "" | *[!0-9]*) fail "upstream_port_must_be_numeric" ;;
    22 | 80 | 443 | 4003) fail "upstream_port_reserved" ;;
esac
[ "$upstream_port" -ge 1 ] 2>/dev/null ||
    fail "upstream_port_out_of_range"
[ "$upstream_port" -le 65535 ] 2>/dev/null ||
    fail "upstream_port_out_of_range"
[ -x "$nft_bin" ] || fail "nft_missing"

rules_path="$(mktemp /tmp/stocker-ibgateway-nft.XXXXXX)"
trap 'rm -f -- "$rules_path"' EXIT HUP INT TERM

if "$nft_bin" list table inet stocker_ibgateway >/dev/null 2>&1; then
    printf '%s\n' \
        "delete table inet stocker_ibgateway" > "$rules_path"
else
    : > "$rules_path"
fi
{
    printf '%s\n' \
        "add table inet stocker_ibgateway" \
        "add chain inet stocker_ibgateway input { type filter hook input priority -300; policy accept; }" \
        "add rule inet stocker_ibgateway input iifname \"lo\" tcp dport $upstream_port accept" \
        "add rule inet stocker_ibgateway input tcp dport $upstream_port drop"
} >> "$rules_path"

"$nft_bin" --check -f "$rules_path" >/dev/null 2>&1 ||
    fail "nft_rules_check_failed"
"$nft_bin" -f "$rules_path" >/dev/null 2>&1 ||
    fail "nft_rules_install_failed"

printf 'ibgateway_boundary_install:installed:%s\n' "$upstream_port"
