set -eu

config_path="${IBGATEWAY_PROXY_CONFIG:-/etc/ibgateway/loopback-proxy.env}"
ufw_bin="${IBGATEWAY_UFW:-/usr/sbin/ufw}"
iptables_bin="${IBGATEWAY_IPTABLES:-/usr/sbin/iptables}"
ip6tables_bin="${IBGATEWAY_IP6TABLES:-/usr/sbin/ip6tables}"

fail() {
    printf '%s\n' \
        "blocked_unsafe_runtime_configuration: ibgateway_loopback_boundary:$1" >&2
    exit 78
}

read_upstream_port() {
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
}

require_executable() {
    [ -x "$1" ] || fail "$2"
}

require_rule_token() {
    padded_rule=" $1 "
    expected_token="$2"
    failure_reason="$3"
    case "$padded_rule" in
        *" $expected_token "*) ;;
        *) fail "$failure_reason" ;;
    esac
}

verify_chain() {
    family="$1"
    chain="$2"
    command_path="$3"

    rules="$("$command_path" -S "$chain" 2>/dev/null)" ||
        fail "${family}_rules_unavailable"
    first_rule="$(
        printf '%s\n' "$rules" |
            awk '/^-A / { print; exit }'
    )"
    second_rule="$(
        printf '%s\n' "$rules" |
            awk '/^-A / { count += 1; if (count == 2) { print; exit } }'
    )"

    require_rule_token \
        "$first_rule" "-A $chain" "${family}_first_rule_not_loopback_allow"
    require_rule_token \
        "$first_rule" "-i lo" "${family}_first_rule_not_loopback_allow"
    require_rule_token \
        "$first_rule" "-p tcp" "${family}_first_rule_not_loopback_allow"
    require_rule_token \
        "$first_rule" "--dport $upstream_port" \
        "${family}_first_rule_not_loopback_allow"
    require_rule_token \
        "$first_rule" "-j ACCEPT" "${family}_first_rule_not_loopback_allow"

    second_padded=" $second_rule "
    case "$second_padded" in
        *" -i "*) fail "${family}_second_rule_not_non_loopback_deny" ;;
    esac
    require_rule_token \
        "$second_rule" "-A $chain" "${family}_second_rule_not_non_loopback_deny"
    require_rule_token \
        "$second_rule" "-p tcp" "${family}_second_rule_not_non_loopback_deny"
    require_rule_token \
        "$second_rule" "--dport $upstream_port" \
        "${family}_second_rule_not_non_loopback_deny"
    require_rule_token \
        "$second_rule" "-j DROP" "${family}_second_rule_not_non_loopback_deny"
}

read_upstream_port
require_executable "$ufw_bin" "ufw_missing"
require_executable "$iptables_bin" "iptables_missing"
require_executable "$ip6tables_bin" "ip6tables_missing"

ufw_status="$("$ufw_bin" status 2>/dev/null)" || fail "ufw_status_unavailable"
[ "$(printf '%s\n' "$ufw_status" | sed -n '1p')" = "Status: active" ] ||
    fail "ufw_inactive"

verify_chain "ipv4" "ufw-user-input" "$iptables_bin"
verify_chain "ipv6" "ufw6-user-input" "$ip6tables_bin"

printf 'ibgateway_loopback_boundary:verified:%s\n' "$upstream_port"
