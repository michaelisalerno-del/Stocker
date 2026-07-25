#!/bin/sh
set -eu

active_link="${IBGATEWAY_ACTIVE_LINK:-/opt/ibgateway/current}"
installer_root="${IBGATEWAY_INSTALLER_ROOT:-/var/lib/ibgateway/installers}"
release_root="${IBGATEWAY_RELEASE_ROOT:-/opt/ibgateway/releases}"
expected_owner="${IBGATEWAY_EXPECTED_OWNER:-root}"

fail() {
    printf '%s\n' \
        "blocked_unsafe_runtime_configuration: ibgateway_integrity:$1" >&2
    exit 1
}

read_field() {
    awk -F= -v wanted="$1" '
        $1 == wanted {
            count += 1
            value = substr($0, length($1) + 2)
        }
        END {
            if (count != 1 || value == "") {
                exit 1
            }
            print value
        }
    ' "$provenance"
}

[ -L "$active_link" ] || fail "active_pointer_missing"
active_path="$(readlink -f -- "$active_link")" || fail "active_pointer_unresolvable"
case "$active_path" in
    "$release_root"/*) ;;
    *) fail "active_path_outside_release_root" ;;
esac

identity="$(basename -- "$active_path")"
provenance="$installer_root/$identity.runtime-provenance"
manifest="$installer_root/$identity.manifest.sha256"

[ -f "$provenance" ] || fail "provenance_missing"
[ -f "$manifest" ] || fail "file_manifest_missing"
[ -x "$active_path/ibgateway" ] || fail "launcher_missing"

manifest_version="$(read_field manifest_version)" ||
    fail "invalid_provenance_manifest_version"
source_url="$(read_field source_url)" || fail "invalid_provenance_source_url"
installer_path="$(read_field installer_path)" || fail "invalid_provenance_installer_path"
installer_sha256="$(read_field installer_sha256)" ||
    fail "invalid_provenance_installer_sha256"
installed_path="$(read_field installed_path)" || fail "invalid_provenance_installed_path"
recorded_manifest="$(read_field file_manifest_path)" ||
    fail "invalid_provenance_manifest_path"
manifest_sha256="$(read_field file_manifest_sha256)" ||
    fail "invalid_provenance_manifest_sha256"

[ "$manifest_version" = "1" ] || fail "unsupported_provenance_version"
[ "$source_url" = \
    "https://download2.interactivebrokers.com/installers/ibgateway/latest-standalone/ibgateway-latest-standalone-linux-x64.sh" ] ||
    fail "unapproved_source_url"
[ "$installed_path" = "$active_path" ] || fail "active_path_mismatch"
[ "$recorded_manifest" = "$manifest" ] || fail "manifest_path_mismatch"
case "$installer_path" in
    "$installer_root"/*) ;;
    *) fail "installer_path_outside_archive_root" ;;
esac
[ -f "$installer_path" ] || fail "installer_archive_missing"

[ "${#installer_sha256}" -eq 64 ] || fail "invalid_installer_sha256"
[ "${#manifest_sha256}" -eq 64 ] || fail "invalid_manifest_sha256"
case "$installer_sha256" in
    *[!0-9a-f]*) fail "invalid_installer_sha256" ;;
esac
case "$manifest_sha256" in
    *[!0-9a-f]*) fail "invalid_manifest_sha256" ;;
esac

actual_installer_sha256="$(sha256sum "$installer_path" | awk '{print $1}')"
[ "$actual_installer_sha256" = "$installer_sha256" ] ||
    fail "installer_hash_mismatch"
actual_manifest_sha256="$(sha256sum "$manifest" | awk '{print $1}')"
[ "$actual_manifest_sha256" = "$manifest_sha256" ] ||
    fail "manifest_hash_mismatch"

if find "$active_path" -xdev ! -user "$expected_owner" -print -quit | grep -q .; then
    fail "installed_file_has_wrong_owner"
fi
if find "$active_path" -xdev \( -type f -o -type d \) \
    \( -perm -020 -o -perm -002 \) -print -quit | grep -q .
then
    fail "installed_file_group_or_other_writable"
fi
(
    cd "$active_path"
    sha256sum --quiet --strict -c "$manifest"
) || fail "installed_file_hash_mismatch"

printf 'IB Gateway integrity verified: %s\n' "$identity"
