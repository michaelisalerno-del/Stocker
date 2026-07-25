#!/bin/sh
set -eu

database_directory="/var/lib/stocker/prospective"
database_path="$database_directory/prospective.sqlite3"
wal_path="$database_path-wal"
shm_path="$database_path-shm"

fail() {
    printf '%s\n' \
        "blocked_unsafe_runtime_configuration: web_sqlite_boundary:$1" >&2
    exit 78
}

[ "$(id -u)" -eq 0 ] || fail "root_required"
getent passwd stocker >/dev/null 2>&1 || fail "recorder_user_missing"
getent passwd stocker-web >/dev/null 2>&1 || fail "web_user_missing"
getent group stocker >/dev/null 2>&1 || fail "shared_group_missing"

[ -d "$database_directory" ] || fail "database_directory_missing"
[ ! -L "$database_directory" ] || fail "database_directory_symlink"
chown stocker:stocker "$database_directory"
chmod 0750 "$database_directory"

[ -f "$database_path" ] || fail "database_missing"
[ ! -L "$database_path" ] || fail "database_symlink"
chown stocker:stocker "$database_path"

prepare_auxiliary_file() {
    auxiliary_path="$1"
    auxiliary_mode="$2"
    auxiliary_label="$3"
    [ ! -L "$auxiliary_path" ] || fail "${auxiliary_label}_symlink"
    if [ ! -e "$auxiliary_path" ]; then
        install -o stocker -g stocker -m "$auxiliary_mode" \
            /dev/null "$auxiliary_path"
    fi
    [ -f "$auxiliary_path" ] || fail "${auxiliary_label}_not_regular"
    chown stocker:stocker "$auxiliary_path"
}

prepare_auxiliary_file "$wal_path" 0640 "wal"
prepare_auxiliary_file "$shm_path" 0660 "shm"
chmod 0640 "$database_path" "$wal_path"
chmod 0660 "$shm_path"

printf '%s\n' "web_sqlite_boundary:verified"
