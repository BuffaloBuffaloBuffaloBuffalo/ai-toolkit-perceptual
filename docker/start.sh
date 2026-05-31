#!/bin/bash
set -e  # Exit the script if any statement returns a non-true return value

# ref https://github.com/runpod/containers/blob/main/container-template/start.sh

# ---------------------------------------------------------------------------- #
#                          Function Definitions                                #
# ---------------------------------------------------------------------------- #


# Setup ssh
setup_ssh() {
    if [[ $PUBLIC_KEY ]]; then
        echo "Setting up SSH..."
        mkdir -p ~/.ssh
        echo "$PUBLIC_KEY" >> ~/.ssh/authorized_keys
        chmod 700 -R ~/.ssh

         if [ ! -f /etc/ssh/ssh_host_rsa_key ]; then
            ssh-keygen -t rsa -f /etc/ssh/ssh_host_rsa_key -q -N ''
            echo "RSA key fingerprint:"
            ssh-keygen -lf /etc/ssh/ssh_host_rsa_key.pub
        fi

        if [ ! -f /etc/ssh/ssh_host_dsa_key ]; then
            ssh-keygen -t dsa -f /etc/ssh/ssh_host_dsa_key -q -N ''
            echo "DSA key fingerprint:"
            ssh-keygen -lf /etc/ssh/ssh_host_dsa_key.pub
        fi

        if [ ! -f /etc/ssh/ssh_host_ecdsa_key ]; then
            ssh-keygen -t ecdsa -f /etc/ssh/ssh_host_ecdsa_key -q -N ''
            echo "ECDSA key fingerprint:"
            ssh-keygen -lf /etc/ssh/ssh_host_ecdsa_key.pub
        fi

        if [ ! -f /etc/ssh/ssh_host_ed25519_key ]; then
            ssh-keygen -t ed25519 -f /etc/ssh/ssh_host_ed25519_key -q -N ''
            echo "ED25519 key fingerprint:"
            ssh-keygen -lf /etc/ssh/ssh_host_ed25519_key.pub
        fi

        service ssh start

        echo "SSH host keys:"
        for key in /etc/ssh/*.pub; do
            echo "Key: $key"
            ssh-keygen -lf $key
        done
    fi
}

# Export env vars
export_env_vars() {
    echo "Exporting environment variables..."
    printenv | grep -E '^RUNPOD_|^PATH=|^_=' | awk -F = '{ print "export " $1 "=\"" $2 "\"" }' >> /etc/rp_environment
    echo 'source /etc/rp_environment' >> ~/.bashrc
}

# ---------------------------------------------------------------------------- #
#                               Main Program                                   #
# ---------------------------------------------------------------------------- #


echo "Pod Started"

setup_ssh
export_env_vars

# Symlink output / datasets / models onto the persistent /workspace volume
# so training results survive pod restart. The trainer writes to
# /app/ai-toolkit/<dir> (the UI's default training root); both that path
# and /workspace/<dir> end up at the same persistent location.
link_to_workspace() {
    local name="$1"
    local app_path="/app/ai-toolkit/${name}"
    local ws_path="/workspace/${name}"
    if [ -d /workspace ]; then
        mkdir -p "${ws_path}"
        # Preserve anything that the image shipped in /app/ai-toolkit/${name}
        # by copying it into /workspace on first boot.
        if [ -d "${app_path}" ] && [ ! -L "${app_path}" ]; then
            cp -an "${app_path}/." "${ws_path}/" 2>/dev/null || true
            rm -rf "${app_path}"
        fi
        ln -sfn "${ws_path}" "${app_path}"
        echo "Linked ${app_path} -> ${ws_path}"
    fi
}

# Persist the UI database on /workspace so jobs, settings, and the saved HF
# token survive a pod stop (the container filesystem is wiped on stop; only the
# /workspace volume persists). Prisma points at /app/ai-toolkit/aitk_db.db
# (file:../../aitk_db.db from ui/prisma/schema.prisma); move that file onto the
# volume on first boot and symlink it back. SQLite resolves the symlink, so the
# -wal/-shm side files are created on /workspace too.
link_db_to_workspace() {
    local db_name="aitk_db.db"
    local app_db="/app/ai-toolkit/${db_name}"
    local ws_db="/workspace/${db_name}"
    if [ -d /workspace ]; then
        # First boot only: seed the volume with the schema-pushed db baked into
        # the image. Never overwrite a db that already exists on the volume.
        if [ ! -e "${ws_db}" ] && [ -f "${app_db}" ] && [ ! -L "${app_db}" ]; then
            mv "${app_db}" "${ws_db}"
        fi
        rm -f "${app_db}"
        ln -sfn "${ws_db}" "${app_db}"
        echo "Linked ${app_db} -> ${ws_db}"
    fi
}
link_to_workspace output
link_to_workspace datasets
link_to_workspace models
link_db_to_workspace

# Bring the persisted db up to date with any additive schema changes shipped in
# a newer image. Guarded so a push failure warns instead of aborting startup.
if [ -d /workspace ]; then
    ( cd /app/ai-toolkit/ui && npx prisma db push --skip-generate ) \
        || echo "WARNING: 'prisma db push' failed; continuing with the existing database schema."
fi

echo "Starting AI Toolkit UI..."
cd /app/ai-toolkit/ui && npm run start