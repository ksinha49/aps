#!/bin/bash
set -euo pipefail

# RHEL on-premise installation script
INSTALL_DIR="/opt/scout-ai"

echo "Installing Scout AI by Ameritas to ${INSTALL_DIR}..."

# Create service user
id -u scout &>/dev/null || useradd -r -s /sbin/nologin scout

# Create directories
mkdir -p "${INSTALL_DIR}"
mkdir -p /var/log/scout-ai
mkdir -p /data/indexes

# Create env file with secure permissions BEFORE setting ownership
touch "${INSTALL_DIR}/.env"
chmod 600 "${INSTALL_DIR}/.env"
echo "# Configure SCOUT_* environment variables here" > "${INSTALL_DIR}/.env"

# Create virtualenv and install
python3 -m venv "${INSTALL_DIR}/venv"
"${INSTALL_DIR}/venv/bin/pip" install --upgrade pip
"${INSTALL_DIR}/venv/bin/pip" install scout-ai[api,otel]

# Copy systemd unit
cp deploy/rhel/systemd/app.service /etc/systemd/system/scout-ai.service

# Copy logrotate config
cp deploy/rhel/logrotate.conf /etc/logrotate.d/scout-ai

# Set ownership
chown -R scout:scout "${INSTALL_DIR}" /data/indexes /var/log/scout-ai

# Enable and start
systemctl daemon-reload
systemctl enable scout-ai
systemctl start scout-ai

echo "Scout AI by Ameritas installed and started."
echo "Check status: systemctl status scout-ai"
echo "View logs: journalctl -u scout-ai -f"
