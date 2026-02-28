#!/bin/bash
set -euo pipefail

# RHEL on-premise installation script
INSTALL_DIR="/opt/pageindex-rag"

echo "Installing PageIndex RAG to ${INSTALL_DIR}..."

# Create service user
id -u pageindex &>/dev/null || useradd -r -s /sbin/nologin pageindex

# Create directories
mkdir -p "${INSTALL_DIR}"
mkdir -p /var/log/pageindex-rag
mkdir -p /data/indexes

# Create virtualenv and install
python3 -m venv "${INSTALL_DIR}/venv"
"${INSTALL_DIR}/venv/bin/pip" install --upgrade pip
"${INSTALL_DIR}/venv/bin/pip" install pageindex-rag[api,otel]

# Copy systemd unit
cp deploy/rhel/systemd/app.service /etc/systemd/system/pageindex-rag.service

# Copy logrotate config
cp deploy/rhel/logrotate.conf /etc/logrotate.d/pageindex-rag

# Set ownership
chown -R pageindex:pageindex "${INSTALL_DIR}" /data/indexes /var/log/pageindex-rag

# Enable and start
systemctl daemon-reload
systemctl enable pageindex-rag
systemctl start pageindex-rag

echo "PageIndex RAG installed and started."
echo "Check status: systemctl status pageindex-rag"
echo "View logs: journalctl -u pageindex-rag -f"
