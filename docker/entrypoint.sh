#!/bin/bash
set -e

# Run any pre-start hooks
if [ -f /app/pre-start.sh ]; then
    echo "Running pre-start script..."
    source /app/pre-start.sh
fi

exec "$@"
