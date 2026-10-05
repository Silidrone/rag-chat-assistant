#!/bin/sh
set -e
cd "$(dirname "$0")"
docker compose -p ragchatassistant -f docker-compose.yml --env-file .env down --remove-orphans
echo "rag-chat-assistant is down."
