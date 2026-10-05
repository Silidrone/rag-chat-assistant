#!/bin/sh
# Build and start the production stack behind the shared Traefik proxy on
# codeise. Requires docker-prod/.env to exist on the server.
set -e
cd "$(dirname "$0")"

docker compose -p ragchatassistant -f docker-compose.yml --env-file .env up --build -d
echo "rag-chat-assistant is up."
echo "  Site:     https://ragchat.silidrone.com"
echo "  Unlisted: https://8rh5yx2gibyk.kaldify.com"
