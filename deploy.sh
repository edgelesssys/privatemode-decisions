#!/bin/sh
# Deploy with docker compose to a host over ssh: sync tracked files, keep
# the host's .env, rebuild, wait, check. The app binds 127.0.0.1:8600 on the
# host; put a reverse proxy with TLS in front of it.
#   HOST=myhost DIR=privatemode-decisions sh deploy.sh
set -eu
HOST="${HOST:?set HOST to the ssh host}"; DIR="${DIR:-privatemode-decisions}"
cd "$(dirname "$0")"
.venv/bin/pytest -q tests
git ls-files > /tmp/privatemode-decisions.files
ssh "$HOST" "mkdir -p $DIR"
rsync -az --files-from=/tmp/privatemode-decisions.files ./ "$HOST:$DIR/"
ssh "$HOST" "test -f $DIR/.env || { echo 'no $DIR/.env on the host: copy .env.example and fill it in'; exit 1; }"
ssh "$HOST" "cd $DIR && docker compose up --build -d --remove-orphans 2>&1 | tail -2"
ssh "$HOST" "for i in \$(seq 1 30); do curl -sf http://127.0.0.1:8600/api/models >/dev/null && exit 0; sleep 2; done; echo 'did not come up'; docker compose -f $DIR/docker-compose.yml logs --tail 30; exit 1"
echo "deployed to $HOST:$DIR, listening on 127.0.0.1:8600 there"
