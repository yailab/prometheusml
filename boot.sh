#!/bin/bash
# this script is used to boot a Docker container
while true; do
    flask db upgrade
    if [[ "$?" == "0" ]]; then
        flask load_roles_db;
        break
    fi
    echo Deploy command failed, retrying in 5 secs...
    sleep 5
done

# Change it to the following so that can override the command in docker-compose
exec "$@"
