#!/usr/bin/bash

export SERVER_NAME="$(hostname)"
export SERVER_PORT="2222"

# the SERVER_ADDRESS env var overrides the client config value : server_address
export SERVER_ADDRESS="${SERVER_NAME}:${SERVER_PORT}"

# the SERVICE env var overrides the server config value : server_address
export SERVICE="0.0.0.0:${SERVER_PORT}"

PYBISCUS_CONF_PATH=configs/linearregression

