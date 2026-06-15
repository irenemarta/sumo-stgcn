#!/bin/bash
# INFO ON SET COMMAND: https://gist.github.com/mohanpedala/1e2ff5661761d3abd0385e8223e16425?permalink_comment_id=3799230
set -e # blocks script execution in case of exception
set -u # checks if referred variable have been defined
set -o pipefail # returns error if any command in a pipeline fails
set -x # executable mode

# installing uv
curl -LsSf https://astral.sh/uv/install.sh | sh

uv sync

check_cuda(){
    # verify the existence of GPU and redirect output to folder that discards received data
    # both stdout and stderr are adresses to /dev/null
    command -v nvidia-smi >/dev/null 2>&1 || return 1 
    nvidia-smi -L > /dev/null 2>&1 || return 1

    count=$(nvidia-smi -L | wc -l) # list NVIDIA file and count lines (1 line = 1 GPU)
    if [ "$count" -ge 1 ]; then 
        return 0
    else 
        return 1
    fi
}

if check_cuda; then
    echo "CUDA GPU available. Installing dependencies"
    uv sync --extra cuda # uv does not sync extras by default
    echo "CUDA dependencies installed."
else
    echo "No CUDA GPU found. Installing CPU dependencies"
    uv sync --extra cpu
    echo "CPU dependencies installed."
fi