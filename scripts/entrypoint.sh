#!/bin/bash

set -e

echo ">>> [QLab] Starting container setup..."

if ! command -v python >/dev/null 2>&1; then
    echo "ERROR: Python is not installed!"
    exit 1
fi

if ! python -c "import sys; sys.exit(0 if sys.version_info >= (3,11) else 1)"; then
    echo "ERROR: Python 3.11+ required!"
    exit 1
fi


if [ -f "requirements.txt" ]; then
    echo ">>> Checking required Python packages..."
    MISSING=0
    while IFS= read -r pkg; do
        [[ -z "$pkg" || "$pkg" =~ ^# ]] && continue
        pkg_name=$(echo "$pkg" | cut -d= -f1 | tr -d ' ')
        echo " - Checking: $pkg_name"
        if ! python -c "import importlib.util, sys; sys.exit(0 if importlib.util.find_spec('$pkg_name') else 1)" ; then
            echo "   Missing: $pkg_name"
            MISSING=1
        fi
    done < requirements.txt

    if [ $MISSING -eq 1 ]; then
        echo "ERROR: Some Python dependencies are missing."
        exit 1
    fi
fi


echo ">>> Checking system libraries..."
for lib in libGL.so.1 libX11.so.6; do
    if ! ldconfig -p | grep -q $lib; then
        echo "WARNING: $lib not found. GUI features may not work properly."
    fi
done

if [ "$1" = "--check" ]; then
    echo ">>> Checks complete. Exiting (--check mode)."
    exit 0
fi

echo ">>> All checks passed. Starting QLab..."
if [ $# -eq 0 ]; then
    exec python -m src
else
    exec "$@"
fi
