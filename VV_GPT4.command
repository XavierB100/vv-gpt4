#!/bin/bash
cd "$(dirname "$0")" || exit 1
if [ ! -x .venv/bin/python ]; then
  echo "Set up VV-GPT4 first. See README.md in this folder."
  read -r -p "Press Enter to close."
  exit 1
fi
exec .venv/bin/python launcher.py "$@"
