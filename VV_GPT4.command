#!/bin/bash
cd "$(dirname "$0")" || exit 1
if [ ! -x .venv/bin/python ]; then
  echo "Set up VV-GPT4 first. See README.md in this folder."
  read -r -p "Press Enter to close."
  exit 1
fi
if [ -d /Applications/Ollama.app ]; then
  open -g -a Ollama
fi
exec .venv/bin/python launcher.py "$@"
