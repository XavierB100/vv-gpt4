"""Optional machine-local paths. No original checkout is required to run the studio."""
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def legacy_models():
    override = os.environ.get('VVGPT_LEGACY_MODELS')
    if override:
        return Path(override).expanduser().resolve()
    local = ROOT / 'local_config.json'
    config = json.loads(local.read_text()) if local.exists() else {}
    return Path(config.get('legacy_models', ROOT.parent / 'vv-gpt3' / 'models')).expanduser().resolve()
