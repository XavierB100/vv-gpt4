"""Create a separate VV-GPT4 launcher. Refuse to replace another application."""
import plistlib
import shlex
import shutil
from pathlib import Path

root=Path(__file__).resolve().parents[1]
app=Path.home()/'Desktop'/'VV-GPT4 Preview.app'
if app.exists():
    raise SystemExit(f'Already exists; left untouched: {app}')
macos=app/'Contents'/'MacOS'
macos.mkdir(parents=True)
launcher=macos/'launch'
launcher.write_text('#!/bin/bash\nexec open -a Terminal '+shlex.quote(str(root/'VV_GPT4.command'))+'\n')
launcher.chmod(0o755)
resources=app/'Contents'/'Resources'
resources.mkdir()
shutil.copy2(root/'static'/'VVGPT.icns', resources/'VVGPT.icns')
with (app/'Contents'/'Info.plist').open('wb') as file:
    plistlib.dump({'CFBundleName':'VV-GPT4 Preview','CFBundleDisplayName':'VV-GPT4 Preview',
                  'CFBundleIdentifier':'local.xavier.vvgpt4.preview','CFBundleVersion':'4.0.0',
                  'CFBundleShortVersionString':'4.0.0','CFBundleExecutable':'launch',
                  'CFBundleIconFile':'VVGPT',
                  'CFBundlePackageType':'APPL'},file)
print(app)
