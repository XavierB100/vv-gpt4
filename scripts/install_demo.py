"""Explicit, checksum-verified installation of the public VV-GPT story checkpoint."""
import argparse
import hashlib
import json
import os
import shutil
import sys
import tempfile
import urllib.request
import zipfile
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from vvgpt.config import ROOT
from vvgpt.lab import load_checkpoint
from vvgpt.model import GPT, GPTConfig
from vvgpt.store import Store

MANIFEST = ROOT / 'examples/demo.json'
LIMIT = 32 * 1024 * 1024


def install(archive, store, manifest):
    archive = Path(archive)
    if archive.stat().st_size > LIMIT or hashlib.sha256(archive.read_bytes()).hexdigest() != manifest['sha256']:
        raise ValueError('Demo checksum mismatch. No files installed.')
    rid = manifest['run_id']
    if len(rid) != 32 or any(c not in '0123456789abcdef' for c in rid):
        raise ValueError('Invalid demo identifier.')
    destination = store.root / 'runs' / rid
    existing = store.rows('SELECT id FROM runs WHERE id=?', (rid,))
    if existing:
        checkpoint = destination / 'best.pt'
        if checkpoint.is_file() and hashlib.sha256(checkpoint.read_bytes()).hexdigest() == manifest['checkpoint_sha256']:
            return False
        raise ValueError('A different run already uses this identifier. Use an empty VVGPT_DATA directory.')
    if destination.exists():
        raise ValueError('A run folder already exists. It has been left untouched.')
    staging = Path(tempfile.mkdtemp(prefix='.demo-', dir=store.root/'runs'))
    installed = False
    try:
        with zipfile.ZipFile(archive) as package:
            names = package.namelist()
            if len(names) != 3 or set(names) != {'best.pt', 'summary.json', 'record.json'}:
                raise ValueError('Unexpected demo contents.')
            if sum(x.file_size for x in package.infolist()) > LIMIT:
                raise ValueError('Demo exceeds the size limit.')
            for name in names:
                (staging/name).write_bytes(package.read(name))
        if hashlib.sha256((staging/'best.pt').read_bytes()).hexdigest() != manifest['checkpoint_sha256']:
            raise ValueError('Checkpoint checksum mismatch.')
        checkpoint = load_checkpoint(staging/'best.pt')
        GPT(GPTConfig(**checkpoint['config'])).load_state_dict(checkpoint['model'])
        record = json.loads((staging/'record.json').read_text())
        run = record['run']
        if run['id'] != rid or run['status'] != 'completed':
            raise ValueError('Invalid demo run record.')
        with store.db() as db:
            db.execute('BEGIN IMMEDIATE')
            db.execute('INSERT INTO runs(id,name,config,status,step,best,created,updated) VALUES(?,?,?,?,?,?,?,?)',
                       tuple(run[k] for k in ('id','name','config','status','step','best','created','updated')))
            for metric in record['metrics']:
                db.execute('INSERT INTO metrics VALUES(?,?,?,?,?,?)', (rid, *[metric[k] for k in ('step','train','val','lr','elapsed')]))
            staging.rename(destination)
            installed = True
        return True
    except BaseException:
        if installed:
            shutil.rmtree(destination)
        raise
    finally:
        if staging.exists():
            shutil.rmtree(staging)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive', type=Path, help='Use an already downloaded official archive')
    parser.add_argument('--data', type=Path, default=Path(os.environ.get('VVGPT_DATA', ROOT/'runtime')))
    args = parser.parse_args()
    manifest = json.loads(MANIFEST.read_text())
    with tempfile.TemporaryDirectory(prefix='vvgpt-download-') as temporary:
        archive = args.archive
        if archive is None:
            print('Downloading the optional public story model (about 13 MB)…', flush=True)
            archive = Path(temporary)/'demo.zip'
            with urllib.request.urlopen(manifest['url'], timeout=60) as response, archive.open('wb') as output:
                received = 0
                while chunk := response.read(1024*1024):
                    received += len(chunk)
                    if received > LIMIT:
                        raise ValueError('Download exceeds the demo size limit.')
                    output.write(chunk)
        changed = install(archive, Store(args.data), manifest)
    print('Demo installed.' if changed else 'Demo already installed; existing run preserved.')
    print('Open the studio, then select Simple stories / further learning. No training data was installed.')


if __name__ == '__main__':
    main()
