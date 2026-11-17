"""Launch only this checkout, reusing its server and choosing a free local port."""
import argparse
import json
import os
import socket
import subprocess
import sys
import time
import urllib.request
import webbrowser
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def identity(port):
    try:
        with urllib.request.build_opener(urllib.request.ProxyHandler({})).open(f'http://127.0.0.1:{port}/health', timeout=.3) as response:
            body = json.load(response)
        data_root=Path(os.environ.get('VVGPT_DATA', ROOT/'runtime')).resolve()
        return body.get('app') == 'VV-GPT4' and body.get('root') == str(ROOT) and body.get('data') == str(data_root)
    except (OSError, ValueError):
        return False


def launch(no_browser=False):
    preferred = int(os.environ.get('VVGPT_PORT', '5054'))
    free = None
    for port in range(preferred, min(preferred+20,65536)):
        if identity(port):
            url=f'http://127.0.0.1:{port}'
            print(f'VV-GPT4 is ready: {url}', flush=True)
            if not no_browser: webbrowser.open(url)
            return 0
        if free is None:
            with socket.socket() as sock:
                try:
                    sock.bind(('127.0.0.1',port)); free=port
                except OSError:
                    pass
    if free is None:
        raise RuntimeError('No free port found. Set VVGPT_PORT to another starting port.')
    env=dict(os.environ,VVGPT_PORT=str(free))
    proc=subprocess.Popen([sys.executable,str(ROOT/'app.py')],cwd=ROOT,env=env)
    try:
        for _ in range(150):
            if proc.poll() is not None:
                raise RuntimeError('The server exited. Check the error above.')
            if identity(free): break
            time.sleep(.2)
        else: raise RuntimeError('Server did not become ready within 30 seconds.')
        url=f'http://127.0.0.1:{free}'
        print(f'VV-GPT4 is ready: {url}\nKeep this Terminal open. Ctrl+C closes the web server; training workers continue.',flush=True)
        if not no_browser and not webbrowser.open(url): print(f'Open {url} in your browser.',flush=True)
        return proc.wait()
    except KeyboardInterrupt:
        return 0
    finally:
        if proc.poll() is None:
            proc.terminate()
            try: proc.wait(timeout=5)
            except subprocess.TimeoutExpired: proc.kill()


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--no-browser',action='store_true')
    sys.exit(launch(parser.parse_args().no_browser))
