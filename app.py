import json
import os
import secrets
import threading
import time
from pathlib import Path
from urllib.parse import urlsplit

from flask import Flask, Response, jsonify, render_template, request, stream_with_context
from werkzeug.exceptions import HTTPException

from vvgpt.store import Store, uid
from vvgpt.knowledge import add_document, context_messages
from vvgpt.providers import LocalProvider
from vvgpt import lab

ROOT = Path(__file__).resolve().parent


def create_app(data_dir=None, provider=None):
    app = Flask(__name__)
    app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024
    store = Store(data_dir or os.environ.get('VVGPT_DATA', ROOT/'runtime'))
    local = provider or LocalProvider()
    csrf = secrets.token_urlsafe(32)
    active, guard, compute = {}, threading.Lock(), threading.Lock()
    lab.reconcile(store)
    app.extensions.update(store=store, provider=local, csrf=csrf)

    @app.before_request
    def local_only():
        host = request.host.split(':')[0]
        if host not in ('127.0.0.1', 'localhost'):
            return jsonify(error='This workspace accepts localhost connections only.'), 403
        if request.method in ('POST', 'PUT', 'DELETE', 'PATCH'):
            if request.headers.get('X-VV-Token') != csrf:
                return jsonify(error='Refresh this page before trying again.'), 403
            origin = request.headers.get('Origin')
            if origin and urlsplit(origin).netloc != request.host:
                return jsonify(error='Cross-origin requests are disabled.'), 403

    @app.after_request
    def headers(response):
        response.headers['X-Content-Type-Options'] = 'nosniff'
        response.headers['Referrer-Policy'] = 'no-referrer'
        response.headers['Content-Security-Policy'] = "default-src 'self'; script-src 'self'; style-src 'self'; img-src 'self' data:; connect-src 'self'; frame-ancestors 'none'; base-uri 'none'; form-action 'self'"
        response.headers['Cache-Control'] = 'no-store'
        return response

    @app.errorhandler(Exception)
    def error(exc):
        if isinstance(exc, HTTPException):
            return jsonify(error=exc.description), exc.code
        if isinstance(exc, (ValueError, TypeError, KeyError)):
            return jsonify(error=str(exc)), 400
        app.logger.exception('Request failed')
        return jsonify(error='The request failed. Check the local server log for details.'), 500

    @app.get('/')
    def index():
        return render_template('index.html', token=csrf)

    @app.get('/health')
    def health():
        return jsonify(app='VV-GPT4', root=str(ROOT), data=str(store.root), status='ready')

    @app.get('/api/state')
    def state():
        lab.reconcile(store)
        return jsonify(provider=local.models(), conversations=store.rows('SELECT * FROM conversations ORDER BY created DESC'),
                       documents=store.rows('SELECT id,name,length(content) AS chars FROM documents ORDER BY created DESC'),
                       datasets=store.rows('SELECT * FROM datasets ORDER BY created DESC'),
                       runs=store.rows('SELECT * FROM runs ORDER BY created DESC'), models=lab.library(store),
                       memory=store.setting('memory'), version='4.0.0-alpha')

    @app.post('/api/conversations')
    def conversation_new():
        cid = uid()
        store.execute('INSERT INTO conversations VALUES(?,?,?)', (cid, 'New conversation', time.time()))
        return jsonify(id=cid)

    @app.get('/api/conversations/<cid>')
    def conversation_get(cid):
        if not store.rows('SELECT id FROM conversations WHERE id=?', (cid,)):
            return jsonify(error='Conversation not found.'), 404
        return jsonify(messages=store.rows('SELECT * FROM messages WHERE conversation=? ORDER BY created,id', (cid,)))

    @app.delete('/api/conversations/<cid>')
    def conversation_delete(cid):
        with guard:
            if cid in active:
                return jsonify(error='Stop generation before deleting this conversation.'), 409
            store.execute('DELETE FROM conversations WHERE id=?', (cid,))
        return jsonify(ok=True)

    @app.post('/api/conversations/<cid>/stop')
    def conversation_stop(cid):
        with guard:
            if cid in active:
                active[cid].set()
        return jsonify(ok=True)

    @app.post('/api/conversations/<cid>/chat')
    def chat(cid):
        data = request.get_json()
        question, model = data.get('message'), data.get('model')
        if not isinstance(question, str) or not question.strip() or len(question) > 8000:
            raise ValueError('Write a message of 1–8,000 characters.')
        if not store.rows('SELECT id FROM conversations WHERE id=?', (cid,)):
            return jsonify(error='Conversation not found.'), 404
        local.validate(model)
        with guard:
            if cid in active or not compute.acquire(blocking=False):
                return jsonify(error='Another generation is running. Stop it or wait for it to finish.'), 409
            cancelled = threading.Event()
            active[cid] = cancelled
        try:
            messages, sources = context_messages(store, cid, question.strip(), bool(data.get('knowledge', True)))
            store.message(cid, 'user', question.strip())
            store.execute("UPDATE conversations SET title=? WHERE id=? AND title='New conversation'", (question.strip()[:60], cid))
        except Exception:
            with guard:
                active.pop(cid, None)
                compute.release()
            raise

        def events():
            answer, status = '', 'interrupted'
            try:
                yield json.dumps({'type':'sources','sources':sources})+'\n'
                for event in local.stream(model, messages, cancelled):
                    if event['type'] == 'token':
                        answer += event['text']
                    if event['type'] == 'metrics':
                        status = 'complete' if event.get('reason') != 'length' else 'truncated'
                    yield json.dumps(event)+'\n'
                if cancelled.is_set():
                    status = 'stopped'
                if not answer and status == 'complete':
                    status = 'failed'
                yield json.dumps({'type':'done','status':status})+'\n'
            except GeneratorExit:
                cancelled.set()
                raise
            except Exception as exc:
                status = 'failed'
                yield json.dumps({'type':'error','error':str(exc)})+'\n'
            finally:
                try:
                    store.message(cid, 'assistant', answer, status, model, sources)
                finally:
                    with guard:
                        active.pop(cid, None)
                        compute.release()
        return Response(stream_with_context(events()), mimetype='application/x-ndjson')

    def upload_text():
        upload = request.files.get('file')
        if not upload or not upload.filename:
            raise ValueError('Choose a text or Markdown file.')
        name = Path(upload.filename).name
        if Path(name).suffix.lower() not in ('.txt', '.md', '.log'):
            raise ValueError('This release supports .txt, .md, and .log files.')
        try:
            content = upload.read().decode('utf-8-sig')
        except UnicodeDecodeError:
            raise ValueError('Save this file as UTF-8 text before importing it.')
        return name, content

    @app.post('/api/documents')
    def document_upload():
        name, content = upload_text()
        if len(content) > 2_000_000:
            raise ValueError('Keep each knowledge file under 2 million characters.')
        return jsonify(id=add_document(store, name, content))

    @app.delete('/api/documents/<did>')
    def document_delete(did):
        with store.db() as db:
            db.execute('DELETE FROM chunks WHERE document=?', (did,))
            db.execute('DELETE FROM documents WHERE id=?', (did,))
        return jsonify(ok=True)

    @app.put('/api/memory')
    def memory_save():
        value = request.get_json().get('memory')
        if not isinstance(value, str) or len(value) > 4000:
            raise ValueError('Preferences must be text under 4,000 characters.')
        store.execute('INSERT OR REPLACE INTO settings VALUES(?,?)', ('memory', value))
        return jsonify(ok=True)

    @app.post('/api/datasets')
    def dataset_upload():
        return jsonify(id=lab.add_dataset(store, *upload_text()))

    @app.post('/api/runs')
    def run_new():
        return jsonify(id=lab.start_run(store, request.get_json()))

    @app.get('/api/runs/<rid>')
    def run_get(rid):
        rows = store.rows('SELECT * FROM runs WHERE id=?', (rid,))
        if not rows:
            return jsonify(error='Experiment not found.'), 404
        return jsonify(run=rows[0], metrics=store.rows('SELECT * FROM metrics WHERE run=? ORDER BY step', (rid,)))

    @app.post('/api/runs/<rid>/<action>')
    def run_control(rid, action):
        lab.control_run(store, rid, action)
        return jsonify(ok=True)

    @app.post('/api/sample')
    def sample():
        if not compute.acquire(blocking=False):
            return jsonify(error='Wait for the current generation to finish.'), 409
        try:
            data = request.get_json()
            return jsonify(lab.generate_sample(store, data['model'], data['prompt'], data.get('tokens', 80)))
        finally:
            compute.release()

    return app


if __name__ == '__main__':
    create_app().run(host='127.0.0.1', port=int(os.environ.get('VVGPT_PORT','5054')), debug=False, threaded=True)
