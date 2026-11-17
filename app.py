"""VV-GPT's own-model studio. No pretrained providers or external inference services."""
import json
import os
import secrets
import threading
import time
from pathlib import Path
from urllib.parse import urlsplit
from flask import Flask,jsonify,render_template,request,send_file
from werkzeug.exceptions import HTTPException
from vvgpt.store import Store,uid
from vvgpt import lab

ROOT=Path(__file__).resolve().parent


def create_app(data_dir=None):
    app=Flask(__name__)
    app.config['MAX_CONTENT_LENGTH']=16*1024*1024
    store=Store(data_dir or os.environ.get('VVGPT_DATA',ROOT/'runtime'))
    csrf=secrets.token_urlsafe(32)
    compute=threading.Lock();cache=lab.ModelCache()
    lab.reconcile(store)
    app.extensions.update(store=store,csrf=csrf)

    @app.before_request
    def local_only():
        if request.host.split(':')[0] not in ('127.0.0.1','localhost'):
            return jsonify(error='This studio accepts localhost connections only.'),403
        if request.method in ('POST','PUT','DELETE','PATCH'):
            if request.headers.get('X-VV-Token')!=csrf:
                return jsonify(error='Refresh the page before trying again.'),403
            origin=request.headers.get('Origin')
            if origin and urlsplit(origin).netloc!=request.host:
                return jsonify(error='Cross-origin requests are disabled.'),403

    @app.after_request
    def headers(response):
        response.headers['X-Content-Type-Options']='nosniff'
        response.headers['Referrer-Policy']='no-referrer'
        response.headers['Content-Security-Policy']="default-src 'self'; script-src 'self'; style-src 'self'; img-src 'self' data:; connect-src 'self'; frame-ancestors 'none'; base-uri 'none'; form-action 'self'"
        response.headers['Cache-Control']='no-store'
        if request.path=='/report':
            response.headers['Content-Security-Policy']="default-src 'none'; style-src 'unsafe-inline'; img-src 'self' data:; frame-ancestors 'none'"
        return response

    @app.errorhandler(Exception)
    def error(exc):
        if isinstance(exc,HTTPException):return jsonify(error=exc.description),exc.code
        if isinstance(exc,(ValueError,TypeError,KeyError,OverflowError)):return jsonify(error=str(exc)),400
        app.logger.exception('Request failed')
        return jsonify(error='The request failed. Check the local server log.'),500

    def payload():
        data=request.get_json()
        if not isinstance(data,dict):raise ValueError('Expected a JSON object.')
        return data

    @app.get('/')
    def index():return render_template('index.html',token=csrf)

    @app.get('/health')
    def health():return jsonify(app='VV-GPT4',root=str(ROOT),data=str(store.root),status='ready',mode='custom-models')

    @app.get('/report')
    def report():return send_file(ROOT/'docs/enhancements.html')

    @app.get('/api/state')
    def state():
        lab.reconcile(store)
        runs=store.rows('SELECT * FROM runs ORDER BY created DESC')
        for run in runs:run['summary']=lab.run_summary(store,run['id'])
        return jsonify(datasets=lab.dataset_info(store),runs=runs,models=lab.library(store),
                       samples=store.rows('SELECT id,model,prompt,created FROM samples ORDER BY created DESC LIMIT 12'),
                       version='4.1-custom',mode='custom-models')

    @app.post('/api/datasets')
    def dataset_upload():
        file=request.files.get('file')
        if not file or not file.filename:raise ValueError('Choose a text, Markdown, or chat-export file.')
        name=Path(file.filename).name
        if Path(name).suffix.lower() not in ('.txt','.md','.log'):raise ValueError('Use a .txt, .md, or .log file.')
        try:text=file.read().decode('utf-8-sig')
        except UnicodeDecodeError:raise ValueError('Save the file as UTF-8 text before importing it.')
        did=lab.add_dataset(store,name,text,request.form.get('deduplicate','true')=='true')
        return jsonify(id=did)

    @app.post('/api/datasets/example')
    def dataset_example():
        rows=store.rows('SELECT id FROM datasets WHERE name=?',('Shakespeare · original corpus',))
        if rows:return jsonify(id=rows[0]['id'])
        path=lab.LEGACY.parent/'data/input.txt'
        if not path.exists():raise ValueError('Original example not found. Import your text file instead.')
        return jsonify(id=lab.add_dataset(store,'Shakespeare · original corpus',path.read_text(encoding='utf-8')))

    @app.post('/api/runs')
    def run_new():return jsonify(id=lab.start_run(store,payload()))

    @app.get('/api/runs/<rid>')
    def run_get(rid):
        rows=store.rows('SELECT * FROM runs WHERE id=?',(rid,))
        if not rows:return jsonify(error='Experiment not found.'),404
        return jsonify(run=rows[0],summary=lab.run_summary(store,rid),
                       metrics=store.rows('SELECT * FROM metrics WHERE run=? ORDER BY step',(rid,)))

    @app.post('/api/runs/<rid>/<action>')
    def run_control(rid,action):
        lab.control_run(store,rid,action)
        return jsonify(ok=True)

    @app.post('/api/sample')
    def sample():
        if not compute.acquire(blocking=False):return jsonify(error='A generation is already running.'),409
        try:
            data=payload()
            result=lab.generate_sample(store,data['model'],data['prompt'],data.get('tokens',120),data,cache)
            results={'enhanced':result}
            if data.get('compare'):
                baseline=dict(data,repetition_penalty=1.,no_repeat_ngram=0)
                results['baseline']=lab.generate_sample(store,data['model'],data['prompt'],data.get('tokens',120),baseline,cache)
            sid=uid()
            store.execute('INSERT INTO samples VALUES(?,?,?,?,?,?)',(sid,data['model'],data['prompt'],json.dumps(results),time.time(),json.dumps(data)))
            return jsonify(id=sid,**results)
        finally:compute.release()

    @app.get('/api/samples/<sid>')
    def sample_get(sid):
        rows=store.rows('SELECT * FROM samples WHERE id=?',(sid,))
        if not rows:return jsonify(error='Sample not found.'),404
        item=rows[0];item['result']=json.loads(item['result']);item['settings']=json.loads(item['settings'])
        return jsonify(item)

    return app


if __name__=='__main__':
    create_app().run(host='127.0.0.1',port=int(os.environ.get('VVGPT_PORT','5054')),debug=False,threaded=True)
