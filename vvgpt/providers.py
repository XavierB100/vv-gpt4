import json
import urllib.error
import urllib.request

OLLAMA = 'http://127.0.0.1:11434'


class LocalProvider:
    def request(self, path, payload=None, timeout=5):
        request = urllib.request.Request(OLLAMA + path,
            data=json.dumps(payload).encode() if payload is not None else None,
            headers={'Content-Type': 'application/json'})
        # Do not send localhost traffic through environment-configured proxies.
        return urllib.request.build_opener(urllib.request.ProxyHandler({})).open(request, timeout=timeout)

    def models(self):
        try:
            with self.request('/api/tags') as response:
                models = json.load(response).get('models', [])
            return {'online': True, 'models': [m for m in models if self.local_candidate(m)]}
        except (OSError, ValueError):
            return {'online': False, 'models': [], 'message': 'Open Ollama, then refresh. Your installed local models will appear here.'}

    @staticmethod
    def local_candidate(model):
        name = model.get('name', '').lower()
        return ('cloud' not in name and not model.get('remote_host') and not model.get('remote_model')
                and model.get('size', 0) > 1_000_000)

    def validate(self, name):
        available = self.models()
        if name not in [m['name'] for m in available['models']]:
            raise ValueError('Choose an installed local model. Open Ollama if it is offline.')
        with self.request('/api/show', {'model': name}) as response:
            details = json.load(response)
        if details.get('remote_model') or details.get('remote_host'):
            raise ValueError('Cloud models are disabled in this local workspace.')

    def stream(self, model, messages, cancelled, max_tokens=768):
        payload = {'model': model, 'messages': messages, 'stream': True, 'think': False,
                   'options': {'temperature': 0.6, 'num_ctx': 8192, 'num_predict': max_tokens}, 'keep_alive': '5m'}
        with self.request('/api/chat', payload, timeout=180) as response:
            for line in response:
                if cancelled.is_set():
                    return
                if not line.strip():
                    continue
                event = json.loads(line)
                if event.get('error'):
                    raise RuntimeError(event['error'])
                if event.get('message', {}).get('content'):
                    yield {'type': 'token', 'text': event['message']['content']}
                if event.get('done'):
                    duration = event.get('eval_duration', 0) / 1e9
                    yield {'type': 'metrics', 'tokens': event.get('eval_count', 0),
                           'tokens_per_second': round(event.get('eval_count', 0) / duration, 1) if duration else None,
                           'reason': event.get('done_reason', 'stop')}
