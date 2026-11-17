import json
import sqlite3
import time
import uuid
from contextlib import contextmanager
from pathlib import Path


def uid():
    return uuid.uuid4().hex


class Store:
    def __init__(self, root):
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        for folder in ('datasets', 'runs'):
            (self.root / folder).mkdir(exist_ok=True)
        self.path = self.root / 'studio.sqlite3'
        with self.db() as db:
            db.executescript('''
                PRAGMA journal_mode=WAL;
                CREATE TABLE IF NOT EXISTS conversations(id TEXT PRIMARY KEY, title TEXT, created REAL);
                CREATE TABLE IF NOT EXISTS messages(id TEXT PRIMARY KEY, conversation TEXT REFERENCES conversations(id) ON DELETE CASCADE,
                    role TEXT, content TEXT, status TEXT, model TEXT, sources TEXT, created REAL);
                CREATE TABLE IF NOT EXISTS documents(id TEXT PRIMARY KEY, name TEXT, content TEXT, created REAL);
                CREATE VIRTUAL TABLE IF NOT EXISTS chunks USING fts5(document UNINDEXED, name UNINDEXED, content);
                CREATE TABLE IF NOT EXISTS settings(key TEXT PRIMARY KEY, value TEXT);
                CREATE TABLE IF NOT EXISTS samples(id TEXT PRIMARY KEY, model TEXT, prompt TEXT, result TEXT, created REAL, settings TEXT);
                CREATE TABLE IF NOT EXISTS datasets(id TEXT PRIMARY KEY, name TEXT, tokens INTEGER, chars INTEGER, sha256 TEXT, created REAL);
                CREATE TABLE IF NOT EXISTS runs(id TEXT PRIMARY KEY, name TEXT, config TEXT, status TEXT, control TEXT DEFAULT '',
                    step INTEGER DEFAULT 0, best REAL, error TEXT, pid INTEGER, created REAL, updated REAL);
                CREATE TABLE IF NOT EXISTS metrics(run TEXT REFERENCES runs(id), step INTEGER, train REAL, val REAL, lr REAL, elapsed REAL,
                    PRIMARY KEY(run,step));
                CREATE INDEX IF NOT EXISTS messages_by_conversation ON messages(conversation,created);
            ''')

    @contextmanager
    def db(self):
        db = sqlite3.connect(self.path, timeout=20)
        db.row_factory = sqlite3.Row
        db.execute('PRAGMA foreign_keys=ON')
        try:
            yield db
            db.commit()
        except BaseException:
            db.rollback()
            raise
        finally:
            db.close()

    def rows(self, query, args=()):
        with self.db() as db:
            return [dict(r) for r in db.execute(query, args)]

    def execute(self, query, args=()):
        with self.db() as db:
            db.execute(query, args)

    def setting(self, key, default=''):
        rows = self.rows('SELECT value FROM settings WHERE key=?', (key,))
        return rows[0]['value'] if rows else default

    def message(self, conversation, role, content, status='complete', model='', sources=None):
        mid = uid()
        self.execute('INSERT INTO messages VALUES(?,?,?,?,?,?,?,?)',
                     (mid, conversation, role, content, status, model, json.dumps(sources or []), time.time()))
        return mid
