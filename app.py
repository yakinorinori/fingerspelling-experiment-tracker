# -*- coding: utf-8 -*-
"""
指文字認識研究 実験記録Webアプリ

- 実験ごとに「行(クラス・モデルなど) × 列(正認識/誤認識/未検出などの条件)」の表を記録
- 列（条件）は実験ごとに自由に追加・削除できる
- 正認識率などを自動計算し、TSV / Markdown でコピーできる
- 記録した実験は後からすべて編集可能（SQLite保存）

起動:  python app.py  →  http://127.0.0.1:5002
"""
import json
import sqlite3
from datetime import datetime
from pathlib import Path

from flask import Flask, g, jsonify, render_template, request

BASE_DIR = Path(__file__).resolve().parent
DB_PATH = BASE_DIR / 'data' / 'experiments.db'

app = Flask(__name__)
app.config['JSON_AS_ASCII'] = False


# ---------------------------------------------------------------- DB
def get_db():
    if 'db' not in g:
        DB_PATH.parent.mkdir(parents=True, exist_ok=True)
        g.db = sqlite3.connect(DB_PATH)
        g.db.row_factory = sqlite3.Row
    return g.db


@app.teardown_appcontext
def close_db(exc):
    db = g.pop('db', None)
    if db is not None:
        db.close()


SCHEMA = """
CREATE TABLE IF NOT EXISTS experiments (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    title       TEXT NOT NULL,
    category    TEXT DEFAULT '',
    description TEXT DEFAULT '',
    status      TEXT DEFAULT '予定',
    class_count INTEGER,
    columns     TEXT DEFAULT '[]',  -- [{"name": "正認識", "count": true}, ...]
    rows        TEXT DEFAULT '[]',  -- [{"label": "クラス0", "values": ["", ...]}, ...]
    notes       TEXT DEFAULT '',
    created_at  TEXT,
    updated_at  TEXT
);
"""

# 初期登録する実験（行うべき実験のまとめ）
SEED_EXPERIMENTS = [
    {
        'title': '手型モデル精度（7クラスSVM）',
        'category': '1. 手型モデル',
        'description': 'MediaPipe骨格32次元特徴量による手型7クラス分類SVMの精度評価。',
        'class_count': 7,
        'columns': [
            {'name': 'サンプル数', 'count': False},
            {'name': '正認識', 'count': True},
            {'name': '誤認識', 'count': True},
        ],
        'rows': [{'label': f'クラス{i}', 'values': ['', '', '']} for i in range(7)],
    },
    {
        'title': '動作モデル比較（Transformer vs 1D-CNN）',
        'category': '2. 動作モデル',
        'description': '手の動き5クラス分類における Transformer と 1D-CNN の比較実験。',
        'class_count': 5,
        'columns': [
            {'name': 'Accuracy', 'count': False},
            {'name': 'Precision', 'count': False},
            {'name': 'Recall', 'count': False},
            {'name': 'F1', 'count': False},
        ],
        'rows': [
            {'label': 'Transformer', 'values': ['', '', '', '']},
            {'label': '1D-CNN', 'values': ['', '', '', '']},
        ],
    },
    {
        'title': '指文字単語分割（3クラス: 静止/動作/遷移）',
        'category': '3. 単語分割',
        'description': 'フレーム単位の 静止状態・動作状態・遷移区間 3クラス分類による単語分割。',
        'class_count': 3,
        'columns': [
            {'name': 'サンプル数', 'count': False},
            {'name': '正認識', 'count': True},
            {'name': '誤認識', 'count': True},
        ],
        'rows': [
            {'label': '静止状態', 'values': ['', '', '']},
            {'label': '動作状態', 'values': ['', '', '']},
            {'label': '遷移区間', 'values': ['', '', '']},
        ],
    },
    {
        'title': '指文字単語分割（2クラス: 文字区間/遷移区間)',
        'category': '3. 単語分割',
        'description': 'フレーム単位の 文字区間・遷移区間 2クラス分類による単語分割。',
        'class_count': 2,
        'columns': [
            {'name': 'サンプル数', 'count': False},
            {'name': '正認識', 'count': True},
            {'name': '誤認識', 'count': True},
        ],
        'rows': [
            {'label': '文字区間', 'values': ['', '', '']},
            {'label': '遷移区間', 'values': ['', '', '']},
        ],
    },
    {
        'title': '指文字単体認識精度（静止/動作/統一モデル）',
        'category': '4. 単体認識',
        'description': '指文字1文字単位の認識精度。静止状態モデル・動作状態モデル・統一モデルを比較。',
        'class_count': None,
        'columns': [
            {'name': 'サンプル数', 'count': False},
            {'name': '正認識', 'count': True},
            {'name': '誤認識', 'count': True},
            {'name': '未検出', 'count': True},
        ],
        'rows': [
            {'label': '静止状態モデル', 'values': ['', '', '', '']},
            {'label': '動作状態モデル', 'values': ['', '', '', '']},
            {'label': '統一モデル', 'values': ['', '', '', '']},
        ],
    },
    {
        'title': '骨格保存導入後の比較実験',
        'category': '5. 骨格保存',
        'description': '骨格保存（ランドマークキャッシュ等）導入前後での精度・処理速度の比較。',
        'class_count': None,
        'columns': [
            {'name': '精度(%)', 'count': False},
            {'name': '処理時間(s)', 'count': False},
            {'name': '備考', 'count': False},
        ],
        'rows': [
            {'label': '導入前', 'values': ['', '', '']},
            {'label': '導入後', 'values': ['', '', '']},
        ],
    },
    {
        'title': '手話文章中の指文字単語判別',
        'category': '6. 文章中判別',
        'description': '手話文章動画の中に含まれる指文字単語の検出・認識精度。',
        'class_count': None,
        'columns': [
            {'name': 'サンプル数', 'count': False},
            {'name': '正認識', 'count': True},
            {'name': '誤認識', 'count': True},
            {'name': '未検出', 'count': True},
        ],
        'rows': [{'label': '全体', 'values': ['', '', '', '']}],
    },
]


def init_db():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    db = sqlite3.connect(DB_PATH)
    db.executescript(SCHEMA)
    n = db.execute('SELECT COUNT(*) FROM experiments').fetchone()[0]
    if n == 0:
        now = datetime.now().isoformat(timespec='seconds')
        for e in SEED_EXPERIMENTS:
            db.execute(
                'INSERT INTO experiments '
                '(title, category, description, status, class_count, columns, rows, notes, created_at, updated_at) '
                'VALUES (?,?,?,?,?,?,?,?,?,?)',
                (e['title'], e['category'], e['description'], '予定',
                 e['class_count'],
                 json.dumps(e['columns'], ensure_ascii=False),
                 json.dumps(e['rows'], ensure_ascii=False),
                 '', now, now))
        db.commit()
    db.close()


def row_to_dict(r, full=True):
    d = {
        'id': r['id'], 'title': r['title'], 'category': r['category'],
        'status': r['status'], 'class_count': r['class_count'],
        'created_at': r['created_at'], 'updated_at': r['updated_at'],
    }
    if full:
        d.update({
            'description': r['description'],
            'columns': json.loads(r['columns']),
            'rows': json.loads(r['rows']),
            'notes': r['notes'],
        })
    return d


# ---------------------------------------------------------------- API
@app.route('/')
def index():
    return render_template('index.html')


@app.route('/api/experiments', methods=['GET'])
def list_experiments():
    rows = get_db().execute(
        'SELECT * FROM experiments ORDER BY category, id').fetchall()
    return jsonify([row_to_dict(r, full=False) for r in rows])


@app.route('/api/experiments', methods=['POST'])
def create_experiment():
    d = request.json or {}
    now = datetime.now().isoformat(timespec='seconds')
    default_cols = [{'name': '正認識', 'count': True}, {'name': '誤認識', 'count': True}]
    cur = get_db().execute(
        'INSERT INTO experiments '
        '(title, category, description, status, class_count, columns, rows, notes, created_at, updated_at) '
        'VALUES (?,?,?,?,?,?,?,?,?,?)',
        (d.get('title', '新しい実験'), d.get('category', ''), d.get('description', ''),
         d.get('status', '予定'), d.get('class_count'),
         json.dumps(d.get('columns', default_cols), ensure_ascii=False),
         json.dumps(d.get('rows', []), ensure_ascii=False),
         d.get('notes', ''), now, now))
    get_db().commit()
    r = get_db().execute('SELECT * FROM experiments WHERE id=?', (cur.lastrowid,)).fetchone()
    return jsonify(row_to_dict(r)), 201


@app.route('/api/experiments/<int:eid>', methods=['GET'])
def get_experiment(eid):
    r = get_db().execute('SELECT * FROM experiments WHERE id=?', (eid,)).fetchone()
    if r is None:
        return jsonify({'error': 'not found'}), 404
    return jsonify(row_to_dict(r))


@app.route('/api/experiments/<int:eid>', methods=['PUT'])
def update_experiment(eid):
    r = get_db().execute('SELECT * FROM experiments WHERE id=?', (eid,)).fetchone()
    if r is None:
        return jsonify({'error': 'not found'}), 404
    d = request.json or {}
    cur = row_to_dict(r)
    for key in ['title', 'category', 'description', 'status', 'class_count', 'notes']:
        if key in d:
            cur[key] = d[key]
    columns = d.get('columns', cur['columns'])
    rows = d.get('rows', cur['rows'])
    get_db().execute(
        'UPDATE experiments SET title=?, category=?, description=?, status=?, '
        'class_count=?, columns=?, rows=?, notes=?, updated_at=? WHERE id=?',
        (cur['title'], cur['category'], cur['description'], cur['status'],
         cur['class_count'],
         json.dumps(columns, ensure_ascii=False),
         json.dumps(rows, ensure_ascii=False),
         cur['notes'], datetime.now().isoformat(timespec='seconds'), eid))
    get_db().commit()
    r = get_db().execute('SELECT * FROM experiments WHERE id=?', (eid,)).fetchone()
    return jsonify(row_to_dict(r))


@app.route('/api/experiments/<int:eid>', methods=['DELETE'])
def delete_experiment(eid):
    get_db().execute('DELETE FROM experiments WHERE id=?', (eid,))
    get_db().commit()
    return jsonify({'success': True})


if __name__ == '__main__':
    init_db()
    print('実験記録アプリ: http://127.0.0.1:5002')
    app.run(debug=True, host='0.0.0.0', port=5002)
