# -*- coding: utf-8 -*-
"""
実験5「指文字単体認識精度（静止/動作/統一モデル）」を、実際の単体文字動画で行う機能

流れ:
  1. 単体文字フォルダの動画を一覧にする
  2. 動画を選ぶ → 右手の骨格を取る（右手が1フレームも取れなければ「未検出」。自動再生しない）
  3. 静止SVM・動作Transformer・統一Transformer で認識する（結果は保存して、何度見ても計算し直さない）
  4. 動画を見て正解の文字を入力する（正解は data/single_char_labels.csv にまとめ、次回以降も使う）
  5. 正解と認識結果を比べて自動判定。おかしければ手で修正できる
  6. モデルごとに 正認識・誤認識・未検出 を集計し、実験5の表に反映する

認識には fingerspelling-character のコード（右手の骨格・特徴量・モデル）をそのまま使う。
そのため、このアプリは fingerspelling-character と同じ Python 環境（venv_hand など）で起動する。
"""
import csv
import hashlib
import importlib
import json
import sqlite3
import sys
import threading
from datetime import datetime
from pathlib import Path

import numpy as np
from flask import Blueprint, abort, jsonify, render_template, request, send_file

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / 'data'
SETTINGS_PATH = DATA_DIR / 'single_char_settings.json'
LABELS_CSV = DATA_DIR / 'single_char_labels.csv'
LANDMARK_CACHE = DATA_DIR / 'single_char_landmarks'
DB_PATH = DATA_DIR / 'experiments.db'

VIDEO_EXTS = {'.mp4', '.mov', '.avi', '.m4v', '.mkv'}
EXPERIMENT_TITLE = '指文字単体認識精度（静止/動作/統一モデル）'

# モデルの種類（実験5の表の行名と対応）
MODEL_KINDS = [
    {'key': 'static', 'name': '静止状態モデル', 'setting': 'static_model'},
    {'key': 'dynamic', 'name': '動作状態モデル', 'setting': 'dynamic_model'},
    {'key': 'unified', 'name': '統一モデル', 'setting': 'unified_model'},
]
JUDGES = ['正認識', '誤認識', '未検出', '対象外']

bp = Blueprint('singlechar', __name__)
_lock = threading.Lock()      # 認識は1本ずつ（MediaPipe・モデルを同時に使わない）
_models = {}                  # 読み込んだモデル {パス: (更新時刻, pack)}


# ---------------------------------------------------------------- 設定
def find_character_repo():
    """
    fingerspelling-character を探す。このアプリのフォルダから上へ4階層まで、
    「<親>/fingerspelling-character」と「<親>/Fingerspelling-repos/fingerspelling-character」を順に見る
    """
    for parent in list(BASE_DIR.parents)[:4]:
        for cand in (parent / 'fingerspelling-character', parent / 'Fingerspelling-repos' / 'fingerspelling-character'):
            if (cand / 'recognize_video.py').is_file():
                return cand
    return BASE_DIR.parent / 'fingerspelling-character'


def default_settings():
    char_repo = find_character_repo()
    models = char_repo / 'results' / 'models'
    # fingerspelling-character と同じく python_画像処理/Fingerspelling/Video を使う
    video = char_repo.parent.parent / 'Fingerspelling' / 'Video' / '単語分割' / '単体文字'
    return {
        'video_dir': str(video),
        'character_repo': str(char_repo),
        'static_model': str(models / 'svm_H3+O.joblib'),
        'dynamic_model': str(models / 'dynamic_transformer_H3+O+M.pt'),
        'unified_model': str(models / 'unified_transformer_S+H3+O+M.pt'),
        'flip': False,
        'show_result': 'after_play',     # after_play: 1回再生し終わってから結果を出す / immediately
    }


def load_settings():
    s = default_settings()
    if SETTINGS_PATH.exists():
        try:
            s.update(json.loads(SETTINGS_PATH.read_text(encoding='utf-8')))
        except (OSError, ValueError):
            pass
    return s


def save_settings(new):
    s = load_settings()
    for k in default_settings():
        if k in new:
            s[k] = new[k]
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    SETTINGS_PATH.write_text(json.dumps(s, ensure_ascii=False, indent=2), encoding='utf-8')
    _models.clear()
    return s


def check_settings(s):
    """設定したパスが存在するか（画面に表示する）"""
    out = {}
    for k in ['video_dir', 'character_repo']:
        out[k] = Path(s[k]).expanduser().is_dir()
    for m in MODEL_KINDS:
        out[m['setting']] = Path(s[m['setting']]).expanduser().is_file()
    return out


# ---------------------------------------------------------------- 動画一覧・正解CSV
def video_root():
    return Path(load_settings()['video_dir']).expanduser()


def list_videos():
    root = video_root()
    if not root.is_dir():
        return []
    vids = [p for p in root.rglob('*') if p.is_file() and p.suffix.lower() in VIDEO_EXTS
            and not p.name.startswith('.')]
    return sorted(str(p.relative_to(root)) for p in vids)


def resolve_video(rel):
    root = video_root().resolve()
    path = (root / rel).resolve()
    if root not in path.parents or not path.is_file():
        abort(404)
    return path


def load_labels():
    """正解CSV: video（単体文字フォルダからの相対パス）, label, updated_at"""
    if not LABELS_CSV.exists():
        return {}
    with open(LABELS_CSV, encoding='utf-8-sig', newline='') as f:
        return {r['video']: r['label'] for r in csv.DictReader(f) if r.get('label')}


def save_label(video, label):
    labels = load_labels()
    if label:
        labels[video] = label
    else:
        labels.pop(video, None)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    now = datetime.now().isoformat(timespec='seconds')
    old = {}
    if LABELS_CSV.exists():
        with open(LABELS_CSV, encoding='utf-8-sig', newline='') as f:
            old = {r['video']: r.get('updated_at', '') for r in csv.DictReader(f)}
    with open(LABELS_CSV, 'w', encoding='utf-8-sig', newline='') as f:   # Excel でも文字化けしないように BOM 付き
        w = csv.writer(f)
        w.writerow(['video', 'label', 'updated_at'])
        for v in sorted(labels):
            w.writerow([v, labels[v], now if v == video else old.get(v, '')])


# ---------------------------------------------------------------- DB（認識結果・手での修正）
SCHEMA = """
CREATE TABLE IF NOT EXISTS single_char_results (
    video       TEXT NOT NULL,
    model       TEXT NOT NULL,      -- static / dynamic / unified
    model_path  TEXT,
    model_mtime REAL,
    predicted   TEXT,               -- 認識した文字（未検出なら空）
    top         TEXT,               -- [[文字, 確率], ...] 上位5
    classes     TEXT,               -- そのモデルが認識できる文字の一覧
    hand_rate   REAL,               -- 右手が取れたフレームの割合
    frames      INTEGER,
    override    TEXT,               -- 手で修正した判定（空なら自動判定）
    updated_at  TEXT,
    PRIMARY KEY (video, model)
);
"""


def db():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(DB_PATH)
    con.row_factory = sqlite3.Row
    con.executescript(SCHEMA)
    return con


def get_results(video):
    with db() as con:
        rows = con.execute('SELECT * FROM single_char_results WHERE video=?', (video,)).fetchall()
    return {r['model']: dict(r) for r in rows}


# ---------------------------------------------------------------- 認識（fingerspelling-character を使う）
def char_modules():
    repo = str(Path(load_settings()['character_repo']).expanduser().resolve())
    if repo not in sys.path:
        sys.path.insert(0, repo)
    try:
        return {name: importlib.import_module(name)
                for name in ['extract_landmarks', 'features', 'recognize_video', 'train_static_svm']}
    except ImportError as e:
        raise RuntimeError(f'fingerspelling-character のコードを読み込めません（{repo}）: {e}')


def load_model(path):
    p = Path(path).expanduser()
    mtime = p.stat().st_mtime
    if path not in _models or _models[path][0] != mtime:
        _models[path] = (mtime, char_modules()['recognize_video'].load(p))
    return _models[path][1]


def landmarks(video):
    """右手の骨格（動画の中身と左右反転の設定ごとに data/single_char_landmarks/ に保存して使い回す）"""
    s = load_settings()
    path = resolve_video(video)
    st = path.stat()
    key = hashlib.md5(f'{path}|{st.st_size}|{st.st_mtime}|{s["flip"]}'.encode()).hexdigest()
    cache = LANDMARK_CACHE / f'{key}.npz'
    if cache.exists():
        d = np.load(cache)
        return d['lm'], d['world'], int(d['width']), int(d['height']), float(d['fps'])
    lm, world, w, h, fps = char_modules()['extract_landmarks'].extract_video(path, bool(s['flip']))
    LANDMARK_CACHE.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache, lm=lm, world=world, width=w, height=h, fps=fps)
    return lm, world, w, h, fps


def predict_all(video, models=None, force=False):
    """3つのモデルで認識して DB に保存。保存済みでモデルが変わっていなければ計算しない"""
    s = load_settings()
    kinds = [m for m in MODEL_KINDS if not models or m['key'] in models]
    saved = get_results(video)
    todo = []
    for m in kinds:
        mp = s[m['setting']]
        if not Path(mp).expanduser().is_file():
            continue
        mtime = Path(mp).expanduser().stat().st_mtime
        old = saved.get(m['key'])
        if force or not old or old['model_path'] != mp or old['model_mtime'] != mtime:
            todo.append((m, mp, mtime))
    if not todo:
        return get_results(video)

    with _lock:
        mods = char_modules()
        F = mods['features']
        RV = mods['recognize_video']
        lm, world, w, h, fps = landmarks(video)
        valid = ~np.isnan(lm).any(axis=(1, 2)) if len(lm) else np.zeros(0, bool)
        hand_rate = float(valid.mean()) if len(valid) else 0.0
        blocks = F.compute_blocks(lm, world, w, h) if valid.any() else None
        now = datetime.now().isoformat(timespec='seconds')
        with db() as con:
            for m, mp, mtime in todo:
                pack = load_model(mp)
                classes = list(pack['classes'])
                predicted, top = '', []
                if blocks is not None:
                    probs = whole_probs(pack, blocks, valid, F, RV, mods['train_static_svm'])
                    order = np.argsort(probs)[::-1][:5]
                    top = [[classes[i], round(float(probs[i]), 4)] for i in order]
                    predicted = top[0][0]
                old = saved.get(m['key']) or {}
                con.execute(
                    'INSERT OR REPLACE INTO single_char_results '
                    '(video, model, model_path, model_mtime, predicted, top, classes, hand_rate, frames, override, updated_at) '
                    'VALUES (?,?,?,?,?,?,?,?,?,?,?)',
                    (video, m['key'], mp, mtime, predicted, json.dumps(top, ensure_ascii=False),
                     json.dumps(classes, ensure_ascii=False), hand_rate, int(len(lm)),
                     old.get('override') or '', now))
    return get_results(video)


def whole_probs(pack, blocks, valid, F, RV, SVM):
    """動画全体の確率（recognize_video.py と同じ計算）"""
    if pack['type'] == 'svm':
        probs = RV.svm_probs(pack, blocks, valid)
        c = SVM.center_frames(valid)
        return probs[c].mean(0) if len(c) else np.zeros(len(pack['classes']))
    extra = {'S': RV.svm_probs(pack['svm'], blocks, valid)} if 'svm' in pack else None
    seq = F.build(pack['features'], blocks, extra)
    if hasattr(F, 'fit_length'):
        seq = F.fit_length(seq, valid, pack.get('max_len', 0))[0]
    return RV.run_seq(pack, seq)


# ---------------------------------------------------------------- 判定・集計
def judge(result, label):
    """自動判定（手での修正があればそちらを優先）。正解が未入力なら None"""
    if result is None:
        return None
    if result.get('override'):
        return result['override']
    classes = json.loads(result.get('classes') or '[]')
    if label and classes and label not in classes:
        return '対象外'          # そのモデルが認識できない文字（例: 静止SVM に動作指文字）
    if not result.get('predicted'):
        return '未検出'          # 右手が1フレームも取れなかった（正解が未入力でも未検出として数える）
    if not label:
        return None
    return '正認識' if result['predicted'] == label else '誤認識'


def auto_judge(result, label):
    r = dict(result or {})
    r['override'] = ''
    return judge(r, label) if result else None


def video_info(video, labels=None, results=None):
    labels = load_labels() if labels is None else labels
    res = get_results(video) if results is None else results
    label = labels.get(video, '')
    out = {'video': video, 'label': label, 'models': {}}
    for m in MODEL_KINDS:
        r = res.get(m['key'])
        if r is None:
            out['models'][m['key']] = None
            continue
        out['models'][m['key']] = {
            'predicted': r['predicted'], 'top': json.loads(r['top'] or '[]'),
            'hand_rate': r['hand_rate'], 'frames': r['frames'],
            'override': r['override'] or '', 'judge': judge(r, label), 'auto': auto_judge(r, label),
            'updated_at': r['updated_at'],
        }
    any_res = [r for r in res.values()]
    out['undetected'] = bool(any_res) and all(not r['predicted'] for r in any_res)
    return out


def all_results():
    with db() as con:
        rows = con.execute('SELECT * FROM single_char_results').fetchall()
    by_video = {}
    for r in rows:
        by_video.setdefault(r['video'], {})[r['model']] = dict(r)
    return by_video


def summary():
    labels = load_labels()
    res = all_results()
    videos = set(list_videos())
    table = {m['key']: {j: 0 for j in JUDGES} | {'未判定': 0} for m in MODEL_KINDS}
    confusions = {m['key']: {} for m in MODEL_KINDS}
    for v in videos:
        for m in MODEL_KINDS:
            r = res.get(v, {}).get(m['key'])
            j = judge(r, labels.get(v, ''))
            if r is None:
                continue
            table[m['key']][j or '未判定'] += 1
            if j == '誤認識':
                key = f"{labels.get(v, '')}→{r['predicted']}"
                confusions[m['key']][key] = confusions[m['key']].get(key, 0) + 1
    out = []
    for m in MODEL_KINDS:
        t = table[m['key']]
        n = t['正認識'] + t['誤認識'] + t['未検出']
        out.append({'key': m['key'], 'name': m['name'], **t, 'サンプル数': n,
                    'rate': round(100 * t['正認識'] / n, 1) if n else None,
                    'confusions': sorted(confusions[m['key']].items(), key=lambda x: -x[1])[:10]})
    return out


def reflect_to_experiment():
    """実験5の表（行=モデル）に サンプル数・正認識・誤認識・未検出 を書き込む"""
    s = summary()
    with db() as con:
        e = con.execute('SELECT * FROM experiments WHERE title=?', (EXPERIMENT_TITLE,)).fetchone()
        if e is None:
            raise RuntimeError(f'実験「{EXPERIMENT_TITLE}」が見つかりません')
        cols = json.loads(e['columns'])
        rows = json.loads(e['rows'])
        names = [c['name'] for c in cols]
        for need in ['サンプル数', '正認識', '誤認識', '未検出']:
            if need not in names:
                cols.append({'name': need, 'count': need != 'サンプル数'})
                names.append(need)
                for r in rows:
                    r['values'].append('')
        for item in s:
            row = next((r for r in rows if r['label'] == item['name']), None)
            if row is None:
                row = {'label': item['name'], 'values': [''] * len(cols)}
                rows.append(row)
            while len(row['values']) < len(cols):
                row['values'].append('')
            for k in ['サンプル数', '正認識', '誤認識', '未検出']:
                row['values'][names.index(k)] = str(item[k])
        note = (e['notes'] or '').rstrip()
        stamp = f"[単体文字動画で集計 {datetime.now().isoformat(timespec='minutes')}] 正解入力済み {len(load_labels())} 本"
        con.execute('UPDATE experiments SET columns=?, rows=?, status=?, notes=?, updated_at=? WHERE id=?',
                    (json.dumps(cols, ensure_ascii=False), json.dumps(rows, ensure_ascii=False),
                     '実施中' if e['status'] == '予定' else e['status'],
                     (note + '\n' if note else '') + stamp,
                     datetime.now().isoformat(timespec='seconds'), e['id']))
        return e['id']


# ---------------------------------------------------------------- 画面・API
@bp.route('/single')
def page():
    return render_template('single.html')


@bp.route('/api/single/settings', methods=['GET', 'PUT'])
def api_settings():
    s = save_settings(request.json or {}) if request.method == 'PUT' else load_settings()
    return jsonify({'settings': s, 'ok': check_settings(s)})


@bp.route('/api/single/videos')
def api_videos():
    labels = load_labels()
    res = all_results()
    out = []
    for v in list_videos():
        info = video_info(v, labels, res.get(v, {}))
        out.append({'video': v, 'label': info['label'], 'undetected': info['undetected'],
                    'recognized': any(x is not None for x in info['models'].values()),
                    'done': {k: (x['judge'] if x else None) for k, x in info['models'].items()}})
    return jsonify(out)


@bp.route('/api/single/classes')
def api_classes():
    """正解の文字の候補（読み込めたモデルの文字の和集合。モデルが無ければ五十音）"""
    s = load_settings()
    classes, kinds = [], {}
    for m in MODEL_KINDS:
        p = s[m['setting']]
        if Path(p).expanduser().is_file():
            try:
                for c in load_model(p)['classes']:
                    if c not in classes:
                        classes.append(c)
                    kinds.setdefault(c, []).append(m['key'])
            except Exception:
                pass
    if not classes:
        classes = list('あいうえおかきくけこさしすせそたちつてとなにぬねのはひふへほまみむめもやゆよらりるれろわん')
    return jsonify({'classes': classes, 'kinds': kinds})


@bp.route('/api/single/video')
def api_video_file():
    return send_file(resolve_video(request.args.get('path', '')), conditional=True)


@bp.route('/api/single/info')
def api_info():
    return jsonify(video_info(request.args.get('path', '')))


@bp.route('/api/single/recognize', methods=['POST'])
def api_recognize():
    d = request.json or {}
    video = d.get('video', '')
    resolve_video(video)
    try:
        predict_all(video, force=bool(d.get('force')))
    except Exception as e:                       # 画面にそのまま理由を出す
        return jsonify({'error': f'{type(e).__name__}: {e}'}), 500
    return jsonify(video_info(video))


@bp.route('/api/single/label', methods=['PUT'])
def api_label():
    d = request.json or {}
    resolve_video(d.get('video', ''))
    save_label(d['video'], d.get('label', ''))
    return jsonify(video_info(d['video']))


@bp.route('/api/single/override', methods=['PUT'])
def api_override():
    """
    判定を手で修正する。「正認識」を選んだときは、そのモデルが認識した文字を正解にする
    （正解が決まれば、ほかのモデルも自動で判定し直される。そのモデル自身は自動判定で正認識になる）
    """
    d = request.json or {}
    if d.get('judge') and d['judge'] not in JUDGES:
        abort(400)
    judge_ = d.get('judge') or ''
    if judge_ == '正認識':
        r = get_results(d['video']).get(d['model'])
        if r and r['predicted']:
            save_label(d['video'], r['predicted'])
            judge_ = ''                       # 正解＝認識結果になったので自動判定に任せる
    with db() as con:
        con.execute('UPDATE single_char_results SET override=? WHERE video=? AND model=?',
                    (judge_, d['video'], d['model']))
    return jsonify(video_info(d['video']))


@bp.route('/api/single/summary')
def api_summary():
    return jsonify(summary())


@bp.route('/api/single/reflect', methods=['POST'])
def api_reflect():
    try:
        eid = reflect_to_experiment()
    except RuntimeError as e:
        return jsonify({'error': str(e)}), 400
    return jsonify({'experiment_id': eid})


@bp.route('/api/single/export.csv')
def api_export():
    """動画ごとの結果（正解・各モデルの認識結果と判定）を CSV で出す"""
    import io
    labels = load_labels()
    res = all_results()
    buf = io.StringIO()
    w = csv.writer(buf)
    head = ['video', 'label']
    for m in MODEL_KINDS:
        head += [f"{m['name']}_認識", f"{m['name']}_判定", f"{m['name']}_確率"]
    w.writerow(head)
    for v in list_videos():
        info = video_info(v, labels, res.get(v, {}))
        row = [v, info['label']]
        for m in MODEL_KINDS:
            x = info['models'][m['key']]
            row += [x['predicted'], x['judge'] or '', x['top'][0][1] if x and x['top'] else ''] if x else ['', '', '']
        w.writerow(row)
    data = ('﻿' + buf.getvalue()).encode('utf-8')
    path = DATA_DIR / 'single_char_results.csv'
    path.write_bytes(data)
    return send_file(path, mimetype='text/csv', as_attachment=True, download_name='single_char_results.csv')
