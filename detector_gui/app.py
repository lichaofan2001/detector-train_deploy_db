"""
Detector Training GUI - Flask Application

Provides a web-based interface for:
- Configuring and launching model training
- Real-time training log streaming
- Managing, downloading, and exporting trained models

This module is standalone and does NOT import from the detector package,
so it starts instantly without loading torch/CUDA/pandas.
"""

import os
import sys
import json
import random
import signal
import subprocess
import threading
import queue
import time
from pathlib import Path
from datetime import datetime

import yaml
from flask import Flask, render_template, request, jsonify, Response, send_file


# ---------------------------------------------------------------------------
# Path utilities (standalone, no detector imports needed)
# ---------------------------------------------------------------------------

def _get_project_root():
    """Return the project root directory (parent of detector_gui/)."""
    return Path(__file__).parent.parent


def _get_yolo7_cfg_base():
    """Return the user-level yolo_train_tool directory."""
    base = Path('./yolo_train_tool').resolve()
    base.mkdir(parents=True, exist_ok=True)
    return base


def _ensure_yolo7_cfg_initialized():
    """Initialize ~/.yolo7_cfg with default configs from project directory if not already done."""
    cfg_base = _get_yolo7_cfg_base()
    project_root = _get_project_root()

    def copy_if_missing(src_dir, dest_subdir, extensions=None):
        dest_dir = cfg_base / dest_subdir
        dest_dir.mkdir(parents=True, exist_ok=True)
        if not extensions:
            return
        for ext in extensions:
            if not list(dest_dir.glob(f'*{ext}')) and src_dir.exists():
                for src_file in src_dir.glob(f'*{ext}'):
                    dest_file = dest_dir / src_file.name
                    if not dest_file.exists():
                        import shutil
                        shutil.copy2(src_file, dest_file)

    copy_if_missing(project_root / 'cfg', 'cfg', ['.yaml'])
    copy_if_missing(project_root / 'data', 'data', ['.yaml'])

    runs_dir = cfg_base / 'runs' / 'train'
    runs_dir.mkdir(parents=True, exist_ok=True)


_ensure_yolo7_cfg_initialized()


def _get_cfg_path(filename=None):
    cfg_dir = _get_yolo7_cfg_base() / 'cfg'
    cfg_dir.mkdir(parents=True, exist_ok=True)
    if filename:
        return cfg_dir / filename
    return cfg_dir


def _get_data_path(filename=None):
    data_dir = _get_yolo7_cfg_base() / 'data'
    data_dir.mkdir(parents=True, exist_ok=True)
    if filename:
        return data_dir / filename
    return data_dir


def _get_detector_home():
    home_dir = Path.home() / '.detector'
    home_dir.mkdir(parents=True, exist_ok=True)
    return home_dir


def _get_ckpoints_path(filename=None):
    ckpoints_dir = _get_yolo7_cfg_base() / 'ckpoints'
    ckpoints_dir.mkdir(parents=True, exist_ok=True)
    if filename:
        return ckpoints_dir / filename
    return ckpoints_dir


def _get_trained_weights_dir():
    return _get_yolo7_cfg_base() / 'runs' / 'train'


_DEFAULT_CFG = 'yolov7-tiny-silu-sqnet.yaml'
_DEFAULT_WEIGHTS = ' '
_DEFAULT_DATA = 'data_test.yaml'
_DEFAULT_HYP = 'hyp.scratch.tiny.yaml'
_IMAGE_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'}


def _default_train_dir():
    return str(_get_yolo7_cfg_base() / 'runs' / 'train')


def _safe_read_yaml(path):
    try:
        with open(path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f) or {}
    except Exception:
        return {}


def _format_size(size_bytes):
    for unit in ('B', 'KB', 'MB', 'GB'):
        if size_bytes < 1024:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024
    return f"{size_bytes:.1f} TB"


# ---------------------------------------------------------------------------
# Flask Application
# ---------------------------------------------------------------------------

app = Flask(__name__,
            template_folder=str(Path(__file__).parent / 'templates'),
            static_folder=str(Path(__file__).parent / 'static'))

def get_version():
    """Read version from VERSION.txt file."""
    version_file = _get_project_root() / 'VERSION.txt'
    if version_file.exists():
        return version_file.read_text(encoding='utf-8').strip()
    return "1.0.0"

# Global state for training process
_train_state = {
    'process': None,
    'log_queue': queue.Queue(),
    'is_running': False,
    'start_time': None,
    'params': None,
    'session_id': None,
    'log_file': None,
}
_log_lock = threading.Lock()

# Global state for ONNX export process
_export_state = {
    'process': None,
    'is_running': False,
    'log_lines': [],
    'result_path': None,
}
_export_lock = threading.Lock()

def _get_train_log_dir():
    """Return the directory for training log files."""
    log_dir = _get_yolo7_cfg_base() / 'logs'
    log_dir.mkdir(exist_ok=True)
    return log_dir

def _get_or_create_session_id():
    """Get existing session ID from state, or create new one if training not running."""
    with _log_lock:
        if _train_state['is_running'] and _train_state['session_id']:
            return _train_state['session_id']
        return f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{random.randint(1000, 9999)}"

def _write_log_to_file(session_id, line, mode='a'):
    """Write a log line to the persistent log file."""
    try:
        log_file = _get_train_log_dir() / f"{session_id}.log"
        with open(log_file, mode, encoding='utf-8') as f:
            f.write(line + '\n')
        return log_file
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Routes - Pages
# ---------------------------------------------------------------------------

@app.route('/')
def index():
    return render_template('index.html', version=get_version())


# ---------------------------------------------------------------------------
# Routes - API: Defaults
# ---------------------------------------------------------------------------

@app.route('/api/defaults')
def api_defaults():
    """Return default configuration paths and parameters."""
    cfg_dir = _get_cfg_path()
    cfg_files = sorted(cfg_dir.glob('*.yaml')) if cfg_dir.exists() else []

    data_dir = _get_data_path()
    hyp_files = sorted(data_dir.glob('hyp.*.yaml')) if data_dir.exists() else []
    data_files = sorted(data_dir.glob('data*.yaml')) if data_dir.exists() else []

    ckpt_dir = _get_ckpoints_path()
    weight_files = sorted(ckpt_dir.glob('*.pt')) if ckpt_dir.exists() else []
    project_ckpts = _get_project_root() / 'ckpoints'
    if project_ckpts.exists():
        weight_files.extend(sorted(project_ckpts.glob('*.pt')))

    trained_weights_dir = _get_trained_weights_dir()
    trained_weights = []
    if trained_weights_dir.exists():
        for exp_dir in sorted(trained_weights_dir.glob('*')):
            if exp_dir.is_dir():
                weights_dir = exp_dir / 'weights'
                if weights_dir.exists():
                    for w in sorted(weights_dir.glob('*.pt')):
                        trained_weights.append({'name': str(exp_dir.name) + '/' + w.name, 'path': str(w), 'img_width': 1024, 'img_height': 1024})

    defaults = {
        'cfg': str(_get_cfg_path(_DEFAULT_CFG)),
        'data': str(_get_data_path(_DEFAULT_DATA)),
        'weights': str(_get_ckpoints_path(_DEFAULT_WEIGHTS)),
        'hyp': str(_get_data_path(_DEFAULT_HYP)),
        'epochs': 20,
        'batch_size': 32,
        'img_size_train': 1024,
        'img_size_test': 1024,
        'device': '0',
        'workers': 16,
        'name': 'exp',
        'freeze': '',
        'available_cfgs': [{'name': f.name, 'path': str(f)} for f in cfg_files],
        'available_hyps': [{'name': f.name, 'path': str(f)} for f in hyp_files],
        'available_data': [{'name': f.name, 'path': str(f)} for f in data_files],
        'available_weights': [{'name': f.name, 'path': str(f)} for f in weight_files],
        'trained_weights': trained_weights,
    }
    return jsonify(defaults)


@app.route('/api/gpu/list')
def api_gpu_list():
    """Get list of available GPU devices with usage info."""
    import torch
    gpu_list = []
    try:
        import pynvml
        pynvml.nvmlInit()
        has_pynvml = True
    except:
        has_pynvml = False

    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            mem_total = torch.cuda.get_device_properties(i).total_memory / (1024**3)
            mem_used = torch.cuda.memory_allocated(i) / (1024**3) if has_pynvml else 0
            util_rate = 0
            if has_pynvml:
                try:
                    handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                    util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                    util_rate = util.gpu
                except:
                    pass
            gpu_list.append({
                'id': str(i),
                'name': torch.cuda.get_device_name(i),
                'memory_total': round(mem_total, 1),
                'memory_used': round(mem_used, 1),
                'utilization': util_rate
            })
    if has_pynvml:
        try:
            pynvml.nvmlShutdown()
        except:
            pass
    return jsonify({'gpus': gpu_list, 'cuda_available': torch.cuda.is_available()})
def api_model_info():
    """Get model input size from a PT model file."""
    weight_path = request.args.get('path', '')
    if not weight_path:
        return jsonify({'error': 'No weight path provided'}), 400

    weight_path = os.path.normpath(os.path.expanduser(weight_path))
    if not os.path.exists(weight_path):
        return jsonify({'error': f'File not found: {weight_path}'}), 400

    try:
        import torch
        from detector.utils.module_loader import torch_load

        ckpt = torch_load(weight_path, map_location='cpu')

        img_size = 640
        if 'model' in ckpt:
            model = ckpt['model']
            if hasattr(model, 'yaml'):
                yaml_dict = model.yaml
                if 'img_size' in yaml_dict:
                    img_size = yaml_dict['img_size']
                    if isinstance(img_size, list):
                        img_size = img_size[0] if len(img_size) >= 1 else 640
                elif 'depth_multiple' in yaml_dict:
                    img_size = 640
        elif 'opt' in ckpt and isinstance(ckpt['opt'], dict):
            img_size = ckpt['opt'].get('img_size', 640)
            if isinstance(img_size, list):
                img_size = img_size[0]

        return jsonify({
            'img_size': img_size,
            'img_width': img_size,
            'img_height': img_size,
        })
    except Exception as e:
        return jsonify({'error': str(e), 'img_size': 640, 'img_width': 640, 'img_height': 640}), 200


# ---------------------------------------------------------------------------
# Routes - API: File Browser
# ---------------------------------------------------------------------------

@app.route('/api/browse')
def api_browse():
    """Browse filesystem for file selection."""
    browse_path = request.args.get('path', '')
    if not browse_path:
        browse_path = str(_get_project_root())
    ext_filter = request.args.get('filter', '')
    browse_path = os.path.expanduser(browse_path)
    browse_path = os.path.normpath(browse_path)
    browse_path = browse_path.replace('\\', '/')
    if not os.path.isdir(browse_path):
        return jsonify({'error': f'Not a directory: {browse_path}'}), 400

    items = []
    try:
        entries = sorted(
            os.listdir(browse_path),
            key=lambda x: (not os.path.isdir(os.path.join(browse_path, x)), x.lower())
        )
        for entry in entries:
            if entry.startswith('.'):
                continue
            full = os.path.join(browse_path, entry)
            is_dir = os.path.isdir(full)
            if not is_dir and ext_filter:
                exts = [e.strip() for e in ext_filter.split(',')]
                if not any(entry.endswith(e) for e in exts):
                    continue
            info = {'name': entry, 'path': full, 'is_dir': is_dir}
            if not is_dir:
                try:
                    info['size'] = _format_size(os.path.getsize(full))
                except OSError:
                    info['size'] = ''
            items.append(info)
    except PermissionError:
        return jsonify({'error': f'Permission denied: {browse_path}'}), 403

    shortcuts = [
        {'name': '项目根目录', 'path': str(_get_project_root()).replace('\\', '/')},
        {'name': '模型配置 (cfg)', 'path': str(_get_cfg_path()).replace('\\', '/')},
        {'name': '数据配置 (data)', 'path': str(_get_data_path()).replace('\\', '/')},
        {'name': '训练权重 (runs/train)', 'path': str(_get_trained_weights_dir()).replace('\\', '/')},
        {'name': '主目录', 'path': str(Path.home()).replace('\\', '/')},
    ]

    drives = []
    if os.name == 'nt':
        import string
        for letter in string.ascii_uppercase:
            drive = f'{letter}:\\'
            if os.path.isdir(drive):
                drives.append({'name': f'{letter}:\\', 'path': drive})

    parent = str(Path(browse_path).parent).replace('\\', '/')
    current_normalized = browse_path.replace('\\', '/')

    is_at_drive_root = os.name == 'nt' and len(browse_path) == 3 and browse_path.endswith(':\\')

    return jsonify({
        'current': current_normalized,
        'parent': parent if (parent != current_normalized and not is_at_drive_root) else None,
        'items': [{'name': i['name'], 'path': i['path'].replace('\\', '/'), 'is_dir': i['is_dir'], 'size': i.get('size', '')} for i in items],
        'shortcuts': shortcuts,
        'drives': drives if drives else None,
        'is_at_drive_root': is_at_drive_root,
    })


# ---------------------------------------------------------------------------
# Routes - API: Dataset Split
# ---------------------------------------------------------------------------

@app.route('/api/dataset/split', methods=['POST'])
def api_dataset_split():
    """Auto-split an image directory into train/val/test sets (6:2:2).

    Generates .txt list files and a data.yaml configuration file.
    Expects JSON body: {img_dir, nc, names, seed?}
    """
    data = request.json or {}
    img_dir = data.get('img_dir', '').strip()
    nc = int(data.get('nc', 1))
    names = data.get('names', [])
    seed = int(data.get('seed', 42))

    if not img_dir or not os.path.isdir(img_dir):
        return jsonify({'error': f'图片目录不存在: {img_dir}'}), 400

    if not names or len(names) != nc:
        return jsonify({'error': f'类别名称数量({len(names)})与类别数({nc})不匹配'}), 400

    # Scan for image files
    img_dir_path = Path(img_dir)
    all_images = sorted([
        str(p) for p in img_dir_path.rglob('*')
        if p.suffix.lower() in _IMAGE_EXTS and p.is_file()
    ])

    if not all_images:
        return jsonify({'error': f'目录中未找到图片文件: {img_dir}'}), 400

    # Shuffle and split 6:2:2
    random.seed(seed)
    random.shuffle(all_images)
    total = len(all_images)
    n_train = int(total * 0.6)
    n_val = int(total * 0.2)

    train_imgs = all_images[:n_train]
    val_imgs = all_images[n_train:n_train + n_val]
    test_imgs = all_images[n_train + n_val:]

    # Output directory: alongside the image directory
    output_dir = img_dir_path.parent / f'{img_dir_path.name}_split'
    output_dir.mkdir(parents=True, exist_ok=True)

    # Write txt files
    for name_str, img_list in [('train', train_imgs), ('val', val_imgs), ('test', test_imgs)]:
        txt_path = output_dir / f'{name_str}.txt'
        with open(txt_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(img_list) + '\n')

    # Write data.yaml
    yaml_path = output_dir / 'data.yaml'
    yaml_content = {
        'train': str(output_dir / 'train.txt'),
        'val': str(output_dir / 'val.txt'),
        'test': str(output_dir / 'test.txt'),
        'nc': nc,
        'names': names,
    }
    with open(yaml_path, 'w', encoding='utf-8') as f:
        yaml.dump(yaml_content, f, default_flow_style=False, allow_unicode=True)

    return jsonify({
        'yaml_path': str(yaml_path),
        'output_dir': str(output_dir),
        'total_images': total,
        'train_count': len(train_imgs),
        'val_count': len(val_imgs),
        'test_count': len(test_imgs),
    })


# ---------------------------------------------------------------------------
# Routes - API: Training
# ---------------------------------------------------------------------------

def _build_train_command(params):
    """Build the detector-train command from params dict."""
    cmd = [sys.executable, '-m', 'detector.train']

    project_root = _get_project_root()

    def to_abs(val):
        if val and not os.path.isabs(val):
            return str(project_root / val)
        return val

    for key in ('cfg', 'data', 'hyp'):
        val = params.get(key, '').strip()
        if val:
            cmd.extend([f'--{key}', to_abs(val)])

    weights = params.get('weights', '').strip()
    if weights:
        cmd.extend(['--weights', to_abs(weights)])

    cmd.extend(['--epochs', str(params.get('epochs', 20))])
    cmd.extend(['--batch-size', str(params.get('batch_size', 32))])
    train_sz = params.get('img_size_train', 1024)
    test_sz = params.get('img_size_test', 1024)
    cmd.extend(['--img-size', str(train_sz), str(test_sz)])
    cmd.extend(['--device', str(params.get('device', '0'))])
    cmd.extend(['--workers', str(params.get('workers', 16))])

    freeze = params.get('freeze', '')
    if freeze:
        cmd.extend(['--freeze']+ freeze.split())

    project = params.get('project', '').strip()
    if project:
        cmd.extend(['--project', project])
    else:
        cmd.extend(['--project', _default_train_dir()])

    name = params.get('name', '').strip()
    if name:
        cmd.extend(['--name', name])

    save_period = params.get('save_period', -1)
    if save_period and int(save_period) > 0:
        cmd.extend(['--save_period', str(save_period)])

    return cmd


def _stream_output(proc, log_queue, session_id=None, log_file=None):
    """Read process stdout and push to the log queue."""
    try:
        for line in iter(proc.stdout.readline, ''):
            if line:
                line = line.rstrip('\n')
                log_queue.put(line)
                if session_id and log_file:
                    _write_log_to_file(session_id, line, 'a')
        proc.stdout.close()
    except Exception:
        pass

    proc.wait()
    end_msg = f'\n===== 训练结束 (返回码: {proc.returncode}) ====='
    log_queue.put(end_msg)
    if session_id and log_file:
        _write_log_to_file(session_id, end_msg, 'a')
    with _log_lock:
        _train_state['is_running'] = False


@app.route('/api/train/start', methods=['POST'])
def api_train_start():
    """Start a training run."""
    with _log_lock:
        if _train_state['is_running']:
            return jsonify({'error': '训练正在进行中，请先停止当前训练'}), 409

    params = request.json or {}
    cmd = _build_train_command(params)

    while not _train_state['log_queue'].empty():
        try:
            _train_state['log_queue'].get_nowait()
        except queue.Empty:
            break

    session_id = _get_or_create_session_id()
    log_file = _write_log_to_file(session_id, '', 'w')

    _train_state['log_queue'].put('===== 启动训练 =====')
    _train_state['log_queue'].put(f'命令: {" ".join(cmd)}')
    _train_state['log_queue'].put('')

    if log_file:
        _write_log_to_file(session_id, '===== 启动训练 =====', 'a')
        _write_log_to_file(session_id, f'命令: {" ".join(cmd)}', 'a')
        _write_log_to_file(session_id, '', 'a')

    try:
        popen_kwargs = dict(
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True, bufsize=1,
            env=os.environ.copy(),
            cwd=str(_get_project_root()),
        )
        if sys.platform != 'win32':
            popen_kwargs['preexec_fn'] = os.setsid
        proc = subprocess.Popen(cmd, **popen_kwargs)
    except Exception as e:
        return jsonify({'error': f'启动训练失败: {e}'}), 500

    with _log_lock:
        _train_state['process'] = proc
        _train_state['is_running'] = True
        _train_state['start_time'] = datetime.now().isoformat()
        _train_state['params'] = params
        _train_state['session_id'] = session_id
        _train_state['log_file'] = str(log_file) if log_file else None

    t = threading.Thread(target=_stream_output, args=(proc, _train_state['log_queue'], session_id, log_file), daemon=True)
    t.start()

    return jsonify({'status': 'started', 'pid': proc.pid, 'command': ' '.join(cmd), 'session_id': session_id})


@app.route('/api/train/stop', methods=['POST'])
def api_train_stop():
    """Stop the running training process."""
    with _log_lock:
        proc = _train_state['process']
        if not proc or not _train_state['is_running']:
            return jsonify({'error': '当前没有正在运行的训练'}), 404

    try:
        if sys.platform != 'win32':
            os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
        else:
            proc.terminate()
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        if sys.platform != 'win32':
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        else:
            proc.kill()
    except Exception:
        proc.kill()

    with _log_lock:
        _train_state['is_running'] = False
        _train_state['log_queue'].put('\n===== 训练已被手动停止 =====')

    return jsonify({'status': 'stopped'})


@app.route('/api/train/status')
def api_train_status():
    with _log_lock:
        return jsonify({
            'is_running': _train_state['is_running'],
            'start_time': _train_state['start_time'],
            'params': _train_state['params'],
            'session_id': _train_state['session_id'],
            'log_file': _train_state['log_file'],
        })


@app.route('/api/train/log')
def api_train_log():
    """SSE endpoint for real-time training log streaming."""
    def generate():
        while True:
            try:
                msg = _train_state['log_queue'].get(timeout=1)
                yield f"data: {json.dumps({'line': msg})}\n\n"
            except queue.Empty:
                yield f"data: {json.dumps({'heartbeat': True})}\n\n"

    return Response(generate(), mimetype='text/event-stream',
                    headers={'Cache-Control': 'no-cache', 'X-Accel-Buffering': 'no'})


@app.route('/api/train/log/history')
def api_train_log_history():
    """Get historical log lines from a session file.

    Query params:
        session_id: The training session ID
        from_line: (optional) Line number to start from (0-indexed)
    """
    session_id = request.args.get('session_id')
    from_line = int(request.args.get('from_line', 0))

    if not session_id:
        return jsonify({'error': 'session_id is required'}), 400

    log_file = _get_train_log_dir() / f"{session_id}.log"
    if not log_file.exists():
        return jsonify({'error': 'Log file not found', 'lines': [], 'total': 0}), 404

    try:
        with open(log_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        lines = [l.rstrip('\n') for l in lines]
        return jsonify({
            'lines': lines[from_line:],
            'total': len(lines),
            'from_line': from_line,
        })
    except Exception as e:
        return jsonify({'error': str(e), 'lines': [], 'total': 0}), 500


# ---------------------------------------------------------------------------
# Routes - API: Model Management
# ---------------------------------------------------------------------------

@app.route('/api/models')
def api_models():
    """List all training experiments."""
    train_dir = Path(request.args.get('project', _default_train_dir()))
    experiments = []

    if not train_dir.exists():
        return jsonify({'experiments': [], 'project': str(train_dir)})

    for exp_dir in sorted(train_dir.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True):
        if not exp_dir.is_dir():
            continue

        exp_info = {
            'name': exp_dir.name,
            'path': str(exp_dir),
            'created': datetime.fromtimestamp(exp_dir.stat().st_mtime).strftime('%Y-%m-%d %H:%M:%S'),
        }

        opt_file = exp_dir / 'opt.yaml'
        if opt_file.exists():
            opt_data = _safe_read_yaml(opt_file)
            exp_info['epochs'] = opt_data.get('epochs', '?')
            exp_info['batch_size'] = opt_data.get('batch_size', '?')
            exp_info['img_size'] = opt_data.get('img_size', '?')
            exp_info['data'] = os.path.basename(str(opt_data.get('data', '?')))
            exp_info['cfg'] = os.path.basename(str(opt_data.get('cfg', '?')))

        weights_dir = exp_dir / 'weights'
        if weights_dir.exists():
            weight_files = list(weights_dir.glob('*.pt'))
            exp_info['weight_files'] = []
            for wf in sorted(weight_files, key=lambda p: p.name):
                exp_info['weight_files'].append({
                    'name': wf.name,
                    'path': str(wf),
                    'size': _format_size(wf.stat().st_size),
                })
            # Also list any .onnx files
            onnx_files = list(weights_dir.glob('*.onnx'))
            exp_info['onnx_files'] = []
            for of in sorted(onnx_files, key=lambda p: p.name):
                exp_info['onnx_files'].append({
                    'name': of.name,
                    'path': str(of),
                    'size': _format_size(of.stat().st_size),
                })

        # Also check onnx in experiment root
        for of in exp_dir.glob('*.onnx'):
            if 'onnx_files' not in exp_info:
                exp_info['onnx_files'] = []
            exp_info['onnx_files'].append({
                'name': of.name,
                'path': str(of),
                'size': _format_size(of.stat().st_size),
            })

        result_images = []
        for img_name in ('results.png', 'confusion_matrix.png', 'F1_curve.png',
                         'PR_curve.png', 'P_curve.png', 'R_curve.png'):
            if (exp_dir / img_name).exists():
                result_images.append(img_name)
        exp_info['result_images'] = result_images

        results_file = exp_dir / 'results.txt'
        if results_file.exists():
            try:
                with open(results_file, 'r') as f:
                    lines = f.readlines()
                if lines:
                    last_line = lines[-1].strip().split()
                    if len(last_line) >= 12:
                        exp_info['final_metrics'] = {
                            'precision': float(last_line[8]),
                            'recall': float(last_line[9]),
                            'mAP_50': float(last_line[10]),
                            'mAP_50_95': float(last_line[11]),
                        }
            except Exception:
                pass

        experiments.append(exp_info)

    return jsonify({'experiments': experiments, 'project': str(train_dir)})


@app.route('/api/models/download')
def api_model_download():
    """Download a model file (.pt or .onnx)."""
    file_path = request.args.get('path', '')
    if not file_path or not os.path.isfile(file_path):
        return jsonify({'error': '文件不存在'}), 404
    if not file_path.endswith(('.pt', '.onnx')):
        return jsonify({'error': '仅支持下载 .pt 和 .onnx 文件'}), 400
    return send_file(file_path, as_attachment=True,
                     download_name=os.path.basename(file_path))


@app.route('/api/models/image')
def api_model_image():
    """Serve a result image."""
    img_path = request.args.get('path', '')
    if not img_path or not os.path.isfile(img_path):
        return jsonify({'error': 'Image not found'}), 404
    if not img_path.endswith(('.png', '.jpg', '.jpeg')):
        return jsonify({'error': 'Invalid file type'}), 400
    return send_file(img_path)


@app.route('/api/models/<exp_name>/detail')
def api_model_detail(exp_name):
    """Get detailed info for a specific experiment."""
    project = request.args.get('project', _default_train_dir())
    exp_dir = Path(project) / exp_name
    if not exp_dir.exists():
        return jsonify({'error': f'Experiment not found: {exp_name}'}), 404

    detail = {'name': exp_name, 'path': str(exp_dir)}

    opt_file = exp_dir / 'opt.yaml'
    if opt_file.exists():
        detail['opt'] = _safe_read_yaml(opt_file)

    hyp_file = exp_dir / 'hyp.yaml'
    if hyp_file.exists():
        detail['hyp'] = _safe_read_yaml(hyp_file)

    results_file = exp_dir / 'results.txt'
    if results_file.exists():
        try:
            with open(results_file, 'r') as f:
                detail['results_raw'] = f.read()
        except Exception:
            pass

    return jsonify(detail)


@app.route('/api/models/<exp_name>/delete', methods=['POST'])
def api_model_delete(exp_name):
    """Delete an experiment directory and all its contents."""
    project = request.args.get('project', _default_train_dir())
    exp_dir = Path(project) / exp_name
    if not exp_dir.exists():
        return jsonify({'error': f'Experiment not found: {exp_name}'}), 404

    import shutil
    try:
        shutil.rmtree(exp_dir)
        return jsonify({'status': 'deleted', 'name': exp_name})
    except Exception as e:
        return jsonify({'error': f'删除失败: {e}'}), 500


# ---------------------------------------------------------------------------
# Routes - API: ONNX Export
# ---------------------------------------------------------------------------

@app.route('/api/models/export-onnx', methods=['POST'])
def api_export_onnx():
    """Export a .pt model to ONNX format.

    JSON body: {weights, img_width, img_height, grid}
    - weights: path to .pt file
    - img_width: integer (e.g. 1024)
    - img_height: integer (e.g. 1024)
    - grid: boolean (true=NPU format, false=GPU format)
    """
    with _export_lock:
        if _export_state['is_running']:
            return jsonify({'error': 'ONNX 导出正在进行中，请等待完成'}), 409

    data = request.json or {}
    weights = data.get('weights', '').strip()
    weights = os.path.normpath(weights)
    img_width = int(data.get('img_width', 1024))
    img_height = int(data.get('img_height', 1024))
    grid = data.get('grid', False)

    if not weights or not os.path.isfile(weights):
        return jsonify({'error': f'权重文件不存在: {weights}'}), 400

    cmd = [sys.executable, '-m', 'detector.cli',
           '--weights', weights,
           '--img-size', str(img_width), str(img_height)]
    if grid:
        cmd.append('--grid')

    expected_onnx = os.path.splitext(weights)[0] + '.onnx'

    with _export_lock:
        _export_state['is_running'] = True
        _export_state['log_lines'] = []
        _export_state['result_path'] = None

    def run_export():
        try:
            popen_kwargs = dict(
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True, bufsize=1,
                env=os.environ.copy(),
                cwd=str(_get_project_root()),
            )
            proc = subprocess.Popen(cmd, **popen_kwargs)
            with _export_lock:
                _export_state['process'] = proc

            for line in iter(proc.stdout.readline, ''):
                if line:
                    with _export_lock:
                        _export_state['log_lines'].append(line.rstrip('\n'))
            proc.stdout.close()
            proc.wait()

            with _export_lock:
                _export_state['is_running'] = False
                if proc.returncode == 0 and os.path.isfile(expected_onnx):
                    _export_state['result_path'] = expected_onnx
                    _export_state['log_lines'].append(f'导出成功: {expected_onnx}')
                else:
                    _export_state['log_lines'].append(f'导出失败 (返回码: {proc.returncode})')
        except Exception as e:
            with _export_lock:
                _export_state['is_running'] = False
                _export_state['log_lines'].append(f'导出异常: {e}')

    t = threading.Thread(target=run_export, daemon=True)
    t.start()

    return jsonify({'status': 'started', 'command': ' '.join(cmd)})


@app.route('/api/models/export-onnx/status')
def api_export_onnx_status():
    """Check ONNX export progress."""
    with _export_lock:
        return jsonify({
            'is_running': _export_state['is_running'],
            'log': _export_state['log_lines'],
            'result_path': _export_state['result_path'],
        })


# ---------------------------------------------------------------------------
# Routes - API: Inference
# ---------------------------------------------------------------------------

_inference_state = {
    'process': None,
    'is_running': False,
    'output_dir': None,
    'result': None,
    'result_queue': None,
}
_inference_lock = threading.Lock()


@app.route('/api/inference/start', methods=['POST'])
def api_inference_start():
    """Start inference on a video file.

    Accepts multipart form data:
    - video: video file
    - weights: path to model weights
    - frame_interval: sample every N frames
    - max_frames: maximum number of frames to process
    - output_folder: output folder name
    - img_width: inference image width
    - img_height: inference image height
    - conf_thres: confidence threshold
    - iou_thres: IOU threshold for NMS
    """
    with _inference_lock:
        if _inference_state['is_running']:
            return jsonify({'error': '推理正在进行中，请等待完成'}), 409

    if 'video' not in request.files:
        return jsonify({'error': '未上传视频文件'}), 400

    video_file = request.files['video']
    inference_mode = request.form.get('inference_mode', 'fast').strip()
    weights = request.form.get('weights', '').strip()
    qwen_model_path = request.form.get('qwen_model_path', '').strip()
    frame_interval = int(request.form.get('frame_interval', 5))
    max_frames = int(request.form.get('max_frames', 100))
    output_folder = request.form.get('output_folder', 'inference_result')
    conf_thres = float(request.form.get('conf_thres', 0.25))
    iou_thres = float(request.form.get('iou_thres', 0.45))
    save_images = request.form.get('save_images', 'true').lower() == 'true'
    only_save_with_objects = request.form.get('only_save_with_objects', 'false').lower() == 'true'
    ignore_class_id = request.form.get('ignore_class_id', '').strip()

    # 不再从前端获取 img_size，设置默认值为 1024，后续会自动从模型获取
    img_size = 1024

    if inference_mode == 'fast':
        if not weights:
            return jsonify({'error': '请选择YOLO模型权重文件'}), 400

        if not os.path.isfile(weights):
            return jsonify({'error': f'YOLO模型文件不存在: {weights}'}), 400
    else:
        if not qwen_model_path:
            return jsonify({'error': '请选择Qwen模型文件夹'}), 400

        if not os.path.isdir(qwen_model_path):
            return jsonify({'error': f'Qwen模型文件夹不存在: {qwen_model_path}'}), 400

    import uuid
    temp_dir = _get_yolo7_cfg_base() / 'temp'
    temp_dir.mkdir(parents=True, exist_ok=True)
    video_filename = f'{uuid.uuid4().hex}.mp4'
    video_path = temp_dir / video_filename
    video_file.save(str(video_path))

    if os.path.isabs(output_folder):
        output_dir = Path(output_folder)
    else:
        base_output_dir = _get_yolo7_cfg_base() / output_folder
        if base_output_dir.exists():
            i = 2
            output_dir = base_output_dir.parent / f'{base_output_dir.name}{i}'
            while output_dir.exists():
                i += 1
                output_dir = base_output_dir.parent / f'{base_output_dir.name}{i}'
        else:
            output_dir = base_output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    with _inference_lock:
        _inference_state['is_running'] = True
        _inference_state['output_dir'] = str(output_dir)
        result_queue = _inference_state.get('result_queue') or queue.Queue()
        _inference_state['result_queue'] = result_queue

    def run_inference():
        nonlocal img_size, conf_thres, iou_thres  # 声明非局部变量
        try:
            import cv2
            import numpy as np
            import torch
            from pathlib import Path
            from detector_gui.model_manager import (
                create_model_manager,
                postprocess_and_draw,
                save_detection_results
            )

            # 解析忽略类别ID
            ignore_classes = []
            if ignore_class_id:
                ignore_classes = [int(c.strip()) for c in ignore_class_id.split(',') if c.strip().isdigit()]
            print(f'忽略类别ID: {ignore_classes}')

            # 创建统一的模型管理器
            if inference_mode == 'fast':
                # YOLO 模型
                model_manager = create_model_manager(
                    'yolo', weights,
                    img_size=img_size,
                    conf_thres=conf_thres,
                    iou_thres=iou_thres
                )
                model_manager.load_model()
                img_size = model_manager.img_size
                print(f'[调试] YOLO模型已加载，输入尺寸: {img_size}')
            else:
                # 大模型
                model_manager = create_model_manager('large', qwen_model_path)
                model_manager.load_model()
                print(f'[调试] 大模型已加载')

            classnames = model_manager.classnames

            print(f'视频路径: {video_path}')
            video_cap = cv2.VideoCapture(str(video_path))
            if not video_cap.isOpened():
                raise Exception(f'无法打开视频文件: {video_path}')
            total_frames = int(video_cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = video_cap.get(cv2.CAP_PROP_FPS)
            print(f'总帧数: {total_frames}, FPS: {fps}')

            frames_dir = output_dir / 'frames'
            frames_dir.mkdir(parents=True, exist_ok=True)
            save_dir = output_dir / 'labels'
            save_dir.mkdir(parents=True, exist_ok=True)
            images_dir = output_dir / 'images' if save_images else None
            if images_dir:
                images_dir.mkdir(parents=True, exist_ok=True)

            frame_count = 0
            saved_count = 0
            frame_idx = 0

            while saved_count < max_frames:
                ret, frame = video_cap.read()
                if not ret:
                    break

                if frame_idx % frame_interval == 0:
                    # 计算当前帧的时间（秒）
                    frame_time = frame_idx / fps
                    # 格式化为 秒_毫秒
                    total_seconds = int(frame_time)
                    milliseconds = int((frame_time % 1) * 1000)
                    time_str = f'{total_seconds:02d}s_{milliseconds:03d}ms'
                    
                    frame_path = frames_dir / f'frame_{saved_count:06d}_{time_str}.jpg'
                    cv2.imwrite(str(frame_path), frame)

                    imgH, imgW = frame.shape[:2]
                    txt_path = save_dir / f'frame_{saved_count:06d}_{time_str}.txt'

                    # 使用统一的模型管理器进行推理
                    try:
                        # 执行推理
                        bboxes = model_manager.inference(str(frame_path))

                        # 使用统一的后处理函数
                        filtered_bboxes, plotted, det_count = postprocess_and_draw(
                            bboxes, (imgH, imgW),
                            classnames=classnames,
                            ignore_classes=ignore_classes,
                            original_image=frame
                        )

                        # 保存结果到标签文件
                        if filtered_bboxes:
                            save_detection_results(
                                filtered_bboxes, str(txt_path),
                                classnames=classnames,
                                ignore_classes=ignore_classes
                            )
                        else:
                            # 无检测结果时创建空文件
                            open(txt_path, 'w').close()
                    except Exception as e:
                        print(f'推理出错: {e}')
                        # 出错时创建空的标签文件
                        open(txt_path, 'w').close()
                        plotted = frame.copy()
                        det_count = 0

                    # 只保存带目标的图像（如果启用）
                    if save_images and images_dir:
                        if not only_save_with_objects or det_count > 0:
                            img_save_path = images_dir / f'frame_{saved_count:06d}_{time_str}.jpg'
                            cv2.imwrite(str(img_save_path), plotted)

                    import posixpath
                    rel_path = f'images/frame_{saved_count:06d}_{time_str}.jpg'
                    url_path = posixpath.join('api', 'inference', 'image', rel_path).replace('\\', '/')
                    result_queue.put({
                        'type': 'image',
                        'url': '/' + url_path,
                        'det_count': det_count,
                        'frame_idx': saved_count,
                    })

                    saved_count += 1

                frame_count += 1
                if frame_count % 50 == 0:
                    print(f'处理帧: {frame_count}/{total_frames}')

                frame_idx += 1

            video_cap.release()
            os.remove(str(video_path))

            # 卸载模型
            model_manager.unload_model()

            result_txt_files = []
            base_dir = str(output_dir)
            for txt_file in sorted(save_dir.glob('*.txt')):
                rel_path = os.path.relpath(str(txt_file), base_dir)
                url_path = posixpath.join('api', 'inference', 'file', rel_path).replace('\\', '/')
                result_txt_files.append('/' + url_path)

            with _inference_lock:
                _inference_state['is_running'] = False
                _inference_state['result'] = {
                    'status': 'completed',
                    'output_dir': str(output_dir),
                    'images': [],
                    'txt_files': result_txt_files
                }
                result_queue.put({'type': 'done'})

            print(f'推理完成！共处理 {saved_count} 帧，结果保存在: {output_dir}')

        except Exception as e:
            import traceback
            traceback.print_exc()
            with _inference_lock:
                _inference_state['is_running'] = False
                _inference_state['result'] = {'status': 'error', 'error': str(e)}
            print(f'推理失败: {e}')

    t = threading.Thread(target=run_inference, daemon=True)
    t.start()

    return jsonify({
        'status': 'started',
        'output_dir': str(output_dir),
        'message': '推理已开始，请稍候...'
    })


@app.route('/api/inference/start-folder', methods=['POST'])
def api_inference_start_folder():
    """Start inference on an image folder.

    Accepts JSON data:
    - weights: path to model weights
    - source_folder: path to image folder
    - output_folder: output folder name
    - img_width: inference image width
    - img_height: inference image height
    - conf_thres: confidence threshold
    - iou_thres: IOU threshold for NMS
    - save_images: whether to save result images
    """
    with _inference_lock:
        if _inference_state['is_running']:
            return jsonify({'error': '推理正在进行中，请等待完成'}), 409

    data = request.json or {}
    inference_mode = data.get('inference_mode', 'fast').strip()
    weights = data.get('weights', '').strip()
    qwen_model_path = data.get('qwen_model_path', '').strip()
    source_folder = data.get('source_folder', '').strip()
    output_folder = data.get('output_folder', 'inference_result')
    conf_thres = float(data.get('conf_thres', 0.25))
    iou_thres = float(data.get('iou_thres', 0.45))
    save_images = data.get('save_images', True)
    only_save_with_objects = data.get('only_save_with_objects', False)
    ignore_class_id = data.get('ignore_class_id', '').strip()

    # 不再从前端获取 img_size，设置默认值为 1024，后续会自动从模型获取
    img_size = 1024

    if not source_folder:
        return jsonify({'error': '请选择图片文件夹'}), 400

    if not os.path.isdir(source_folder):
        return jsonify({'error': f'图片文件夹不存在: {source_folder}'}), 400

    if inference_mode == 'fast':
        if not weights:
            return jsonify({'error': '请选择YOLO模型权重文件'}), 400

        if not os.path.isfile(weights):
            return jsonify({'error': f'YOLO模型文件不存在: {weights}'}), 400
    else:
        if not qwen_model_path:
            return jsonify({'error': '请选择Qwen模型文件夹'}), 400

        if not os.path.isdir(qwen_model_path):
            return jsonify({'error': f'Qwen模型文件夹不存在: {qwen_model_path}'}), 400

    supported_exts = ['.jpg', '.jpeg', '.png', '.bmp', '.webp']
    image_files = []
    for ext in supported_exts:
        image_files.extend(list(Path(source_folder).glob(f'*{ext}')))
        image_files.extend(list(Path(source_folder).glob(f'*{ext.upper()}')))
    image_files = sorted(image_files)

    if not image_files:
        return jsonify({'error': f'文件夹中没有找到图片文件: {source_folder}'}), 400

    if os.path.isabs(output_folder):
        output_dir = Path(output_folder)
    else:
        base_output_dir = _get_yolo7_cfg_base() / output_folder
        if base_output_dir.exists():
            i = 2
            output_dir = base_output_dir.parent / f'{base_output_dir.name}{i}'
            while output_dir.exists():
                i += 1
                output_dir = base_output_dir.parent / f'{base_output_dir.name}{i}'
        else:
            output_dir = base_output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    with _inference_lock:
        _inference_state['is_running'] = True
        _inference_state['output_dir'] = str(output_dir)
        result_queue = _inference_state.get('result_queue') or queue.Queue()
        _inference_state['result_queue'] = result_queue

    def run_inference_folder():
        try:
            import cv2
            import numpy as np
            from detector_gui.model_manager import (
                create_model_manager,
                postprocess_and_draw,
                save_detection_results
            )

            # 解析忽略类别ID
            ignore_classes = []
            if ignore_class_id:
                ignore_classes = [int(c.strip()) for c in ignore_class_id.split(',') if c.strip().isdigit()]
            print(f'忽略类别ID: {ignore_classes}')

            # 创建统一的模型管理器
            if inference_mode == 'fast':
                # YOLO 模型
                model_manager = create_model_manager(
                    'yolo', weights,
                    img_size=1024,
                    conf_thres=conf_thres,
                    iou_thres=iou_thres
                )
                model_manager.load_model()
                img_size = model_manager.img_size
                print(f'[调试] YOLO模型已加载，输入尺寸: {img_size}')
            else:
                # 大模型
                model_manager = create_model_manager('large', qwen_model_path)
                model_manager.load_model()
                print(f'[调试] 大模型已加载')

            classnames = model_manager.classnames

            test_list_file = output_dir / 'test_images.txt'
            with open(test_list_file, 'w', encoding='utf-8') as f:
                for img_file in image_files:
                    f.write(str(img_file) + '\n')
            print(f'test_images.txt 已写入，共 {len(image_files)} 张图片')

            save_dir = output_dir / 'labels'
            save_dir.mkdir(parents=True, exist_ok=True)
            images_dir = output_dir / 'images' if save_images else None
            if images_dir:
                images_dir.mkdir(parents=True, exist_ok=True)

            def on_result(image_path, det_count):
                import posixpath
                rel_path = os.path.relpath(image_path, str(output_dir))
                url_path = posixpath.join('api', 'inference', 'image', rel_path).replace('\\', '/')
                result_queue.put({
                    'type': 'image',
                    'url': '/' + url_path,
                    'det_count': det_count,
                })

            # 遍历图片进行推理
            for img_file in image_files:
                img_path = str(img_file)
                img = cv2.imread(img_path)
                if img is None:
                    print(f'无法读取图片: {img_path}')
                    continue

                imgH, imgW = img.shape[:2]
                img_name = os.path.basename(img_file)
                txt_name = os.path.splitext(img_name)[0] + '.txt'
                txt_path = save_dir / txt_name

                try:
                    # 执行推理
                    bboxes = model_manager.inference(img_path)

                    # 使用统一的后处理函数
                    filtered_bboxes, plotted, det_count = postprocess_and_draw(
                        bboxes, (imgH, imgW),
                        classnames=classnames,
                        ignore_classes=ignore_classes,
                        original_image=img
                    )

                    # 保存结果到标签文件
                    if filtered_bboxes:
                        save_detection_results(
                            filtered_bboxes, str(txt_path),
                            classnames=classnames,
                            ignore_classes=ignore_classes
                        )
                    else:
                        # 无检测结果时创建空文件
                        open(txt_path, 'w').close()
                except Exception as e:
                    print(f'推理出错: {e}')
                    # 出错时创建空的标签文件
                    open(txt_path, 'w').close()
                    plotted = img.copy()
                    det_count = 0

                # 只保存带目标的图像（如果启用）
                if save_images and images_dir:
                    if not only_save_with_objects or det_count > 0:
                        img_save_path = images_dir / img_name
                        cv2.imwrite(str(img_save_path), plotted)
                        on_result(str(img_save_path), det_count)

            # 卸载模型
            model_manager.unload_model()

            import posixpath
            result_txt_files = []
            base_dir = str(output_dir)

            labels_dir = output_dir / 'labels'
            for txt_file in sorted(labels_dir.glob('*.txt')):
                rel_path = os.path.relpath(str(txt_file), base_dir)
                url_path = posixpath.join('api', 'inference', 'file', rel_path).replace('\\', '/')
                result_txt_files.append('/' + url_path)

            with _inference_lock:
                _inference_state['is_running'] = False
                _inference_state['result'] = {
                    'status': 'completed',
                    'output_dir': str(output_dir),
                    'images': [],
                    'txt_files': result_txt_files
                }
                result_queue.put({'type': 'done'})

            print(f'推理完成！结果保存在: {output_dir} (images/ 和 labels/)')

        except Exception as e:
            import traceback
            traceback.print_exc()
            with _inference_lock:
                _inference_state['is_running'] = False
                _inference_state['result'] = {'status': 'error', 'error': str(e)}
            print(f'推理失败: {e}')

    t = threading.Thread(target=run_inference_folder, daemon=True)
    t.start()

    return jsonify({
        'status': 'started',
        'output_dir': str(output_dir),
        'message': '推理已开始，请稍候...'
    })


@app.route('/api/inference/result')
def api_inference_result():
    with _inference_lock:
        result = _inference_state.get('result')
        if result:
            _inference_state['result'] = None
        return jsonify(result or {'status': 'no_result'})


@app.route('/api/inference/file/<path:filename>')
def api_inference_file(filename):
    from flask import send_from_directory
    filename = filename.replace('\\', '/')
    output_dir = _inference_state.get('output_dir')
    if output_dir:
        directory = output_dir
    else:
        directory = str(_get_yolo7_cfg_base())
    safe_path = os.path.normpath(os.path.join(directory, filename))
    if not safe_path.startswith(directory):
        return 'Forbidden', 403
    return send_from_directory(directory, filename)


@app.route('/api/inference/image/<path:filename>')
def api_inference_image(filename):
    from flask import send_from_directory
    filename = filename.replace('\\', '/')
    output_dir = _inference_state.get('output_dir')
    if output_dir:
        directory = output_dir
    else:
        directory = str(_get_yolo7_cfg_base())
    safe_path = os.path.normpath(os.path.join(directory, filename))
    if not safe_path.startswith(directory):
        return 'Forbidden', 403
    return send_from_directory(directory, filename)


@app.route('/api/inference/status')
def api_inference_status():
    """Check inference status."""
    with _inference_lock:
        return jsonify({
            'is_running': _inference_state['is_running'],
            'output_dir': _inference_state['output_dir'],
        })


@app.route('/api/inference/stream')
def api_inference_stream():
    """SSE endpoint for real-time inference result streaming."""
    result_queue = _inference_state.get('result_queue')

    def generate():
        if result_queue is None:
            yield f"data: {json.dumps({'error': 'no inference started'})}\n\n"
            return

        while True:
            try:
                msg = result_queue.get(timeout=1)
                if msg is None:
                    break
                yield f"data: {json.dumps(msg)}\n\n"
            except queue.Empty:
                yield f"data: {json.dumps({'heartbeat': True})}\n\n"

    return Response(generate(), mimetype='text/event-stream',
                    headers={'Cache-Control': 'no-cache', 'X-Accel-Buffering': 'no'})


_realtime_state = {
    'is_running': False,
    'thread': None,
    'stop_event': None,
    'weights': '',
    'conf_thres': 0.25,
    'iou_thres': 0.45,
    'img_size': 1024,
    'device': '',
    'target_module': '',
    'mmap_shared_dir': '',
    'mmap_category': 'eo',
    'last_frame_jpg': None,
    'last_result': [],
    'last_error': '',
    'last_frame_time': '',
    'last_frame_id': 0,
}
_realtime_lock = threading.Lock()


class MmapCacheReader:
    """读取 Node.js MmapCacheManager 写入的共享内存缓存文件。

    二进制文件格式（与 cache.js 完全一致）：
      Header (256B) → Slot Info Table (max_slots × 64B) → Data Area (max_slots × slot_capacity)
    """
    import struct as _struct

    HEADER_SIZE = 256
    SLOT_INFO_SIZE = 64
    MAGIC = b'DSC1'

    # Header 偏移
    H_MAGIC = 0           # 4B ascii
    H_VERSION = 4         # 4B uint32
    H_MAX_SLOTS = 8       # 4B uint32
    H_SLOT_CAP = 12       # 4B uint32
    H_WRITE_INDEX = 16    # 4B uint32
    H_TOTAL_WRITTEN = 20  # 8B uint64

    # Slot Info 偏移
    S_FRAME_ID = 0        # 8B uint64
    S_JPG_SIZE = 8        # 4B uint32
    S_JSON_SIZE = 12      # 4B uint32
    S_FORMAT = 16         # 1B uint8 (0=jpg, 1=tif)
    S_TIMESTAMP = 24      # 8B uint64 (对齐到 offset 24)

    def __init__(self, shared_dir):
        self.shared_dir = shared_dir
        self.fds = {}
        self.headers = {}

    def _read_at(self, fd, offset, size):
        """从指定偏移量读取指定字节数。"""
        os.lseek(fd, offset, os.SEEK_SET)
        data = b''
        while len(data) < size:
            chunk = os.read(fd, size - len(data))
            if not chunk:
                break
            data += chunk
        return bytearray(data)

    def _read_header(self, fd):
        buf = self._read_at(fd, 0, self.HEADER_SIZE)
        magic = bytes(buf[self.H_MAGIC:self.H_MAGIC + 4])
        if magic != self.MAGIC:
            raise ValueError(f'Invalid cache file magic: {magic}')
        import struct
        return {
            'max_slots': struct.unpack_from('<I', buf, self.H_MAX_SLOTS)[0],
            'slot_capacity': struct.unpack_from('<I', buf, self.H_SLOT_CAP)[0],
            'write_index': struct.unpack_from('<I', buf, self.H_WRITE_INDEX)[0],
            'total_written': struct.unpack_from('<Q', buf, self.H_TOTAL_WRITTEN)[0],
        }

    def init(self, category):
        file_path = os.path.join(self.shared_dir, f'{category}_cache.bin')
        if not os.path.exists(file_path):
            raise FileNotFoundError(f'Cache file not found: {file_path}')

        fd = os.open(file_path, os.O_RDONLY | os.O_BINARY)
        self.fds[category] = fd
        self.headers[category] = self._read_header(fd)
        hdr = self.headers[category]
        print(f'[mmap] {category} initialized: {file_path} '
              f'(max_slots={hdr["max_slots"]}, slot_cap={hdr["slot_capacity"]}, '
              f'write_idx={hdr["write_index"]}, total={hdr["total_written"]})', flush=True)
        return True

    def read_latest(self, category):
        """对照 cache.js readLatest() 实现：重新读取 header，计算最新槽位，读取数据。"""
        if category not in self.fds:
            return None

        import struct
        try:
            fd = self.fds[category]

            # 每次重新读取 header（因为写入进程会更新 write_index 和 total_written）
            hdr = self._read_header(fd)
            self.headers[category] = hdr

            if hdr['total_written'] == 0:
                return None

            # 与 cache.js readLatest() 完全一致的槽位计算
            write_index = hdr['write_index']
            max_slots = hdr['max_slots']
            latest_slot = (write_index - 1) if write_index > 0 else (max_slots - 1)

            # 读取 Slot Info
            info_offset = self.HEADER_SIZE + latest_slot * self.SLOT_INFO_SIZE
            slot_info = self._read_at(fd, info_offset, self.SLOT_INFO_SIZE)

            frame_id = struct.unpack_from('<Q', slot_info, self.S_FRAME_ID)[0]
            if frame_id == 0:
                return None

            jpg_size = struct.unpack_from('<I', slot_info, self.S_JPG_SIZE)[0]
            json_size = struct.unpack_from('<I', slot_info, self.S_JSON_SIZE)[0]
            fmt = slot_info[self.S_FORMAT]
            timestamp = struct.unpack_from('<Q', slot_info, self.S_TIMESTAMP)[0]

            if jpg_size == 0:
                return None

            # 读取 Data Area
            data_area_offset = self.HEADER_SIZE + max_slots * self.SLOT_INFO_SIZE
            slot_data_offset = data_area_offset + latest_slot * hdr['slot_capacity']

            data_buf = self._read_at(fd, slot_data_offset, jpg_size + json_size)

            img_buffer = bytes(data_buf[:jpg_size])
            json_buffer = bytes(data_buf[jpg_size:jpg_size + json_size])

            return {
                'img_buffer': img_buffer,
                'json_buffer': json_buffer,
                'metadata': {
                    'format': 'tif' if fmt == 1 else 'jpg',
                    'timestamp': timestamp,
                    'frame_id': frame_id,
                }
            }
        except Exception as e:
            print(f'[mmap] read_latest error: {e}', flush=True)
            return None

    def close(self):
        for fd in self.fds.values():
            try:
                os.close(fd)
            except OSError:
                pass
        self.fds = {}
        self.headers = {}

_recording_state = {
    'is_recording': False,
    'output_path': '',
    'start_time': None,
}
_recording_lock = threading.Lock()


_ws_client = None
_ws_lock = threading.Lock()
_ws_connected = False


def _init_electron_ws_client():
    global _ws_client, _ws_connected
    ws_port = os.environ.get('ELECTRON_WS_PORT', '')
    if not ws_port:
        print('[WS] ELECTRON_WS_PORT not set, skipping WebSocket connection')
        return

    try:
        import websocket
    except ImportError:
        try:
            import pip
            pip.install('websocket-client')
            import websocket
        except:
            print('[WS] websocket-client not available')
            return

    def on_message(ws, message):
        try:
            msg = json.loads(message)
            method = msg.get('method', '')
            msg_id = msg.get('id')
            params = msg.get('params', {})

            if method == 'module.receive_result':
                result_data = params.get('result_data', {})
                print(f'[WS] Received result from other module: {result_data}')
                with _ws_lock:
                    _ws_connected = True
                ws.send(json.dumps({
                    'jsonrpc': '2.0',
                    'result': {'status': 'received'},
                    'id': msg_id
                }))
        except Exception as e:
            print(f'[WS] Message error: {e}')

    def on_error(ws, error):
        print(f'[WS] Error: {error}')
        with _ws_lock:
            _ws_connected = False

    def on_close(ws, *args):
        print('[WS] Connection closed')
        with _ws_lock:
            _ws_connected = False

    def on_open(ws):
        print('[WS] Connected to Electron宿主')
        ws.send(json.dumps({
            'jsonrpc': '2.0',
            'method': 'module.register',
            'params': {
                'id': 'yolo_train_tool',
                'name': 'YOLO训练工具',
                'version': '1.0.0',
                'services': [{
                    'name': 'realtimeResult',
                    'methods': ['sendResult', 'receiveResult']
                }]
            },
            'id': 1
        }))
        with _ws_lock:
            _ws_connected = True

        def heartbeat():
            while True:
                time.sleep(20)
                try:
                    ws.send(json.dumps({
                        'jsonrpc': '2.0',
                        'method': 'module.heartbeat',
                        'params': {'id': 'yolo_train_tool'}
                    }))
                except:
                    break
        t = threading.Thread(target=heartbeat, daemon=True)
        t.start()

    try:
        ws_url = f'ws://127.0.0.1:{ws_port}'
        print(f'[WS] Connecting to {ws_url}...')
        _ws_client = websocket.WebSocketApp(ws_url)
        _ws_client.on_message = on_message
        _ws_client.on_error = on_error
        _ws_client.on_close = on_close
        _ws_client.on_open = on_open
        t = threading.Thread(target=_ws_client.run_forever, daemon=True)
        t.start()
    except Exception as e:
        print(f'[WS] Failed to connect: {e}')


def _send_to_module(target_module, method, params=None):
    global _ws_client, _ws_connected
    if not _ws_connected:
        print('[WS] Not connected to Electron宿主')
        return None
    try:
        import uuid
        msg_id = str(uuid.uuid4())
        _ws_client.send(json.dumps({
            'jsonrpc': '2.0',
            'method': 'module.relay',
            'params': {
                'target': target_module,
                'method': method,
                'params': params or {}
            },
            'id': msg_id
        }))
        print(f'[WS] Sent relay to {target_module}.{method}')
        return {'status': 'sent'}
    except Exception as e:
        print(f'[WS] Send error: {e}')
        return {'error': str(e)}


def _publish_topic(topic, message):
    """通过 WebSocket 向 Electron 宿主发布 topic 消息。"""
    global _ws_client, _ws_connected
    if not _ws_connected:
        print('[WS] Not connected, cannot publish topic')
        return False
    try:
        import uuid
        msg_id = str(uuid.uuid4())
        _ws_client.send(json.dumps({
            'jsonrpc': '2.0',
            'method': 'topic.publish',
            'params': {
                'topic': topic,
                'message': message if isinstance(message, str) else json.dumps(message),
            },
            'id': msg_id
        }))
        print(f'[WS] Published to topic "{topic}"', flush=True)
        return True
    except Exception as e:
        print(f'[WS] Publish error: {e}')
        return False


def _compose_stream_source(stream_url, stream_port):
    stream_url = (stream_url or '').strip()
    stream_port = str(stream_port or '').strip()
    if not stream_port:
        return stream_url
    if '://' in stream_url:
        from urllib.parse import urlparse, urlunparse
        parsed = urlparse(stream_url)
        host_part = parsed.netloc.split('@')[-1]
        if parsed.netloc and ':' not in host_part:
            netloc = f'{parsed.netloc}:{stream_port}'
            return urlunparse((parsed.scheme, netloc, parsed.path, parsed.params, parsed.query, parsed.fragment))
    return f'{stream_url}:{stream_port}'


def _stop_realtime_stream():
    with _realtime_lock:
        stop_event = _realtime_state.get('stop_event')
        thread = _realtime_state.get('thread')

    if stop_event:
        stop_event.set()
    if thread and thread.is_alive():
        thread.join(timeout=2)

    with _realtime_lock:
        _realtime_state['is_running'] = False
        _realtime_state['thread'] = None
        _realtime_state['stop_event'] = None


def _run_realtime_stream_worker():
    import sys
    print('[realtime] worker thread started', flush=True)
    import cv2
    import numpy as np
    import torch
    from detector.models.experimental import attempt_load
    from detector.utils.datasets import letterbox
    from detector.utils.general import non_max_suppression, scale_coords, check_img_size
    from detector.utils.torch_utils import select_device

    with _realtime_lock:
        weights = _realtime_state['weights']
        conf_thres = _realtime_state['conf_thres']
        iou_thres = _realtime_state['iou_thres']
        device_str = _realtime_state['device']
        stop_event = _realtime_state['stop_event']
        mmap_shared_dir = _realtime_state['mmap_shared_dir']
        mmap_category = _realtime_state['mmap_category']
        _realtime_state['last_error'] = ''

    print(f'[realtime] weights={weights}, device={device_str}, mmap_dir={mmap_shared_dir}, category={mmap_category}', flush=True)

    try:
        if device_str != 'cpu' and not torch.cuda.is_available():
            device_str = 'cpu'
        device = select_device(device_str or '')
        print('[realtime] loading model...', flush=True)
        model = attempt_load(weights, map_location=device)
        stride = max(int(model.stride.max()), 32)
        img_size = int(check_img_size(640, s=stride))
        model.eval()
        half = device.type != 'cpu'
        if half:
            model.half()
        names = model.module.names if hasattr(model, 'module') else model.names
        print(f'[realtime] model loaded, stride={stride}, img_size={img_size}', flush=True)
    except Exception as e:
        print(f'[realtime] model load FAILED: {e}', flush=True)
        with _realtime_lock:
            _realtime_state['last_error'] = f'模型加载失败: {e}'
            _realtime_state['is_running'] = False
        return

    try:
        mmap_reader = MmapCacheReader(mmap_shared_dir)
        mmap_reader.init(mmap_category)
        print(f'[realtime] mmap initialized: {mmap_shared_dir}/{mmap_category}', flush=True)
    except Exception as e:
        print(f'[realtime] mmap init FAILED: {e}', flush=True)
        with _realtime_lock:
            _realtime_state['last_error'] = f'共享内存初始化失败: {e}'
            _realtime_state['is_running'] = False
        return

    last_frame_id = 0
    loop_count = 0
    while not stop_event.is_set():
        try:
            data = mmap_reader.read_latest(mmap_category)
            loop_count += 1
            if loop_count <= 3:
                print(f'[realtime-debug] loop={loop_count}, data={data is not None}, last_fid={last_frame_id}')
                if data:
                    print(f'[realtime-debug] frame_id={data["metadata"]["frame_id"]}, jpg_size={len(data["img_buffer"])}')
            if not data or data['metadata']['frame_id'] == last_frame_id:
                time.sleep(0.01)
                continue

            last_frame_id = data['metadata']['frame_id']
            print(f'[realtime-debug] processing frame_id={last_frame_id}')
            img_buffer = data['img_buffer']
            nparr = np.frombuffer(img_buffer, np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if frame is None:
                time.sleep(0.01)
                continue

            img = letterbox(frame, new_shape=img_size, stride=stride, auto=False)[0]
            img = img[:, :, ::-1].transpose(2, 0, 1)
            img = np.ascontiguousarray(img)
            img = torch.from_numpy(img).to(device)
            img = img.half() if half else img.float()
            img /= 255.0
            if img.ndimension() == 3:
                img = img.unsqueeze(0)

            with torch.no_grad():
                pred = model(img)[0]
            det = non_max_suppression(pred, conf_thres=conf_thres, iou_thres=iou_thres)[0]

            plotted = frame.copy()
            img_h, img_w = frame.shape[:2]
            result_items = []
            if det is not None and len(det):
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], plotted.shape).round()
                for row in det.tolist():
                    x1, y1, x2, y2, conf, cls = row
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                    cls_id = int(cls)
                    label = names[cls_id] if names and cls_id < len(names) else str(cls_id)
                    cv2.rectangle(plotted, (x1, y1), (x2, y2), (255, 0, 0), 2)
                    cv2.putText(plotted, f'{label} {conf:.2f}', (x1, max(20, y1 - 8)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                    result_items.append({
                        'label': label,
                        'conf': float(conf),
                        'xyxy': [x1, y1, x2, y2],
                        'cls': cls_id,
                        'cx': (x1 + x2) / 2.0 / img_w,
                        'cy': (y1 + y2) / 2.0 / img_h,
                        'w': (x2 - x1) / float(img_w),
                        'h': (y2 - y1) / float(img_h),
                    })

            ok_jpg, encoded = cv2.imencode('.jpg', plotted, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
            if not ok_jpg:
                continue

            with _realtime_lock:
                _realtime_state['last_frame_jpg'] = encoded.tobytes()
                _realtime_state['last_result'] = result_items
                _realtime_state['last_frame_time'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                _realtime_state['last_frame_id'] = data['metadata']['frame_id']
                _realtime_state['last_error'] = ''
        except Exception as e:
            with _realtime_lock:
                _realtime_state['last_error'] = f'推理异常: {e}'
            time.sleep(0.2)

    try:
        mmap_reader.close()
    except Exception:
        pass

    with _realtime_lock:
        _realtime_state['is_running'] = False


@app.route('/api/realtime/connect', methods=['POST'])
def api_realtime_connect():
    data = request.json or {}
    weights = (data.get('weights') or '').strip()
    conf_thres = float(data.get('conf_thres', 0.25))
    iou_thres = float(data.get('iou_thres', 0.45))
    device = (data.get('device') or '').strip()
    target_module = (data.get('target_module') or '').strip()
    mmap_shared_dir = (data.get('mmap_shared_dir') or '').strip()
    mmap_category = 'eo'  # 只支持 EO 可见光

    # 未指定共享内存目录时，从当前工作目录向上查找 modules/data-service/shared
    if not mmap_shared_dir:
        search_dir = os.getcwd()
        found = None
        for _ in range(6):  # 最多向上查找 6 层
            candidate = os.path.join(search_dir, 'modules', 'data-service', 'shared')
            if os.path.isdir(candidate):
                found = candidate.replace('\\', '/')
                break
            parent = os.path.dirname(search_dir)
            if parent == search_dir:
                break
            search_dir = parent
        if found:
            mmap_shared_dir = found
        else:
            return jsonify({'error': '未指定共享内存目录，且未找到默认路径 (modules/data-service/shared)'}), 400
    if not os.path.isdir(mmap_shared_dir):
        return jsonify({'error': f'共享内存目录不存在: {mmap_shared_dir}'}), 400
    if not weights:
        return jsonify({'error': '请选择模型权重文件'}), 400
    if not os.path.isfile(weights):
        return jsonify({'error': f'模型文件不存在: {weights}'}), 400

    _stop_realtime_stream()

    stop_event = threading.Event()
    with _realtime_lock:
        _realtime_state['is_running'] = True
        _realtime_state['stop_event'] = stop_event
        _realtime_state['weights'] = weights
        _realtime_state['conf_thres'] = conf_thres
        _realtime_state['iou_thres'] = iou_thres
        _realtime_state['device'] = device
        _realtime_state['target_module'] = target_module
        _realtime_state['mmap_shared_dir'] = mmap_shared_dir
        _realtime_state['mmap_category'] = mmap_category
        _realtime_state['last_frame_jpg'] = None
        _realtime_state['last_result'] = []
        _realtime_state['last_error'] = ''
        _realtime_state['last_frame_time'] = ''
        _realtime_state['last_frame_id'] = 0

    worker = threading.Thread(target=_run_realtime_stream_worker, daemon=True)
    with _realtime_lock:
        _realtime_state['thread'] = worker
    worker.start()

    return jsonify({'status': 'started', 'mmap_dir': mmap_shared_dir})


@app.route('/api/realtime/disconnect', methods=['POST'])
def api_realtime_disconnect():
    _stop_realtime_stream()
    return jsonify({'status': 'stopped'})


@app.route('/api/realtime/status')
def api_realtime_status():
    with _realtime_lock:
        return jsonify({
            'is_running': _realtime_state['is_running'],
            'mmap_dir': _realtime_state['mmap_shared_dir'],
            'last_frame_time': _realtime_state['last_frame_time'],
            'last_result': _realtime_state['last_result'],
            'error': _realtime_state['last_error'],
        })


@app.route('/api/realtime/frame')
def api_realtime_frame():
    with _realtime_lock:
        frame = _realtime_state.get('last_frame_jpg')
    if not frame:
        return Response(status=204)
    return Response(frame, mimetype='image/jpeg',
                    headers={'Cache-Control': 'no-store, no-cache, must-revalidate, max-age=0'})


@app.route('/api/realtime/save-current', methods=['POST'])
def api_realtime_save_current():
    with _realtime_lock:
        frame = _realtime_state.get('last_frame_jpg')
        result = list(_realtime_state.get('last_result') or [])
    if not frame:
        return jsonify({'error': '当前没有可保存的帧'}), 400

    save_dir = _get_yolo7_cfg_base() / 'realtime_captures'
    save_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime('%Y%m%d_%H%M%S_%f')[:-3]
    frame_path = save_dir / f'realtime_{ts}.jpg'
    result_path = save_dir / f'realtime_{ts}.json'

    with open(frame_path, 'wb') as f:
        f.write(frame)
    with open(result_path, 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    return jsonify({
        'status': 'saved',
        'frame_path': str(frame_path),
        'result_path': str(result_path),
    })


@app.route('/api/realtime/send-current', methods=['POST'])
def api_realtime_send_current():
    with _realtime_lock:
        result = list(_realtime_state.get('last_result') or [])
        frame_id = _realtime_state.get('last_frame_id', 0)

    if not result:
        return jsonify({'error': '当前帧暂无推理结果可发送'}), 400

    # 构造检测消息
    detections = []
    for item in result:
        detections.append({
            'cls': item.get('cls', 0),
            'score': round(item.get('conf', 0), 6),
            'cx': round(item.get('cx', 0), 6),
            'cy': round(item.get('cy', 0), 6),
            'w': round(item.get('w', 0), 6),
            'h': round(item.get('h', 0), 6),
        })

    message = {
        'from_module_id': 'eo_web',
        'frame_id': str(frame_id),
        'roi': [],
        'detections': detections,
    }

    print(f'[realtime] Sending detection: frame_id={frame_id}, detections={len(detections)}', flush=True)
    return jsonify({
        'status': 'ready',
        'message': f'检测结果已发送 (frame_id={frame_id}, {len(detections)} detections)',
        'payload': message,
    })


@app.route('/api/recording/start', methods=['POST'])
def api_recording_start():
    data = request.json or {}
    output_path = (data.get('output_path') or '').strip()

    with _recording_lock:
        if _recording_state['is_recording']:
            return jsonify({'error': '已经在录制中'}), 400

        if not output_path:
            output_path = str(_get_yolo7_cfg_base() / 'recordings')
        _recording_state['output_path'] = output_path
        _recording_state['is_recording'] = True
        _recording_state['start_time'] = datetime.now()

    return jsonify({
        'status': 'started',
        'output_path': output_path,
        'start_time': _recording_state['start_time'].isoformat(),
    })


@app.route('/api/recording/stop', methods=['POST'])
def api_recording_stop():
    data = request.json or {}
    video_data = data.get('video_data')
    filename = data.get('filename', '')

    with _recording_lock:
        if not _recording_state['is_recording']:
            return jsonify({'error': '当前没有在录制'}), 400

        output_path = _recording_state['output_path']
        start_time = _recording_state['start_time']
        _recording_state['is_recording'] = False

    if not video_data:
        return jsonify({'error': '未收到视频数据'}), 400

    Path(output_path).mkdir(parents=True, exist_ok=True)

    if not filename:
        ts = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f'screen_record_{ts}.webm'

    file_path = os.path.join(output_path, filename)

    try:
        if isinstance(video_data, str):
            import base64
            video_bytes = base64.b64decode(video_data)
        else:
            video_bytes = video_data

        with open(file_path, 'wb') as f:
            f.write(video_bytes)

        duration = (datetime.now() - start_time).total_seconds() if start_time else 0

        return jsonify({
            'status': 'saved',
            'file_path': file_path,
            'filename': filename,
            'duration': round(duration, 2),
            'size': len(video_bytes),
        })
    except Exception as e:
        return jsonify({'error': f'保存失败: {str(e)}'}), 500


@app.route('/api/recording/status', methods=['GET'])
def api_recording_status():
    with _recording_lock:
        return jsonify({
            'is_recording': _recording_state['is_recording'],
            'output_path': _recording_state['output_path'],
            'start_time': _recording_state['start_time'].isoformat() if _recording_state['start_time'] else None,
        })


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def run_gui(host='0.0.0.0', port=5000, debug=False):
    """Launch the training GUI web server."""
    _init_electron_ws_client()
    print(f"\n{'='*50}")
    print(f"  Detector 训练管理界面")
    print(f"  访问地址: http://{host}:{port}")
    print(f"{'='*50}\n")
    app.run(host=host, port=port, debug=debug, threaded=True)


if __name__ == '__main__':
    run_gui(debug=True)
