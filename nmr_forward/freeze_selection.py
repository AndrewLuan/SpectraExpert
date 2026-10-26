# 功能概述：只据已完成候选的固定验证集结果冻结最终测试前模型选择，不读取任何测试标签。
# 主要职责：freeze_selection 验证候选并原子创建不可覆盖的JSON；verify_manifest 复核全部
# 冻结文件SHA与选择规则；_inspect_run核对best epoch、history对应验证指标、complete及缓存身份。
# _read_json/_file_sha/_selection分别负责拒绝非有限数、流式摘要及确定性最优选择。
# 代码库关系：读取train.py产生的best.pt、validation.json、provenance.json、complete.json、
# history.jsonl；可信PyTorch checkpoint用CPU加载。没有导入train或数据缓存，不触发模型推理。
# 输入输出与副作用：CLI --runs DIR... --out FILE；--verify FILE只读核验。所有候选必须
# 注册相同evaluation cache SHA且validation n相同；按W1最小选择primary和有辅助头候选。
# W1并列以绝对run目录字典序打破，规则与全部候选SHA写入manifest；临时文件+硬链接拒绝覆盖。
# 边界：SHA只能检测冻结文件随后改变，不能证明模型科学有效或历史上从未查看过test；
# manifest中的test未访问声明仅描述本工具执行过程，不能替代实验人员的独立测试隔离纪律。
import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile

import torch


_METRIC = 'integral_weighted_W1_ppm'
_FILES = ('best.pt', 'validation.json', 'provenance.json', 'complete.json', 'history.jsonl')
_RULE = {'metric': _METRIC, 'direction': 'minimize',
         'primary_eligible': 'all_candidates', 'auxiliary_eligible': 'aux_classes > 0',
         'tie_breaker': 'absolute_run_directory_lexicographic'}


def _reject_constant(value):
    raise ValueError(f'nonfinite JSON constant: {value}')


def _read_json(path):
    value = json.loads(Path(path).read_text(), parse_constant=_reject_constant)
    # 1e999虽符合JSON语法但在Python中成为inf，因此重新编码核验有限数。
    json.dumps(value, allow_nan=False)
    return value


def _file_sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def _number(value, label):
    if isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value):
        raise ValueError(f'{label} must be a finite number')
    return float(value)


def _sha(value, label):
    if not isinstance(value, str) or len(value) != 64 or any(c not in '0123456789abcdef' for c in value):
        raise ValueError(f'invalid {label} SHA256')
    return value


def _inspect_run(directory):
    run = Path(directory).resolve()
    # 先收集摘要，再解码，再重核，拒绝仍在更新的候选。
    hashes = {name: _file_sha(run / name) for name in _FILES}
    complete = _read_json(run / 'complete.json')
    validation = _read_json(run / 'validation.json')
    provenance = _read_json(run / 'provenance.json')
    checkpoint = torch.load(run / 'best.pt', map_location='cpu', weights_only=False)
    epoch, epochs = checkpoint.get('epoch'), complete.get('epochs')
    if type(epoch) is not int or type(epochs) is not int or not 1 <= epoch <= epochs:
        raise ValueError(f'{run}: invalid best/completed epoch')
    n = validation.get('n')
    if type(n) is not int or n <= 0:
        raise ValueError(f'{run}: validation n must be positive integer')
    value = _number(validation.get(_METRIC), 'validation W1')
    if value < 0 or value != _number(complete.get('best_validation_W1_ppm'), 'complete best W1'):
        raise ValueError(f'{run}: validation/complete best mismatch')
    history = []
    for line in (run / 'history.jsonl').read_text().splitlines():
        entry = json.loads(line, parse_constant=_reject_constant)
        json.dumps(entry, allow_nan=False)
        if entry.get('epoch') != len(history) + 1:
            raise ValueError(f'{run}: noncontiguous history')
        history.append(entry)
    if len(history) != epochs:
        raise ValueError(f'{run}: completed epochs/history length mismatch')
    matching = history[epoch - 1].get('validation', {})
    if matching.get('n') != n or _number(matching.get(_METRIC), 'best epoch W1') != value:
        raise ValueError(f'{run}: best checkpoint epoch does not match validation')
    history_best = min(_number(row.get('validation', {}).get(_METRIC), 'history W1') for row in history)
    if history_best != value:
        raise ValueError(f'{run}: selected checkpoint is not history validation minimum')
    cache = _sha(provenance.get('cache_sha256'), 'training cache')
    evaluation = _sha(provenance.get('validation_cache_sha256', cache), 'evaluation cache')
    embedded = checkpoint.get('provenance')
    if embedded is not None:
        if (embedded.get('cache_sha256') != cache or
                embedded.get('validation_cache_sha256', embedded.get('cache_sha256')) != evaluation):
            raise ValueError(f'{run}: checkpoint/registered cache provenance mismatch')
    aux_classes = checkpoint.get('aux_classes', 0)
    if type(aux_classes) is not int or aux_classes < 0:
        raise ValueError(f'{run}: invalid aux_classes')
    if not isinstance(checkpoint.get('model'), dict) or not checkpoint['model']:
        raise ValueError(f'{run}: missing model state')
    for name, tensor in checkpoint['model'].items():
        if torch.is_tensor(tensor) and not torch.isfinite(tensor).all():
            raise ValueError(f'{run}: nonfinite model parameter {name}')
    if hashes != {name: _file_sha(run / name) for name in _FILES}:
        raise ValueError(f'{run}: candidate changed while freezing')
    return {'run_directory': str(run), 'checkpoint_epoch': epoch, 'completed_epochs': epochs,
            'aux_classes': aux_classes, 'validation': validation,
            'training_cache_sha256': cache, 'evaluation_cache_sha256': evaluation,
            'files': {name: {'path': str(run / name), 'sha256': hashes[name]} for name in _FILES}}


def _selection(candidates):
    if not candidates:
        raise ValueError('at least one candidate required')
    directories = [item['run_directory'] for item in candidates]
    if len(set(directories)) != len(directories):
        raise ValueError('duplicate candidate run')
    if len({item['evaluation_cache_sha256'] for item in candidates}) != 1:
        raise ValueError('candidate evaluation cache SHA mismatch')
    if len({item['validation']['n'] for item in candidates}) != 1:
        raise ValueError('candidate validation n mismatch')
    ordered = sorted(candidates, key=lambda item: (item['validation'][_METRIC], item['run_directory']))
    auxiliary = [item for item in ordered if item['aux_classes'] > 0]
    return {'primary': ordered[0]['run_directory'],
            'auxiliary': auxiliary[0]['run_directory'] if auxiliary else None}


def freeze_selection(run_dirs, out):
    """冻结全部候选与验证集选择，返回写出的manifest；不覆写已有路径。"""
    out = Path(out)
    if out.exists():
        raise FileExistsError(out)
    candidates = sorted((_inspect_run(path) for path in run_dirs), key=lambda item: item['run_directory'])
    selected = _selection(candidates)
    manifest = {'schema_version': 1, 'created_at_utc': datetime.now(timezone.utc).isoformat(),
                'selection_rule': _RULE.copy(), 'selected': selected, 'candidates': candidates,
                'evaluation_cache_sha256': candidates[0]['evaluation_cache_sha256'],
                'validation_n': candidates[0]['validation']['n'],
                'test_access': {'accessed_by_selection_tool': False,
                    'statement': 'Selection reads only completed model and validation artifacts; no test cache, labels, or metrics are read.',
                    'scope': 'this_tool_execution_only_not_a_claim_about_prior_human_or_other_tool_access'}}
    encoded = json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False)
    out.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile('w', encoding='utf-8', dir=out.parent,
                                          prefix=out.name + '.', suffix='.tmp', delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, out)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return manifest


def verify_manifest(manifest):
    """评估前重核所有冻结文件、候选内容及选择；接收JSON路径或已解析dict。"""
    manifest = _read_json(manifest) if isinstance(manifest, (str, Path)) else manifest
    json.dumps(manifest, allow_nan=False)
    if manifest.get('schema_version') != 1 or manifest.get('selection_rule') != _RULE:
        raise ValueError('unsupported manifest version or selection rule')
    candidates = manifest['candidates']
    selected = _selection(candidates)
    if (manifest.get('selected') != selected or
            manifest.get('evaluation_cache_sha256') != candidates[0]['evaluation_cache_sha256'] or
            manifest.get('validation_n') != candidates[0]['validation']['n']):
        raise ValueError('manifest selection/summary mismatch')
    for frozen in candidates:
        current = _inspect_run(frozen['run_directory'])
        if current != frozen:
            raise ValueError(f'frozen candidate changed: {frozen["run_directory"]}')
    return manifest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--runs', nargs='+')
    parser.add_argument('--out')
    parser.add_argument('--verify')
    args = parser.parse_args()
    if args.verify:
        if args.runs or args.out:
            parser.error('--verify cannot combine with --runs/--out')
        result = verify_manifest(args.verify)
    else:
        if not args.runs or not args.out:
            parser.error('--runs and --out are required to freeze')
        result = freeze_selection(args.runs, args.out)
    print(json.dumps({'selected': result['selected'], 'candidates': len(result['candidates'])}))


if __name__ == '__main__':
    main()
