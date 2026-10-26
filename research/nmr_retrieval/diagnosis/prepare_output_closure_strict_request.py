"""功能概述：生成零前向产物闭包核验的精确请求与未签 job。

main 只登记已封存文件的路径和字节 SHA，保留源 NN 阶段 inner0/external1。
请求仅授权原始字节与元数据完整性检查；独审输入门和 job 门通过后才可执行。
"""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
REMOTE = '/home/user/IRfingerprints/SpectrExpert/outputs/nmr_top10_barrier_diagnosis_20261025'

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def save(name, value):
    with (ROOT / name).open('x') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')

def main():
    helper = ROOT / 'close_failed_supervision_outputs_strict.py'
    fail = json.loads((ROOT / 'replay_device_indexing_independent_failure_review.json').read_text())
    assert fail['independent'] is True and fail['accounting_status'] == 'PASS'
    paths = {
        'protocol': ('protocol_device_indexing.json', 'protocol_device_indexing.json'),
        'job': ('replay_device_indexing_job.json', 'replay_device_indexing_job.json'),
        'replay_admission': ('device_index_replay_independent_admission.json', 'device_index_replay_independent_admission.json'),
        'worker': ('runtime_device_indexing.py', 'runtime_device_indexing.py'),
        'supervisor': ('supervise_stage_rss_exit_boundary.py', 'supervise_stage_rss_exit_boundary.py'),
        'query_manifest': ('query_manifest.json', 'selected_query_manifest.json'),
        'remote_manifest': ('collected/replay_device_indexing_remote_manifest.json', 'replay_device_indexing_remote_manifest.json')
    }
    refs = {key: {'path': REMOTE + '/' + remote, 'sha256': sha(ROOT / local)}
            for key, (local, remote) in paths.items()}
    assert refs['remote_manifest']['sha256'] == '8cc079d4b9a936c82994eb0c888010d2145d71dc3a8ffd6ff49675ea6ed953b7'
    request = {
        'schema': 'failed_supervision_output_closure_request_v1',
        'execution_scope': 'FAILED_SUPERVISION_OUTPUT_CLOSURE_ONLY',
        'execution_authorized': True, 'helper_sha256': sha(helper),
        'stage_root': REMOTE + '/replay_device_indexing_attempt',
        'write_root': REMOTE,
        'refs': refs, 'prior_role_row_forwards': 2,
        'source_consumed_role_row_forwards': 8208, 'cumulative_role_row_forwards': 8210,
        'global_remaining_role_row_forwards': 430, 'new_role_row_forwards': 0,
        'scientific_aggregation_authorized': False,
        'source_stage_expected_inner_exit': 0, 'source_stage_expected_external_exit': 1,
        'resource_monitor_gap_status': 'UNKNOWN_NOT_RECERTIFIED'
    }
    save('output_closure_strict_request.json', request)
    stage = REMOTE + '/failed_supervision_output_closure_attempt'
    job = {
        'schema': 'top10_bounded_stage_v1', 'write_root': REMOTE, 'stage_dir': stage,
        'max_seconds': 3600, 'max_rss_bytes': 4 * 1024 ** 3,
        'max_artifact_bytes': 4 * 1024 ** 3, 'max_cpu_threads': 4,
        'cwd': '/home/user/IRfingerprints/SpectrExpert',
        'source_sha256': {REMOTE + '/close_failed_supervision_outputs_strict.py': sha(helper),
                          REMOTE + '/output_closure_strict_request.json': sha(ROOT / 'output_closure_strict_request.json')},
        'argv': ['/home/user/.venvs/spectrexpert/bin/python', '-B',
                 REMOTE + '/close_failed_supervision_outputs_strict.py',
                 '--request', REMOTE + '/output_closure_strict_request.json',
                 '--request-sha256', sha(ROOT / 'output_closure_strict_request.json'),
                 '--admission', REMOTE + '/output_closure_strict_input_independent_admission.json',
                 '--admission-sha256', 'REVIEW_SHA_TO_BE_BOUND',
                 '--out', stage + '/worker', '--execute', '--threads', '4']
    }
    save('output_closure_strict_job_unsigned.json', job)
    print(json.dumps({'request_sha256': sha(ROOT / 'output_closure_strict_request.json'),
                      'job_unsigned_sha256': sha(ROOT / 'output_closure_strict_job_unsigned.json'),
                      'helper_sha256': sha(helper)}))

if __name__ == '__main__':
    main()

