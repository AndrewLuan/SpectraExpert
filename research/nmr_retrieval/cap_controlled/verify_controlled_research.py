"""功能概述：验证新受控研究的最终证据闭合，不替换旧诊断零 test 接触门槛。
main 检查最终 completion、报告 manifest 和独审，确认唯一验证规则的预选、
完整预算/阶段收据、test 后无重选，以及曝光处置和正式系统状态。文件尚未齐全
时返回 MISSING；只读证据，不执行训练、评价或外部调用。
"""
import hashlib
import json
from pathlib import Path

M = Path(__file__).resolve().parent


def main():
    names = ['completion.json', 'research_report.md', 'research_report.input_sha256.json',
             'research_report_final_review.json', 'training_results_independent_review.json',
             'validation_results_independent_review.json', 'validation_selection.json',
             'test_freeze_independent_review.json', 'test_results_independent_review.json']
    missing = [x for x in names if not (M / x).is_file()]
    if missing:
        print('MISSING:' + ','.join(missing))
        return 2
    read = lambda name: json.loads((M / name).read_text())
    for name in names:
        if name.endswith('_review.json'):
            r = read(name)
            assert r['status'] == 'PASS' and r['independent'] is True, name
    c = read('completion.json')
    assert c['status'] == 'COMPLETE_CONTROLLED_RESEARCH'
    assert c['fits_completed'] == 8 and c['draws_per_fit'] == 524288
    assert c['validation_jobs_completed'] == 16 and c['mechanism_jobs_completed'] == 4
    assert c['test_matrix_cells_completed'] == 12
    assert c['all_stage_strict_double_exit0'] is True
    assert c['formal_system_changed'] is False and c['promotion'] is False
    assert c['external_api_calls'] == 0 and c['test_reselection_performed'] is False
    assert c['prior_zero_test_contact_readiness_actual_exit'] == 1
    assert c['test_blind_claim'] is False
    s = read('validation_selection.json')
    assert s['preselected'] in ('cap10_dual', 'fixedcenter_mixed_listwise_single')
    assert c['preselected'] == s['preselected']
    manifest = read('research_report.input_sha256.json')
    for field in ('input_sha256', 'output_sha256'):
        for path, expected in manifest[field].items():
            assert hashlib.sha256(Path(path).read_bytes()).hexdigest() == expected, path
    print('PASS')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
