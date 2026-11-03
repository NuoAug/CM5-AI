"""Read-only, task-specific cleanup inventory; never deletes/stops/uninstalls anything."""
import hashlib
import json
import os
from pathlib import Path
import subprocess

ROOT = Path('/home/mylin/CM5-AI-M')
BASELINE = Path('/home/mylin/.cm5-ai-m-task-20261103/baseline.json')
MARKER = 'CM5-AI-M authorized test 20261103'
UNITS = ('cm5-ai-m-test-web.service', 'cm5-ai-m-test-vlm.service')


def digest(path):
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError:
        return None


def command(arguments, limit=16000):
    try:
        result = subprocess.run(arguments, capture_output=True, text=True, timeout=8,
                                env=dict(os.environ, LC_ALL='C'))
        return {'exit_code': result.returncode, 'stdout': result.stdout[:limit],
                'stderr': result.stderr.strip()[:500], 'stdout_truncated': len(result.stdout) > limit}
    except (OSError, subprocess.TimeoutExpired) as error:
        return {'exit_code': None, 'error': type(error).__name__}


def root_processes():
    found = []
    for directory in Path('/proc').glob('[0-9]*'):
        if int(directory.name) == os.getpid():
            continue
        try:
            raw = (directory / 'cmdline').read_bytes()
            if str(ROOT).encode() not in raw:
                continue
            # Deliberately omit argv: the local runner uses an API key in its arguments.
            found.append({'pid': int(directory.name), 'name': (directory / 'comm').read_text().strip(),
                          'references_test_root': True, 'ownership': 'candidate_only_check_before_stopping'})
        except (OSError, ValueError):
            continue
    return found


def parse_packages(result):
    if not isinstance(result, dict) or result.get('exit_code') != 0 or not isinstance(result.get('stdout'), str) or result.get('stdout_truncated'):
        return None
    packages = {}
    for line in result['stdout'].splitlines():
        parts = line.split('\t', 2)
        if len(parts) != 3 or not parts[0]:
            return None
        packages[parts[0]] = {'version': parts[1].strip(), 'status': parts[2].strip()}
    return packages


def manual_packages(result):
    if not isinstance(result, dict) or result.get('exit_code') != 0 or not isinstance(result.get('stdout'), str) or result.get('stdout_truncated'):
        return None
    return set(result['stdout'].splitlines())


def package_diff(baseline):
    before = parse_packages(baseline.get('packages'))
    current_result = command(['dpkg-query','-W','-f=${binary:Package}\t${Version}\t${db:Status-Abbrev}\n'], limit=4*1024*1024)
    current = parse_packages(current_result)
    if before is None or current is None:
        return {'verified': False, 'reason': 'baseline_or_current_dpkg_inventory_unavailable'}
    prior_manual = manual_packages(baseline.get('manual_packages'))
    current_manual = manual_packages(command(['apt-mark','showmanual'], limit=4*1024*1024))
    changed = [{'package': name, 'before': before[name], 'current': current[name]}
               for name in sorted(before.keys() & current.keys()) if before[name] != current[name]]
    result = {'verified': True, 'added_since_baseline': sorted(current.keys() - before.keys()),
              'removed_since_baseline': sorted(before.keys() - current.keys()), 'changed_since_baseline': changed,
              'manual_mark_comparison_verified': prior_manual is not None and current_manual is not None,
              'differences_are_not_task_ownership_proof': True}
    if result['manual_mark_comparison_verified']:
        result['manual_added_since_baseline'] = sorted(current_manual - prior_manual)
        result['manual_removed_since_baseline'] = sorted(prior_manual - current_manual)
    return result


def main():
    marker = ROOT / '.task-owner'
    try:
        marker_match = marker.read_text(encoding='utf-8').strip() == MARKER
    except OSError:
        marker_match = False
    try:
        resolved = str(ROOT.resolve(strict=True))
    except OSError:
        resolved = None
    report = {'schema_version': 'cm5-ai-m-test.cleanup-audit.v1', 'read_only': True,
              'restored_initial_state': None, 'test_root': str(ROOT), 'root_exists': ROOT.exists(),
              'root_is_symlink': ROOT.is_symlink(), 'resolved_test_root': resolved,
              'owner_marker_exists': marker.exists(), 'owner_marker_matches': marker_match,
              'baseline_path': str(BASELINE), 'baseline_exists': BASELINE.is_file(),
              'baseline_sha256': digest(BASELINE), 'units': {}, 'candidate_processes': root_processes(),
              'package_restore_status': 'manual_comparison_required',
              'configuration_restore_status': 'manual_comparison_required',
              'notes': ['Matching a marker or process path alone does not authorize deletion.',
                        'Compare exact task changes against the saved baseline; never autoremove.',
                        'No raw argv, private env, SSH state or image/audio content is printed.']}
    if ROOT.exists():
        stat = ROOT.stat()
        report['test_root_uid'] = stat.st_uid
        report['test_root_gid'] = stat.st_gid
    if BASELINE.is_file():
        try:
            if BASELINE.stat().st_size <= 16*1024*1024:
                baseline = json.loads(BASELINE.read_text(encoding='utf-8'))
                report['baseline_top_level_keys'] = sorted(baseline) if isinstance(baseline, dict) else None
                if isinstance(baseline, dict):
                    report['packages'] = package_diff(baseline)
        except (OSError, ValueError):
            report['baseline_read_error'] = True
    for name in UNITS:
        destination = Path('/etc/systemd/system') / name
        source = ROOT / 'deploy' / name
        current_hash, expected_hash = digest(destination), digest(source)
        report['units'][name] = {
            'path': str(destination), 'exists': destination.exists(), 'is_symlink': destination.is_symlink(),
            'sha256': current_hash, 'matches_current_task_template': bool(current_hash and current_hash == expected_hash),
            'systemd': command(['systemctl','show',name,'--no-pager',
                                '--property=LoadState,ActiveState,SubState,UnitFileState,MainPID,NRestarts,FragmentPath'])}
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
