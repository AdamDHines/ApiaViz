"""Export an explicitly labelled UV-off report from a completed frozen study.

Presentation only: reuse immutable inputs through links in a fresh output,
retain the original report, and verify that every numerical result is unchanged.
"""
import argparse
import json
from pathlib import Path
import subprocess
import sys
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from apiaviz.research import uv_trial_report
from apiaviz.research.spectral_input import file_sha


def run(study, output):
    if output.exists(): raise FileExistsError('Use a fresh presentation output')
    p = json.loads((study/'protocol.json').read_text())
    assert p['methods'] == ['apiaviz_uv'] and p['encoder']['uv_config']['uv_enabled'] is False
    assert json.loads((study/'readiness.json').read_text())['complete']
    assert json.loads((study/'audit.json').read_text())['passed']
    rows = json.loads((study/'report/trials.json').read_text())
    old = json.loads((study/'report/summary.json').read_text())
    output.mkdir(parents=True)
    for name in ['worlds', 'trials', 'protocol.json', 'audit.json', 'readiness.json']:
        (output/name).symlink_to(study/name, target_is_directory=(study/name).is_dir())
    labels = dict(apiaviz_uv='ApiaViz UV off')
    with patch.dict(uv_trial_report.NAMES, labels):
        uv_trial_report.report(output, p, rows, movies=True)
    new = json.loads((output/'report/summary.json').read_text())
    assert new == old  # Display labels do not alter method IDs or statistics.
    for name, sha in json.loads((output/'report/artifacts.json').read_text()).items():
        assert file_sha(output/'report'/name) == sha
    movies = list((output/'report/movies').glob('*.mp4'))
    for movie in movies:
        subprocess.run(['ffmpeg', '-v', 'error', '-xerror', '-i', str(movie), '-f', 'null', '-'], check=True)
    manifest = dict(kind='Presentation export only; not an experiment execution directory',
        source_study=str(study), protocol_sha256=file_sha(study/'protocol.json'),
        original_artifacts_sha256=file_sha(study/'report/artifacts.json'),
        labelled_artifacts_sha256=file_sha(output/'report/artifacts.json'),
        labels=labels, numerical_results_identical=True, movies_decoded=len(movies),
        script_sha256=file_sha(Path(__file__)))
    (output/'PRESENTATION.json').write_text(json.dumps(manifest, indent=2)+'\n')
    print(json.dumps(manifest, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--study', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    run(args.study.resolve(), args.output.resolve())
