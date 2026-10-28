"""Full-state resume provenance and crash-tail recovery, using local temp files."""
from __future__ import annotations

import copy
import json
import random
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import training_resume as resume


class ResumeStateTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.args = SimpleNamespace(
            output_dir=self.root, resume_state=None, resume_adapter=None, start_step=10,
            model=str(self.root / 'sft'), lr=1e-5, kl_beta=0.05, adv_std_floor=0.2,
            lora_r=64, lora_alpha=128, max_grad_norm=0.3, warmup_steps=2,
            prompts_per_step=8, num_generations=6, suites=['suite'], tasks=[0, 1],
            seeds=[1], seed=17, max_seq_tokens=76000, hard_max_tokens=76000,
        )
        self.state = {
            'sample_format': 'per_request_v1', 'global_step': 10, 'world_size': 2,
            'model': self.args.model, 'args': copy.deepcopy(vars(self.args)),
            'optimizer': {'state': {}, 'param_groups': []},
            'prompts': [('suite', 0, 1), ('suite', 1, 1)], 'prompt_order': [1, 0],
            'prompt_cursor': 80, 'prompt_rng': random.Random(17).getstate(),
            'rank_rng': [{'python': random.Random(17 + i).getstate(),
                          'torch': torch.get_rng_state(), 'cuda': torch.tensor([1], dtype=torch.uint8)} for i in range(2)],
        }
        self.checkpoint = self.make_checkpoint(10)
        self.args.resume_state = self.checkpoint / 'training_state.pt'
        self.args.resume_adapter = self.checkpoint / 'adapter'

    def make_checkpoint(self, step):
        path = self.root / f'checkpoint-{step}'
        (path / 'adapter').mkdir(parents=True)
        (path / 'adapter' / 'adapter_config.json').write_text(json.dumps({'r': 64, 'lora_alpha': 128}))
        (path / 'adapter' / 'adapter_model.safetensors').write_bytes(b'fixture-weight-placeholder')
        (path / 'trainer_state.json').write_text(json.dumps({'global_step': step}))
        state = dict(self.state, global_step=step)
        torch.save(state, path / 'training_state.pt')
        return path


    def test_latest_complete_only_ignores_tmp_and_incomplete_directories(self):
        incomplete = self.root / 'checkpoint-30'
        incomplete.mkdir()
        (incomplete / 'training_state.pt').write_bytes(b'partial')
        tmp = self.make_checkpoint(40)
        tmp.rename(self.root / 'checkpoint-40.tmp')
        self.assertEqual(resume.latest_complete_checkpoint(self.root), self.checkpoint)
        latest = self.make_checkpoint(20)
        self.assertEqual(resume.latest_complete_checkpoint(self.root), latest)
        with self.assertRaisesRegex(resume.ResumeError, 'older rollback'):
            resume.validate_resume(self.state, self.args, world=2)

    def test_adapter_must_be_the_same_checkpoint_sibling(self):
        self.args.resume_adapter = self.root / 'other-adapter'
        with self.assertRaisesRegex(resume.ResumeError, 'same checkpoint'):
            resume.validate_resume(self.state, self.args, world=2)


    def test_format_step_world_model_and_rng_provenance_are_required(self):
        for key, value in [('sample_format', 'legacy'), ('global_step', 9), ('world_size', 4),
                           ('model', str(self.root / 'other-sft')), ('rank_rng', []),
                           ('prompt_order', [1, 1]), ('prompt_cursor', -1), ('optimizer', None)]:
            state = copy.deepcopy(self.state)
            state[key] = value
            with self.subTest(key=key), self.assertRaises(resume.ResumeError):
                resume.validate_resume(state, self.args, world=2)


    def test_cannot_resume_a_checkpoint_into_a_different_run_directory(self):
        self.args.output_dir = self.root / 'different-run'
        with self.assertRaisesRegex(resume.ResumeError, "this run's"):
            resume.validate_resume(self.state, self.args, world=2)


class ResumeLogTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def test_archives_original_bytes_and_keeps_exact_checkpoint_prefix(self):
        metrics = b'{ "step": 1, "reward": 0.10 }\n{"step":2}\n{"step":3}\n'
        rollouts = b'{"step":1,"sid":"a"}\n{"step":2,"sid":"b"}\n{"step":3,"sid":"c"}\n{"step":3,"sid":"d"}\n'
        for name, data in [('metrics.jsonl', metrics), ('rollouts.jsonl', rollouts)]:
            (self.root / name).write_bytes(data)
        result = resume.prepare_resume_logs(self.root, 2)
        self.assertEqual(len(result['trimmed']), 2)
        for info in result['trimmed']:
            original = metrics if info['file'] == 'metrics.jsonl' else rollouts
            self.assertEqual(Path(info['archive']).read_bytes(), original)
            self.assertEqual((self.root / info['file']).read_bytes(), b''.join(original.splitlines(keepends=True)[:2]))
        self.assertEqual(resume.prepare_resume_logs(self.root, 2)['trimmed'], [])

    def test_step_zero_probe_row_is_kept_and_other_step_zero_rows_are_rejected(self):
        path = self.root / 'metrics.jsonl'
        path.write_bytes(b'{"step":0,"probe_only":true}\n{"step":1}\n{"step":8}\n')
        info = resume.prepare_resume_logs(self.root, 0)['trimmed'][0]
        self.assertEqual(path.read_bytes(), b'{"step":0,"probe_only":true}\n')
        self.assertEqual(Path(info['archive']).read_bytes(), b'{"step":0,"probe_only":true}\n{"step":1}\n{"step":8}\n')
        path.write_bytes(b'{"step":0}\n{"step":1}\n')
        with self.assertRaises(resume.ResumeError):
            resume.prepare_resume_logs(self.root, 0)

    def test_torn_final_append_is_archived_and_removed(self):
        path = self.root / 'metrics.jsonl'
        original = b'{"step":1}\n{"step":2}\n{"step":3,"reward":'
        path.write_bytes(original)
        info = resume.prepare_resume_logs(self.root, 2)['trimmed'][0]
        self.assertTrue(info['dropped_partial_line'])
        self.assertEqual(Path(info['archive']).read_bytes(), original)
        self.assertEqual(path.read_bytes(), b'{"step":1}\n{"step":2}\n')

    def test_complete_last_row_without_newline_will_not_merge_with_next_append(self):
        path = self.root / 'metrics.jsonl'
        path.write_bytes(b'{"step":1}')
        info = resume.prepare_resume_logs(self.root, 1)['trimmed'][0]
        self.assertTrue(info['added_final_newline'])
        self.assertEqual(path.read_bytes(), b'{"step":1}\n')

    def test_validate_all_logs_before_modifying_any(self):
        metrics = b'{"step":1}\n{"step":2}\n'
        broken = b'not json\n{"step":2}\n'
        (self.root / 'metrics.jsonl').write_bytes(metrics)
        (self.root / 'rollouts.jsonl').write_bytes(broken)
        with self.assertRaises(resume.ResumeError):
            resume.prepare_resume_logs(self.root, 1)
        self.assertEqual((self.root / 'metrics.jsonl').read_bytes(), metrics)
        self.assertEqual((self.root / 'rollouts.jsonl').read_bytes(), broken)
        self.assertFalse(list(self.root.glob('*.resume-*')))

    def test_mixed_lineages_are_not_silently_rewritten(self):
        path = self.root / 'metrics.jsonl'
        data = b'{"step":2}\n{"step":1}\n'
        path.write_bytes(data)
        with self.assertRaisesRegex(resume.ResumeError, 'nonmonotonic'):
            resume.prepare_resume_logs(self.root, 1)
        self.assertEqual(path.read_bytes(), data)

    def test_concurrent_append_is_detected_before_replacement(self):
        path = self.root / 'metrics.jsonl'
        original = b'{"step":1}\n{"step":2}\n'
        path.write_bytes(original)
        def append_after_archive(_):
            with path.open('ab') as stream:
                stream.write(b'{"step":3}\n')
        with mock.patch.object(resume, '_sync_directory', side_effect=append_after_archive):
            with self.assertRaisesRegex(resume.ResumeError, 'another writer'):
                resume.prepare_resume_logs(self.root, 1)
        self.assertEqual(path.read_bytes(), original + b'{"step":3}\n')


if __name__ == '__main__':
    unittest.main()
