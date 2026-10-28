"""Shared-filesystem visibility and whole-batch failure regression checks."""
from __future__ import annotations

import ast
import errno
import json
import os
import stat
import sys
import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import rollout_storage as storage
from infra_pipeline import pack_plan
from raw_request_training import InvalidRollout


class FakeClock:
    def __init__(self):
        self.now = 0.0
        self.sleeps = []

    def monotonic(self):
        return self.now

    def sleep(self, duration):
        self.sleeps.append(duration)
        self.now += duration


class RolloutStorageTest(unittest.TestCase):
    def test_delayed_visibility_then_partial_json_recovers_within_deadline(self):
        clock = FakeClock()
        missing = FileNotFoundError(errno.ENOENT, 'not visible on this NFS client')
        with mock.patch.object(Path, 'read_text', side_effect=[missing, '{"ready":', '{"ready":true}']) as read, \
             mock.patch.object(storage.time, 'monotonic', clock.monotonic), \
             mock.patch.object(storage.time, 'sleep', clock.sleep):
            result = storage.read_rollout_json('node-b/rollout.json', timeout_s=1)
        self.assertEqual(result, {'ready': True})
        self.assertEqual(read.call_count, 3)
        self.assertEqual(sum(clock.sleeps), 0.75)

    def test_persistently_corrupt_json_times_out_without_resetting_deadline(self):
        clock = FakeClock()
        with mock.patch.object(Path, 'read_text', return_value='{broken'), \
             mock.patch.object(storage.time, 'monotonic', clock.monotonic), \
             mock.patch.object(storage.time, 'sleep', clock.sleep):
            with self.assertRaises(storage.RolloutVisibilityTimeout) as caught:
                storage.read_rollout_json('corrupt/rollout.json', timeout_s=1, initial_delay_s=0.4, max_delay_s=0.6)
        self.assertEqual(clock.now, 1)
        self.assertIn('corrupt/rollout.json', str(caught.exception))
        self.assertIn('1.000s deadline', str(caught.exception))
        self.assertIn('JSONDecodeError', str(caught.exception))
        self.assertIsInstance(caught.exception.__cause__, json.JSONDecodeError)

    def test_persistently_missing_file_has_the_same_bounded_timeout(self):
        clock = FakeClock()
        error = FileNotFoundError(errno.ENOENT, 'missing')
        with mock.patch.object(Path, 'read_text', side_effect=error), \
             mock.patch.object(storage.time, 'monotonic', clock.monotonic), \
             mock.patch.object(storage.time, 'sleep', clock.sleep):
            with self.assertRaisesRegex(storage.RolloutVisibilityTimeout, 'FileNotFoundError'):
                storage.read_rollout_json('missing.json', timeout_s=0.3)
        self.assertEqual(clock.now, 0.3)
        self.assertLessEqual(max(clock.sleeps), 0.25)

    def test_permission_and_other_io_errors_are_not_retried(self):
        for code in (errno.EACCES, errno.EIO, errno.ESTALE):
            with self.subTest(errno=code), mock.patch.object(Path, 'read_text', side_effect=OSError(code, 'I/O failure')) as read, \
                 mock.patch.object(storage.time, 'sleep') as sleep:
                with self.assertRaises(OSError) as caught:
                    storage.read_rollout_json('rollout.json')
                self.assertEqual(caught.exception.errno, code)
                self.assertEqual(read.call_count, 1)
                sleep.assert_not_called()

    def test_valid_json_with_wrong_shape_is_not_retried(self):
        with mock.patch.object(Path, 'read_text', return_value='[]') as read, mock.patch.object(storage.time, 'sleep') as sleep:
            with self.assertRaisesRegex(ValueError, 'must be an object'):
                storage.read_rollout_json('rollout.json')
        self.assertEqual(read.call_count, 1)
        sleep.assert_not_called()

    def test_atomic_publication_fsyncs_file_before_replace_and_directory_after(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'rollout.json'
            path.write_text('{"old":true}')
            events = []
            real_fsync, real_replace = os.fsync, os.replace

            def fsync(fd):
                events.append('directory' if stat.S_ISDIR(os.fstat(fd).st_mode) else 'file')
                real_fsync(fd)

            def replace(source, destination):
                self.assertEqual(Path(source).parent, path.parent)
                self.assertEqual(json.loads(path.read_text()), {'old': True})
                self.assertEqual(json.loads(Path(source).read_text()), {'complete': [1, 2, 3]})
                events.append('replace')
                real_replace(source, destination)

            with mock.patch.object(storage.os, 'fsync', side_effect=fsync), mock.patch.object(storage.os, 'replace', side_effect=replace):
                storage.write_rollout_json_atomic(path, {'complete': [1, 2, 3]})
            self.assertEqual(events, ['file', 'replace', 'directory'])
            self.assertEqual(storage.read_rollout_json(path), {'complete': [1, 2, 3]})
            self.assertEqual(list(Path(tmp).glob('*.tmp')), [])

    def test_failed_replace_never_exposes_partial_output_or_leaves_temp_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'rollout.json'
            original = b'{"old":true}\n'
            path.write_bytes(original)
            with mock.patch.object(storage.os, 'replace', side_effect=OSError(errno.EIO, 'replace failed')):
                with self.assertRaises(OSError):
                    storage.write_rollout_json_atomic(path, {'new': True})
            self.assertEqual(path.read_bytes(), original)
            self.assertEqual(list(Path(tmp).glob('*.tmp')), [])


class TrainerWholeBatchFailureTest(unittest.TestCase):
    @staticmethod
    def build_stage():
        # Execute the actual main-loop validation block without loading a model,
        # initializing NCCL, or reaching its optimizer section.
        path = Path(__file__).resolve().parents[1] / 'train.py'
        module = ast.parse(path.read_text())
        main = next(n for n in module.body if isinstance(n, ast.FunctionDef) and n.name == 'main')
        loop = next(n for n in ast.walk(main) if isinstance(n, ast.For) and isinstance(n.target, ast.Name) and n.target.id == 'step')

        def assigns(node, name):
            return isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == name for t in node.targets)

        start = next(i for i, n in enumerate(loop.body) if assigns(n, 'active'))
        end = next(i for i, n in enumerate(loop.body) if assigns(n, 'n_tokens_local'))
        body = loop.body[start:end] + [ast.Expr(value=ast.Call(func=ast.Name(id='_update_probe', ctx=ast.Load()), args=[], keywords=[]))]
        return compile(ast.fix_missing_locations(ast.Module(body=body, type_ignores=[])), str(path), 'exec')

    @staticmethod
    def trainer_function(name):
        path = Path(__file__).resolve().parents[1] / 'train.py'
        node = next(n for n in ast.parse(path.read_text()).body if isinstance(n, ast.FunctionDef) and n.name == name)
        scope: dict = {}
        exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), 'exec'), scope)
        return scope[name]

    def check_failed_batch(self, remote_failure):
        class Dist:
            def all_gather_object(self, output, local):
                output[:] = [local] + ([[{'rank': 1, 'path': 'bad.json', 'error': 'visibility deadline expired'}]] if remote_failure else [])

        reader = mock.Mock(side_effect=[{}, storage.RolloutVisibilityTimeout('bad.json deadline expired')])
        update = mock.Mock()
        namespace = {
            'results': [{'sample_path': name, 'verdict': 'request_cap_exhausted', 'advantage': 0.5} for name in ('good.json', 'bad.json')],
            'rank': 0, 'world': 2 if remote_failure else 1, 'dist': Dist(), 'image_pad_id': -1, 'zero_signal': False,
            'processor': None, 'args': SimpleNamespace(max_seq_tokens=100, hard_max_tokens=100, pack_prefixes=False,
                                                       pack_assignment='trajectory', advantage_mode='grpo', critic_cotrain=False),
            'time': time, 'pack_plan': pack_plan,
            'read_rollout_json': reader, 'build_request_samples': lambda *a, **kw: ([{'n_tokens': 1}], 1),
            'InvalidRollout': InvalidRollout, 'json': json, 'log': mock.Mock(), '_update_probe': update,
            'balanced_assignment': self.trainer_function('balanced_assignment'),
        }
        with self.assertRaisesRegex(InvalidRollout, 'refusing a biased partial update'):
            exec(self.build_stage(), namespace)
        self.assertEqual(len(namespace['samples']), 1)
        self.assertTrue(namespace['errors'])
        update.assert_not_called()



    def test_pack_level_assignment_builds_each_chain_once_and_counts_each_trajectory_once(self):
        # the real main-loop block on every rank of a 3-rank world: the union of the built chains is every chain exactly
        # once, and each trajectory's completion tokens enter exactly one rank's objective denominator
        lengths = {0: [900, 50, 400], 1: [700], 2: [300, 300], 3: [1000, 20, 20, 20]}
        full_tokens = {0: 11, 1: 13, 2: 17, 3: 19}

        def build(raw, processor, *, chains=None, **kw):
            assert chains is not None and kw['pack_prefixes']
            return [{'n_tokens': 1, 'chain': c, 'traj': raw['traj']} for c in sorted(chains)], full_tokens[raw['traj']]

        class Dist:
            def all_gather_object(self, output, local):
                output[:] = [local]

        built, objective = [], 0
        for rank in range(3):
            namespace = {
                'results': [{'sample_path': f'{i}.json', 'verdict': 'progress', 'advantage': 0.5,
                             'diagnostics': {'packed_lengths': lengths[i], 'packed_cost': sum(lengths[i])}} for i in range(4)],
                'rank': rank, 'world': 3, 'dist': Dist(), 'image_pad_id': -1, 'zero_signal': False, 'processor': None,
                'args': SimpleNamespace(max_seq_tokens=100, hard_max_tokens=100, pack_prefixes=True, pack_assignment='pack',
                                        advantage_mode='grpo', critic_cotrain=False),
                'read_rollout_json': lambda path: {'traj': int(path.split('.')[0])}, 'build_request_samples': build,
                'InvalidRollout': InvalidRollout, 'json': json, 'log': mock.Mock(), '_update_probe': mock.Mock(),
                'balanced_assignment': self.trainer_function('balanced_assignment'), 'time': time, 'pack_plan': pack_plan,
            }
            exec(self.build_stage(), namespace)
            self.assertTrue(namespace['pack_level'])
            built += [(s['traj'], s['chain']) for s in namespace['samples']]
            objective += namespace['objective_tokens_local']
        self.assertEqual(sorted(built), sorted((i, c) for i in lengths for c in range(len(lengths[i]))))
        self.assertEqual(objective, sum(full_tokens.values()))




if __name__ == '__main__':
    unittest.main()
