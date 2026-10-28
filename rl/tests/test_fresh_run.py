"""Launcher contracts, health validation and own-process lifecycle without GPUs."""
import copy
import json
import socket
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import fresh_run as run


class FreshRunTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        nodes = copy.deepcopy(run.DEFAULTS['nodes'])
        nodes['a'].update(hostname=socket.gethostname().split('.')[0], host='127.0.0.1')
        self.overrides = {'run_dir': str(self.root / 'run'), 'repo': str(self.root / 'repo'),
                          'harness': str(self.root / 'harness'), 'base_model': str(self.root / 'sft'), 'nodes': nodes}
        for name in run.SOURCE_FILES:
            path = self.root / 'repo' / 'rl' / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text('# fixture\n')
        for name in ('server/env_server.py', 'adapters/libero/hidden_object_audit.py', 'adapters/libero/backend.py'):
            path = self.root / 'harness' / 'src/captain_harness' / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text('# fixture\n')
        entrypoint = self.root / 'harness/scripts/react_episode.py'
        entrypoint.parent.mkdir(parents=True)
        entrypoint.write_text('# fixture entrypoint\n')
        (self.root / 'sft').mkdir()
        (self.root / 'sft/config.json').write_text('{}')
        self.cfg = run.normalize_config(self.overrides)
        self.manifest_path = self.root / 'run' / 'manifest.json'
        self.manifest = run.initialize(self.manifest_path, self.overrides, 'a')

    def test_smoke_overrides_and_fresh_commands_have_no_old_adapter(self):
        cfg = run.normalize_config(dict(self.overrides, nnodes=1, train_gpus=[5], max_steps=1, save_steps=1,
                                       prompts_per_step=2, num_generations=2))
        commands = run.role_commands(cfg, 'a')
        train = commands['train']
        self.assertEqual(train[train.index('--nproc-per-node') + 1], '1')
        self.assertEqual(train[train.index('--max-steps') + 1], '1')
        self.assertEqual(train[train.index('--num-generations') + 1], '2')
        self.assertNotIn('--resume-adapter', train)
        self.assertNotIn('--no-sync-at-start', train)
        env = commands['env']
        self.assertEqual(env[env.index('--max-actions') + 1], '150')
        self.assertEqual(env[env.index('--max-tool-calls') + 1], '600')
        self.assertEqual(env[env.index('--enable-thinking') + 1], 'off')
        self.assertIn('--max-training-requests', env)

    def test_code_reward_config_is_validated_and_passed_to_env_and_trainer(self):
        cfg = run.normalize_config(dict(self.overrides, code_reward_mode='python_exec_cost_v1', code_reward_weight=0.25))
        commands = run.role_commands(cfg, 'a')
        env, train = commands['env'], commands['train']
        self.assertEqual(env[env.index('--code-reward') + 1], 'python_exec_cost_v1')
        self.assertEqual(train[train.index('--code-reward-weight') + 1], '0.25')
        self.assertEqual(train[train.index('--code-reward-budget') + 1], '40')
        self.assertNotIn('--code-reward', run.role_commands(self.cfg, 'a')['train'])
        for bad in ({'code_reward_mode': 'python_exec_cost_v1'}, {'code_reward_weight': 0.5}, {'code_reward_mode': 'other', 'code_reward_weight': 1.0}):
            with self.assertRaises(run.RunError):
                run.normalize_config(dict(self.overrides, **bad))

    def test_physcritic_advantage_mode_reaches_the_trainer_and_the_hashes(self):
        import torch
        import physics_critic
        critic = self.root / 'critic'
        critic.mkdir()
        (critic / 'feature_spec.json').write_text(json.dumps({'feature_version': physics_critic.cf.FEATURE_VERSION, 'features': ['eef_z'],
                                                              'mean': [0.0], 'std': [1.0], 'clip': 10, 'task_vocab': ['libero_goal:0']}))
        torch.save({'arch': {'in_dim': 2}, 'state_dict': physics_critic.mlp(2)[0].state_dict()}, critic / 'critic_mlp.pt')
        cfg = run.normalize_config(dict(self.overrides, advantage_mode='physcritic', critic_dir=str(critic), gae_lambda=0.9,
                                        reward_mode='final_success_binary_v1', score_centering=True, sampler_topk=20,
                                        ratio_mask_low=0.2, ratio_mask_high=0.28))
        train = run.role_commands(cfg, 'a')['train']
        for flag, value in (('--advantage-mode', 'physcritic'), ('--critic-dir', str(critic.resolve())), ('--gae-lambda', '0.9'),
                            ('--ratio-mask-low', '0.2'), ('--ratio-mask-high', '0.28')):
            self.assertEqual(train[train.index(flag) + 1], value)
        hashes = run.source_hashes(cfg)
        self.assertIn(str(critic.resolve() / 'critic_mlp.pt'), hashes)
        for flag in ('--advantage-mode', '--critic-dir', '--ratio-mask-low'):
            self.assertNotIn(flag, run.role_commands(self.cfg, 'a')['train'])
        with self.assertRaises(run.RunError):
            run.normalize_config(dict(self.overrides, advantage_mode='physcritic', critic_dir=str(critic)))   # needs the 0/1 reward

    def test_venv_interpreter_symlinks_are_not_dereferenced(self):
        venv_python = self.root / 'venv/bin/python'
        venv_python.parent.mkdir(parents=True)
        venv_python.symlink_to(Path(sys.executable).resolve())
        cfg = run.normalize_config(dict(self.overrides, python_env=str(venv_python), python_train=str(venv_python)))
        self.assertEqual(cfg['python_env'], str(venv_python))
        self.assertEqual(cfg['python_train'], str(venv_python))
        self.assertNotEqual(cfg['python_env'], str(venv_python.resolve()))
        for argv in run.role_commands(cfg, 'a').values():
            self.assertEqual(argv[0], str(venv_python))

    def test_single_training_node_can_keep_both_rollout_nodes(self):
        cfg = run.normalize_config(dict(self.overrides, nnodes=1, rollout_nodes=['a', 'b']))
        manifest = dict(self.manifest, config=cfg)
        self.assertEqual(run.active_nodes(cfg), ['a'])
        self.assertNotIn('train', run.role_commands(cfg, 'b'))
        with mock.patch.object(run.socket, 'gethostname', return_value=cfg['nodes']['b']['hostname']):
            with mock.patch.object(run, 'check_port'), mock.patch.object(run, 'validate_health', return_value={}), mock.patch.object(run, 'start_role', return_value={}) as start:
                run.run_action(manifest, 'b', 'services')
                self.assertEqual([call.args[2] for call in start.call_args_list], ['proxy', 'env'])
            with self.assertRaisesRegex(run.RunError, 'not enabled for training'):
                run.run_action(manifest, 'b', 'train')
        with mock.patch.object(run, 'check_port'), mock.patch.object(run, 'check_gpu_memory', return_value={4: 80}), mock.patch.object(run, 'validate_health', return_value={}) as health, mock.patch.object(run, 'start_role', return_value={}) as start:
            run.run_action(manifest, 'a', 'train')
            self.assertEqual([call.args[1] for call in health.call_args_list], ['a', 'b'])
            argv, child_env = start.call_args.args[3:5]
            self.assertEqual(argv[argv.index('--nnodes') + 1], '1')
            self.assertEqual(argv[argv.index('--nproc-per-node') + 1], '4')
            self.assertEqual(child_env['RL_POLICY_URL'], 'http://127.0.0.1:8500,http://node-b:8500')
            self.assertEqual(len(child_env['RL_ENV_SERVER'].split(',')), 2)

    def test_insufficient_gpu_memory_rejects_launch_without_eviction(self):
        cfg = run.normalize_config(dict(self.overrides, nnodes=1, train_gpus=[4, 5]))
        manifest = dict(self.manifest, config=cfg)
        low = type('Result', (), {'stdout': '4, 65536\n5, 60000\n'})()
        with mock.patch.object(run.subprocess, 'run', return_value=low), mock.patch.object(run, 'validate_health', return_value={}), mock.patch.object(run, 'start_role') as start, mock.patch.object(run.os, 'kill') as kill:
            with self.assertRaisesRegex(run.RunError, 'GPU 5 .*no process was evicted'):
                run.run_action(manifest, 'a', 'train')
            start.assert_not_called()
            kill.assert_not_called()
        high = type('Result', (), {'stdout': '4, 65536\n5, 70000\n'})()
        with mock.patch.object(run.subprocess, 'run', return_value=high):
            self.assertEqual(run.check_gpu_memory(cfg)[4], 64)

    def test_node_identity_is_checked_before_status_or_stop(self):
        with self.assertRaisesRegex(run.RunError, 'requires hostname'):
            run.run_action(self.manifest, 'b', 'stop')

    def test_manifest_configuration_and_hashes_freeze_after_metrics(self):
        (self.root / 'run/metrics.jsonl').write_text('{"step":1}\n')
        with self.assertRaisesRegex(run.RunError, 'immutable'):
            run.initialize(self.manifest_path, dict(self.overrides, lr=2e-5), 'a')
        path = self.root / 'repo/rl/reward.py'
        path.write_text('# changed\n')
        with self.assertRaisesRegex(run.RunError, 'source hashes changed'):
            run.run_action(self.manifest, 'a', 'services')

    def test_main_configuration_uses_a_training_and_both_rollout_nodes(self):
        cfg = run.normalize_config(json.loads((Path(__file__).parent / 'data/minimal_config.json').read_text()))
        self.assertEqual(cfg['nnodes'], 1)
        self.assertEqual(cfg['single_node'], 'a')
        self.assertEqual(cfg['rollout_nodes'], ['a', 'b'])
        self.assertEqual(cfg['train_gpus'], [4, 5, 6, 7])
        self.assertEqual(cfg['minimum_free_gpu_gib'], 60)

    def test_source_hashes_cover_public_goal_contract_and_react_runtime(self):
        root = self.root / 'harness/src/captain_harness'
        files = ('adapters/capx/completion_gate.py', 'adapters/capx/object_geometry.py',
                 'react/context.py', 'react/chat_llm.py', 'react/loop.py', 'react/tools.py',
                 'react/code_mode_runner.js', 'react/prompts/captain.request.json')
        for name in files:
            path = root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text('fixture source\n')
        hashes = run.source_hashes(self.cfg)
        self.assertTrue(all(str(root / name) in hashes for name in files))
        self.assertIn(str(self.root / 'harness/scripts/react_episode.py'), hashes)
        self.assertEqual(list(hashes), sorted(hashes))
        self.assertEqual(hashes, run.source_hashes(self.cfg))
        self.assertNotEqual(hashes, self.manifest['source_hashes'])
        cache = root / 'react/__pycache__/transient.py'
        cache.parent.mkdir(parents=True)
        cache.write_text('runtime cache ignored\n')
        self.assertEqual(hashes, run.source_hashes(self.cfg))
        (root / files[0]).write_text('changed public finish predicate\n')
        self.assertNotEqual(hashes, run.source_hashes(self.cfg))

    def test_reinitialize_before_training_when_every_recorded_role_is_stopped(self):
        records = []
        for node, role in [('a', 'train'), ('a', 'env'), ('b', 'env'), ('b', 'proxy')]:
            path = run.record_path(self.cfg, node, role)
            run.atomic_json(path, {'stopped_at': time.time(), 'node': node, 'role': role})
            records.append((path, path.read_bytes()))
        run.atomic_json(run.preflight_path(self.cfg, 'b'), {'ok': True})   # a receipt, not a role record
        (self.root / 'repo/rl/train.py').write_text('# updated before any checkpoint\n')
        updated = run.initialize(self.manifest_path, dict(self.overrides, nnodes=1, single_node='a', rollout_nodes=['a', 'b']), 'a')
        self.assertEqual(updated['config']['nnodes'], 1)
        self.assertEqual(updated['config']['rollout_nodes'], ['a', 'b'])
        self.assertEqual(updated['run_id'], self.manifest['run_id'])
        self.assertEqual(updated['created_at'], self.manifest['created_at'])
        self.assertEqual(updated['source_hashes'], run.source_hashes(updated['config']))
        self.assertTrue(all(path.read_bytes() == contents for path, contents in records))
        self.assertFalse(run.training_history_exists(updated['config']['run_dir']))

    def test_reinitialize_refuses_a_role_that_has_not_been_stopped(self):
        path = run.record_path(self.cfg, 'b', 'env')
        run.atomic_json(path, {'node': 'b', 'role': 'env'})
        before = self.manifest_path.read_bytes()
        with self.assertRaisesRegex(run.RunError, 'stop all owned roles'):
            run.initialize(self.manifest_path, dict(self.overrides, nnodes=1, rollout_nodes=['a', 'b']), 'a')
        self.assertEqual(self.manifest_path.read_bytes(), before)

    def test_relocate_changes_only_the_layout_of_a_run_with_history(self):
        (self.root / 'run' / 'metrics.jsonl').write_text('{"step": 1}\n')
        for node, role in [('a', 'train'), ('a', 'env'), ('b', 'env'), ('b', 'proxy')]:
            run.atomic_json(run.record_path(self.cfg, node, role), {'stopped_at': time.time(), 'node': node, 'role': role})
        nodes = copy.deepcopy(self.overrides['nodes'])
        nodes['b'].update(egl='2', sam3_url='http://127.0.0.1:8114')
        layout = dict(self.overrides, nnodes=1, single_node='a', rollout_nodes=['b'], train_gpus=list(range(8)), nodes=nodes,
                      policy_workers=['http://node-b:8500', 'http://node-b:8501'])
        with self.assertRaisesRegex(run.RunError, 'only change the layout; these would change: lr'):
            run.relocate(self.manifest_path, dict(layout, lr=2e-5), 'a')
        moved = run.relocate(self.manifest_path, layout, 'a')
        self.assertEqual(moved['config']['train_gpus'], list(range(8)))
        self.assertEqual(moved['run_id'], self.manifest['run_id'])
        self.assertEqual(moved['relocations'][0]['previous']['train_gpus'], [4, 5, 6, 7])
        self.assertEqual(set(moved['relocations'][0]['changed']), {'nnodes', 'rollout_nodes', 'train_gpus', 'nodes', 'policy_workers'})
        self.assertEqual(json.loads(self.manifest_path.read_text()), moved)
        cfg = moved['config']
        train = run.role_commands(cfg, 'a')['train']
        self.assertEqual(train[train.index('--nproc-per-node') + 1], '8')
        self.assertEqual(train[train.index('--nnodes') + 1], '1')
        proxy = run.role_commands(cfg, 'b')['proxy']
        self.assertEqual(proxy[proxy.index('--upstream') + 1], 'http://node-b:8500,http://node-b:8501')

    def test_relocate_may_extend_max_steps_and_resume_uses_the_new_horizon(self):
        (self.root / 'run' / 'metrics.jsonl').write_text('{"step": 5}\n')
        base = dict(self.overrides, max_steps=64)
        manifest = dict(json.loads(self.manifest_path.read_text()), config=run.normalize_config(base))
        run.atomic_json(self.manifest_path, manifest)
        with self.assertRaisesRegex(run.RunError, 'may only grow'):
            run.relocate(self.manifest_path, dict(base, max_steps=32), 'a')
        with self.assertRaisesRegex(run.RunError, 'may only grow'):
            run.relocate(self.manifest_path, dict(base, max_steps=5), 'a')
        with self.assertRaisesRegex(run.RunError, 'lr'):
            run.relocate(self.manifest_path, dict(base, max_steps=256, lr=2e-5), 'a')
        moved = run.relocate(self.manifest_path, dict(base, max_steps=256), 'a')
        self.assertEqual(moved['relocations'][-1]['previous'], {'max_steps': 64})
        ck = self.root / 'run' / 'checkpoint-8'
        ck.mkdir()
        saved = {key: run.DEFAULTS[key] for key in run.TRAIN_KEYS + ('suites', 'tasks', 'seeds')}
        saved.update(model=str(self.root / 'sft'), output_dir=str(self.root / 'run'), max_steps=64)
        (ck / 'trainer_state.json').write_text(json.dumps({'global_step': 8, 'args': saved}))
        (ck / 'training_state.pt').write_bytes(b'state')
        (ck / 'adapter').mkdir()
        (ck / 'adapter' / 'adapter_config.json').write_text('{}')
        (ck / 'adapter' / 'adapter_model.safetensors').write_bytes(b'w')
        train = run.role_commands(moved['config'], 'a', resume=True)['train']
        self.assertEqual(train[train.index('--max-steps') + 1], '256')
        self.assertEqual(train[train.index('--lr') + 1], str(saved['lr']))

    def test_relocate_may_change_save_steps_and_resume_uses_it(self):
        (self.root / 'run' / 'metrics.jsonl').write_text('{"step": 5}\n')
        base = dict(self.overrides, save_steps=2)
        manifest = dict(json.loads(self.manifest_path.read_text()), config=run.normalize_config(base))
        run.atomic_json(self.manifest_path, manifest)
        with self.assertRaisesRegex(run.RunError, 'positive integer'):
            run.relocate(self.manifest_path, dict(base, save_steps=0), 'a')
        moved = run.relocate(self.manifest_path, dict(base, save_steps=10), 'a')
        self.assertEqual(moved['relocations'][-1]['previous'], {'save_steps': 2})
        ck = self.root / 'run' / 'checkpoint-8'
        ck.mkdir()
        saved = {key: run.DEFAULTS[key] for key in run.TRAIN_KEYS + ('suites', 'tasks', 'seeds')}
        saved.update(model=str(self.root / 'sft'), output_dir=str(self.root / 'run'), save_steps=2)
        (ck / 'trainer_state.json').write_text(json.dumps({'global_step': 8, 'args': saved}))
        (ck / 'training_state.pt').write_bytes(b'state')
        (ck / 'adapter').mkdir()
        (ck / 'adapter' / 'adapter_config.json').write_text('{}')
        (ck / 'adapter' / 'adapter_model.safetensors').write_bytes(b'w')
        train = run.role_commands(moved['config'], 'a', resume=True)['train']
        self.assertEqual(train[train.index('--save-steps') + 1], '10')

    def test_relocate_refuses_running_roles_and_source_changes(self):
        (self.root / 'run' / 'metrics.jsonl').write_text('{"step": 1}\n')
        run.atomic_json(run.record_path(self.cfg, 'b', 'env'), {'node': 'b', 'role': 'env', 'pid': 1})
        before = self.manifest_path.read_bytes()
        with self.assertRaisesRegex(run.RunError, 'stop all owned roles'):
            run.relocate(self.manifest_path, dict(self.overrides, train_gpus=list(range(8))), 'a')
        run.atomic_json(run.record_path(self.cfg, 'b', 'env'), {'node': 'b', 'role': 'env', 'pid': 1, 'stopped_at': 1.0})
        (self.root / 'repo/rl/train.py').write_text('# changed after training started\n')
        with self.assertRaisesRegex(run.RunError, 'source hashes changed'):
            run.relocate(self.manifest_path, dict(self.overrides, train_gpus=list(range(8))), 'a')
        self.assertEqual(self.manifest_path.read_bytes(), before)
        import shutil
        shutil.copytree(self.root / 'repo', self.root / 'repo2')
        old_hash = self.manifest['source_hashes'][str(self.root / 'repo/rl/train.py')]
        moved = run.relocate(self.manifest_path, dict(self.overrides, train_gpus=list(range(8)), repo=str(self.root / 'repo2')), 'a')
        self.assertEqual(moved['relocations'][0]['changed_sources'], ['repo/rl/train.py'])
        self.assertEqual(moved['relocations'][0]['previous_source_hashes'], {'repo/rl/train.py': old_hash})
        self.assertEqual(moved['source_hashes'], run.source_hashes(moved['config']))

    def _source_checkpoint(self, step=20, model=None):
        source = self.root / 'other-run' / f'checkpoint-{step}'
        (source / 'adapter').mkdir(parents=True)
        (source / 'training_state.pt').write_bytes(b'state')
        args = {key: run.DEFAULTS[key] for key in run.TRAIN_KEYS + ('suites', 'tasks', 'seeds')}
        args.update(model=model or str(self.root / 'sft'), output_dir=str(self.root / 'other-run'))
        (source / 'trainer_state.json').write_text(json.dumps({'global_step': step, 'args': args}))
        (source / 'adapter' / 'adapter_config.json').write_text('{}')
        (source / 'adapter' / 'adapter_model.safetensors').write_bytes(b'weights')
        (source / 'model-00001-of-00004.safetensors').write_bytes(b'merged shard, not needed to resume')
        return source

    def test_continue_from_seeds_the_source_checkpoint_and_resumes_it(self):
        source = self._source_checkpoint()
        path = self.root / 'cont' / 'manifest.json'
        overrides = dict(self.overrides, run_dir=str(self.root / 'cont'), continue_from=str(source))
        manifest = run.initialize(path, overrides, 'a')
        cfg = manifest['config']
        self.assertIn(str(source / 'training_state.pt'), manifest['source_hashes'])
        seeded = run.seed_from_checkpoint(cfg)
        self.assertEqual(seeded, self.root / 'cont' / 'checkpoint-20')
        self.assertFalse((seeded / 'model-00001-of-00004.safetensors').exists())
        lineage = json.loads((self.root / 'cont' / 'lineage.json').read_text())
        self.assertEqual((lineage['continued_from'], lineage['step']), (str(source), 20))
        train = run.role_commands(cfg, 'a', resume=True)['train']
        self.assertEqual(train[train.index('--resume-state') + 1], str(seeded / 'training_state.pt'))
        self.assertEqual(train[train.index('--start-step') + 1], '20')
        self.assertEqual(train[train.index('--output-dir') + 1], cfg['run_dir'])

    def test_continue_from_a_full_finetune_checkpoint_copies_its_node_local_state_and_passes_pipeline_options(self):
        source = self.root / 'other-run' / 'checkpoint-24'
        source.mkdir(parents=True)
        state = self.root / 'local' / 'other-run' / 'checkpoint-24'
        state.mkdir(parents=True)
        (state / '.metadata').write_bytes(b'dcp')
        (state / '__0_0.distcp').write_bytes(b'shard')
        (source / 'fsdp').symlink_to(state)
        args = {key: run.DEFAULTS[key] for key in run.TRAIN_KEYS + ('suites', 'tasks', 'seeds')}
        args.update(model=str(self.root / 'sft'), output_dir=str(self.root / 'other-run'), full_finetune=True)
        (source / 'trainer_state.json').write_text(json.dumps({'global_step': 24, 'full_finetune': True, 'args': args}))
        (source / 'training_state.pt').write_bytes(b'state')
        (source / 'model.safetensors.index.json').write_text('{}')
        (source / 'model-00001-of-00004.safetensors').write_bytes(b'bf16')
        overrides = dict(self.overrides, run_dir=str(self.root / 'async'), continue_from=str(source), full_finetune=True,
                         full_state_dir=str(self.root / 'local' / 'async'), score_centering=True, sampler_topk=20,
                         offload_activations='off', async_lookahead=1, is_cap=2.0, pack_prefixes=True)
        manifest = run.initialize(self.root / 'async' / 'manifest.json', overrides, 'a')
        self.assertFalse(any('.metadata' in k or 'distcp' in k for k in manifest['source_hashes']))   # node-local, not hashed
        seeded = run.seed_from_checkpoint(manifest['config'])
        self.assertTrue((seeded / 'fsdp').is_symlink())
        self.assertEqual((seeded / 'fsdp').resolve(), self.root / 'local' / 'async' / 'checkpoint-24')
        self.assertEqual((seeded / 'fsdp' / '__0_0.distcp').read_bytes(), b'shard')
        self.assertEqual((seeded / 'model-00001-of-00004.safetensors').read_bytes(), b'bf16')
        self.assertTrue((state / '.metadata').exists())               # the source keeps its own resume state
        self.assertEqual(run.complete_checkpoint_step(seeded), 24)
        train = run.role_commands(manifest['config'], 'a', resume=True)['train']
        for flag, value in (('--offload-activations', 'off'), ('--async-lookahead', '1'), ('--is-cap', '2.0'), ('--start-step', '24')):
            self.assertEqual(train[train.index(flag) + 1], value)
        self.assertIn('--pack-prefixes', train)
        self.assertEqual(train[train.index('--full-state-dir') + 1], str(self.root / 'local' / 'async'))   # not the source run's
        with self.assertRaisesRegex(run.RunError, 'already exists'):
            run.seed_from_checkpoint(manifest['config'])
        with self.assertRaisesRegex(run.RunError, 'is_cap'):
            run.normalize_config(dict(self.overrides, is_cap=2.0))      # needs score_centering

    def test_continue_from_requires_a_complete_checkpoint_of_the_same_base_model(self):
        source = self._source_checkpoint(model=str(self.root / 'another-sft'))
        overrides = dict(self.overrides, run_dir=str(self.root / 'cont'), continue_from=str(source))
        with self.assertRaisesRegex(run.RunError, 'different base model'):
            run.initialize(self.root / 'cont' / 'manifest.json', overrides, 'a')
        (source / 'training_state.pt').write_bytes(b'')
        with self.assertRaisesRegex(run.RunError, 'not a complete checkpoint'):
            run.initialize(self.root / 'cont' / 'manifest.json', overrides, 'a')

    def test_segment_windows_suites_and_probe_reach_the_trainer_and_the_hashes(self):
        bank, probe = self.root / 'bank.json', self.root / 'probe.json'
        bank.write_text('{}')
        probe.write_text('{}')
        for name in ('configs/calibration/libero_franka_v3.json', 'third_party/manifest.json'):   # hashed under python-runtime/v1
            (self.root / 'harness' / name).parent.mkdir(parents=True, exist_ok=True)
            (self.root / 'harness' / name).write_text('{}')
        cfg = run.normalize_config(dict(self.overrides, runtime_tool_contract='python-runtime/v1', segment_bank=str(bank),
                                        segment_horizon=4, curriculum_fresh_horizon=40, curriculum_suites=['libero_goal', 'libero_10'],
                                        curriculum_active_min_delta=0.05, probe_set=str(probe), probe_every=10))
        train = run.role_commands(cfg, 'a')['train']
        self.assertEqual(train[train.index('--segment-horizon') + 1], '4')
        self.assertEqual(train[train.index('--curriculum-fresh-horizon') + 1], '40')
        self.assertEqual(train[train.index('--curriculum-suites') + 1:train.index('--curriculum-suites') + 3], ['libero_goal', 'libero_10'])
        self.assertEqual(train[train.index('--curriculum-active-min-delta') + 1], '0.05')
        self.assertEqual(train[train.index('--probe-set') + 1], str(probe))
        focused = run.role_commands(run.normalize_config(dict(self.overrides, runtime_tool_contract='python-runtime/v1', segment_bank=str(bank),
                                                              probe_set=str(probe), curriculum_focus_set='train')), 'a')['train']
        self.assertEqual(focused[focused.index('--curriculum-focus-set') + 1], 'train')
        with self.assertRaisesRegex(run.RunError, 'curriculum_focus_set'):
            run.normalize_config(dict(self.overrides, runtime_tool_contract='python-runtime/v1', segment_bank=str(bank), curriculum_focus_set='train'))
        self.assertIn(str(probe), run.source_hashes(cfg))
        plain = run.role_commands(run.normalize_config(dict(self.overrides, runtime_tool_contract='python-runtime/v1', segment_bank=str(bank))), 'a')['train']
        self.assertNotIn('--curriculum-fresh-horizon', plain)
        self.assertNotIn('--probe-set', plain)
        with self.assertRaisesRegex(run.RunError, 'curriculum_suites'):
            run.normalize_config(dict(self.overrides, runtime_tool_contract='python-runtime/v1', segment_bank=str(bank), curriculum_suites=[]))

    def test_remaining_task_budget_clean_handoff_and_student_python_reach_the_trainer_env_and_preflight(self):
        bank = self.root / 'bank.json'
        bank.write_text('{}')
        for name in ('configs/calibration/libero_franka_v3.json', 'third_party/manifest.json'):
            (self.root / 'harness' / name).parent.mkdir(parents=True, exist_ok=True)
            (self.root / 'harness' / name).write_text('{}')
        student = self.root / 'student' / 'python'
        student.parent.mkdir()
        student.write_text('#!/bin/sh\n')
        student.chmod(0o755)
        base = dict(self.overrides, runtime_tool_contract='python-runtime/v1', segment_bank=str(bank))
        cfg = run.normalize_config(dict(base, segment_budget=40, segment_handoff='clean', student_python=str(student),
                                        sandbox_hide=['/data/teacher'], wall_time=1500))
        commands = run.role_commands(cfg, 'a')
        train, env = commands['train'], commands['env']
        self.assertEqual(train[train.index('--segment-budget') + 1], '40')
        self.assertEqual(train[train.index('--segment-handoff') + 1], 'clean')
        self.assertEqual(train[train.index('--segment-min-live') + 1], '4')
        self.assertEqual(env[env.index('--student-python') + 1], str(student))
        self.assertEqual(env[env.index('--sandbox-hide') + 1], '/data/teacher')
        self.assertEqual(env[env.index('--wall-time') + 1], '1500')
        sc = run.role_commands(run.normalize_config(dict(base, score_centering=True, sampler_topk=20)), 'a')
        self.assertEqual(sc['proxy'][sc['proxy'].index('--sampler-topk') + 1], '20')
        self.assertIn('--score-centering', sc['train'])
        self.assertNotIn('--score-centering', train)
        with self.assertRaisesRegex(run.RunError, 'sampler_topk'):
            run.normalize_config(dict(base, score_centering=True))
        multi = run.role_commands(run.normalize_config(dict(base, extra_policy_ports=[8501, 8502])), 'a')
        self.assertEqual(multi['proxy'][multi['proxy'].index('--upstream') + 1],
                         'http://127.0.0.1:8500,http://127.0.0.1:8501,http://127.0.0.1:8502')
        with self.assertRaisesRegex(run.RunError, 'distinct'):
            run.normalize_config(dict(base, extra_policy_ports=[8512]))
        two = run.normalize_config(dict(base, extra_policy_ports=[8501], full_finetune=True))
        two["nodes"]["b"] = dict(two["nodes"]["b"], extra_policy_ports=[], sam3_url="http://127.0.0.1:8124", molmo_url="http://127.0.0.1:8132/v1")
        cb = run.role_commands(two, 'b')
        self.assertEqual(cb['proxy'][cb['proxy'].index('--upstream') + 1], 'http://127.0.0.1:8500')
        self.assertEqual(cb['env'][cb['env'].index('--sam3-url') + 1], 'http://127.0.0.1:8124')
        ca = run.role_commands(two, 'a')
        self.assertEqual(ca['proxy'][ca['proxy'].index('--upstream') + 1], 'http://127.0.0.1:8500,http://127.0.0.1:8501')
        self.assertIn('--full-finetune', ca['train'])
        remote = run.role_commands(run.normalize_config(dict(base, policy_workers=['http://node-b:8500', 'http://node-b:8501'],
                                                             curriculum_manifest='/m.json', reward_mode='final_success_binary_v1',
                                                             loss_normalization='trajectory_mean', skip_zero_signal=True,
                                                             warmup_attempts=320, boundary_fresh_only=True)), 'a')
        self.assertEqual(remote['proxy'][remote['proxy'].index('--upstream') + 1], 'http://node-b:8500,http://node-b:8501')
        for flag in ('--curriculum-manifest', '--reward-mode', '--loss-normalization', '--skip-zero-signal', '--warmup-attempts',
                     '--boundary-fresh-only'):
            self.assertIn(flag, remote['train'])
        plain = run.role_commands(run.normalize_config(base), 'a')
        self.assertNotIn('--segment-budget', plain['train'])
        self.assertEqual(plain['train'][plain['train'].index('--segment-handoff') + 1], 'continuation')
        self.assertNotIn('--student-python', plain['env'])
        with self.assertRaisesRegex(run.RunError, 'student_python'):
            run.normalize_config(dict(base, segment_handoff='clean'))
        with self.assertRaisesRegex(run.RunError, 'segment_budget'):
            run.normalize_config(dict(base, segment_budget=80))
        with self.assertRaisesRegex(run.RunError, 'student_python'):
            run.normalize_config(dict(base, student_python='relative/python'))

    def test_checkpoint_zero_and_other_training_artifacts_freeze_manifest(self):
        artifacts = [('checkpoint-0', True), ('.checkpoint-0-staging', True), ('rollouts.jsonl', False),
                     ('training_state.pt', False), ('trainer_state.json', False), ('final-adapter', True)]
        before = self.manifest_path.read_bytes()
        for name, directory in artifacts:
            with self.subTest(name=name):
                path = self.root / 'run' / name
                path.mkdir() if directory else path.write_bytes(b'preserved training history')
                with self.assertRaisesRegex(run.RunError, 'immutable'):
                    run.initialize(self.manifest_path, dict(self.overrides, nnodes=1, rollout_nodes=['a', 'b']), 'a')
                self.assertEqual(self.manifest_path.read_bytes(), before)
                self.assertTrue(path.exists())
                path.rmdir() if directory else path.unlink()

    def test_missing_manifest_cannot_be_recreated_over_training_history(self):
        self.manifest_path.unlink()
        (self.root / 'run/checkpoint-0').mkdir()
        with self.assertRaisesRegex(run.RunError, 'training history already exists'):
            run.initialize(self.manifest_path, self.overrides, 'a')
        self.assertFalse(self.manifest_path.exists())

    def test_unrelated_occupied_port_is_rejected(self):
        with socket.socket() as sock:
            sock.bind(('127.0.0.1', 0))
            sock.listen()
            with self.assertRaisesRegex(run.RunError, 'occupied'):
                run.check_port(sock.getsockname()[1])

    def test_proxy_http_success_does_not_hide_dead_upstream(self):
        with mock.patch.object(run, 'json_get', return_value={'ok': True, 'upstream_health': 500}):
            with self.assertRaisesRegex(run.RunError, 'upstream policy'):
                run.validate_health(self.cfg, 'a')

    def test_env_budget_and_reward_schema_must_match(self):
        proxy = {'ok': True, 'upstream_health': 200, 'sampling': {
            'temperature': 1.0, 'top_p': 1.0, 'top_k': -1, 'min_p': 0.0,
            'repetition_penalty': 1.0, 'presence_penalty': 0.0, 'frequency_penalty': 0.0}}
        env = {'ok': True, 'sample_format': 'per_request_v1', 'reward_version': 'v12a',
               'goal_audit_schema': 'hidden-object-audit/v2', 'max_training_requests': 4, 'episode': run.expected_episode(self.cfg)}
        with mock.patch.object(run, 'json_get', side_effect=[proxy, env]):
            run.validate_health(self.cfg, 'a')
        env['episode']['max_actions'] = 60
        with mock.patch.object(run, 'json_get', side_effect=[proxy, env]):
            with self.assertRaisesRegex(run.RunError, 'budgets/context'):
                run.validate_health(self.cfg, 'a')

    def test_status_tolerates_an_in_progress_metric_append(self):
        (self.root / 'run/metrics.jsonl').write_bytes(b'{"step":1,"reward":0}\n{"step":2,')
        status = run.run_action(self.manifest, 'a', 'status')
        self.assertEqual(status['last_metric']['step'], 1)

    def test_stop_refuses_a_reused_pid_and_keeps_unrelated_process_alive(self):
        proc = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(30)'])
        self.addCleanup(lambda: (proc.terminate(), proc.wait()) if proc.poll() is None else None)
        identity = run.process_identity(proc.pid)
        path = self.root / 'pid.json'
        run.atomic_json(path, dict(identity, owner='not-owned', start_ticks=identity['start_ticks'] + 1))
        with self.assertRaisesRegex(run.RunError, 'reused'):
            run.stop_record(path, timeout=0.2)
        self.assertIsNone(proc.poll())

    def test_detached_child_sessions_are_stopped_by_verified_ownership(self):
        script = 'import subprocess,sys,time; subprocess.Popen([sys.executable,"-c","import time; time.sleep(30)"],start_new_session=True); time.sleep(30)'
        argv = [sys.executable, '-c', script]
        saved = run.start_role(self.manifest, 'a', 'env', argv, dict(os_environ()))
        path = run.record_path(self.cfg, 'a', 'env')
        self.addCleanup(lambda: run.stop_record(path, timeout=0.2))
        deadline = time.monotonic() + 3
        while len(run.owned_processes(saved)) < 2 and time.monotonic() < deadline:
            time.sleep(0.05)
        owned = run.owned_processes(saved)
        self.assertGreaterEqual(len(owned), 2)
        self.assertTrue(run.start_role(self.manifest, 'a', 'env', argv, dict(os_environ()))['already_running'])
        result = run.stop_record(path, timeout=0.5)
        self.assertTrue(result['stopped'])
        self.assertTrue(all(not run.same_process(p) for p in owned))


def os_environ():
    import os
    return os.environ


if __name__ == '__main__':
    unittest.main()
