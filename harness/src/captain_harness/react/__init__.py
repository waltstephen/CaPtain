"""Codex-style bare ReAct driver (V1.1).

Task-agnostic by contract: this package knows about a workspace directory, a sandbox, tools,
text, images, budgets and a Responses-API model. It must never learn anything about the task
domain; see tests/test_import_boundaries.py for the enforced vocabulary and import rules.
"""
