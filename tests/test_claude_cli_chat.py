"""
Fuzz tests for `ClaudeCliChat` in `src/rag_chain.py`.

The class shells out to the local `claude -p` CLI via `subprocess.run`. We mock
the module-imported `rag_chain.subprocess.run` so the tests never touch a real
binary or network.
"""

import importlib
import os
import subprocess
import sys
import unittest
from unittest import mock

# Make `src/` importable.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))
# Spec requires the literal form too.
sys.path.insert(0, "src")

# Other test files (e.g. test_manual_bug_fixes) install a MagicMock at
# sys.modules["rag_chain"] to skip heavyweight imports. Evict any cached mock,
# import the real module fresh, and keep a stable reference. Patching attributes
# on this captured module object (not via the "rag_chain" string) survives any
# later sys.modules replacement done by a sibling test file.
sys.modules.pop("rag_chain", None)
import rag_chain  # noqa: E402

# Defend against later eviction: re-import if a sibling replaces it.
if not hasattr(rag_chain, "ClaudeCliChat"):
    rag_chain = importlib.import_module("rag_chain")  # pragma: no cover

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage  # noqa: E402

ClaudeCliChat = rag_chain.ClaudeCliChat


def _completed(returncode=0, stdout="", stderr=""):
    """Build a fake CompletedProcess for `subprocess.run` to return."""
    return subprocess.CompletedProcess(
        args=["claude"], returncode=returncode, stdout=stdout, stderr=stderr
    )


class _RagChainTestBase(unittest.TestCase):
    """
    Pin `sys.modules["rag_chain"]` to the real module for each test.

    Sibling test files (notably test_manual_bug_fixes) install a MagicMock at
    sys.modules["rag_chain"] at import time and never restore it. If unittest
    later runs our tests with that MagicMock cached, `mock.patch
    ("rag_chain.subprocess.run")` would patch a no-op attribute on the
    MagicMock — leaving `_generate` calling the real subprocess. Restoring the
    real module per-test makes the patch land where we expect.
    """

    def setUp(self):
        self._prev_rag_chain = sys.modules.get("rag_chain")
        sys.modules["rag_chain"] = rag_chain

    def tearDown(self):
        if self._prev_rag_chain is None:
            sys.modules.pop("rag_chain", None)
        else:
            sys.modules["rag_chain"] = self._prev_rag_chain


class TestClaudeCliChatNormal(_RagChainTestBase):
    def test_normal_success_strips_trailing_newline(self):
        with mock.patch("rag_chain.subprocess.run") as run:
            run.return_value = _completed(returncode=0, stdout="hello\n", stderr="")
            chat = ClaudeCliChat(model="opus")
            result = chat._generate([HumanMessage(content="hi")])
        self.assertEqual(result.generations[0].message.content, "hello")

    def test_subprocess_args_shape(self):
        with mock.patch("rag_chain.subprocess.run") as run:
            run.return_value = _completed(returncode=0, stdout="ok\n", stderr="")
            chat = ClaudeCliChat(model="opus")
            chat._generate([HumanMessage(content="hi")])
        args, kwargs = run.call_args
        # Positional argv list.
        argv = args[0]
        self.assertEqual(argv[0], "claude")
        self.assertEqual(argv[1], "-p")
        self.assertEqual(argv[-2], "--model")
        self.assertEqual(argv[-1], "opus")
        # Keyword args.
        self.assertTrue(kwargs.get("capture_output"))
        self.assertTrue(kwargs.get("text"))
        self.assertEqual(kwargs.get("timeout"), 180)


class TestClaudeCliChatErrors(_RagChainTestBase):
    def test_timeout_returns_friendly_message(self):
        with mock.patch("rag_chain.subprocess.run") as run:
            run.side_effect = subprocess.TimeoutExpired(cmd="claude", timeout=180)
            chat = ClaudeCliChat(model="opus")
            result = chat._generate([HumanMessage(content="hi")])
        content = result.generations[0].message.content
        self.assertIn("timed out", content.lower())

    def test_file_not_found_says_cli_not_found(self):
        with mock.patch("rag_chain.subprocess.run") as run:
            run.side_effect = FileNotFoundError()
            chat = ClaudeCliChat(model="opus")
            result = chat._generate([HumanMessage(content="hi")])
        content = result.generations[0].message.content
        # Must clearly indicate CLI was not found.
        self.assertIn("not found", content.lower())
        self.assertIn("claude", content.lower())

    def test_nonzero_exit_surfaces_stderr_when_no_stdout(self):
        with mock.patch("rag_chain.subprocess.run") as run:
            run.return_value = _completed(returncode=1, stdout="", stderr="rate limit")
            chat = ClaudeCliChat(model="opus")
            result = chat._generate([HumanMessage(content="hi")])
        content = result.generations[0].message.content
        self.assertIn("rate limit", content)

    def test_nonzero_exit_with_stdout_returns_stdout(self):
        # Non-fatal: if the CLI printed something, surface it rather than the error.
        with mock.patch("rag_chain.subprocess.run") as run:
            run.return_value = _completed(
                returncode=1, stdout="partial", stderr="warning"
            )
            chat = ClaudeCliChat(model="opus")
            result = chat._generate([HumanMessage(content="hi")])
        content = result.generations[0].message.content
        self.assertEqual(content, "partial")


class TestClaudeCliChatEdgeCases(_RagChainTestBase):
    def test_empty_stdout_returns_empty_string(self):
        with mock.patch("rag_chain.subprocess.run") as run:
            run.return_value = _completed(returncode=0, stdout="", stderr="")
            chat = ClaudeCliChat(model="opus")
            result = chat._generate([HumanMessage(content="hi")])
        self.assertEqual(result.generations[0].message.content, "")

    def test_very_long_stdout_passes_through(self):
        big = "x" * 100000
        with mock.patch("rag_chain.subprocess.run") as run:
            run.return_value = _completed(returncode=0, stdout=big, stderr="")
            chat = ClaudeCliChat(model="opus")
            result = chat._generate([HumanMessage(content="hi")])
        self.assertEqual(result.generations[0].message.content, big)


class TestClaudeCliChatMessageFlattening(_RagChainTestBase):
    def test_system_and_human_messages_tagged(self):
        with mock.patch("rag_chain.subprocess.run") as run:
            run.return_value = _completed(returncode=0, stdout="ok", stderr="")
            chat = ClaudeCliChat(model="opus")
            chat._generate([
                SystemMessage(content="you are helpful"),
                HumanMessage(content="hi"),
            ])
        prompt = run.call_args.args[0][2]
        self.assertIn("<system>", prompt)
        self.assertIn("you are helpful", prompt)
        self.assertIn("</system>", prompt)
        self.assertIn("<user>", prompt)
        self.assertIn("hi", prompt)
        self.assertIn("</user>", prompt)

    def test_ai_message_in_history_tagged(self):
        with mock.patch("rag_chain.subprocess.run") as run:
            run.return_value = _completed(returncode=0, stdout="ok", stderr="")
            chat = ClaudeCliChat(model="opus")
            chat._generate([
                HumanMessage(content="hi"),
                AIMessage(content="hello back"),
                HumanMessage(content="follow up"),
            ])
        prompt = run.call_args.args[0][2]
        self.assertIn("<assistant>", prompt)
        self.assertIn("hello back", prompt)
        self.assertIn("</assistant>", prompt)


if __name__ == "__main__":
    unittest.main()
