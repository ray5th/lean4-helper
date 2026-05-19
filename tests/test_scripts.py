"""
Unit tests for scripts/run_agent.py and scripts/build_index.py CLIs.

We mock out heavy dependencies (LangGraphAgent, MathLibRetriever) so the tests
exercise only the argparse / control-flow surface of each script.
"""
import contextlib
import importlib
import io
import os
import sys
import unittest
from unittest import mock

_HERE = os.path.dirname(__file__)
_ROOT = os.path.abspath(os.path.join(_HERE, ".."))
sys.path.insert(0, os.path.join(_ROOT, "scripts"))
sys.path.insert(0, os.path.join(_ROOT, "src"))


def _fresh_import(module_name: str):
    """(Re)import a script module so that patches applied via sys.modules
    take effect for top-level `from x import Y` statements."""
    if module_name in sys.modules:
        del sys.modules[module_name]
    return importlib.import_module(module_name)


# ---------------------------------------------------------------------------
# run_agent.py
# ---------------------------------------------------------------------------

class TestRunAgent(unittest.TestCase):
    def _run_main(self, argv, agent_success=True):
        """Patch LangGraphAgent + sys.argv, import run_agent, run main(),
        return (captured_stdout, MockAgent class, mock_instance)."""
        mock_agent_cls = mock.MagicMock(name="LangGraphAgent")
        mock_instance = mock_agent_cls.return_value
        mock_instance.solve_file.return_value = agent_success

        # Inject the mock into the source module so `from langgraph_agent import LangGraphAgent`
        # picks it up at import-time inside run_agent.
        with mock.patch.dict(sys.modules, {"langgraph_agent": mock.MagicMock(LangGraphAgent=mock_agent_cls)}):
            run_agent = _fresh_import("run_agent")
            buf = io.StringIO()
            with mock.patch.object(sys, "argv", argv), contextlib.redirect_stdout(buf):
                run_agent.main()
            return buf.getvalue(), mock_agent_cls, mock_instance

    def test_defaults(self):
        out, cls, inst = self._run_main(["run_agent.py", "problems/simple_add.lean"])
        cls.assert_called_once_with(
            model_name="qwen3-vl:4b",
            max_retries=5,
            index_dir=None,
        )
        inst.solve_file.assert_called_once_with("problems/simple_add.lean")
        self.assertIn("Success", out)

    def test_custom_model(self):
        _, cls, _ = self._run_main(
            ["run_agent.py", "problems/simple_add.lean", "--model", "custom-model"]
        )
        _, kwargs = cls.call_args
        self.assertEqual(kwargs["model_name"], "custom-model")

    def test_retries_forwarded(self):
        _, cls, _ = self._run_main(
            ["run_agent.py", "problems/simple_add.lean", "--retries", "3"]
        )
        _, kwargs = cls.call_args
        self.assertEqual(kwargs["max_retries"], 3)

    def test_index_dir_forwarded(self):
        _, cls, _ = self._run_main(
            ["run_agent.py", "problems/simple_add.lean", "--index-dir", "/tmp/idx"]
        )
        _, kwargs = cls.call_args
        self.assertEqual(kwargs["index_dir"], "/tmp/idx")

    def test_failure_prints_failed_message(self):
        out, _, _ = self._run_main(
            ["run_agent.py", "problems/simple_add.lean"], agent_success=False
        )
        self.assertIn("Failed", out)
        self.assertNotIn("Success", out)

    def test_missing_file_arg_exits(self):
        mock_agent_cls = mock.MagicMock(name="LangGraphAgent")
        with mock.patch.dict(sys.modules, {"langgraph_agent": mock.MagicMock(LangGraphAgent=mock_agent_cls)}):
            run_agent = _fresh_import("run_agent")
            with mock.patch.object(sys, "argv", ["run_agent.py"]):
                # argparse writes the error to stderr; swallow it for clean test output.
                with contextlib.redirect_stderr(io.StringIO()):
                    with self.assertRaises(SystemExit) as ctx:
                        run_agent.main()
                self.assertEqual(ctx.exception.code, 2)


# ---------------------------------------------------------------------------
# build_index.py
# ---------------------------------------------------------------------------

class TestBuildIndex(unittest.TestCase):
    def _run_main(self, argv, is_built=False):
        """Patch MathLibRetriever + sys.argv, import build_index, run main(),
        return (captured_stdout, MockRetriever class, mock_instance)."""
        mock_retr_cls = mock.MagicMock(name="MathLibRetriever")
        mock_instance = mock_retr_cls.return_value
        mock_instance.is_index_built.return_value = is_built
        mock_instance.index_dir = "/fake/index/dir"

        with mock.patch.dict(sys.modules, {"retriever": mock.MagicMock(MathLibRetriever=mock_retr_cls)}):
            build_index = _fresh_import("build_index")
            buf = io.StringIO()
            with mock.patch.object(sys, "argv", argv), contextlib.redirect_stdout(buf):
                build_index.main()
            return buf.getvalue(), mock_retr_cls, mock_instance

    def test_defaults(self):
        out, cls, inst = self._run_main(["build_index.py"])
        cls.assert_called_once_with(index_dir=None)
        inst.build.assert_called_once_with(mathlib_root=None, max_files=None)
        self.assertIn("Done", out)

    def test_mathlib_root_forwarded(self):
        _, _, inst = self._run_main(["build_index.py", "--mathlib-root", "/tmp/fake"])
        _, kwargs = inst.build.call_args
        self.assertEqual(kwargs["mathlib_root"], "/tmp/fake")

    def test_max_files_forwarded(self):
        _, _, inst = self._run_main(["build_index.py", "--max-files", "5"])
        _, kwargs = inst.build.call_args
        self.assertEqual(kwargs["max_files"], 5)

    def test_index_dir_forwarded(self):
        _, cls, _ = self._run_main(["build_index.py", "--index-dir", "/tmp/idx"])
        cls.assert_called_once_with(index_dir="/tmp/idx")

    def test_existing_index_skips_rebuild(self):
        out, _, inst = self._run_main(["build_index.py"], is_built=True)
        inst.build.assert_not_called()
        self.assertIn("Index already exists", out)

    def test_existing_index_rebuilds_when_max_files_set(self):
        out, _, inst = self._run_main(
            ["build_index.py", "--max-files", "5"], is_built=True
        )
        inst.build.assert_called_once_with(mathlib_root=None, max_files=5)
        self.assertNotIn("Index already exists", out)


if __name__ == "__main__":
    unittest.main()
