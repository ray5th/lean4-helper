"""
Tests for the two-stage retry / fast_model logic in src/langgraph_agent.py.

Covers:
- make_generate_node(strong, fast): which chain runs on attempt 0 vs >=1, based
  on whether fast_chain is None or set.
- LangGraphAgent.__init__: when _fast_chain is built vs left None, based on
  fast_model being None / equal / different from model_name. Confirms
  RAGProofChain construction count matches the policy.
"""

import os
import sys
import unittest
from unittest.mock import MagicMock, patch

# Add src to Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
sys.path.insert(0, 'src')

import langgraph_agent
from langgraph_agent import LangGraphAgent, make_generate_node


def _base_state(**overrides):
    """Build a minimal ProofState dict, allowing field overrides."""
    state = {
        "file_path": "/tmp/fake.lean",
        "lean_code": "import Mathlib\n\ntheorem foo : True := by sorry",
        "goals": ["⊢ True"],
        "errors": ["err"],
        "attempt": 0,
        "max_retries": 5,
        "status": "pending",
        "retrieved_lemmas": [{"name": "trivial", "doc": ""}],
        "solved_at_attempt": 0,
    }
    state.update(overrides)
    return state


def _make_chain(output="```lean\nimport Mathlib\n\ntheorem foo : True := by trivial\n```"):
    """Mock chain whose .generate() returns `output`."""
    chain = MagicMock()
    chain.generate.return_value = output
    return chain


# ---------------------------------------------------------------------------
# make_generate_node: fast_chain is None -> always strong, every attempt
# ---------------------------------------------------------------------------

class TestGenerateNodeNoFastChain(unittest.TestCase):
    """When fast_chain is None, every attempt (0, 2, 5) uses strong_chain."""

    def setUp(self):
        self._write_patcher = patch.object(langgraph_agent, "_write_file")
        self._read_patcher = patch.object(langgraph_agent, "_read_file")
        self.mock_write = self._write_patcher.start()
        self.mock_read = self._read_patcher.start()
        self.addCleanup(self._write_patcher.stop)
        self.addCleanup(self._read_patcher.stop)

    def test_attempt_0_uses_strong_when_fast_is_none(self):
        strong = _make_chain()
        node = make_generate_node(strong, None)
        state = _base_state(attempt=0)

        node(state)

        strong.generate.assert_called_once()

    def test_attempt_2_uses_strong_when_fast_is_none(self):
        strong = _make_chain()
        node = make_generate_node(strong, None)
        state = _base_state(attempt=2)

        node(state)

        strong.generate.assert_called_once()

    def test_attempt_5_uses_strong_when_fast_is_none(self):
        strong = _make_chain()
        node = make_generate_node(strong, None)
        state = _base_state(attempt=5)

        node(state)

        strong.generate.assert_called_once()

    def test_default_fast_chain_arg_is_none(self):
        """make_generate_node(strong) — without fast_chain — must default to None."""
        strong = _make_chain()
        node = make_generate_node(strong)  # no second arg
        state = _base_state(attempt=0)

        node(state)

        strong.generate.assert_called_once()


# ---------------------------------------------------------------------------
# make_generate_node: fast_chain on attempt 0 only
# ---------------------------------------------------------------------------

class TestGenerateNodeFastOnAttemptZero(unittest.TestCase):
    """On attempt=0, fast_chain.generate is called and strong is not consulted."""

    def setUp(self):
        self._write_patcher = patch.object(langgraph_agent, "_write_file")
        self._read_patcher = patch.object(langgraph_agent, "_read_file")
        self.mock_write = self._write_patcher.start()
        self.mock_read = self._read_patcher.start()
        self.addCleanup(self._write_patcher.stop)
        self.addCleanup(self._read_patcher.stop)

    def test_attempt_0_uses_fast_chain(self):
        strong = _make_chain()
        fast = _make_chain()
        node = make_generate_node(strong, fast)
        state = _base_state(attempt=0)

        node(state)

        fast.generate.assert_called_once()
        strong.generate.assert_not_called()

    def test_attempt_0_passes_state_fields_to_fast(self):
        strong = _make_chain()
        fast = _make_chain()
        node = make_generate_node(strong, fast)
        state = _base_state(
            attempt=0,
            lean_code="import Mathlib\n\ntheorem t : 1 + 1 = 2 := by sorry",
            goals=["⊢ 1 + 1 = 2"],
            errors=["unsolved goals"],
            retrieved_lemmas=[{"name": "Nat.add_comm", "doc": ""}],
        )

        node(state)

        strong.generate.assert_not_called()
        kwargs = fast.generate.call_args.kwargs
        self.assertEqual(kwargs["lean_code"], state["lean_code"])
        self.assertEqual(kwargs["goals"], state["goals"])
        self.assertEqual(kwargs["errors"], state["errors"])
        self.assertEqual(kwargs["retrieved_lemmas"], state["retrieved_lemmas"])


# ---------------------------------------------------------------------------
# make_generate_node: strong on attempt >= 1, even if fast_chain is given
# ---------------------------------------------------------------------------

class TestGenerateNodeStrongOnRetry(unittest.TestCase):
    """On attempt >= 1, strong_chain is called and fast is not consulted."""

    def setUp(self):
        self._write_patcher = patch.object(langgraph_agent, "_write_file")
        self._read_patcher = patch.object(langgraph_agent, "_read_file")
        self.mock_write = self._write_patcher.start()
        self.mock_read = self._read_patcher.start()
        self.addCleanup(self._write_patcher.stop)
        self.addCleanup(self._read_patcher.stop)

    def test_attempt_1_uses_strong_not_fast(self):
        strong = _make_chain()
        fast = _make_chain()
        node = make_generate_node(strong, fast)
        state = _base_state(attempt=1)

        node(state)

        strong.generate.assert_called_once()
        fast.generate.assert_not_called()

    def test_attempt_2_uses_strong_not_fast(self):
        strong = _make_chain()
        fast = _make_chain()
        node = make_generate_node(strong, fast)
        state = _base_state(attempt=2)

        node(state)

        strong.generate.assert_called_once()
        fast.generate.assert_not_called()

    def test_attempt_5_uses_strong_not_fast(self):
        strong = _make_chain()
        fast = _make_chain()
        node = make_generate_node(strong, fast)
        state = _base_state(attempt=5)

        node(state)

        strong.generate.assert_called_once()
        fast.generate.assert_not_called()


# ---------------------------------------------------------------------------
# LangGraphAgent.__init__: fast_chain construction policy
# ---------------------------------------------------------------------------

class TestLangGraphAgentFastChainInit(unittest.TestCase):
    """Whether _fast_chain is created depends on fast_model vs model_name."""

    def setUp(self):
        # Patch heavyweight collaborators so __init__ doesn't try to boot
        # Lean / FAISS / network for real.
        self._env_patcher = patch.object(langgraph_agent, "LeanEnvironment")
        self._ret_patcher = patch.object(langgraph_agent, "MathLibRetriever")
        self._chain_patcher = patch.object(langgraph_agent, "RAGProofChain")
        self.mock_env_cls = self._env_patcher.start()
        self.mock_ret_cls = self._ret_patcher.start()
        self.mock_chain_cls = self._chain_patcher.start()
        self.addCleanup(self._env_patcher.stop)
        self.addCleanup(self._ret_patcher.stop)
        self.addCleanup(self._chain_patcher.stop)

        # Each RAGProofChain() call returns a fresh mock so the agent's
        # _chain and _fast_chain are distinguishable objects when both built.
        self.mock_chain_cls.side_effect = lambda *a, **kw: MagicMock(
            name=f"chain({kw.get('model_name', a[0] if a else '?')})"
        )

    def test_fast_model_none_leaves_fast_chain_none(self):
        agent = LangGraphAgent(model_name="strong-model", fast_model=None)
        self.assertIsNone(agent._fast_chain)
        # RAGProofChain should be constructed exactly once (for _chain only).
        self.assertEqual(self.mock_chain_cls.call_count, 1)

    def test_fast_model_equal_to_model_name_leaves_fast_chain_none(self):
        agent = LangGraphAgent(model_name="same-model", fast_model="same-model")
        self.assertIsNone(agent._fast_chain)
        # No point building a second identical chain.
        self.assertEqual(self.mock_chain_cls.call_count, 1)

    def test_fast_model_distinct_builds_separate_fast_chain(self):
        agent = LangGraphAgent(model_name="strong-model", fast_model="fast-model")
        self.assertIsNotNone(agent._fast_chain)
        self.assertIsNot(agent._chain, agent._fast_chain)
        # Both chains constructed: strong + fast.
        self.assertEqual(self.mock_chain_cls.call_count, 2)
        # Verify the two model names were both used.
        model_names_used = []
        for call in self.mock_chain_cls.call_args_list:
            kwargs = call.kwargs
            args = call.args
            name = kwargs.get("model_name", args[0] if args else None)
            model_names_used.append(name)
        self.assertIn("strong-model", model_names_used)
        self.assertIn("fast-model", model_names_used)

    def test_fast_model_empty_string_leaves_fast_chain_none(self):
        """Empty-string fast_model is falsy and should NOT build a second chain."""
        agent = LangGraphAgent(model_name="strong-model", fast_model="")
        self.assertIsNone(agent._fast_chain)
        self.assertEqual(self.mock_chain_cls.call_count, 1)


if __name__ == "__main__":
    unittest.main()
