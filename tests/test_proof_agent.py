import os
import sys
import unittest
from unittest.mock import MagicMock, patch

# Add src to Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

# Mock langgraph_agent.LangGraphAgent before importing proof_agent (so its
# top-level `from langgraph_agent import LangGraphAgent` doesn't pull in the
# heavy real module). Restore sys.modules afterward so this file doesn't
# poison the langgraph_agent import for other test files in the same run.
_orig_langgraph_agent = sys.modules.get('langgraph_agent')
sys.modules['langgraph_agent'] = MagicMock()

from proof_agent import ProofAgent

if _orig_langgraph_agent is not None:
    sys.modules['langgraph_agent'] = _orig_langgraph_agent
else:
    del sys.modules['langgraph_agent']


class TestProofAgentConstructor(unittest.TestCase):
    def test_default_args_constructs_langgraph_agent_with_defaults(self):
        with patch('proof_agent.LangGraphAgent') as MockLangGraphAgent:
            ProofAgent()
            MockLangGraphAgent.assert_called_once_with(
                model_name="llama-3.3-70b-versatile",
                max_retries=5,
            )

    def test_custom_model_name_forwarded(self):
        with patch('proof_agent.LangGraphAgent') as MockLangGraphAgent:
            ProofAgent(model_name="custom-model:7b")
            MockLangGraphAgent.assert_called_once_with(
                model_name="custom-model:7b",
                max_retries=5,
            )

    def test_custom_max_retries_forwarded(self):
        with patch('proof_agent.LangGraphAgent') as MockLangGraphAgent:
            ProofAgent(max_retries=10)
            MockLangGraphAgent.assert_called_once_with(
                model_name="llama-3.3-70b-versatile",
                max_retries=10,
            )

    def test_all_custom_args_forwarded(self):
        with patch('proof_agent.LangGraphAgent') as MockLangGraphAgent:
            ProofAgent(model_name="other-model:13b", max_retries=3)
            MockLangGraphAgent.assert_called_once_with(
                model_name="other-model:13b",
                max_retries=3,
            )

    def test_underlying_agent_stored_on_instance(self):
        with patch('proof_agent.LangGraphAgent') as MockLangGraphAgent:
            mock_instance = MagicMock()
            MockLangGraphAgent.return_value = mock_instance
            agent = ProofAgent()
            self.assertIs(agent._agent, mock_instance)


class TestProofAgentSolveFile(unittest.TestCase):
    def test_solve_file_delegates_returning_true(self):
        with patch('proof_agent.LangGraphAgent') as MockLangGraphAgent:
            mock_instance = MagicMock()
            mock_instance.solve_file.return_value = True
            MockLangGraphAgent.return_value = mock_instance

            agent = ProofAgent()
            result = agent.solve_file("path/to/file.lean")

            self.assertTrue(result)
            mock_instance.solve_file.assert_called_once_with("path/to/file.lean")

    def test_solve_file_delegates_returning_false(self):
        with patch('proof_agent.LangGraphAgent') as MockLangGraphAgent:
            mock_instance = MagicMock()
            mock_instance.solve_file.return_value = False
            MockLangGraphAgent.return_value = mock_instance

            agent = ProofAgent()
            result = agent.solve_file("path/to/file.lean")

            self.assertFalse(result)
            mock_instance.solve_file.assert_called_once_with("path/to/file.lean")

    def test_solve_file_path_is_forwarded_exactly(self):
        with patch('proof_agent.LangGraphAgent') as MockLangGraphAgent:
            mock_instance = MagicMock()
            mock_instance.solve_file.return_value = True
            MockLangGraphAgent.return_value = mock_instance

            agent = ProofAgent()
            agent.solve_file("/absolute/path/proof.lean")
            mock_instance.solve_file.assert_called_once_with("/absolute/path/proof.lean")

    def test_solve_file_propagates_exception(self):
        with patch('proof_agent.LangGraphAgent') as MockLangGraphAgent:
            mock_instance = MagicMock()
            mock_instance.solve_file.side_effect = RuntimeError("boom")
            MockLangGraphAgent.return_value = mock_instance

            agent = ProofAgent()
            with self.assertRaises(RuntimeError):
                agent.solve_file("foo.lean")


class TestProofAgentNoExtraState(unittest.TestCase):
    def test_no_unexpected_public_methods(self):
        with patch('proof_agent.LangGraphAgent'):
            agent = ProofAgent()
            public_attrs = {a for a in dir(agent) if not a.startswith('_')}
            # The wrapper should only expose solve_file
            self.assertEqual(public_attrs, {"solve_file"})

    def test_solve_file_does_not_mutate_wrapper_state(self):
        with patch('proof_agent.LangGraphAgent') as MockLangGraphAgent:
            mock_instance = MagicMock()
            mock_instance.solve_file.return_value = True
            MockLangGraphAgent.return_value = mock_instance

            agent = ProofAgent()
            before = set(vars(agent).keys())
            agent.solve_file("a.lean")
            after = set(vars(agent).keys())
            self.assertEqual(before, after)

    def test_multiple_calls_use_same_underlying_agent(self):
        with patch('proof_agent.LangGraphAgent') as MockLangGraphAgent:
            mock_instance = MagicMock()
            mock_instance.solve_file.return_value = True
            MockLangGraphAgent.return_value = mock_instance

            agent = ProofAgent()
            agent.solve_file("a.lean")
            agent.solve_file("b.lean")

            # Only one underlying agent constructed
            self.assertEqual(MockLangGraphAgent.call_count, 1)
            self.assertEqual(mock_instance.solve_file.call_count, 2)


if __name__ == '__main__':
    unittest.main()
