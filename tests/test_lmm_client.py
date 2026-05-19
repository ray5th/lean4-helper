import os
import sys
import unittest
from unittest.mock import MagicMock, patch

# Make `src/` importable.
sys.path.insert(0, 'src')
sys.path.insert(
    0,
    os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')),
)

# Patch `groq.Groq` BEFORE importing lmm_client because LMMClient.__init__
# instantiates Groq() at construction time. If we did not patch first, the
# real Groq client would try to read GROQ_API_KEY from the environment and
# raise during test collection.
_groq_patcher = patch('groq.Groq')
_MockGroq = _groq_patcher.start()
_MockGroq.return_value = MagicMock()

import lmm_client  # noqa: E402
from lmm_client import LMMClient  # noqa: E402


def _make_response(content):
    """Build a fake Groq chat.completions response with the given content."""
    response = MagicMock()
    choice = MagicMock()
    choice.message.content = content
    response.choices = [choice]
    return response


def _make_empty_choices_response():
    response = MagicMock()
    response.choices = []
    return response


class TestLMMClientInit(unittest.TestCase):
    def test_default_model_name(self):
        with patch.object(lmm_client, 'Groq') as MockGroq:
            MockGroq.return_value = MagicMock()
            client = LMMClient()
        self.assertEqual(client.model_name, "llama-3.3-70b-versatile")

    def test_custom_model_name(self):
        with patch.object(lmm_client, 'Groq') as MockGroq:
            MockGroq.return_value = MagicMock()
            client = LMMClient(model_name="custom-model")
        self.assertEqual(client.model_name, "custom-model")

    def test_init_instantiates_groq_client(self):
        with patch.object(lmm_client, 'Groq') as MockGroq:
            fake = MagicMock()
            MockGroq.return_value = fake
            client = LMMClient()
            MockGroq.assert_called_once_with()
            self.assertIs(client._client, fake)


class TestLMMClientChat(unittest.TestCase):
    def setUp(self):
        self.groq_patcher = patch.object(lmm_client, 'Groq')
        MockGroq = self.groq_patcher.start()
        self.fake_groq = MagicMock()
        MockGroq.return_value = self.fake_groq
        self.client = LMMClient()
        # Default: every call returns a non-empty completion.
        self.fake_groq.chat.completions.create.return_value = _make_response("ok")

    def tearDown(self):
        self.groq_patcher.stop()

    def test_chat_no_system_prompt(self):
        self.fake_groq.chat.completions.create.return_value = _make_response("hi there")
        result = self.client.chat("hello")
        self.assertEqual(result, "hi there")
        self.fake_groq.chat.completions.create.assert_called_once()
        kwargs = self.fake_groq.chat.completions.create.call_args.kwargs
        self.assertEqual(kwargs["model"], "llama-3.3-70b-versatile")
        self.assertEqual(kwargs["max_tokens"], 1024)
        self.assertEqual(
            kwargs["messages"],
            [{"role": "user", "content": "hello"}],
        )

    def test_chat_with_system_prompt(self):
        self.fake_groq.chat.completions.create.return_value = _make_response("reply")
        result = self.client.chat("hello", system_prompt="you are a bot")
        self.assertEqual(result, "reply")
        kwargs = self.fake_groq.chat.completions.create.call_args.kwargs
        self.assertEqual(
            kwargs["messages"],
            [
                {"role": "system", "content": "you are a bot"},
                {"role": "user", "content": "hello"},
            ],
        )

    def test_chat_empty_prompt(self):
        # Locks in current behavior: empty prompts are forwarded to Groq
        # rather than rejected. The downstream model can choose how to
        # respond; our wrapper just relays.
        self.fake_groq.chat.completions.create.return_value = _make_response("")
        result = self.client.chat("")
        self.assertEqual(result, "")
        kwargs = self.fake_groq.chat.completions.create.call_args.kwargs
        self.assertEqual(
            kwargs["messages"],
            [{"role": "user", "content": ""}],
        )

    def test_chat_uses_custom_model_name(self):
        with patch.object(lmm_client, 'Groq') as MockGroq:
            fake = MagicMock()
            MockGroq.return_value = fake
            fake.chat.completions.create.return_value = _make_response("x")
            client = LMMClient(model_name="my-model")
            client.chat("ping")
            kwargs = fake.chat.completions.create.call_args.kwargs
            self.assertEqual(kwargs["model"], "my-model")

    def test_chat_handles_empty_choices(self):
        # Regression guard for a real bug: Groq can return an empty
        # `choices` list (rate-limited, content-filtered, etc.). Indexing
        # `[0]` would raise IndexError. We now return an empty string.
        self.fake_groq.chat.completions.create.return_value = _make_empty_choices_response()
        result = self.client.chat("hello")
        self.assertEqual(result, "")

    def test_chat_handles_missing_message(self):
        response = MagicMock()
        choice = MagicMock()
        choice.message = None
        response.choices = [choice]
        self.fake_groq.chat.completions.create.return_value = response
        self.assertEqual(self.client.chat("hello"), "")

    def test_chat_handles_none_content(self):
        response = MagicMock()
        choice = MagicMock()
        choice.message.content = None
        response.choices = [choice]
        self.fake_groq.chat.completions.create.return_value = response
        self.assertEqual(self.client.chat("hello"), "")


class TestGenerateProofSteps(unittest.TestCase):
    def setUp(self):
        self.groq_patcher = patch.object(lmm_client, 'Groq')
        MockGroq = self.groq_patcher.start()
        self.fake_groq = MagicMock()
        MockGroq.return_value = self.fake_groq
        self.fake_groq.chat.completions.create.return_value = _make_response("proof")
        self.client = LMMClient()

    def tearDown(self):
        self.groq_patcher.stop()

    def test_prompt_contains_code_goals_and_errors(self):
        lean_code = "import Mathlib\n\ntheorem foo : True := sorry"
        goals = ["⊢ True"]
        errors = ["uses sorry"]

        result = self.client.generate_proof_steps(lean_code, goals=goals, errors=errors)
        self.assertEqual(result, "proof")

        kwargs = self.fake_groq.chat.completions.create.call_args.kwargs
        messages = kwargs["messages"]
        # generate_proof_steps always sends a system prompt + user message
        self.assertEqual(len(messages), 2)
        self.assertEqual(messages[0]["role"], "system")
        self.assertIn("Lean 4 proof assistant", messages[0]["content"])

        user_content = messages[1]["content"]
        self.assertIn("import Mathlib", user_content)
        self.assertIn("theorem foo : True := sorry", user_content)
        self.assertIn("⊢ True", user_content)
        self.assertIn("uses sorry", user_content)
        self.assertIn("Current Proof Goals:", user_content)
        self.assertIn("Lean Errors:", user_content)

    def test_empty_goals_and_errors_do_not_crash(self):
        result = self.client.generate_proof_steps("anything", goals=[], errors=[])
        self.assertEqual(result, "proof")
        # The formatted prompt still has the section headers, just with empty
        # bodies right after them.
        user_content = self.fake_groq.chat.completions.create.call_args.kwargs[
            "messages"
        ][1]["content"]
        self.assertIn("Current Proof Goals:\n\n", user_content)
        self.assertIn("Lean Errors:\n\n", user_content)

    def test_multiple_goals_and_errors_joined_with_newlines(self):
        self.client.generate_proof_steps(
            "code",
            goals=["g1", "g2"],
            errors=["e1", "e2", "e3"],
        )
        user_content = self.fake_groq.chat.completions.create.call_args.kwargs[
            "messages"
        ][1]["content"]
        self.assertIn("g1\ng2", user_content)
        self.assertIn("e1\ne2\ne3", user_content)


class TestNoEnvVarLeak(unittest.TestCase):
    """Constructing LMMClient must not require GROQ_API_KEY when Groq is mocked."""

    def test_construct_without_env_var(self):
        original = os.environ.pop("GROQ_API_KEY", None)
        try:
            with patch.object(lmm_client, 'Groq') as MockGroq:
                MockGroq.return_value = MagicMock()
                LMMClient()  # should not raise
        finally:
            if original is not None:
                os.environ["GROQ_API_KEY"] = original


if __name__ == '__main__':
    unittest.main()
