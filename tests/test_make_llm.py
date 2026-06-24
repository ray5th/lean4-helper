import os
import sys
import unittest
from unittest.mock import patch

# Add src to Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

import rag_chain
from rag_chain import _make_llm


class TestMakeLLMClaudeCli(unittest.TestCase):
    def test_claude_cli_opus(self):
        """claude-cli-opus -> ClaudeCliChat(model='opus'); Anthropic + Groq NOT called."""
        with patch.object(rag_chain, 'ClaudeCliChat') as MockCli, \
             patch.object(rag_chain, 'ChatAnthropic') as MockAnthropic, \
             patch.object(rag_chain, 'ChatGroq') as MockGroq:
            _make_llm("claude-cli-opus", None)
            MockCli.assert_called_once_with(model="opus")
            MockAnthropic.assert_not_called()
            MockGroq.assert_not_called()

    def test_claude_cli_prefix_stripped(self):
        """claude-cli-sonnet-4-6 -> ClaudeCliChat(model='sonnet-4-6') (prefix stripped)."""
        with patch.object(rag_chain, 'ClaudeCliChat') as MockCli, \
             patch.object(rag_chain, 'ChatAnthropic') as MockAnthropic, \
             patch.object(rag_chain, 'ChatGroq') as MockGroq:
            _make_llm("claude-cli-sonnet-4-6", None)
            MockCli.assert_called_once_with(model="sonnet-4-6")
            MockAnthropic.assert_not_called()
            MockGroq.assert_not_called()


class TestMakeLLMAnthropic(unittest.TestCase):
    def test_anthropic_no_api_key(self):
        """claude-opus-4-7, api_key=None -> ChatAnthropic(model=..., max_tokens=512), no anthropic_api_key."""
        with patch.object(rag_chain, 'ClaudeCliChat') as MockCli, \
             patch.object(rag_chain, 'ChatAnthropic') as MockAnthropic, \
             patch.object(rag_chain, 'ChatGroq') as MockGroq:
            _make_llm("claude-opus-4-7", None)
            MockAnthropic.assert_called_once()
            kwargs = MockAnthropic.call_args.kwargs
            self.assertEqual(kwargs.get("model"), "claude-opus-4-7")
            self.assertEqual(kwargs.get("max_tokens"), 512)
            self.assertNotIn("anthropic_api_key", kwargs)
            MockCli.assert_not_called()
            MockGroq.assert_not_called()

    def test_anthropic_with_api_key(self):
        """claude-opus-4-7, api_key='sk-ant-xxx' -> ChatAnthropic with anthropic_api_key='sk-ant-xxx'."""
        with patch.object(rag_chain, 'ClaudeCliChat') as MockCli, \
             patch.object(rag_chain, 'ChatAnthropic') as MockAnthropic, \
             patch.object(rag_chain, 'ChatGroq') as MockGroq:
            _make_llm("claude-opus-4-7", "sk-ant-xxx")
            MockAnthropic.assert_called_once()
            kwargs = MockAnthropic.call_args.kwargs
            self.assertEqual(kwargs.get("model"), "claude-opus-4-7")
            self.assertEqual(kwargs.get("max_tokens"), 512)
            self.assertEqual(kwargs.get("anthropic_api_key"), "sk-ant-xxx")
            MockCli.assert_not_called()
            MockGroq.assert_not_called()

    def test_claude_sonnet_anthropic_not_cli(self):
        """claude-sonnet-4-6, api_key=None -> Anthropic path (not CLI)."""
        with patch.object(rag_chain, 'ClaudeCliChat') as MockCli, \
             patch.object(rag_chain, 'ChatAnthropic') as MockAnthropic, \
             patch.object(rag_chain, 'ChatGroq') as MockGroq:
            _make_llm("claude-sonnet-4-6", None)
            MockAnthropic.assert_called_once()
            kwargs = MockAnthropic.call_args.kwargs
            self.assertEqual(kwargs.get("model"), "claude-sonnet-4-6")
            self.assertEqual(kwargs.get("max_tokens"), 512)
            MockCli.assert_not_called()
            MockGroq.assert_not_called()


class TestMakeLLMGroq(unittest.TestCase):
    def test_groq_no_api_key(self):
        """llama-3.3-70b-versatile, api_key=None -> ChatGroq(model=..., max_tokens=512), no groq_api_key."""
        with patch.object(rag_chain, 'ClaudeCliChat') as MockCli, \
             patch.object(rag_chain, 'ChatAnthropic') as MockAnthropic, \
             patch.object(rag_chain, 'ChatGroq') as MockGroq:
            _make_llm("llama-3.3-70b-versatile", None)
            MockGroq.assert_called_once()
            kwargs = MockGroq.call_args.kwargs
            self.assertEqual(kwargs.get("model"), "llama-3.3-70b-versatile")
            self.assertEqual(kwargs.get("max_tokens"), 512)
            self.assertNotIn("groq_api_key", kwargs)
            MockCli.assert_not_called()
            MockAnthropic.assert_not_called()

    def test_groq_with_api_key(self):
        """llama-3.3-70b-versatile, api_key='gsk_xxx' -> ChatGroq with groq_api_key='gsk_xxx'."""
        with patch.object(rag_chain, 'ClaudeCliChat') as MockCli, \
             patch.object(rag_chain, 'ChatAnthropic') as MockAnthropic, \
             patch.object(rag_chain, 'ChatGroq') as MockGroq:
            _make_llm("llama-3.3-70b-versatile", "gsk_xxx")
            MockGroq.assert_called_once()
            kwargs = MockGroq.call_args.kwargs
            self.assertEqual(kwargs.get("model"), "llama-3.3-70b-versatile")
            self.assertEqual(kwargs.get("max_tokens"), 512)
            self.assertEqual(kwargs.get("groq_api_key"), "gsk_xxx")
            MockCli.assert_not_called()
            MockAnthropic.assert_not_called()

    def test_gemma_falls_to_groq_default(self):
        """gemma2-9b-it -> ChatGroq (default branch)."""
        with patch.object(rag_chain, 'ClaudeCliChat') as MockCli, \
             patch.object(rag_chain, 'ChatAnthropic') as MockAnthropic, \
             patch.object(rag_chain, 'ChatGroq') as MockGroq:
            _make_llm("gemma2-9b-it", None)
            MockGroq.assert_called_once()
            kwargs = MockGroq.call_args.kwargs
            self.assertEqual(kwargs.get("model"), "gemma2-9b-it")
            self.assertEqual(kwargs.get("max_tokens"), 512)
            MockCli.assert_not_called()
            MockAnthropic.assert_not_called()


class TestMaxTokensAlwaysSet(unittest.TestCase):
    def test_max_tokens_in_anthropic_kwargs(self):
        """max_tokens=512 always in Anthropic kwargs (with and without api_key)."""
        with patch.object(rag_chain, 'ChatAnthropic') as MockAnthropic, \
             patch.object(rag_chain, 'ChatGroq'), \
             patch.object(rag_chain, 'ClaudeCliChat'):
            _make_llm("claude-opus-4-7", None)
            self.assertEqual(MockAnthropic.call_args.kwargs.get("max_tokens"), 512)
            MockAnthropic.reset_mock()
            _make_llm("claude-opus-4-7", "sk-ant-xxx")
            self.assertEqual(MockAnthropic.call_args.kwargs.get("max_tokens"), 512)

    def test_max_tokens_in_groq_kwargs(self):
        """max_tokens=512 always in Groq kwargs (with and without api_key)."""
        with patch.object(rag_chain, 'ChatAnthropic'), \
             patch.object(rag_chain, 'ChatGroq') as MockGroq, \
             patch.object(rag_chain, 'ClaudeCliChat'):
            _make_llm("llama-3.3-70b-versatile", None)
            self.assertEqual(MockGroq.call_args.kwargs.get("max_tokens"), 512)
            MockGroq.reset_mock()
            _make_llm("llama-3.3-70b-versatile", "gsk_xxx")
            self.assertEqual(MockGroq.call_args.kwargs.get("max_tokens"), 512)


class TestRAGProofChainForwardsApiKey(unittest.TestCase):
    def test_init_forwards_api_key_to_make_llm(self):
        """RAGProofChain(model_name='claude-opus-4-7', api_key='test-key')
        should call _make_llm('claude-opus-4-7', 'test-key')."""
        with patch.object(rag_chain, '_make_llm') as mock_make_llm:
            mock_make_llm.return_value = unittest.mock.MagicMock()
            rag_chain.RAGProofChain(model_name="claude-opus-4-7", api_key="test-key")
            mock_make_llm.assert_called_once_with("claude-opus-4-7", "test-key")


if __name__ == "__main__":
    unittest.main()
