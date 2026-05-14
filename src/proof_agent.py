from langgraph_agent import LangGraphAgent


class ProofAgent:
    """Thin compatibility wrapper around LangGraphAgent."""

    def __init__(self, model_name: str = "qwen3-vl:4b", max_retries: int = 5):
        self._agent = LangGraphAgent(model_name=model_name, max_retries=max_retries)

    def solve_file(self, file_path: str) -> bool:
        return self._agent.solve_file(file_path)
