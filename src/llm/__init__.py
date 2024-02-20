import os

from langchain_openai import AzureChatOpenAI


class LLM:
    def __init__(self, model: str, config: dict):
        if model == "gpt35-turbo":
            required_env_vars = [
                "OPENAI_API_VERSION",
                "OPENAI_API_KEY",
                "AZURE_OPENAI_ENDPOINT",
                "AZURE_OPENAI_DEPLOYMENT",
            ]

            for env_var in required_env_vars:
                assert os.getenv(env_var, None) is not None, f"{env_var} not set"

            self.llm = AzureChatOpenAI(
                deployment_name=os.getenv("AZURE_OPENAI_DEPLOYMENT"),
                temperature=config.get("temperature", 0.5),
                max_retries=config.get("max_retries", 3),
            )
        else:
            raise ValueError(f"Unknown model: {model}")
