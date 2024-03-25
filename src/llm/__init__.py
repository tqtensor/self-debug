import os

from langchain_openai import AzureChatOpenAI


class LLM:
    def __init__(self, model: str, config: dict):
        if model == "gpt35-turbo":
            self.llm = AzureChatOpenAI(
                openai_api_key=os.getenv("AZURE_OPENAI_API_KEY"),
                openai_api_version=os.getenv("OPENAI_API_VERSION"),
                azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
                deployment_name="gpt-35",
                temperature=config.get("temperature", 0.5),
                max_retries=config.get("max_retries", 3),
                verbose=config.get("verbose", False),
            )
        else:
            raise ValueError(f"Unknown model: {model}")
