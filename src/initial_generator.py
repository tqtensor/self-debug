# Code module for the initial generator which creates the initial attempt


import re

from langchain_core.prompts import ChatPromptTemplate

from src.llm import LLM
from src.utils import syntax_check


class InitialGenerator:
    def __init__(self, model: str, strategy: str):
        self.strategy = strategy
        self.llm = LLM(model=model, config={"temperature": 0, "max_retries": 10}).llm

    def generate(self, problem: str) -> str:
        if self.strategy == "zero-shot":
            # Zero-shot strategy
            system = "You are a helpful Self-debugging assistant that can understand and solve programming problems."
            human = """
            Given the problem description with the code, you need to fullfill the task by writing the code that solves the problem.

            If the problem asks to return the result via a variable, you need to write the code that assigns the result to the variable.
            For example:
                x = ... # put solution in this variable

            The problem is: {problem_description}.

            Please solve the problem by defining a function f that takes the input and returns the output.
            Then assign the result of calling the function to the variable x.
            For example:
            ```python
            def f(...):
                # your solution here
            x = f(...)
            ```"""
            prompt = ChatPromptTemplate.from_messages(
                [("system", system), ("human", human)]
            )

            # Invoke the LLM
            answer = self.llm.invoke(
                prompt.invoke(
                    {
                        "problem_description": problem,
                    }
                ),
            ).content

            # Extract code from the answer
            match = re.search(r"```python\n(.*?)\n```", answer, re.DOTALL)
            if match:
                code = match.group(1)
                if syntax_check(code)["status"] == "success":
                    return code
                else:
                    # TODO: Handle this case
                    return ""
            else:
                # TODO: Handle this case
                return ""
        elif self.strategy == "plan-and-solve":
            return self._empty(problem)
        else:
            raise ValueError(f"Unknown strategy: {self.strategy}")
