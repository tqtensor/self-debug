import re
from typing import Optional

import openai
from langchain_core.prompts import ChatPromptTemplate

from src.dataset.stack_overflow import StackOverflowDataset
from src.llm import LLM
from src.utils import syntax_check


class CodeGenerator:
    def __init__(self, model: str, strategy: str):
        self.strategy = strategy
        self.llm = LLM(
            model=model, config={"temperature": 0, "max_retries": 10, "verbose": True}
        ).llm

    def generate(
        self,
        problem: str,
        stack_overflow: Optional[StackOverflowDataset],
        feedback: Optional[str],
    ) -> str:
        if self.strategy == "zero-shot":
            # Zero-shot strategy
            system = "You are a helpful Self-debugging assistant that can understand and solve programming problems."
            human = """
            Given the problem description with the code, you need to fullfill the task by writing the code that solves the problem.

            If the problem asks to return the result via a variable, you need to write the code that assigns the result to the variable.
            For example:
                x = ... # put solution in this variable

            The problem is: {problem_description}.
            """

            if feedback:
                if "traceback" in feedback[1].lower():
                    human += "\n\nIn the previous attempt, you generated the following code inside the block ###BEGIN SOLUTION and ###END SOLUTION:\n```\n{generated_code}\n```"
                    human += "\n\nHowever, the code has an error. The error message is:\n```\n{feedback}\n```"
                    human += (
                        "Please analyze the error message and fix the code accordingly."
                    )
                elif ("executed" in feedback[1].lower()) and (
                    "expected" in feedback[1].lower()
                ):
                    human += "\n\nIn the previous attempt, you generated the following code inside the block ###BEGIN SOLUTION and ###END SOLUTION:\n```\n{generated_code}\n```"
                    human += "\n\nThe code executed successfully but failed the test case. Please analyze the difference between executed result versus the test expectation to fix the code accordingly."
            else:
                human += """
            Please solve the problem by defining a function f that takes the input and returns the output.
            Then identify the variable name that the question asks you to return the result via. It could be x, or result.
            Then assign the result of calling the function to that variable.
            For example:
            ```python
            def f(...):
                # your solution here
            identified_variable = f(...)
            ```"""

            prompt = ChatPromptTemplate.from_messages(
                [("system", system), ("human", human)]
            )

            # Invoke the LLM
            if feedback:
                try:
                    answer = self.llm.invoke(
                        prompt.invoke(
                            {
                                "problem_description": problem,
                                "generated_code": feedback[0],
                                "feedback": feedback[1],
                            }
                        ),
                    ).content
                except openai.BadRequestError:
                    answer = "assert True # I am sorry, I am unable to generate the code. Please try again later."
            else:
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
        elif self.strategy == "cot":
            # Retrieve relevant Stack Overflow post
            post = stack_overflow.retrieve(query=problem, k=1)

            # CoT generation
            system = "You are a helpful Chain-of-Thought generator that can understand the reasoning behind programming problems."
            human = """
            Given the problem description with the code, and a Stack Overflow post, you need to learn from the comments to generate step-by-step suggestion that help another agent to solve the problem.

            The give problem is: {problem_description}.

            The Stack Overflow post with supportive comments is: {post}.

            Please generate a series of suggestions that help another agent to solve the problem step-by-step.
            Suggestion 1: [...], then
            Suggestion 2: [...], then
            Suggestion 3: [...], then
            Final suggestion: [...]
            ...
            ```"""
            prompt = ChatPromptTemplate.from_messages(
                [("system", system), ("human", human)]
            )

            # Invoke the LLM
            cot_suggestions = self.llm.invoke(
                prompt.invoke(
                    {
                        "problem_description": problem,
                        "post": post,
                    }
                ),
            ).content

            # CoT strategy
            system = "You are a helpful Self-debugging assistant that can understand and solve programming problems."
            human = """
            Given the problem description with the code, you need to fullfill the task by writing the code that solves the problem.

            If the problem asks to return the result via a variable, you need to write the code that assigns the result to the variable.
            For example:
                x = ... # put solution in this variable

            The problem is: {problem_description}.

            Please solve the problem by defining a function f that takes the input and returns the output.
            Then identify the variable name that the question asks you to return the result via. It could be x, or result.
            Then assign the result of calling the function to that variable.
            For example:
            ```python
            def f(...):
                # your solution here
            identified_variable = f(...)

            To support you in solving the problem, here are the suggestions from the Stack Overflow post:
            {cot_suggestions}
            ```"""
            prompt = ChatPromptTemplate.from_messages(
                [("system", system), ("human", human)]
            )

            # Invoke the LLM
            answer = self.llm.invoke(
                prompt.invoke(
                    {
                        "problem_description": problem,
                        "cot_suggestions": cot_suggestions,
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
        else:
            raise ValueError(f"Unknown strategy: {self.strategy}")
