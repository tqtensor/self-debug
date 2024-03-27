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
        code_context: str,
        stack_overflow: Optional[StackOverflowDataset],
        feedback: Optional[str],
    ) -> str:
        if self.strategy == "zero-shot":
            # Zero-shot strategy
            system = "You are a helpful Self-debugging assistant that can understand and solve programming problems. You have the ability to analyze and execute code, providing feedback and suggestions to help users debug and improve their code. By leveraging your knowledge and expertise, you can assist users in solving complex programming problems and guide them towards writing correct and efficient code. Your goal is to empower users to become better programmers by providing them with valuable insights and assistance throughout their coding journey."
            human = """
            Given the problem description with the code, you need to fulfill the task by writing the code that solves the problem.

            The problem is: {problem_description}.

            Your solution will be evaluated by replacing the code inside the block
            ```
            # SOLUTION START
            [insert]
            # SOLUTION END
            ```
            with your code as follows:
            ```python
            {code_context}
            ```

            Make sure your code is correct and complete to solve the problem.
            """

            if feedback:
                if "traceback" in feedback[1].lower():
                    human += """
                    In the previous attempt, you generated the following code inside the block ###BEGIN SOLUTION and ###END SOLUTION:
                    ```
                    {generated_code}
                    ```
                    However, the code has an error. The error message is:
                    ```
                    {feedback}
                    ```
                    Please analyze the error message and fix the code accordingly.
                    """
                elif ("executed" in feedback[1].lower()) and (
                    "expected" in feedback[1].lower()
                ):
                    human += """
                    In the previous attempt, you generated the following code inside the block ###BEGIN SOLUTION and ###END SOLUTION:
                    ```
                    {generated_code}
                    ```
                    The code executed successfully but failed the test case. Please analyze the difference between the executed result and the expected result to fix the code accordingly. The deviation from the expected result is:
                    ```
                    {feedback}
                    ```
                    """
            else:
                pass

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
                                "code_context": code_context,
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
                            "code_context": code_context,
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
            system = "You are a helpful Chain-of-Thought generator that can understand the reasoning behind programming problems and provide step-by-step guidance to solve them."
            human = """
            Given the problem description with the code, and a Stack Overflow post, you need to learn from the comments to generate step-by-step suggestions that help another agent to solve the problem.

            The given problem is: {problem_description}.

            The Stack Overflow post with supportive comments is: {post}.

            Please generate a series of suggestions that help another agent to solve the problem step-by-step.
            Here are some suggestions:
            - Suggestion 1: [...]
            - Suggestion 2: [...]
            - Suggestion 3: [...]
            - Final suggestion: [...]
            """
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
            system = "You are a helpful Self-debugging assistant that can understand and solve programming problems. You have the ability to analyze and execute code, providing feedback and suggestions to help users debug and improve their code. By leveraging your knowledge and expertise, you can assist users in solving complex programming problems and guide them towards writing correct and efficient code. Your goal is to empower users to become better programmers by providing them with valuable insights and assistance throughout their coding journey."
            human = """
            Given the problem description with the code, you need to fulfill the task by writing the code that solves the problem.

            The problem is: {problem_description}.

            Your solution will be evaluated by replacing the code inside the block
            ```
            # SOLUTION START
            [insert]
            # SOLUTION END
            ```
            with your code as follows:
            ```python
            {code_context}
            ```

            To support you in solving the problem, here are the suggestions from the Stack Overflow post comments:
            {cot_suggestions}

            Make sure your code is correct and complete to solve the problem.
            """
            prompt = ChatPromptTemplate.from_messages(
                [("system", system), ("human", human)]
            )

            # Invoke the LLM
            answer = self.llm.invoke(
                prompt.invoke(
                    {
                        "problem_description": problem,
                        "code_context": code_context,
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
