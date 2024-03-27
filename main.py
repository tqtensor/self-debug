# Reset the environment variables
import os

for key in list(os.environ.keys()):
    if ("AZURE" in key) or ("OPENAI" in key):
        del os.environ[key]

from dotenv import load_dotenv

load_dotenv(".env")


import glob
import os
import time
from collections import defaultdict

from tqdm.auto import tqdm

from src.code_generator import CodeGenerator
from src.dataset import Dataset

LIBRARIES = [
    "Pandas",
    "Numpy",
    "Tensorflow",
    "Scipy",
    "Sklearn",
    "Pytorch",
    "Matplotlib",
]

MAX_SELF_CORRECTION_ATTEMPTS = 5

if __name__ == "__main__":
    # Prepare datasets
    stack_overflow_dataset = Dataset(
        dataset="stack_overflow", kwargs={"download": False}
    ).dataset

    problem_dataset = Dataset(dataset="ds-1000", kwargs=None).dataset

    # Create directories
    strategy = "v4"  # improved prompt following `SELFEVOLVE` paper
    if not os.path.exists(f"generated_code/initial/{strategy}"):
        os.makedirs(f"generated_code/initial/{strategy}")
    if not os.path.exists(f"generated_code/correcting/{strategy}"):
        os.makedirs(f"generated_code/correcting/{strategy}")

    for lib in LIBRARIES:
        for i in tqdm(
            range(len(problem_dataset[lib])), desc=f"Solving problem for {lib}"
        ):
            # Skip solved problems
            if os.path.exists(
                f"generated_code/correcting/{strategy}/{lib}_{str(i).zfill(3)}.txt"
            ):
                continue

            challenge = problem_dataset[lib][i]
            problem = challenge["prompt"]
            code_context = challenge["code_context"]

            initial_generator = CodeGenerator(model="gpt35-turbo", strategy="cot")

            generated_code = initial_generator.generate(
                problem=problem,
                code_context=code_context,
                stack_overflow=stack_overflow_dataset,
                feedback=None,
            )
            dataset_dir = os.getcwd()
            with open(
                f"generated_code/initial/{strategy}/{lib}_{str(i).zfill(3)}.py", "w"
            ) as f:
                f.write(generated_code)

            is_correct = challenge.test(generated_code)

            # Handle self-correction
            attempt = 0
            while (attempt < MAX_SELF_CORRECTION_ATTEMPTS) and (is_correct != True):
                if isinstance(is_correct, tuple):
                    correcting_generator = CodeGenerator(
                        model="gpt35-turbo", strategy="zero-shot"
                    )

                    generated_code = correcting_generator.generate(
                        problem=problem,
                        code_context=code_context,
                        stack_overflow=stack_overflow_dataset,
                        feedback=is_correct,
                    )
                    os.chdir(dataset_dir)
                    with open(
                        f"generated_code/correcting/{strategy}/{lib}_{str(i).zfill(3)}.py",
                        "w",
                    ) as f:
                        f.write(generated_code)

                    is_correct = challenge.test(generated_code)

                    attempt += 1
                elif isinstance(is_correct, bool):
                    break

            os.chdir(dataset_dir)
            with open(
                f"generated_code/correcting/{strategy}/{lib}_{str(i).zfill(3)}.txt", "w"
            ) as f:
                if isinstance(is_correct, bool) and (is_correct == True):
                    f.write("Correct")
                else:
                    f.write("Incorrect")

            time.sleep(5)

    results = defaultdict(lambda: [0, 0])
    for file in glob.glob(f"generated_code/correcting/{strategy}/*.txt"):
        lib = os.path.basename(file).split("_")[0]
        is_correct = open(file, "r").read().strip() == "Correct"
        if is_correct:
            results[lib][0] += 1
            results[lib][1] += 1
        else:
            results[lib][1] += 1

    for lib, (correct, total) in results.items():
        accuracy = correct / total if total else 0
        print(f"{lib}: {accuracy * 100:.2f}%")
