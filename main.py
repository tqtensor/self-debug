from dotenv import load_dotenv

load_dotenv(".env")


import glob
import os
import time
from collections import defaultdict

from tqdm.auto import tqdm

from src.dataset import Dataset
from src.initial_generator import InitialGenerator

LIBRARIES = [
    "Pandas",
    "Numpy",
    "Tensorflow",
    "Scipy",
    "Sklearn",
    "Pytorch",
    "Matplotlib",
]

if __name__ == "__main__":
    # Prepare datasets
    stack_overflow_dataset = Dataset(
        dataset="stack_overflow", kwargs={"download": False}
    ).dataset
    print(stack_overflow_dataset.retrieve(tags=["numpy"], k=3))

    problem_dataset = Dataset(dataset="ds-1000", kwargs=None).dataset

    # Create directories
    if not os.path.exists("generated_code/initial/zero-shot"):
        os.makedirs("generated_code/initial/zero-shot")

    # Run initial generator
    for lib in LIBRARIES:
        for i in tqdm(
            range(len(problem_dataset[lib])), desc=f"Solving problem for {lib}"
        ):
            challenge = problem_dataset[lib][i]
            problem = challenge["prompt"]

            initial_generator = InitialGenerator(
                model="gpt35-turbo", strategy="zero-shot"
            )

            generated_code = initial_generator.generate(problem=problem)
            with open(
                f"generated_code/initial/zero-shot/{lib}_{str(i).zfill(3)}.py", "w"
            ) as f:
                f.write(generated_code)

            is_correct = challenge.test(generated_code)
            with open(
                f"generated_code/initial/zero-shot/{lib}_{str(i).zfill(3)}.txt", "w"
            ) as f:
                if is_correct:
                    f.write("Correct")
                else:
                    f.write("Incorrect")

            time.sleep(5)

    results = defaultdict(lambda: [0, 0])
    for file in glob.glob("generated_code/initial/zero-shot/*.txt"):
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
