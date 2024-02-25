from src.dataset import Dataset
from src.initial_generator import InitialGenerator

dataset = Dataset(dataset="ds-1000").dataset

challenge = dataset["Scipy"][5]
problem = challenge["prompt"]
print(problem)

initial_generator = InitialGenerator(model="gpt35-turbo", strategy="zero-shot")

generated_code = initial_generator.generate(problem=problem)
print("-" * 50)
print(generated_code)

is_correct = challenge.test(generated_code)
print(is_correct)
