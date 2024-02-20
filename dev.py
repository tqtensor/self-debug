from src.dataset import Dataset
from src.initial_generator import InitialGenerator

dataset = Dataset(dataset="ds-1000").dataset

problem = dataset["Pandas"][5]["prompt"]
print(problem)

initial_generator = InitialGenerator(model="gpt35-turbo", strategy="zero-shot")

generated_code = initial_generator.generate(problem=problem)
print("-" * 50)
print(generated_code)

is_correct = dataset["Pandas"][5].test(generated_code)
print(is_correct)
