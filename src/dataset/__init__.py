import os

import gdown

from src.dataset.ds1000 import DS1000Dataset

# Change to current directory
os.chdir(os.path.dirname(os.path.abspath(__file__)))


class Dataset:
    def __init__(self, dataset: str) -> None:
        if dataset == "ds-1000":
            if not os.path.exists("ds1000_data"):
                # Download dataset
                gdown.download(
                    id="1sR0Bl4pVHCe9UltBVyhloE8Reztn72VD",
                    output="ds-1000.zip",
                    quiet=False,
                )
                os.system("unzip ds-1000.zip")
                os.system("rm ds-1000.zip")
            self.dataset = DS1000Dataset(source_dir="ds1000_data", mode="Completion")
        else:
            raise ValueError("Invalid dataset name")
