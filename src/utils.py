# Code module for the utility functions


import json
import os
import platform
import shutil
import tarfile

import pyzstd
import requests


def syntax_check(code: str) -> dict:
    try:
        compile(code, "<string>", "exec")
        return {"status": "success"}
    except SyntaxError as e:
        return {"status": "error", "line": e.lineno, "message": e.msg}


def setup_test_env():
    if platform.system() != "Linux":
        raise OSError("This script only supports Linux.")
    else:
        # Download Python binary
        url = "https://github.com/indygreg/python-build-standalone/releases/download/20200418/cpython-3.7.7-x86_64-unknown-linux-gnu-pgo-20200418T2226.tar.zst"
        response = requests.get(url, stream=True)
        response.raise_for_status()

        destination = "python-build-standalone.tar.zst"
        with open(destination, "wb") as file:
            for chunk in response.iter_content(chunk_size=8192):
                file.write(chunk)

        # Decompress .zst file
        decompressed_file = "python-build-standalone.tar"
        with open(destination, "rb") as f_in, open(decompressed_file, "wb") as f_out:
            dctx = pyzstd.ZstdDecompressor()
            f_out.write(dctx.decompress(f_in.read()))

        # Open decompressed .tar file
        if os.path.exists(os.path.expanduser("~/.self-debug")):
            shutil.rmtree(os.path.expanduser("~/.self-debug"))
        with tarfile.open(decompressed_file, "r") as tar:
            tar.extractall(os.path.expanduser("~/.self-debug"))

        # Clean up
        os.remove(destination)
        os.remove(decompressed_file)

        # Set the environment variable
        python_metadata = json.load(
            open(os.path.expanduser("~/.self-debug/python/PYTHON.json"))
        )
        os.environ["SELFDEBUG_PYTHON"] = os.path.join(
            os.path.expanduser("~/.self-debug/python"),
            python_metadata["python_exe"],
        )


if __name__ == "__main__":
    setup_test_env()
    print("Setup complete.")
