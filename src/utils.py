# Code module for the utility functions


import json
import os
import platform
import shutil
import subprocess
import tarfile

import pyzstd
import requests


def syntax_check(code: str) -> dict:
    try:
        compile(code, "<string>", "exec")
        return {"status": "success"}
    except SyntaxError as e:
        return {"status": "error", "line": e.lineno, "message": e.msg}


def _subprocess_run(command: list, cwd: str = None) -> None:
    process = subprocess.Popen(
        command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, cwd=cwd
    )
    _, stderr = process.communicate()
    print(stderr.decode())


def setup_test_env():
    if platform.system() != "Linux":
        raise OSError("This script only supports Linux.")
    elif not os.path.exists(os.path.expanduser("~/.self-debug")):
        # Download Python binary
        url = "https://github.com/indygreg/python-build-standalone/releases/download/20210506/cpython-3.8.10-x86_64-unknown-linux-gnu-pgo-20210506T0943.tar.zst"
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

    # Set the Python interpreter path
    python_metadata = json.load(
        open(os.path.expanduser("~/.self-debug/python/PYTHON.json"))
    )
    python_interpreter = os.path.join(
        os.path.expanduser("~/.self-debug/python"), python_metadata["python_exe"]
    )

    # Install venv
    _subprocess_run([python_interpreter, "-m", "venv", ".venv"])

    # Install dependencies
    _subprocess_run([".venv/bin/pip", "install", "-U", "pip"])
    _subprocess_run([".venv/bin/pip", "install", "-U", "poetry"])
    _subprocess_run([".venv/bin/python", "-m", "poetry", "install", "--no-root"])
