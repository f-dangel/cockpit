"""Utility functions for running tests."""

import os
import subprocess

REPO_ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def run_command(cmd, filter_download_progress=True):
    """Run a command line command and print stdout. Print stderr if command fails.

    Remove print statements from download progress (less than 7 chars, contains "%").
    """
    result = subprocess.run(cmd, capture_output=True)
    stdout = result.stdout.decode("utf-8").splitlines()

    def is_download(line):
        """Return whether a string is from a download in progress."""
        return "%" in line and len(line) < 7

    if filter_download_progress:
        stdout = [line for line in stdout if not is_download(line)]

    stdout = "\n".join(stdout)
    print(stdout)

    if result.returncode != 0:
        stderr = result.stderr.decode("utf-8").splitlines()

        if filter_download_progress:
            stderr = [line for line in stderr if not is_download(line)]

        stderr = "\n".join(stderr)
        print(stderr)

        raise RuntimeError(f"Command {cmd} crashed")
