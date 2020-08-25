"""Utility functions for running tests."""

import subprocess


def run_command(cmd):
    """Run a command line command and print stdout. Print stderr if command fails."""
    result = subprocess.run(cmd, capture_output=True)

    print(result.stdout.decode("utf-8"))

    if result.returncode != 0:
        print(result.stderr.decode("utf-8"))
        raise RuntimeError(f"Command {cmd} crashed")
