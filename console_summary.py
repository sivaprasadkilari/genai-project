"""
Utility to print the Result Summary accuracy table from README to console.
No existing code is modified; this is an additive helper.
"""

from pathlib import Path

README_PATH = Path(__file__).parent / "README.md"


def print_result_summary() -> None:
    """
    Extract and print the Result Summary table section (lines 104-111).
    Keeps behavior deterministic by simple line slicing.
    """
    with README_PATH.open("r", encoding="utf-8") as f:
        lines = f.readlines()

    start, end = 103, 111  # zero-based indices to capture lines 104-111 inclusive
    snippet = lines[start : end + 1]
    print("".join(snippet).rstrip())


if __name__ == "__main__":
    print_result_summary()

