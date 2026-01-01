"""
Run both baseline and improved retrieval pipelines.
Saves results into baseline/ and improved/ directories.
"""
import json
from config import BASELINE_DIR, IMPROVED_DIR
from baseline.baseline_retrieval import run_baseline
from improved.improved_retrieval import run_improved


def main():
    baseline_results = run_baseline()
    with open(BASELINE_DIR / "baseline_results.json", "w", encoding="utf-8") as f:
        json.dump(baseline_results, f, indent=2, ensure_ascii=False)

    improved_results = run_improved()
    with open(IMPROVED_DIR / "improved_results.json", "w", encoding="utf-8") as f:
        json.dump(improved_results, f, indent=2, ensure_ascii=False)

    print("Done. Results written to baseline/ and improved/")


if __name__ == "__main__":
    main()
