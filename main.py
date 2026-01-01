from pathlib import Path
import argparse

from baseline.baseline_retrieval import run_baseline
from improved.improved_retrieval import run_improved
from config import ROOT_DIR


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--query",
        type=str,
        default="How does retrieval-augmented generation reduce hallucinations in LLMs?",
    )
    args = parser.parse_args()

    baseline_out = ROOT_DIR / "baseline" / "baseline_results.json"
    improved_out = ROOT_DIR / "improved" / "improved_results.json"

    run_baseline(args.query, baseline_out)
    run_improved(args.query, improved_out)

    print(f"Baseline results written to {baseline_out}")
    print(f"Improved results written to {improved_out}")


if __name__ == "__main__":
    main()
