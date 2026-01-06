from pathlib import Path
import argparse

from baseline.baseline_retrieval import run_baseline
from improved.improved_retrieval import run_improved
from evaluation.evaluator import evaluate_and_report
from generation import write_generation_report
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
    evaluation_dir = ROOT_DIR / "evaluation"

    baseline_payload = run_baseline(args.query, baseline_out)
    improved_payload = run_improved(args.query, improved_out)

    evaluate_and_report(args.query, baseline_payload, improved_payload, evaluation_dir)
    write_generation_report(
        baseline_payload,
        improved_payload,
        evaluation_dir / "generation_report.md",
    )

    print(f"Baseline results written to {baseline_out}")
    print(f"Improved results written to {improved_out}")
    print(f"Evaluation artifacts written to {evaluation_dir}")


if __name__ == "__main__":
    main()
