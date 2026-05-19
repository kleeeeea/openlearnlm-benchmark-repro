#!/usr/bin/env python3
"""
OpenLearnLM Benchmark Evaluation Runner

Usage:
    # Full evaluation with test dataset (6,122 items)
    python run_evaluation.py

    # Pilot test with limited items
    python run_evaluation.py --pilot --limit 100

    # Resume from previous progress
    python run_evaluation.py --resume

    # Generate report only (from existing results)
    python run_evaluation.py --report-only

    # Evaluate specific models only
    python run_evaluation.py --models gpt-5.2 deepseek/deepseek-v3.2
"""

import argparse
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from evaluation.config import EvalConfig
from evaluation.engine.orchestrator import EvaluationOrchestrator
from evaluation.results.report_generator import ReportGenerator


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="OpenLearnLM Benchmark Evaluation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    parser.add_argument(
        "--pilot",
        action="store_true",
        help="Run pilot test with limited items"
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of items to evaluate"
    )

    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from previous progress (default: True)"
    )

    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Start fresh, ignore previous progress"
    )

    parser.add_argument(
        "--report-only",
        action="store_true",
        help="Generate report from existing results without running evaluation"
    )

    parser.add_argument(
        "--models",
        nargs="+",
        help="Specific models to evaluate (space-separated)"
    )

    parser.add_argument(
        "--workers",
        type=int,
        default=5,
        help="Workers per model (default: 5)"
    )

    parser.add_argument(
        "--category",
        type=str,
        default="01_기능_skills",
        choices=[
            "01_기능_skills",
            "02_교과지식_content",
            "03_교육학지식_pedagogical",
            "04_태도_attitude"
        ],
        help="Benchmark category to evaluate (default: 01_기능_skills)"
    )

    return parser.parse_args()


def main():
    """Main entry point"""
    args = parse_args()

    # Initialize configuration
    config = EvalConfig()

    # Override config based on arguments
    if args.workers:
        config.WORKERS_PER_MODEL = args.workers

    if args.category:
        config.BENCHMARK_CATEGORY = args.category
        # Ensure output directory exists for the selected category
        config.RESPONSES_DIR.mkdir(parents=True, exist_ok=True)

    if args.models:
        config.MODELS = [m for m in args.models if m]
        if not config.MODELS:
            print("Error: No valid models specified")
            return 1

    # Pilot mode
    if args.pilot:
        limit = args.limit or 100
        print(f"Running pilot test with {limit} items...")
    else:
        limit = args.limit

    # Resume mode
    resume = not args.no_resume

    # Report only mode
    if args.report_only:
        print("Generating report from existing results...")
        generator = ReportGenerator(config.RESPONSES_DIR, config.REPORTS_DIR)
        result = generator.generate_report()

        if "error" in result:
            print(f"Error: {result['error']}")
            return 1

        print(f"Reports generated:")
        print(f"  JSON: {result['json_report']}")
        print(f"  Markdown: {result['markdown_report']}")
        return 0

    # Check API keys
    if not config.OPENAI_API_KEY and any("gpt" in m for m in config.MODELS):
        print("Error: OPENAI_API_KEY not set but GPT model is requested")
        return 1

    if not config.OPENROUTER_API_KEY and any("/" in m for m in config.MODELS):
        print("Error: OPENROUTER_API_KEY not set but OpenRouter models are requested")
        return 1

    # Check test data file
    if not config.TEST_DATA_FILE.exists():
        print(f"Error: Test data file not found: {config.TEST_DATA_FILE}")
        return 1

    # Initialize orchestrator
    print("=" * 60)
    print("OpenLearnLM Benchmark Evaluation")
    print("=" * 60)
    print(f"Category: {config.BENCHMARK_CATEGORY}")
    print(f"Test data: {config.TEST_DATA_FILE}")
    print(f"Output: {config.RESPONSES_DIR}")
    print(f"Models: {', '.join(config.MODELS)}")
    print(f"Workers per model: {config.WORKERS_PER_MODEL}")
    print(f"Resume mode: {resume}")
    print("=" * 60)

    orchestrator = EvaluationOrchestrator(config)

    # Load test data
    items = orchestrator.load_test_data(limit=limit)

    if not items:
        print("Error: No test items loaded")
        return 1

    # Run evaluation
    try:
        summary = orchestrator.run(items, resume=resume)

        # Generate report
        print("\nGenerating final report...")
        generator = ReportGenerator(config.RESPONSES_DIR, config.REPORTS_DIR)
        result = generator.generate_report()

        if "markdown_report" in result:
            print(f"\nReport saved to: {result['markdown_report']}")

        print("\nEvaluation completed successfully!")
        return 0

    except KeyboardInterrupt:
        print("\n\nEvaluation interrupted by user")
        print("Progress has been saved. Run with --resume to continue.")
        return 130

    except Exception as e:
        print(f"\nError during evaluation: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
