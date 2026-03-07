#!/usr/bin/env python3
"""Light Curve Diagnosis Agent - Entry Point

Example usage:
    python run_agent.py ./data/example_lc.csv
    python run_agent.py ./data/example_lc.csv --output report.md
    python run_agent.py ./data/example_lc.csv --time-col time --mag-col mag
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Add src to path for development
sys.path.insert(0, str(Path(__file__).parent / "src"))

from lightcurve_agent import create_diagnosis_agent
from lightcurve_agent.config import get_settings


def main():
    parser = argparse.ArgumentParser(
        description="Diagnose astronomical light curves using AI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s ./data/example_lc.csv
  %(prog)s ./data/example_lc.csv -o my_report.md
  %(prog)s ./data/example_lc.csv --time-col jd --mag-col flux --scale flux
        """,
    )

    parser.add_argument(
        "csv_path",
        type=str,
        help="Path to the light curve CSV file",
    )

    parser.add_argument(
        "-o", "--output",
        type=str,
        default="diagnosis_report.md",
        help="Output markdown file path (default: diagnosis_report.md)",
    )

    parser.add_argument(
        "--time-col",
        type=str,
        default="time",
        help="Name of the time column (default: time)",
    )

    parser.add_argument(
        "--mag-col",
        type=str,
        default="mag",
        help="Name of the magnitude/flux column (default: mag)",
    )

    parser.add_argument(
        "--err-col",
        type=str,
        default="mag_err",
        help="Name of the error column (default: mag_err)",
    )

    parser.add_argument(
        "--scale",
        type=str,
        choices=["mag", "flux"],
        default="mag",
        help="Data scale: magnitude (inverted y-axis) or flux (default: mag)",
    )

    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Model name to use (overrides env var)",
    )

    parser.add_argument(
        "--temperature",
        type=float,
        default=0.6,
        help="Model temperature (default: 0.6)",
    )

    args = parser.parse_args()

    # Validate input file exists
    csv_path = Path(args.csv_path)
    if not csv_path.exists():
        print(f"Error: File not found: {csv_path}", file=sys.stderr)
        sys.exit(1)

    # Load settings to ensure directories exist
    settings = get_settings()
    print(f"Artifacts directory: {settings.artifacts_dir}")
    print(f"Using model: {args.model or settings.openai_model}")
    print(f"Analyzing: {csv_path}")
    print("-" * 50)

    # Create agent
    agent = create_diagnosis_agent(
        model_name=args.model,
        temperature=args.temperature,
    )

    # Build user message
    user_msg = (
        f"Diagnose the light curve in {args.csv_path}. "
        f"Columns: {args.time_col}, {args.mag_col}"
        + (f", {args.err_col}" if args.err_col else "")
        + f". Assume {args.scale} scale."
    )

    print(f"User query: {user_msg}")
    print("-" * 50)

    # Run agent
    try:
        result = agent.invoke(
            {"messages": [{"role": "user", "content": user_msg}]}
        )

        # Extract final response
        final_response = result["messages"][-1].content

        # Save report
        output_path = Path(args.output)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write("# Light Curve Diagnosis Report\n\n")
            f.write(final_response)

        print("-" * 50)
        print(f"Report saved to: {output_path}")

        # Also print a preview
        preview = final_response[:500] + "..." if len(final_response) > 500 else final_response
        print("\nPreview:")
        print(preview)

    except Exception as e:
        print(f"Error during analysis: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
