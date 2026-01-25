#!/usr/bin/env python3
"""
Master script to generate all figures and tables for the LaTeX report.

Usage:
    python scripts/latex_analysis/generate_all.py

Or run individual scripts:
    python scripts/latex_analysis/fig_data_distribution.py
    python scripts/latex_analysis/fig_rating_citation_analysis.py
    ...
"""

import subprocess
import sys
from pathlib import Path

# Define all scripts in order
SCRIPTS = [
    # Phase 1: Data Analysis (Section 3)
    ("fig_data_distribution.py", "Data distribution figures"),
    ("fig_rating_citation_analysis.py", "Rating analysis (normalized metrics only)"),
    ("fig_arxiv_analysis.py", "ArXiv category analysis"),
    ("fig_structural_features.py", "Structural feature extraction"),

    # Phase 2: Baselines (Section 4)
    ("fig_tfidf_analysis.py", "TF-IDF baseline analysis"),
    ("fig_gemini_comparison.py", "Gemini vs fine-tuned comparison"),

    # Phase 3: Experiments (Section 5)
    ("fig_hyperparam_analysis.py", "Hyperparameter sweep analysis"),
    ("fig_model_comparison.py", "Model comparison figures"),
    ("fig_temporal_analysis.py", "Temporal analysis"),

    # Phase 4: Deep Analysis (Section 6)
    ("fig_modality_disagreement.py", "Modality disagreement analysis with structural features"),
    ("fig_category_performance.py", "Accuracy by ArXiv category"),
    ("fig_performance_stratification.py", "Performance stratification with CIs"),
    ("fig_2025_analysis.py", "2025 difficulty analysis"),
    ("fig_top_model_decomposition.py", "Best model decomposition"),
    ("fig_mediation_analysis.py", "Mediation analysis: Features -> Quality -> Predictions"),

    # Phase 5: Tables
    ("generate_tables.py", "LaTeX tables"),
]


def run_script(script_name: str, description: str) -> bool:
    """Run a single script and return success status."""
    script_path = Path(__file__).parent / script_name

    if not script_path.exists():
        print(f"  Warning: {script_name} not found")
        return False

    print(f"\n{'='*60}")
    print(f"Running: {script_name}")
    print(f"Description: {description}")
    print(f"{'='*60}")

    try:
        result = subprocess.run(
            [sys.executable, str(script_path)],
            capture_output=False,
            check=True,
        )
        return True
    except subprocess.CalledProcessError as e:
        print(f"  Error running {script_name}: {e}")
        return False
    except Exception as e:
        print(f"  Exception running {script_name}: {e}")
        return False


def main():
    print("="*60)
    print("LaTeX Report Figure and Table Generation")
    print("="*60)
    print(f"\nTotal scripts to run: {len(SCRIPTS)}")

    # Create output directories
    output_dirs = [
        Path("figures/latex/data"),
        Path("figures/latex/baseline"),
        Path("figures/latex/models"),
        Path("figures/latex/ablations"),
        Path("figures/latex/analysis"),
        Path("figures/latex/tables"),
        Path("latex/tables"),
    ]

    for d in output_dirs:
        d.mkdir(parents=True, exist_ok=True)
        print(f"Created: {d}")

    # Run all scripts
    successful = 0
    failed = 0
    failed_scripts = []

    for script_name, description in SCRIPTS:
        if run_script(script_name, description):
            successful += 1
        else:
            failed += 1
            failed_scripts.append(script_name)

    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"Successful: {successful}/{len(SCRIPTS)}")
    print(f"Failed: {failed}/{len(SCRIPTS)}")

    if failed_scripts:
        print(f"\nFailed scripts:")
        for s in failed_scripts:
            print(f"  - {s}")

    # List generated files
    print("\nGenerated figures:")
    for d in output_dirs:
        if d.exists():
            pdfs = list(d.glob("*.pdf"))
            if pdfs:
                print(f"\n  {d}/ ({len(pdfs)} files)")
                for pdf in sorted(pdfs)[:5]:
                    print(f"    - {pdf.name}")
                if len(pdfs) > 5:
                    print(f"    ... and {len(pdfs) - 5} more")

    print("\n" + "="*60)
    print("Generation complete!")
    print("="*60)

    # Compilation instructions
    print("""
Next steps to compile the LaTeX report:

1. cd latex/
2. pdflatex report.tex
3. biber report
4. pdflatex report.tex
5. pdflatex report.tex

Or use latexmk:
    cd latex/ && latexmk -pdf report.tex
""")


if __name__ == "__main__":
    main()
