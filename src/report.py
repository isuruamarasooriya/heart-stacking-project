"""Turn numbers into readable Markdown, and keep the README results section in sync."""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

RESULTS_START = "<!-- RESULTS_START -->"
RESULTS_END = "<!-- RESULTS_END -->"


def df_to_md(df: pd.DataFrame, digits: int = 3) -> str:
    """Minimal DataFrame -> Markdown table (avoids needing the 'tabulate' package)."""
    def fmt(v):
        if isinstance(v, (float, np.floating)):
            return "" if np.isnan(v) else f"{v:.{digits}f}"
        return "" if v is None else str(v)

    cols = list(df.columns)
    lines = ["| " + " | ".join(map(str, cols)) + " |",
             "| " + " | ".join("---" for _ in cols) + " |"]
    for row in df.astype(object).itertuples(index=False):   # astype(object) keeps ints as ints
        lines.append("| " + " | ".join(fmt(v) for v in row) + " |")
    return "\n".join(lines)


def cv_table_md(cv_summary: pd.DataFrame, metrics: list[str]) -> str:
    """'mean ± std' per model and metric."""
    out = pd.DataFrame({"Model": cv_summary.index})
    for m in metrics:
        out[m] = [f"{cv_summary.loc[name, (m, 'mean')]:.3f} ± {cv_summary.loc[name, (m, 'std')]:.3f}"
                  for name in cv_summary.index]
    return df_to_md(out)


def verdict_text(comparison: dict, boot: dict, primary: str, best_name: str, stack_name: str) -> str:
    """Plain-language verdict that never claims more than the evidence supports."""
    d, p = comparison["mean_diff"], comparison["p_value"]
    if d <= 0:
        cv_part = (f"In cross-validation, {stack_name} did NOT beat {best_name} on {primary} "
                   f"(mean difference {d:+.4f}).")
    elif not np.isnan(p) and p < 0.05:
        cv_part = (f"In cross-validation, {stack_name} beat {best_name} on {primary} "
                   f"(mean difference {d:+.4f}, corrected p = {p:.3f}) - a statistically reliable gain.")
    else:
        cv_part = (f"In cross-validation, {stack_name} was numerically ahead of {best_name} on {primary} "
                   f"(mean difference {d:+.4f}) but the gain is NOT statistically reliable "
                   f"(corrected p = {p:.3f}).")
    if boot["ci_low"] <= 0 <= boot["ci_high"]:
        test_part = (f"On the held-out test set the difference was {boot['observed_diff']:+.4f} with a 95% bootstrap "
                     f"interval of [{boot['ci_low']:+.4f}, {boot['ci_high']:+.4f}], which includes zero, so the "
                     f"test set cannot separate the two models.")
    else:
        test_part = (f"On the held-out test set the difference was {boot['observed_diff']:+.4f} with a 95% bootstrap "
                     f"interval of [{boot['ci_low']:+.4f}, {boot['ci_high']:+.4f}], which excludes zero.")
    return cv_part + " " + test_part


INTERP_START = "<!-- INTERPRETATION_START -->"
INTERP_END = "<!-- INTERPRETATION_END -->"


def update_readme_results(readme_path: Path, results_md: str,
                          start: str = RESULTS_START, end: str = RESULTS_END) -> bool:
    """Replace the text between two markers in README.md. Returns True if updated."""
    if not readme_path.exists():
        return False
    text = readme_path.read_text(encoding="utf-8")
    pattern = re.compile(re.escape(start) + r".*?" + re.escape(end), re.DOTALL)
    if not pattern.search(text):
        return False
    new_block = f"{start}\n{results_md}\n{end}"
    readme_path.write_text(pattern.sub(lambda _m: new_block, text), encoding="utf-8")
    return True
