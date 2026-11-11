"""Tables 7 and 8 of the revised manuscript: every paired comparison of the Results, built from
results/tables/paper_numbers.json (compute_paper_numbers.py); no new computation.

Run from the project root:  python figures_tables/make_comparison_tables.py
Writes results/tables/table7_comparisons.json, table8_comparisons.json and a .md of each.

Each cell is 'estimate [lo, hi] (n)': mean difference A − B in score points at matched budget, or,
in rows marked '(budget saving, %)', the budget saving of A relative to B at matched accuracy.
"""
import re

M = ["mistral", "llama", "qwen"]
GROUPS = [
    ("Allocation and press", [
        ("Proposed − SnapKV", "propsnap_{}_acc"),
        ("Proposed − Uniform", "uni_{}_acc"),
        ("Proposed − Uniform (budget saving, %)", "uni_{}"),
        ("Uniform − SnapKV", "press_uni_snap_{}_acc"),
    ]),
    ("Observation window", [
        ("SnapKV-32 − SnapKV", "press_snap32_snap_{}_acc"),
        ("Uniform − SnapKV-32", "press_uni_snap32_{}_acc"),
        ("Uniform-64 − SnapKV", "press_uni64_snap_{}_acc"),
        ("Proposed − SnapKV-32", "press_prop_snap32_{}_acc"),
    ]),
    ("Other baselines with a 32-token window", [
        ("Proposed − PyramidKV-32", "w32_prop_pyr_{}_acc"),
        ("Proposed − PyramidKV-32 (budget saving, %)", "w32_prop_pyr_{}"),
        ("PyramidKV-32 − PyramidKV", "w32_pyr_self_{}_acc"),
        ("Proposed − Ada-SnapKV-32", "w32_prop_ada_{}_acc"),
        ("Proposed − Ada-SnapKV-32 (budget saving, %)", "w32_prop_ada_{}"),
        ("Ada-SnapKV-32 − Ada-SnapKV", "w32_ada_self_{}_acc"),
    ]),
    ("Normalization of the layer statistic", [
        ("log T variant − Proposed", "logT_{}_acc"),
    ]),
    ("CAKE", [
        ("Proposed − CAKE", "cake_{}_acc"),
        ("Proposed − CAKE (budget saving, %)", "cake_{}"),
        ("CAKE − Uniform", "cakeuni_{}_acc"),
        ("Proposed allocation + CAKE score − CAKE", "decomp_alloc_cakescore_{}_acc"),
        ("Proposed − CAKE allocation + window score", "decomp_alloc_meanscore_{}_acc"),
        ("Proposed allocation + CAKE score − Proposed", "decomp_score_prop_{}_acc"),
        ("CAKE − CAKE allocation + window score", "decomp_score_cake_{}_acc"),
        ("CAKE allocation + window score − Uniform", "decomp_cakealloc_uni_{}_acc"),
        ("Fixed-total variant − CAKE allocation + window score", "norm_cakealloc_{}_acc"),
        ("Fixed-total variant − Uniform", "norm_uni_{}_acc"),
        ("Fixed-total variant − Proposed", "norm_prop_{}_acc"),
    ]),
    ("Combination with Ada-KV", [
        ("Proposed + Ada-KV − Ada-SnapKV", "combo_{}_acc"),
        ("Proposed + Ada-KV − Ada-SnapKV (budget saving, %)", "combo_{}"),
        ("Proposed + Ada-KV − Ada-SnapKV-32", "w32_combo_{}_acc"),
    ]),
]
PAT = re.compile(r"^([+\-]?\d+(?:\.\d+)?)(%| points) \(95% CI \[([+\-]?\d+(?:\.\d+)?)%?, ([+\-]?\d+(?:\.\d+)?)%?\]; (\d+) matched points\)$")


def sign(v):
    return v if v.startswith(("-", "+")) else "+" + v


def cell(NUM, key):
    if key not in NUM: return "—"
    m = PAT.match(str(NUM[key]))
    assert m, (key, NUM[key])
    est, unit, lo, hi, n = m.groups()
    if unit == "%":   # percentages carry no sign in numbers.json; sign them as in Table 6
        est, lo, hi = sign(est), sign(lo), sign(hi)
    return f"{est} [{lo}, {hi}] ({n})"


def build(NUM, part):
    """part 1: Table 7 (first three groups); part 2: Table 8 (the rest)."""
    rows, bold = [], []
    for g, items in (GROUPS[:3] if part == 1 else GROUPS[3:]):
        bold.append(len(rows)); rows.append([g, "", "", ""])
        for label, key in items:
            rows.append([label] + [cell(NUM, key.format(m)) for m in M])
    return {"headers": ["Comparison (A − B)", "Mistral-7B", "Llama-3.1-8B", "Qwen3-8B"], "rows": rows, "bold": bold}


if __name__ == "__main__":
    import json, os, sys
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    src = sys.argv[1] if len(sys.argv) > 1 else os.path.join(root, "results", "tables", "paper_numbers.json")
    NUM = json.load(open(src))
    for part, name in ((1, "table7_comparisons"), (2, "table8_comparisons")):
        t = build(NUM, part)
        out = os.path.join(root, "results", "tables", name)
        json.dump(t, open(out + ".json", "w"), indent=1, ensure_ascii=False)
        md = ["| " + " | ".join(t["headers"]) + " |", "|" + "---|" * len(t["headers"])]
        md += ["| " + " | ".join(f"**{c}**" if i in t["bold"] and c else c for c in r) + " |" for i, r in enumerate(t["rows"])]
        open(out + ".md", "w").write("\n".join(md) + "\n")
        print("saved", out + ".json")
