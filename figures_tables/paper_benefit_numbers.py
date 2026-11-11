"""
Manuscript numbers on the distribution of the discarded attention mass, read from
results/tables/allocation_benefit.json (experiments/revision/analyze_allocation_benefit.py).
Used by compute_paper_numbers.py; can also be run alone to update those keys:
    python figures_tables/paper_benefit_numbers.py [results/tables/paper_numbers.json]
"""
import json, os, sys
REL = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
M = ["Mistral-7B-Instruct-v0.3", "Llama-3.1-8B-Instruct", "Qwen3-8B"]


def three(vals):
    return f"{vals[0]}, {vals[1]}, and {vals[2]}"


def benefit_keys():
    d = json.load(open(os.path.join(REL, "results", "tables", "allocation_benefit.json")))
    e = [d[m]["by_epsilon"]["0.5"] for m in M]; n = [d[m]["normalization"] for m in M]; s = {}
    pc = lambda x: f"{x['est']:.1f}% (95% CI [{x['ci'][0]:.1f}%, {x['ci'][1]:.1f}%])"
    s["ab_worst_values"] = three([f"{x['worst_uniform']['est']:.3f} to {x['worst_rule']['est']:.3f}" for x in e])
    s["ab_worst_pct"] = three([pc(x["worst_reduction_pct"]) for x in e])
    s["ab_mean_pct"] = three([f"{x['mean_reduction_pct']['est']:.1f}%" for x in e])
    s["ab_saving_pct"] = three([pc(x["equal_delta_saving_pct"]) for x in e])
    s["ab_captured"] = three([f"{x['captured_pct']['est']:.0f}%" for x in e])
    s["ab_fixed"] = three([f"{x['delta_fixed']['est']:.4f}" for x in e])
    s["ab_cv"] = three([f"{x['cv_uniform']['est']:.2f} to {x['cv_rule']['est']:.2f}" for x in e])
    w = [d[m]["by_epsilon"][k]["worst_reduction_pct"]["est"] for m in M for k in d[m]["by_epsilon"]]
    s["ab_worst_all_eps"] = f"{min(w):.0f}% to {max(w):.0f}%"
    s["ab_spearman"] = three([f"{x['sample_spearman_phi_H_mean']:.2f}" for x in n])
    s["ab_low_third"] = three([f"{x['low_H_third_in_top_phi_third_pct']:.1f}%" for x in n])
    corr = lambda k: three([f"{x['within_task_corr_logT_' + k]['r']:+.2f}" for x in n])
    s["ab_ctx_phi"], s["ab_ctx_hlogt"], s["ab_ctx_h"] = corr("phi"), corr("H_logT"), corr("H")
    s["ab_logt_delta_pct"] = three([f"{x['logt_vs_rule_delta_pct']['est']:.1f}%" for x in e])
    p = lambda k: d["scores"][k]["pooled"]
    pp = lambda k: f"{p(k)['diff']:+.2f} points (95% CI [{p(k)['ci'][0]:+.2f}, {p(k)['ci'][1]:+.2f}])"
    s["ab_logt_pooled"] = pp("Proposed-logT|Proposed"); s["ab_uni_pooled"] = pp("Proposed|Uniform")
    s["ab_pyr32_pooled"] = pp("Proposed|PyramidKV-w32"); s["ab_uni_pyr32_pooled"] = pp("Uniform|PyramidKV-w32")
    b = [d["budget_variability"][m] for m in M]
    s["ab_bud_sd_phi"] = three([f"{x['proposed_sd']:.3f}" for x in b])
    s["ab_bud_sd_logt"] = three([f"{x['logt_sd']:.3f}" for x in b])
    s["ab_bud_ratio"] = three([f"{x['sd_ratio']:.1f} ([{x['sd_ratio_ci'][0]:.1f}, {x['sd_ratio_ci'][1]:.1f}])" for x in b])
    wr = [x["worst_reduction_pct"]["est"] for x in e]; mr = [x["mean_reduction_pct"]["est"] for x in e]
    s["ab_worst_short"] = f"{min(wr):.0f}% to {max(wr):.0f}%"; s["ab_mean_short"] = f"{min(mr):.0f}% to {max(mr):.0f}%"
    rr = [x["sd_ratio"] for x in b]; lo = min(x["sd_ratio_ci"][0] for x in b)
    s["ab_bud_ratio_short"] = f"{min(rr):.1f} to {max(rr):.1f} (every 95% CI above {lo - 0.05:.1f})"
    return s


if __name__ == "__main__":
    out = sys.argv[1] if len(sys.argv) > 1 else os.path.join(REL, "results", "tables", "paper_numbers.json")
    num = json.load(open(out)); k = benefit_keys(); num.update(k)
    json.dump(num, open(out, "w"), indent=2, ensure_ascii=False)
    for a, v in sorted(k.items()): print(f"{a}: {v}")
