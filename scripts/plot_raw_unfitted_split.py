"""
Regenerate the 8-feature raw unfitted distributions as two 4×3 plots
instead of one large 8×3. Overwrites the originals in data/plots/dist_8feat/.
"""
import os
import re
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PLOT_DIR = "data/plots/dist_8feat"
ERA_ORDER = ["Pre-1800", "1800-1900", "Post-1900"]
ERA_LABELS = ["Pre-1800", "1800–1900", "Post-1900"]
ERA_COLORS = {"Pre-1800": "#4e79a7", "1800-1900": "#f28e2b", "Post-1900": "#59a14f"}

# (feature column, display label, scale for display, unit string)
CONTINUOUS = [
    ("cap_rate", "Line capitalisation rate", 100, "%"),
    ("punct_rate", "Line-end punctuation rate", 100, "%"),
    ("colon_density", "Colon / semicolon density", 1, "/100 words"),
    ("concrete_abstract_ratio", "Concrete-to-abstract ratio", 1, ""),
    ("adv_verb_ratio", "Adverb-to-verb ratio", 1, ""),
    ("rhyme_rate", "Rhyme rate", 100, "%"),
    ("archaic_density", "Archaic word density", 1, "/100 words"),
    ("imageability", "Imageability (Glasgow Norms)", 1, "/ 7"),
]


def main():
    os.makedirs(PLOT_DIR, exist_ok=True)

    df = pd.read_csv("data/poem_features_hurdle_7feat_3class.csv")
    df["era"] = df["era_3class"]
    df = df[df["era"].isin(ERA_ORDER)].copy()

    glasgow = pd.read_csv("data/glasgow_norms.csv").set_index("word")
    img_lex = glasgow["imageability"].to_dict()
    poem_df = pd.read_csv("data/PoetryFoundationData_with_year.csv")[["Title", "Poet", "Poem"]]

    def img_score(text):
        ws = re.findall(r"[a-z']+", str(text).lower())
        vs = [img_lex[w] for w in ws if w in img_lex]
        return float(np.mean(vs)) if len(vs) >= 5 else float("nan")

    poem_df["imageability"] = poem_df["Poem"].apply(img_score)
    img_lookup = poem_df.set_index(["Title", "Poet"])["imageability"].to_dict()
    df["imageability"] = df.apply(lambda r: img_lookup.get((r["Title"], r["Poet"]), float("nan")), axis=1)
    for era in ERA_ORDER:
        m = df.loc[df["era"] == era, "imageability"].mean()
        df.loc[(df["era"] == era) & df["imageability"].isna(), "imageability"] = m

    def save_raw_grid(features_subset, title_suffix, filename):
        n_feat = len(features_subset)
        fig, axes = plt.subplots(n_feat, 3, figsize=(12, 3.5 * n_feat))
        fig.suptitle(f"Feature distributions by era (raw, unfitted) — {title_suffix}", fontsize=13, fontweight="bold", y=1.01)
        if n_feat == 1:
            axes = axes.reshape(1, -1)
        for row, (feat, label, scale, unit) in enumerate(features_subset):
            for col, (era, elabel) in enumerate(zip(ERA_ORDER, ERA_LABELS)):
                ax = axes[row, col]
                vals = df.loc[df["era"] == era, feat].dropna().values * scale
                if len(vals) == 0:
                    ax.set_visible(False)
                    continue
                color = ERA_COLORS[era]
                ax.hist(vals, bins=35, density=True, color=color, alpha=0.7, edgecolor="white", linewidth=0.4)
                ax.axvline(vals.mean(), color="black", lw=1.5, linestyle="--", alpha=0.8)
                ax.set_title(f"{elabel}  (n={len(vals):,})", fontsize=9)
                if col == 0:
                    ax.set_ylabel(f"{label}\n[{unit}]" if unit else label, fontsize=8)
                ax.tick_params(labelsize=7)
                ax.spines[["top", "right"]].set_visible(False)
        plt.tight_layout()
        path = os.path.join(PLOT_DIR, filename)
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Saved: {path}")

    save_raw_grid(CONTINUOUS[:4], "features 1–4", "all_features_raw_unfitted_part1.png")
    save_raw_grid(CONTINUOUS[4:8], "features 5–8", "all_features_raw_unfitted_part2.png")

    old_path = os.path.join(PLOT_DIR, "all_features_raw_unfitted.png")
    if os.path.exists(old_path):
        os.remove(old_path)
        print(f"Removed: {old_path}")
    print("Done.")


if __name__ == "__main__":
    main()
