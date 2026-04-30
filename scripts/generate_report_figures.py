"""Generate all publication-quality figures for the YODEP experiment report.

Phase 1 (runs immediately from existing CSVs):
  fig01  — F1 heatmap: all classifiers × all conditions (ALL language)
  fig02  — AUC heatmap: all classifiers × all conditions (ALL language)
  fig03  — Bar chart with error bars: top classifiers per condition (ALL)
  fig04  — EN vs YO F1 comparison across conditions
  fig05  — C1 vs C2 (F0 removal effect) per classifier
  fig06  — Handcrafted (C1) vs SSL (C4) per classifier
  fig07  — Transfer experiment: delta bar chart
  fig08  — Classifier ranking (bump chart) across conditions
  fig09  — F1 variance (std) heatmap — stability analysis
  fig10  — Precision vs Recall scatter per condition

Phase 2 (requires results/predictions/*.npz from a full re-run):
  fig11  — ROC curves: one subplot per condition, top-8 classifiers
  fig12  — Confusion matrices: best classifier per condition

Phase 3 (requires results/training_curves/*.json and fold_metrics_detail.csv):
  fig13  — Per-fold F1 violin/box plots across LOSO speakers
  fig14  — MLP training loss + validation score curves
  fig15  — GBM / Hist-GBM staged training scores

Usage::

    python scripts/generate_report_figures.py
    python scripts/generate_report_figures.py --phase 1   # CSV-only figures
    python scripts/generate_report_figures.py --phase 2   # prediction figures
    python scripts/generate_report_figures.py --phase 3   # training-curve figures
"""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.lines import Line2D

warnings.filterwarnings("ignore")

# ── paths ──────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).parent.parent
TABLES_DIR   = PROJECT_ROOT / "results" / "tables"
PRED_DIR     = PROJECT_ROOT / "results" / "predictions"
CURVES_DIR   = PROJECT_ROOT / "results" / "training_curves"
FIG_DIR      = PROJECT_ROOT / "results" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

# ── style ───────────────────────────────────────────────────────────────────
PALETTE      = "Blues_r"
CLF_PALETTE  = sns.color_palette("tab20", 16)
COND_COLORS  = {"C1": "#1B4F72", "C2": "#1A5276", "C3": "#154360", "C4": "#E8A020"}
LANG_COLORS  = {"EN": "#2E86C1", "YO": "#E74C3C", "ALL": "#27AE60"}
DPI          = 180
FIGSIZE_FULL = (14, 7)
FIGSIZE_SQ   = (10, 8)

CLF_ORDER = [
    "svm", "logistic", "random_forest", "extra_trees",
    "gradient_boosting", "hist_gradient_boosting", "adaboost", "decision_tree",
    "nu_svc", "sgd", "lda", "qda", "knn", "naive_bayes", "mlp_sklearn", "xgboost",
]
CLF_LABELS = {
    "svm": "SVM", "logistic": "Logistic", "random_forest": "Random Forest",
    "extra_trees": "Extra Trees", "gradient_boosting": "Grad. Boost",
    "hist_gradient_boosting": "Hist-GB", "adaboost": "AdaBoost",
    "decision_tree": "Dec. Tree", "nu_svc": "Nu-SVC", "sgd": "SGD",
    "lda": "LDA", "qda": "QDA", "knn": "KNN", "naive_bayes": "Naive Bayes",
    "mlp_sklearn": "MLP (sklearn)", "xgboost": "XGBoost",
}
COND_ORDER  = ["C1", "C2", "C3", "C4"]
COND_LABELS = {
    "C1": "C1\n(All features)",
    "C2": "C2\n(No F0/pitch)",
    "C3": "C3\n(Reduced set)",
    "C4": "C4\n(SSL/HuBERT)",
}


def _save(fig: plt.Figure, name: str) -> None:
    path = FIG_DIR / name
    fig.savefig(path, dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  saved → {path.name}")


def load_tables() -> tuple[pd.DataFrame, pd.DataFrame]:
    t2 = pd.read_csv(TABLES_DIR / "table2_yodep_main.csv")
    t3 = pd.read_csv(TABLES_DIR / "table3_transfer.csv")
    return t2, t3


# ═══════════════════════════════════════════════════════════════════════════
#  PHASE 1 — figures from aggregated CSV data
# ═══════════════════════════════════════════════════════════════════════════

def fig01_f1_heatmap(df: pd.DataFrame) -> None:
    """F1 macro heatmap: classifiers × conditions, ALL language."""
    subset = df[df["Language"] == "ALL"].copy()
    pivot  = subset.pivot_table(index="Classifier", columns="Condition",
                                values="F1_macro_mean", aggfunc="mean")
    pivot  = pivot.reindex([c for c in CLF_ORDER if c in pivot.index])
    pivot.index = [CLF_LABELS.get(c, c) for c in pivot.index]
    pivot.columns = [COND_LABELS.get(c, c) for c in pivot.columns]

    fig, ax = plt.subplots(figsize=FIGSIZE_SQ)
    sns.heatmap(pivot, annot=True, fmt=".3f", cmap="YlOrRd",
                vmin=0.5, vmax=1.0, linewidths=0.5, linecolor="#cccccc",
                annot_kws={"size": 9}, ax=ax, cbar_kws={"label": "F1 macro"})
    ax.set_title("F1 Macro — All Classifiers × All Conditions (ALL languages)",
                 fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("Feature Condition", fontsize=11)
    ax.set_ylabel("Classifier", fontsize=11)
    ax.tick_params(axis="x", labelsize=9)
    ax.tick_params(axis="y", labelsize=9, rotation=0)
    fig.tight_layout()
    _save(fig, "fig01_f1_heatmap_classifiers_conditions.png")


def fig02_auc_heatmap(df: pd.DataFrame) -> None:
    """AUC heatmap: classifiers × conditions, ALL language."""
    subset = df[df["Language"] == "ALL"].copy()
    pivot  = subset.pivot_table(index="Classifier", columns="Condition",
                                values="AUC_mean", aggfunc="mean")
    pivot  = pivot.reindex([c for c in CLF_ORDER if c in pivot.index])
    pivot.index = [CLF_LABELS.get(c, c) for c in pivot.index]
    pivot.columns = [COND_LABELS.get(c, c) for c in pivot.columns]

    fig, ax = plt.subplots(figsize=FIGSIZE_SQ)
    sns.heatmap(pivot, annot=True, fmt=".3f", cmap="Blues",
                vmin=0.6, vmax=1.0, linewidths=0.5, linecolor="#cccccc",
                annot_kws={"size": 9}, ax=ax, cbar_kws={"label": "AUC"})
    ax.set_title("AUC — All Classifiers × All Conditions (ALL languages)",
                 fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("Feature Condition", fontsize=11)
    ax.set_ylabel("Classifier", fontsize=11)
    ax.tick_params(axis="x", labelsize=9)
    ax.tick_params(axis="y", labelsize=9, rotation=0)
    fig.tight_layout()
    _save(fig, "fig02_auc_heatmap_classifiers_conditions.png")


def fig03_barplot_f1_per_condition(df: pd.DataFrame) -> None:
    """Bar chart with error bars: F1 per classifier, one subplot per condition, ALL language."""
    subset = df[df["Language"] == "ALL"].copy()
    subset["clf_label"] = subset["Classifier"].map(CLF_LABELS)
    conditions = [c for c in COND_ORDER if c in subset["Condition"].unique()]

    fig, axes = plt.subplots(2, 2, figsize=(16, 10), sharey=False)
    axes = axes.flatten()

    for i, cond in enumerate(conditions):
        ax  = axes[i]
        sub = subset[subset["Condition"] == cond].sort_values("F1_macro_mean", ascending=False)
        colors = [CLF_PALETTE[CLF_ORDER.index(c) % len(CLF_PALETTE)]
                  if c in CLF_ORDER else "#999999"
                  for c in sub["Classifier"]]
        bars = ax.bar(sub["clf_label"], sub["F1_macro_mean"],
                      yerr=sub["F1_macro_std"], capsize=4,
                      color=colors, edgecolor="white", linewidth=0.5,
                      error_kw={"elinewidth": 1.2, "ecolor": "#555555"})
        ax.set_title(f"Condition {cond} — {COND_LABELS[cond].replace(chr(10), ' ')}",
                     fontsize=11, fontweight="bold")
        ax.set_ylim(0.4, 1.05)
        ax.axhline(0.5, color="grey", linestyle="--", linewidth=0.8, alpha=0.6)
        ax.set_ylabel("F1 Macro (mean ± SD)", fontsize=9)
        ax.tick_params(axis="x", rotation=45, labelsize=8)
        ax.tick_params(axis="y", labelsize=8)
        # annotate best
        best_idx = sub["F1_macro_mean"].idxmax()
        best_val = sub.loc[best_idx, "F1_macro_mean"]
        ax.annotate(f"Best: {best_val:.3f}",
                    xy=(0.97, 0.95), xycoords="axes fraction",
                    ha="right", va="top", fontsize=8,
                    bbox=dict(boxstyle="round,pad=0.3", fc="#FFF9C4", ec="#E0E0E0"))

    fig.suptitle("F1 Macro by Classifier per Feature Condition (ALL languages, LOSO)",
                 fontsize=13, fontweight="bold", y=1.01)
    fig.tight_layout()
    _save(fig, "fig03_barplot_f1_per_condition.png")


def fig04_en_vs_yo(df: pd.DataFrame) -> None:
    """Grouped bar: EN vs YO F1 for top-8 classifiers, across conditions."""
    top_clfs = CLF_ORDER[:8]
    subset = df[df["Language"].isin(["EN", "YO"]) &
                df["Classifier"].isin(top_clfs)].copy()
    subset["clf_label"] = subset["Classifier"].map(CLF_LABELS)

    conditions = [c for c in COND_ORDER if c in subset["Condition"].unique()]
    fig, axes  = plt.subplots(1, len(conditions), figsize=(16, 5), sharey=True)

    for ax, cond in zip(axes, conditions):
        sub  = subset[subset["Condition"] == cond]
        wide = sub.pivot(index="clf_label", columns="Language", values="F1_macro_mean")
        wide = wide.reindex([CLF_LABELS[c] for c in top_clfs if CLF_LABELS[c] in wide.index])
        x    = np.arange(len(wide))
        w    = 0.35
        ax.bar(x - w/2, wide.get("EN", 0), w, label="EN",
               color=LANG_COLORS["EN"], edgecolor="white")
        ax.bar(x + w/2, wide.get("YO", 0), w, label="YO",
               color=LANG_COLORS["YO"], edgecolor="white")
        ax.set_xticks(x)
        ax.set_xticklabels(wide.index, rotation=45, ha="right", fontsize=8)
        ax.set_title(f"Condition {cond}", fontsize=10, fontweight="bold")
        ax.set_ylim(0.4, 1.05)
        ax.axhline(0.5, color="grey", linestyle="--", linewidth=0.7, alpha=0.6)
        if ax == axes[0]:
            ax.set_ylabel("F1 Macro", fontsize=10)
        ax.legend(fontsize=8)

    fig.suptitle("English vs Yoruba — F1 Macro per Condition (top-8 classifiers)",
                 fontsize=12, fontweight="bold", y=1.02)
    fig.tight_layout()
    _save(fig, "fig04_en_vs_yo_comparison.png")


def fig05_f0_effect(df: pd.DataFrame) -> None:
    """Paired bar: C1 vs C2 (F0 removal effect) for all classifiers, ALL language."""
    subset = df[(df["Language"] == "ALL") & (df["Condition"].isin(["C1", "C2"]))].copy()
    wide   = subset.pivot_table(index="Classifier", columns="Condition",
                                values="F1_macro_mean")
    wide   = wide.reindex([c for c in CLF_ORDER if c in wide.index])
    wide.index = [CLF_LABELS.get(c, c) for c in wide.index]
    wide["delta"] = wide.get("C2", np.nan) - wide.get("C1", np.nan)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(13, 9),
                                   gridspec_kw={"height_ratios": [3, 1]})

    x = np.arange(len(wide))
    w = 0.35
    ax1.bar(x - w/2, wide.get("C1", 0), w, label="C1 (with F0)",
            color=COND_COLORS["C1"], edgecolor="white")
    ax1.bar(x + w/2, wide.get("C2", 0), w, label="C2 (no F0)",
            color="#5DADE2", edgecolor="white")
    ax1.set_xticks(x)
    ax1.set_xticklabels(wide.index, rotation=45, ha="right", fontsize=9)
    ax1.set_ylabel("F1 Macro", fontsize=10)
    ax1.set_ylim(0.4, 1.0)
    ax1.legend(fontsize=9)
    ax1.set_title("Effect of Removing F0/Pitch Features: C1 vs C2 (ALL languages, LOSO)",
                  fontsize=12, fontweight="bold")

    delta_colors = ["#E74C3C" if d < -0.01 else "#27AE60" if d > 0.01 else "#95A5A6"
                    for d in wide["delta"]]
    ax2.bar(x, wide["delta"], color=delta_colors, edgecolor="white")
    ax2.axhline(0, color="black", linewidth=0.8)
    ax2.set_xticks(x)
    ax2.set_xticklabels(wide.index, rotation=45, ha="right", fontsize=9)
    ax2.set_ylabel("Δ F1 (C2−C1)", fontsize=9)
    ax2.set_title("Δ F1 when F0 removed (positive = F0 removal improves; negative = hurts)",
                  fontsize=9)

    fig.tight_layout()
    _save(fig, "fig05_f0_removal_effect_c1_vs_c2.png")


def fig06_handcrafted_vs_ssl(df: pd.DataFrame) -> None:
    """Paired bar: C1 (handcrafted) vs C4 (SSL) for all classifiers, ALL language."""
    subset = df[(df["Language"] == "ALL") & (df["Condition"].isin(["C1", "C4"]))].copy()
    wide   = subset.pivot_table(index="Classifier", columns="Condition",
                                values="F1_macro_mean")
    wide   = wide.reindex([c for c in CLF_ORDER if c in wide.index])
    wide.index = [CLF_LABELS.get(c, c) for c in wide.index]

    fig, ax = plt.subplots(figsize=(13, 6))
    x = np.arange(len(wide))
    w = 0.35
    ax.bar(x - w/2, wide.get("C1", 0), w, label="C1 (Handcrafted)",
           color=COND_COLORS["C1"], edgecolor="white")
    ax.bar(x + w/2, wide.get("C4", 0), w, label="C4 (SSL/HuBERT)",
           color=COND_COLORS["C4"], edgecolor="white")
    ax.set_xticks(x)
    ax.set_xticklabels(wide.index, rotation=45, ha="right", fontsize=9)
    ax.set_ylabel("F1 Macro", fontsize=11)
    ax.set_ylim(0.4, 1.0)
    ax.legend(fontsize=10)
    ax.set_title("Handcrafted Features (C1) vs SSL Embeddings (C4) — ALL languages, LOSO",
                 fontsize=12, fontweight="bold")
    ax.axhline(0.5, color="grey", linestyle="--", linewidth=0.8, alpha=0.6)
    fig.tight_layout()
    _save(fig, "fig06_handcrafted_vs_ssl_c1_vs_c4.png")


def fig07_transfer_delta(t3: pd.DataFrame) -> None:
    """Bar chart: cross-lingual transfer delta per condition and direction."""
    df = t3.copy()
    df["direction"] = df["Train_language"] + "→" + df["Test_language"]
    cross = df[df["Train_language"] != df["Test_language"]].copy()
    cross["Condition"] = cross["Condition"].str.replace("_XFER", "", regex=False)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Left: absolute F1 (within vs cross)
    ax = axes[0]
    within = df[df["Train_language"] == df["Test_language"]].copy()
    within["Condition"] = within["Condition"].str.replace("_XFER", "", regex=False)
    all_rows = pd.concat([within.assign(type="Within-language"),
                          cross.assign(type="Cross-lingual")])
    colors_map = {"Within-language": "#2E86C1", "Cross-lingual": "#E74C3C"}
    for i, (cond, grp) in enumerate(all_rows.groupby("Condition")):
        for j, (dtype, dgrp) in enumerate(grp.groupby("type")):
            x_pos = i * 3 + j
            ax.bar(x_pos, dgrp["F1_macro"].mean(),
                   color=colors_map[dtype], edgecolor="white", width=0.8)
    conditions = sorted(all_rows["Condition"].unique())
    ax.set_xticks([i * 3 + 0.5 for i in range(len(conditions))])
    ax.set_xticklabels(conditions, fontsize=10)
    ax.set_ylabel("F1 Macro", fontsize=10)
    ax.set_title("Within-language vs Cross-lingual F1", fontsize=11, fontweight="bold")
    patches = [mpatches.Patch(color=c, label=l) for l, c in colors_map.items()]
    ax.legend(handles=patches, fontsize=9)

    # Right: delta bar chart
    ax2 = axes[1]
    bar_colors = ["#27AE60" if d >= 0 else "#E74C3C"
                  for d in cross["Delta_vs_within_language"]]
    x = np.arange(len(cross))
    ax2.bar(x, cross["Delta_vs_within_language"], color=bar_colors, edgecolor="white")
    ax2.axhline(0, color="black", linewidth=0.9)
    labels = [f"{row.Condition}\n{row.Train_language}→{row.Test_language}"
              for _, row in cross.iterrows()]
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, fontsize=9)
    ax2.set_ylabel("Δ F1 vs Within-language", fontsize=10)
    ax2.set_title("Cross-lingual Transfer Δ (green=gain, red=loss)", fontsize=11,
                  fontweight="bold")
    for xi, (val, row) in enumerate(zip(cross["Delta_vs_within_language"],
                                        cross.itertuples())):
        ax2.text(xi, val + (0.005 if val >= 0 else -0.012),
                 f"{val:+.3f}", ha="center", va="bottom" if val >= 0 else "top",
                 fontsize=8, fontweight="bold")

    fig.suptitle("Cross-lingual Transfer Experiment (SVM classifier)",
                 fontsize=12, fontweight="bold", y=1.02)
    fig.tight_layout()
    _save(fig, "fig07_transfer_delta.png")


def fig08_classifier_ranking(df: pd.DataFrame) -> None:
    """Bump chart: classifier rank across conditions (ALL language)."""
    subset = df[df["Language"] == "ALL"].copy()
    subset["clf_label"] = subset["Classifier"].map(CLF_LABELS)
    conditions = [c for c in COND_ORDER if c in subset["Condition"].unique()]

    ranks = {}
    for cond in conditions:
        sub  = subset[subset["Condition"] == cond].sort_values("F1_macro_mean",
                                                                ascending=False)
        sub  = sub.reset_index(drop=True)
        for i, row in sub.iterrows():
            ranks.setdefault(row["clf_label"], {})[cond] = i + 1

    rank_df = pd.DataFrame(ranks).T
    rank_df = rank_df.reindex(columns=conditions)

    fig, ax = plt.subplots(figsize=(10, 8))
    n_clfs  = len(rank_df)

    for i, (clf, row) in enumerate(rank_df.iterrows()):
        color  = CLF_PALETTE[i % len(CLF_PALETTE)]
        vals   = [row.get(c, np.nan) for c in conditions]
        x_vals = list(range(len(conditions)))
        ax.plot(x_vals, vals, "o-", color=color, linewidth=1.8,
                markersize=7, label=clf)
        ax.annotate(clf, xy=(len(conditions) - 1, vals[-1]),
                    xytext=(3, 0), textcoords="offset points",
                    fontsize=7, va="center", color=color)

    ax.set_xticks(range(len(conditions)))
    ax.set_xticklabels([COND_LABELS[c] for c in conditions], fontsize=10)
    ax.set_ylabel("Rank (1 = best)", fontsize=10)
    ax.invert_yaxis()
    ax.set_yticks(range(1, n_clfs + 1))
    ax.set_title("Classifier Ranking Across Feature Conditions (ALL languages)",
                 fontsize=12, fontweight="bold")
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    fig.tight_layout()
    _save(fig, "fig08_classifier_ranking_bump_chart.png")


def fig09_f1_std_heatmap(df: pd.DataFrame) -> None:
    """F1 std heatmap — stability / variance analysis, ALL language."""
    subset = df[df["Language"] == "ALL"].copy()
    pivot  = subset.pivot_table(index="Classifier", columns="Condition",
                                values="F1_macro_std", aggfunc="mean")
    pivot  = pivot.reindex([c for c in CLF_ORDER if c in pivot.index])
    pivot.index = [CLF_LABELS.get(c, c) for c in pivot.index]
    pivot.columns = [COND_LABELS.get(c, c) for c in pivot.columns]

    fig, ax = plt.subplots(figsize=FIGSIZE_SQ)
    sns.heatmap(pivot, annot=True, fmt=".3f", cmap="Oranges",
                vmin=0.0, vmax=0.3, linewidths=0.5, linecolor="#cccccc",
                annot_kws={"size": 9}, ax=ax,
                cbar_kws={"label": "F1 Std Dev (lower = more stable)"})
    ax.set_title("F1 Macro Standard Deviation — Stability Across LOSO Folds (ALL languages)",
                 fontsize=12, fontweight="bold", pad=12)
    ax.set_xlabel("Feature Condition", fontsize=11)
    ax.set_ylabel("Classifier", fontsize=11)
    ax.tick_params(axis="x", labelsize=9)
    ax.tick_params(axis="y", labelsize=9, rotation=0)
    fig.tight_layout()
    _save(fig, "fig09_f1_std_stability_heatmap.png")


def fig10_precision_recall_scatter(df: pd.DataFrame) -> None:
    """Precision vs Recall scatter plot, ALL language, coloured by condition."""
    subset = df[df["Language"] == "ALL"].copy()
    subset["clf_label"] = subset["Classifier"].map(CLF_LABELS)

    fig, ax = plt.subplots(figsize=(9, 7))
    for cond, grp in subset.groupby("Condition"):
        ax.scatter(grp["Recall_mean"], grp["Precision_mean"],
                   label=f"Condition {cond}", alpha=0.75, s=70,
                   color=COND_COLORS.get(cond, "#999999"), edgecolors="white",
                   linewidth=0.5)
        for _, row in grp.iterrows():
            ax.annotate(row["clf_label"],
                        (row["Recall_mean"], row["Precision_mean"]),
                        textcoords="offset points", xytext=(4, 2),
                        fontsize=6, alpha=0.7)

    ax.plot([0.4, 1.0], [0.4, 1.0], "k--", linewidth=0.8, alpha=0.4,
            label="Precision = Recall")
    ax.set_xlabel("Recall (macro)", fontsize=11)
    ax.set_ylabel("Precision (macro)", fontsize=11)
    ax.set_xlim(0.4, 1.05)
    ax.set_ylim(0.4, 1.05)
    ax.legend(fontsize=9)
    ax.set_title("Precision vs Recall — All Classifiers, All Conditions (ALL languages)",
                 fontsize=12, fontweight="bold")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    _save(fig, "fig10_precision_recall_scatter.png")


# ═══════════════════════════════════════════════════════════════════════════
#  PHASE 2 — figures from fold-level predictions (.npz files)
# ═══════════════════════════════════════════════════════════════════════════

def _load_predictions(condition: str, language: str, clf: str):
    path = PRED_DIR / f"{condition}_{language}_{clf}.npz"
    if not path.exists():
        return None
    data = np.load(path)
    return data["y_true"], data["y_pred"], data["y_prob"]


def fig11_roc_curves(df: pd.DataFrame) -> None:
    """ROC curves per condition, ALL language, all available classifiers."""
    from sklearn.metrics import roc_curve, auc

    conditions = [c for c in COND_ORDER if c in df["Condition"].unique()]
    fig, axes  = plt.subplots(2, 2, figsize=(13, 11))
    axes       = axes.flatten()

    for ax, cond in zip(axes, conditions):
        clfs_plotted = 0
        for i, clf in enumerate(CLF_ORDER):
            preds = _load_predictions(cond, "ALL", clf)
            if preds is None:
                continue
            y_true, _, y_prob = preds
            if y_prob.ndim == 2:
                y_score = y_prob[:, 1]
            else:
                y_score = y_prob
            fpr, tpr, _ = roc_curve(y_true, y_score)
            roc_auc = auc(fpr, tpr)
            ax.plot(fpr, tpr, linewidth=1.5,
                    color=CLF_PALETTE[i % len(CLF_PALETTE)],
                    label=f"{CLF_LABELS.get(clf, clf)} (AUC={roc_auc:.3f})")
            clfs_plotted += 1

        ax.plot([0, 1], [0, 1], "k--", linewidth=0.8, alpha=0.5)
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel("False Positive Rate", fontsize=9)
        ax.set_ylabel("True Positive Rate", fontsize=9)
        ax.set_title(f"ROC Curves — {cond} ({COND_LABELS[cond].replace(chr(10), ' ')})",
                     fontsize=10, fontweight="bold")
        if clfs_plotted > 0:
            ax.legend(fontsize=6, loc="lower right", ncol=2)
        else:
            ax.text(0.5, 0.5, "No predictions saved yet\n(re-run experiment first)",
                    ha="center", va="center", transform=ax.transAxes,
                    fontsize=10, color="grey")
        ax.grid(alpha=0.3)

    fig.suptitle("ROC Curves — All Classifiers per Feature Condition (ALL languages)",
                 fontsize=13, fontweight="bold", y=1.01)
    fig.tight_layout()
    _save(fig, "fig11_roc_curves_all_conditions.png")


def fig12_confusion_matrices(df: pd.DataFrame) -> None:
    """Confusion matrix for the best classifier per condition, ALL language."""
    from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

    conditions = [c for c in COND_ORDER if c in df["Condition"].unique()]
    best_clfs  = {}
    for cond in conditions:
        sub = df[(df["Condition"] == cond) & (df["Language"] == "ALL")]
        if not sub.empty:
            best_clfs[cond] = sub.loc[sub["F1_macro_mean"].idxmax(), "Classifier"]

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes      = axes.flatten()

    for ax, cond in zip(axes, conditions):
        clf   = best_clfs.get(cond, "svm")
        preds = _load_predictions(cond, "ALL", clf)
        if preds is None:
            ax.text(0.5, 0.5,
                    f"No predictions saved yet\nBest clf: {CLF_LABELS.get(clf, clf)}",
                    ha="center", va="center", transform=ax.transAxes,
                    fontsize=10, color="grey")
            ax.set_title(f"{cond} — {CLF_LABELS.get(clf, clf)}", fontsize=10)
            continue

        y_true, y_pred, _ = preds
        cm  = confusion_matrix(y_true, y_pred)
        disp = ConfusionMatrixDisplay(cm, display_labels=["NORMAL", "DEPRESSED"])
        disp.plot(ax=ax, colorbar=False, cmap="Blues")
        f1_val = df[(df["Condition"] == cond) &
                    (df["Language"] == "ALL") &
                    (df["Classifier"] == clf)]["F1_macro_mean"].values
        f1_str = f" (F1={f1_val[0]:.3f})" if len(f1_val) else ""
        ax.set_title(f"{cond} — Best: {CLF_LABELS.get(clf, clf)}{f1_str}",
                     fontsize=10, fontweight="bold")

    fig.suptitle("Confusion Matrices — Best Classifier per Condition (ALL languages, LOSO)",
                 fontsize=13, fontweight="bold", y=1.01)
    fig.tight_layout()
    _save(fig, "fig12_confusion_matrices_best_per_condition.png")


# ═══════════════════════════════════════════════════════════════════════════
#  PHASE 3 — training-curve figures
# ═══════════════════════════════════════════════════════════════════════════

def _load_curve_json(clf_key: str) -> dict:
    """Return {label: curve_data} for all JSON files matching *_{clf_key}.json."""
    import json
    results = {}
    if not CURVES_DIR.exists():
        return results
    for path in sorted(CURVES_DIR.glob(f"*_{clf_key}.json")):
        stem_parts = path.stem.split("_")
        # filename: {cond}_{lang}_{clf_key}  (clf_key may contain underscores)
        # split off trailing len(clf_key.split('_')) tokens for the label
        n_clf_parts = len(clf_key.split("_"))
        label = "_".join(stem_parts[: len(stem_parts) - n_clf_parts])
        try:
            with open(path) as f:
                results[label] = json.load(f)
        except Exception:
            pass
    return results


def fig13_fold_f1_violin(df: pd.DataFrame) -> None:
    """Violin + box plots: per-speaker LOSO F1 distribution for each condition."""
    fold_csv = TABLES_DIR / "fold_metrics_detail.csv"
    if not fold_csv.exists():
        print("  SKIP fig13 — fold_metrics_detail.csv not found (re-run experiment first)")
        return

    fold_df = pd.read_csv(fold_csv)
    # Focus on ALL language, handcrafted+SSL conditions
    conditions = [c for c in COND_ORDER if c in fold_df["Condition"].unique()]
    languages  = ["ALL", "EN", "YO"]

    for lang in languages:
        sub = fold_df[fold_df["Language"] == lang].copy() if lang != "ALL" else fold_df.copy()
        if sub.empty:
            continue
        sub = sub[sub["Condition"].isin(conditions)]
        if sub.empty:
            continue

        n_conds = len(conditions)
        fig, axes = plt.subplots(1, n_conds, figsize=(5 * n_conds, 6), sharey=False)
        if n_conds == 1:
            axes = [axes]

        for ax, cond in zip(axes, conditions):
            cond_sub = sub[sub["Condition"] == cond].copy()
            if cond_sub.empty:
                ax.set_visible(False)
                continue
            # Order classifiers by median F1 descending
            order = (cond_sub.groupby("Classifier")["F1_macro"]
                     .median().sort_values(ascending=False).index.tolist())
            order_labels = [CLF_LABELS.get(c, c) for c in order]
            cond_sub["clf_label"] = cond_sub["Classifier"].map(CLF_LABELS)

            sns.violinplot(data=cond_sub, x="clf_label", y="F1_macro",
                           order=order_labels, palette="tab20",
                           inner="box", cut=0, ax=ax)
            ax.axhline(0.5, color="grey", linestyle="--", linewidth=0.8, alpha=0.6)
            ax.set_title(f"{cond} — {COND_LABELS[cond].replace(chr(10),' ')}",
                         fontsize=10, fontweight="bold")
            ax.set_xlabel("")
            ax.set_ylabel("F1 Macro (per LOSO fold)" if ax == axes[0] else "")
            ax.tick_params(axis="x", rotation=60, labelsize=7)
            ax.set_ylim(0.0, 1.05)

        lang_title = lang if lang != "ALL" else "ALL languages"
        fig.suptitle(
            f"Per-Speaker LOSO F1 Distribution — {lang_title}",
            fontsize=12, fontweight="bold", y=1.01
        )
        fig.tight_layout()
        _save(fig, f"fig13_fold_f1_violin_{lang}.png")


def fig14_mlp_training_curves() -> None:
    """MLP training loss and validation score curves per condition/language."""
    curves = _load_curve_json("mlp_sklearn")
    if not curves:
        print("  SKIP fig14 — no mlp_sklearn training curve JSONs found")
        return

    n = len(curves)
    ncols = min(n, 3)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 4 * nrows),
                             squeeze=False)
    axes_flat = [ax for row in axes for ax in row]

    for ax, (label, data) in zip(axes_flat, curves.items()):
        plotted = False
        if "loss" in data:
            epochs = list(range(1, len(data["loss"]) + 1))
            ax.plot(epochs, data["loss"], label="Train loss",
                    color="#1B4F72", linewidth=1.8)
            plotted = True
        if "val_score" in data:
            epochs = list(range(1, len(data["val_score"]) + 1))
            ax.plot(epochs, data["val_score"], label="Val score",
                    color="#E74C3C", linewidth=1.8, linestyle="--")
            plotted = True
        if not plotted:
            ax.text(0.5, 0.5, "No curve data", ha="center", va="center",
                    transform=ax.transAxes, color="grey")
        ax.set_title(label.replace("_", " "), fontsize=9, fontweight="bold")
        ax.set_xlabel("Epoch", fontsize=8)
        ax.set_ylabel("Loss / Score", fontsize=8)
        ax.legend(fontsize=7)
        ax.grid(alpha=0.3)

    # Hide unused axes
    for ax in axes_flat[n:]:
        ax.set_visible(False)

    fig.suptitle("MLP (sklearn) Training Loss & Validation Score per Condition/Language",
                 fontsize=12, fontweight="bold", y=1.01)
    fig.tight_layout()
    _save(fig, "fig14_mlp_training_curves.png")


def fig15_gbm_staged_scores() -> None:
    """GradientBoosting and HistGradientBoosting staged training scores."""
    curves_gb   = _load_curve_json("gradient_boosting")
    curves_hgb  = _load_curve_json("hist_gradient_boosting")

    all_curves = {
        f"GB/{k}": v for k, v in curves_gb.items()
    }
    all_curves.update({
        f"HGB/{k}": v for k, v in curves_hgb.items()
    })

    if not all_curves:
        print("  SKIP fig15 — no gradient_boosting/hist_gradient_boosting curve JSONs found")
        return

    n = len(all_curves)
    ncols = min(n, 3)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 4 * nrows),
                             squeeze=False)
    axes_flat = [ax for row in axes for ax in row]

    for ax, (label, data) in zip(axes_flat, all_curves.items()):
        plotted = False
        if "train_score" in data:
            stages = list(range(1, len(data["train_score"]) + 1))
            ax.plot(stages, data["train_score"], label="Train score",
                    color="#1B4F72", linewidth=1.8)
            plotted = True
        if "val_score" in data:
            stages = list(range(1, len(data["val_score"]) + 1))
            ax.plot(stages, data["val_score"], label="Val score",
                    color="#E74C3C", linewidth=1.8, linestyle="--")
            plotted = True
        if not plotted:
            ax.text(0.5, 0.5, "No score data", ha="center", va="center",
                    transform=ax.transAxes, color="grey")
        ax.set_title(label.replace("_", " "), fontsize=9, fontweight="bold")
        ax.set_xlabel("Boosting Stage", fontsize=8)
        ax.set_ylabel("Score (F1 macro)", fontsize=8)
        ax.legend(fontsize=7)
        ax.grid(alpha=0.3)

    for ax in axes_flat[n:]:
        ax.set_visible(False)

    fig.suptitle("GBM / Hist-GBM Staged Training Scores per Condition/Language",
                 fontsize=12, fontweight="bold", y=1.01)
    fig.tight_layout()
    _save(fig, "fig15_gbm_staged_scores.png")


# ═══════════════════════════════════════════════════════════════════════════
#  ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Generate YODEP report figures.")
    parser.add_argument("--phase", type=int, choices=[1, 2, 3], default=None,
                        help="1=CSV figures, 2=prediction figures, 3=training curves, omit=all")
    args = parser.parse_args()

    print("Loading tables...")
    t2, t3 = load_tables()

    run_phase1 = args.phase in (None, 1)
    run_phase2 = args.phase in (None, 2)
    run_phase3 = args.phase in (None, 3)

    if run_phase1:
        print("\n── Phase 1: aggregated-data figures ──")
        fig01_f1_heatmap(t2)
        fig02_auc_heatmap(t2)
        fig03_barplot_f1_per_condition(t2)
        fig04_en_vs_yo(t2)
        fig05_f0_effect(t2)
        fig06_handcrafted_vs_ssl(t2)
        fig07_transfer_delta(t3)
        fig08_classifier_ranking(t2)
        fig09_f1_std_heatmap(t2)
        fig10_precision_recall_scatter(t2)

    if run_phase2:
        print("\n── Phase 2: fold-prediction figures ──")
        npz_files = list(PRED_DIR.glob("*.npz"))
        if not npz_files:
            print("  WARNING: No .npz prediction files found in results/predictions/")
            print("  Re-run the main experiment first, then run this script again.")
        else:
            print(f"  Found {len(npz_files)} prediction file(s).")
        fig11_roc_curves(t2)
        fig12_confusion_matrices(t2)

    if run_phase3:
        print("\n── Phase 3: training-curve figures ──")
        fold_csv = TABLES_DIR / "fold_metrics_detail.csv"
        curves_jsons = list(CURVES_DIR.glob("*.json")) if CURVES_DIR.exists() else []
        if not fold_csv.exists() and not curves_jsons:
            print("  WARNING: No fold_metrics_detail.csv or training curve JSONs found.")
            print("  Re-run the main experiment first, then run this script again.")
        else:
            if fold_csv.exists():
                print(f"  Found fold_metrics_detail.csv")
            if curves_jsons:
                print(f"  Found {len(curves_jsons)} training curve JSON file(s).")
        fig13_fold_f1_violin(t2)
        fig14_mlp_training_curves()
        fig15_gbm_staged_scores()

    print(f"\nAll figures saved to: {FIG_DIR}")


if __name__ == "__main__":
    main()
