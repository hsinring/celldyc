import os
import scanpy as sc
import numpy as np
import pandas as pd
import scvelo as scv
import seaborn as sns
import matplotlib.pyplot as plt
from typing import List, Tuple, Optional
from anndata import AnnData
from scipy import stats
from celldyc.tools.utils import calculate_gene_avg
from celldyc.plotting.utils import calculate_ovl
from celldyc.plotting.core import velocity_dynamics, velocity_hot


def sample_along_time(
    adata: AnnData,
    num: int = 3,
    time_key: str = "pseudotime",
    save_path: Optional[str] = None,
) -> AnnData:
    import matplotlib.colors as mcolors

    arr = adata.obs[time_key].to_numpy()

    points = np.linspace(0.1, 0.9, num)
    std_devs = 0.05
    samples_per_point = 500

    all_samples = []
    select_points = []

    np.random.seed(0)

    for point in points:
        targets = np.random.normal(loc=point, scale=std_devs, size=samples_per_point)
        for target in targets:
            idx = np.abs(arr - target).argmin()
            all_samples.append(idx)
            select_points.append(point)

    all_samples = np.array(all_samples)
    select_points = np.array(select_points)

    _, idx_unique = np.unique(all_samples, return_index=True)
    all_samples = all_samples[idx_unique]
    select_points = select_points[idx_unique]

    adata_sampled = adata[all_samples]

    adata_sampled.obs["sampling_time"] = select_points

    unique_times = sorted(adata_sampled.obs["sampling_time"].astype(float).unique())
    n_cats = len(unique_times)

    time_to_label = {time: str(i) for i, time in enumerate(unique_times)}
    adata_sampled.obs["sampling_time"] = (
        adata_sampled.obs["sampling_time"].astype(float).map(time_to_label)
    )

    correct_order = [str(i) for i in range(n_cats)]
    adata_sampled.obs["sampling_time"] = adata_sampled.obs["sampling_time"].astype(
        pd.CategoricalDtype(categories=correct_order, ordered=True)
    )

    time_cat_mapping = {str(i): f"t{i + 1}" for i in range(n_cats)}
    adata_sampled.obs["sampling_time_cat"] = adata_sampled.obs["sampling_time"].map(
        time_cat_mapping
    )
    adata_sampled.obs["sampling_time_cat"] = adata_sampled.obs[
        "sampling_time_cat"
    ].astype(
        pd.CategoricalDtype(
            categories=[f"t{i + 1}" for i in range(n_cats)], ordered=True
        )
    )

    tab10_colors = plt.cm.tab10.colors[:n_cats]
    adata_sampled.uns["sampling_time_colors"] = [
        mcolors.to_hex(c) for c in tab10_colors
    ]
    adata_sampled.uns["sampling_time_cat_colors"] = [
        mcolors.to_hex(c) for c in tab10_colors
    ]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    sc.pl.umap(adata, color=time_key, show=False, ax=ax1)
    ax1.set_title(f"Original Data\n{adata.shape[0]} cells")

    sc.pl.umap(adata_sampled, color="sampling_time_cat", show=False, ax=ax2)
    ax2.set_title(f"Sampled Data ({num} points)\n{adata_sampled.shape[0]} cells")

    plt.tight_layout()
    plt.show()

    if save_path:
        adata_sampled.write(save_path)
        print(f"\nSaved sampled data to: {save_path}")

    return adata_sampled


def mask_getime_genes(
    adata: AnnData,
    thresholds: Tuple[float, float, float, float] = (0.4, 0.4, 0.1, 0.35),
    group: str = "all",
) -> np.ndarray:
    if "Treatment" not in adata.obs.columns:
        raise ValueError("Column 'Treatment' not found in adata.obs")

    if group not in ["all", "balanced", "igg-preferential"]:
        raise ValueError("group must be 'all', 'balanced', or 'igg-preferential'")

    igg_cells = adata[adata.obs["Treatment"] == "IgG"]
    atrem2_cells = adata[adata.obs["Treatment"] == "aTrem2"]

    igg_gene_avg = calculate_gene_avg(igg_cells)
    atrem2_gene_avg = calculate_gene_avg(atrem2_cells)

    adata.var["avg_igg"] = igg_gene_avg
    adata.var["avg_atrem2"] = atrem2_gene_avg

    igg_thresh, atrem2_thresh, balanced_thresh, biased_thresh = thresholds

    mask1 = (adata.var["avg_igg"] > igg_thresh) | (
        adata.var["avg_atrem2"] > atrem2_thresh
    )
    mask2 = abs(adata.var["avg_igg"] - adata.var["avg_atrem2"]) < balanced_thresh
    mask3 = abs(adata.var["avg_igg"] - adata.var["avg_atrem2"]) > biased_thresh

    if group == "all":
        mask = (mask1 & mask2) | (mask1 & mask3)
    elif group == "balanced":
        mask = mask1 & mask2
    else:  # group == "igg-preferential"
        mask = mask1 & mask3

    return mask


def analyze_gene_contributions(
    adata: AnnData,
    thresholds: Tuple[float, float, float, float] = (0.4, 0.4, 0.1, 0.35),
) -> pd.DataFrame:
    if "Treatment" not in adata.obs.columns:
        raise ValueError("Column 'Treatment' not found in adata.obs")

    igg_thresh, atrem2_thresh, balanced_thresh, biased_thresh = thresholds

    igg_cells = adata[adata.obs["Treatment"] == "IgG"]
    atrem2_cells = adata[adata.obs["Treatment"] == "aTrem2"]

    igg_gene_avg = calculate_gene_avg(igg_cells)
    atrem2_gene_avg = calculate_gene_avg(atrem2_cells)

    assert len(igg_gene_avg) == len(adata.var_names), (
        "Gene average array length mismatch!"
    )
    assert len(atrem2_gene_avg) == len(adata.var_names), (
        "Gene average array length mismatch!"
    )

    igg_idx = adata.obs["Treatment"] == "IgG"
    atrem2_idx = adata.obs["Treatment"] == "aTrem2"

    mean_expr_igg = adata.layers["X"][igg_idx].mean(axis=0)
    mean_expr_atrem2 = adata.layers["X"][atrem2_idx].mean(axis=0)

    epsilon = 1e-10
    fold_change = (mean_expr_atrem2 + epsilon) / (mean_expr_igg + epsilon)
    log2_fc = np.log2(fold_change)

    results_df = pd.DataFrame(
        {
            "gene": adata.var_names,
            "igg_avg": igg_gene_avg,
            "atrem2_avg": atrem2_gene_avg,
            "log2_fold_change": log2_fc,
        }
    )

    mask1 = (results_df["igg_avg"] > igg_thresh) | (
        results_df["atrem2_avg"] > atrem2_thresh
    )
    mask2 = abs(results_df["igg_avg"] - results_df["atrem2_avg"]) < balanced_thresh
    mask3 = abs(results_df["igg_avg"] - results_df["atrem2_avg"]) > biased_thresh

    results_df["group"] = "other"
    results_df.loc[mask1 & mask2, "group"] = "balanced"
    results_df.loc[mask1 & mask3, "group"] = "igg-preferential"

    print(f"Gene analysis complete. Total genes: {len(results_df)}")
    print(f"Balanced group: {len(results_df[results_df['group'] == 'balanced'])}")
    print(
        f"IgG-preferential group: {len(results_df[results_df['group'] == 'igg-preferential'])}"
    )

    return results_df


def getime_contrib(adata: AnnData, layer: Optional[str] = None) -> AnnData:
    if "getime_weights" not in adata.var.columns:
        raise ValueError("Column 'getime_weights' not found in adata.var")

    # Get expression matrix
    if layer and layer in adata.layers:
        X = adata.layers[layer]
    else:
        X = adata.X

    if X is None:
        raise ValueError("No expression matrix found in adata")

    getime_weights = adata.var["getime_weights"].values

    # Convert to array if sparse
    if hasattr(X, "toarray"):
        X_array = X.toarray()
    else:
        X_array = X

    origin_getime = X_array.dot(getime_weights)
    contributions = X_array * getime_weights

    pert_clock = origin_getime[:, np.newaxis] - contributions

    # Normalize to [0, 1]
    origin_norm = (origin_getime - origin_getime.min()) / (
        origin_getime.max() - origin_getime.min()
    )
    pert_norm = (pert_clock - pert_clock.min()) / (pert_clock.max() - pert_clock.min())

    gene_avg = np.linalg.norm(origin_norm[:, np.newaxis] - pert_norm, axis=0)
    gene_avg = (gene_avg - gene_avg.min()) / (gene_avg.max() - gene_avg.min())

    gene_peak = np.abs(origin_norm[:, np.newaxis] - pert_norm).max(axis=0)
    gene_peak = (gene_peak - gene_peak.min()) / (gene_peak.max() - gene_peak.min())

    adata.var["avg_contribution"] = gene_avg
    adata.var["peak_contribution"] = gene_peak

    return adata


def getime_violin_mono2tam(
    adata: AnnData,
    value_col: str,
    ylabel: Optional[str] = None,
    xlabel: Optional[str] = None,
    show_means: bool = False,
    title: Optional[str] = None,
    group_type: str = "cluster",
    time_key: str = "zmanseq_time",
    fontsize: int = 13,
) -> plt.Axes:
    import matplotlib.patheffects as pe

    if value_col not in adata.obs.columns:
        raise ValueError(f"Column '{value_col}' not found in adata.obs")

    if group_type not in ["cluster", "zmanseq_time"]:
        raise ValueError("group_type must be 'cluster' or 'zmanseq_time'")

    plt.rcParams["xtick.labelsize"] = fontsize
    plt.rcParams["ytick.labelsize"] = fontsize
    if group_type == "cluster":
        cluster_col = "Treatment_cluster"
        if cluster_col not in adata.obs.columns:
            raise ValueError(f"Column '{cluster_col}' not found in adata.obs")

        set_order = [
            "Monocytes_aTrem2",
            "MoMac1_aTrem2",
            "Acp5_TAM_aTrem2",
            "Monocytes_IgG",
            "MoMac2_IgG",
            "Arg1_TAM_IgG",
            "Gpnmb_TAM_IgG",
        ]

        color_dict = {
            "Acp5_TAM_aTrem2": "#75609c",
            "Arg1_TAM_IgG": "#bd2c8d",
            "Gpnmb_TAM_IgG": "#6e90b0",
            "MoMac1_aTrem2": "#f49472",
            "MoMac2_IgG": "#e2405b",
            "Monocytes_IgG": "#c34f13",
            "Monocytes_aTrem2": "#c34f13",
        }

        plot_data = []
        for group in set_order:
            group_data = adata[adata.obs[cluster_col] == group]
            for idx in range(len(group_data)):
                val = group_data.obs[value_col].iloc[idx]
                plot_data.append({cluster_col: group, value_col: val})

        plot_df = pd.DataFrame(plot_data)

        fig, ax = plt.subplots(figsize=(5, 3))

        color_list = [color_dict[cat] for cat in set_order]

        sns.violinplot(
            data=plot_df,
            x=cluster_col,
            y=value_col,
            order=set_order,
            ax=ax,
            inner=None,
            palette=color_list,
            linewidth=0,
        )

        ax.set_xticklabels([])
        ax.set_xticks([])
        ax.set_xlabel("")

        ax.axvline(x=2.5, color="black", linestyle="--", linewidth=1.5, alpha=0.8)

        ax.text(
            1,
            ax.get_ylim()[1] * 1.02,
            "aTrem2",
            ha="center",
            fontsize=fontsize + 1,
            fontweight="bold",
            color="darkred",
        )
        ax.text(
            4.5,
            ax.get_ylim()[1] * 1.02,
            "IgG",
            ha="center",
            fontsize=fontsize + 1,
            fontweight="bold",
            color="darkblue",
        )

        if show_means:
            for i, group in enumerate(set_order):
                group_mean = plot_df[plot_df[cluster_col] == group][value_col].mean()
                ax.plot(
                    [i - 0.3, i + 0.3],
                    [group_mean, group_mean],
                    color="black",
                    alpha=0.6,
                    linewidth=2,
                    zorder=10,
                )
                ax.scatter(
                    i,
                    group_mean,
                    color="maroon",
                    s=40,
                    zorder=10,
                    edgecolor="none",
                    linewidth=1.0,
                )
                ax.text(
                    i,
                    group_mean + (ax.get_ylim()[1] - ax.get_ylim()[0]) * 0.03,
                    f"{group_mean:.2f}",
                    ha="center",
                    va="bottom",
                    fontsize=fontsize,
                    color="black",
                    path_effects=[pe.withStroke(linewidth=0.5, alpha=0.85)],
                )

    elif group_type == "zmanseq_time":
        if time_key not in adata.obs.columns:
            raise ValueError(f"Column '{time_key}' not found in adata.obs")

        plot_data = []
        for time_val in sorted(adata.obs[time_key].unique()):
            time_data = adata[adata.obs[time_key] == time_val]
            for clock_val in time_data.obs[value_col]:
                plot_data.append({time_key: time_val, value_col: clock_val})

        plot_df = pd.DataFrame(plot_data)

        fig, ax = plt.subplots(figsize=(3, 3))

        time_nums = sorted(plot_df[time_key].unique())

        sns.violinplot(
            data=plot_df,
            x=time_key,
            y=value_col,
            order=time_nums,
            ax=ax,
            inner=None,
            linewidth=0,
            color="#A8B5C7",
        )

        ax.set_xlabel("")
        ax.set_xticks(range(len(time_nums)))
        ax.set_xticklabels([f"{int(t)}" for t in time_nums])

        if show_means:
            for i, time_val in enumerate(time_nums):
                time_mean = plot_df[plot_df[time_key] == time_val][value_col].mean()
                ax.plot(
                    [i - 0.3, i + 0.3],
                    [time_mean, time_mean],
                    color="black",
                    alpha=0.6,
                    linewidth=2,
                    zorder=10,
                )
                ax.scatter(
                    i,
                    time_mean,
                    color="maroon",
                    s=40,
                    zorder=10,
                    edgecolor="none",
                    linewidth=1.0,
                )
                ax.text(
                    i,
                    time_mean + (ax.get_ylim()[1] - ax.get_ylim()[0]) * 0.03,
                    f"{time_mean:.2f}",
                    ha="center",
                    va="bottom",
                    fontsize=fontsize,
                    color="black",
                    path_effects=[pe.withStroke(linewidth=0.5, alpha=0.85)],
                )

    if ylabel is not None:
        ax.set_ylabel(ylabel, fontsize=fontsize)
        if value_col == "zmanseq_time":
            min_val = plot_df[value_col].min()
            max_val = plot_df[value_col].max()
            time_ticks = [12, 24, 36, 48]
            time_ticks = [t for t in time_ticks if min_val <= t <= max_val]
            ax.set_yticks(time_ticks)
            ax.set_yticklabels([f"{t}" for t in time_ticks])
        else:
            ax.set_yticks([0, 0.5, 1])
            ax.set_yticklabels(["0.0", "0.5", "1.0"])
    if xlabel is not None:
        ax.set_xlabel(xlabel, fontsize=fontsize)

    if title:
        ax.set_title(title)
    plt.tight_layout()
    plt.show()

    return ax


def getime_density(
    adata: AnnData, xlabel: str = "Gene-embedded time", show_title: bool = True
) -> None:
    required_columns = ["Treatment_cluster"]
    if "zman" in xlabel.lower():
        data_column = "zmanseq_time"
    elif "reconstructed" in xlabel.lower():
        data_column = "remake_getime"
    else:
        data_column = "getime"

    required_columns.append(data_column)
    plt.rcParams["xtick.labelsize"] = 11
    plt.rcParams["ytick.labelsize"] = 11

    for col in required_columns:
        if col not in adata.obs.columns:
            raise ValueError(f"Column '{col}' not found in adata.obs")

    set_order = [
        "Monocytes_aTrem2",
        "Acp5_TAM_aTrem2",
        "Arg1_TAM_IgG",
        "Gpnmb_TAM_IgG",
        "Monocytes_IgG",
    ]

    for cluster in set_order:
        if cluster not in adata.obs["Treatment_cluster"].unique():
            raise ValueError(f"Cluster '{cluster}' not found in Treatment_cluster")

    extracted_data = {}
    for cluster in set_order:
        cluster_data = adata[adata.obs["Treatment_cluster"] == cluster]
        extracted_data[cluster] = cluster_data.obs[data_column].values.astype(float)

    merged_arg1_gpnmb = np.concatenate(
        [extracted_data["Arg1_TAM_IgG"], extracted_data["Gpnmb_TAM_IgG"]]
    )

    data_box1 = {
        "Monocytes_aTrem2": extracted_data["Monocytes_aTrem2"],
        "Acp5_TAM_aTrem2": extracted_data["Acp5_TAM_aTrem2"],
    }

    data_box2 = {
        "Monocytes_IgG": extracted_data["Monocytes_IgG"],
        "Arg1+Gpnmb_TAM_IgG": merged_arg1_gpnmb,
    }

    ovl1 = calculate_ovl(data_box1["Monocytes_aTrem2"], data_box1["Acp5_TAM_aTrem2"])
    ovl_dist1 = 1 - ovl1

    ovl2 = calculate_ovl(data_box2["Monocytes_IgG"], data_box2["Arg1+Gpnmb_TAM_IgG"])
    ovl_dist2 = 1 - ovl2

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 3.2))

    colors1 = ["#c34f13", "#75609c"]
    labels1 = ["Monocytes", "Acp5_TAM"]

    colors2 = ["#c34f13", "#A9A9A9"]
    labels2 = ["Monocytes", "Arg1+Gpnmb_TAM"]

    for i, (key, label, color) in enumerate(
        zip(["Monocytes_aTrem2", "Acp5_TAM_aTrem2"], labels1, colors1)
    ):
        sns.kdeplot(
            data_box1[key],
            ax=ax1,
            color=color,
            label=label,
            linewidth=2.0,
            fill=True,
            alpha=0.3,
        )

    ax1.set_xlabel(xlabel, fontsize=11, fontweight="bold")
    ax1.set_ylabel("Density", fontsize=11, fontweight="bold")

    if data_column in ["getime", "remake_getime"]:
        ax1.set_xlim(0, 1)

    ylim_ax1 = ax1.get_ylim()

    ax1.legend(
        loc="upper right",
        fontsize=11,
        frameon=False,
        borderpad=0.5,
        handlelength=0.6,
        handleheight=0.6,
        bbox_to_anchor=(1, 1),
    )

    ax1.text(
        0.03,
        0.95,
        f"OVL dist = {ovl_dist1:.2f}",
        transform=ax1.transAxes,
        ha="left",
        va="top",
        fontsize=11,
        fontweight="bold",
    )

    if show_title:
        ax1.set_title("aTrem2", fontsize=11, pad=10, fontweight="bold")

    for i, (key, label, color) in enumerate(
        zip(["Monocytes_IgG", "Arg1+Gpnmb_TAM_IgG"], labels2, colors2)
    ):
        sns.kdeplot(
            data_box2[key],
            ax=ax2,
            color=color,
            label=label,
            linewidth=2.0,
            fill=True,
            alpha=0.3,
        )

    ax2.set_xlabel(xlabel, fontsize=11, fontweight="bold")
    ax2.set_ylabel("Density", fontsize=11, fontweight="bold")

    if data_column in ["getime", "remake_getime"]:
        ax2.set_xlim(0, 1)

    ylim_ax2 = ax2.get_ylim()

    y_max = max(ylim_ax1[1], ylim_ax2[1]) * 1.3

    ax1.set_ylim(0, y_max)
    ax2.set_ylim(0, y_max)

    ax2.legend(
        loc="upper right",
        fontsize=10,
        frameon=False,
        borderpad=0.5,
        handlelength=0.6,
        handleheight=0.6,
        bbox_to_anchor=(1, 1),
    )

    ax2.text(
        0.03,
        0.95,
        f"OVL dist = {ovl_dist2:.2f}",
        transform=ax2.transAxes,
        ha="left",
        va="top",
        fontsize=10,
        fontweight="bold",
    )

    if show_title:
        ax2.set_title("IgG", fontsize=11, pad=10, fontweight="bold")

    if data_column == "zmanseq_time":
        ax1.set_xticks([12, 24, 36, 48])
        ax1.set_xticklabels(["12", "24", "36", "48"])
        ax2.set_xticks([12, 24, 36, 48])
        ax2.set_xticklabels(["12", "24", "36", "48"])

    for ax in (ax1, ax2):
        nbins = 2 if ax.get_ylim()[1] < 0.2 else 3
        ax.locator_params(axis="y", nbins=nbins)
        ax.set_yticklabels([f"{x:.1f}" for x in ax.get_yticks()])

    plt.tight_layout()
    plt.show()

    print(f"aTrem2 - OVL dist: {ovl_dist1:.2f}")
    print(f"IgG - OVL dist: {ovl_dist2:.2f}")


def plot_contrib_comparison(
    results_df: pd.DataFrame,
    x_label: str = "Average contribution in IgG",
    y_label: str = "Average contribution in aTrem2",
) -> None:
    import matplotlib.colors as mcolors
    from adjustText import adjust_text
    from matplotlib.lines import Line2D

    plt.rcParams["xtick.labelsize"] = 12
    plt.rcParams["ytick.labelsize"] = 12

    cmap = plt.cm.RdBu_r

    log2_fc = results_df["log2_fold_change"].values
    data_max = max(abs(log2_fc.min()), abs(log2_fc.max()))
    vmax = min(data_max, 3.0)
    vmin = -vmax

    norm = mcolors.TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)

    if vmax <= 1.5:
        tick_positions = [-1, 0, 1]
        tick_labels = ["-1.0", "0.0", "1.0"]
    else:
        tick_positions = [-2, 0, 2]
        tick_labels = ["-2.0", "0.0", "2.0"]

    fig, ax = plt.subplots(figsize=(6.5, 5.5))

    scatter = ax.scatter(
        results_df["igg_avg"],
        results_df["atrem2_avg"],
        c=results_df["log2_fold_change"],
        cmap=cmap,
        norm=norm,
        s=30,
        alpha=1,
        edgecolors="gray",
        linewidths=0.5,
    )

    ax.plot([0, 1], [0, 1], "k--", alpha=0.7, linewidth=1)

    ax.axvline(
        x=0.4,
        ymin=0,
        ymax=0.4,
        color="darkgray",
        linestyle="--",
        linewidth=1.5,
        alpha=0.7,
    )
    ax.axhline(
        y=0.4,
        xmin=0,
        xmax=0.4,
        color="darkgray",
        linestyle="--",
        linewidth=1.5,
        alpha=0.7,
    )

    ax.set_xlabel(x_label, fontsize=14)
    ax.set_ylabel(y_label, fontsize=14)
    ax.set_xlim(-0.01, 1.01)
    ax.set_ylim(-0.01, 1.01)

    texts = []

    balance_genes = results_df[results_df["group"] == "balanced"]
    for _, row in balance_genes.iterrows():
        text = ax.annotate(
            row["gene"],
            (row["igg_avg"], row["atrem2_avg"]),
            fontsize=12,
            alpha=0.9,
            color="black",
        )
        texts.append(text)

    bias_genes = results_df[results_df["group"] == "igg-preferential"]
    for _, row in bias_genes.iterrows():
        text = ax.annotate(
            row["gene"],
            (row["igg_avg"], row["atrem2_avg"]),
            fontsize=12,
            alpha=0.9,
            color="dodgerblue",
        )
        texts.append(text)

    if len(texts) > 0:
        adjust_text(
            texts,
            ax=ax,
            arrowprops=dict(arrowstyle="-", color="gray", lw=0.5, alpha=0.5),
            expand_points=(1.5, 1.5),
            force_text=(0.5, 0.5),
            force_points=(0.2, 0.2),
            lim=1000,
            precision=0.001,
        )

    legend_elements = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor="black",
            markersize=12,
            label=f"Balanced Genes ({len(balance_genes)})",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor="dodgerblue",
            markersize=12,
            label=f"IgG-Preferential Genes ({len(bias_genes)})",
        ),
    ]

    ax.legend(
        handles=legend_elements,
        loc="upper left",
        fontsize=13,
        frameon=False,
        title_fontsize=14,
    )

    # cax = fig.add_axes([0.9, 0.15, 0.02, 0.2])
    cax = fig.add_axes([0.88, 0.65, 0.02, 0.25])
    cbar = plt.colorbar(scatter, cax=cax)
    cbar.set_ticks(tick_positions)
    cbar.set_ticklabels(tick_labels)
    cbar.ax.tick_params(labelsize=12)

    plt.tight_layout(rect=[0, 0, 0.88, 1])
    plt.show()


def plot_gene_density(
    results_df: pd.DataFrame,
    x_label: str = "Average contribution in IgG",
    y_label: str = "Average contribution in aTrem2",
) -> None:
    from scipy.stats import gaussian_kde
    from matplotlib.colors import LinearSegmentedColormap
    import matplotlib.colors as mcolors

    plt.rcParams["xtick.labelsize"] = 18
    plt.rcParams["ytick.labelsize"] = 18

    x = results_df["igg_avg"].values
    y = results_df["atrem2_avg"].values

    total_genes = len(results_df)

    above_diagonal = results_df[results_df["atrem2_avg"] > results_df["igg_avg"]]
    above_count = len(above_diagonal)
    above_percentage = (above_count / total_genes) * 100

    below_diagonal = results_df[results_df["atrem2_avg"] < results_df["igg_avg"]]
    below_count = len(below_diagonal)
    below_percentage = (below_count / total_genes) * 100

    on_diagonal = results_df[results_df["atrem2_avg"] == results_df["igg_avg"]]
    on_count = len(on_diagonal)
    on_percentage = (on_count / total_genes) * 100 if on_count > 0 else 0

    print(f"Genes above diagonal: {above_count} ({above_percentage:.1f}%)")
    print(f"Genes below diagonal: {below_count} ({below_percentage:.1f}%)")
    if on_count > 0:
        print(f"Genes on diagonal: {on_count} ({on_percentage:.1f}%)")
    print(f"Total genes: {total_genes}")

    xy = np.vstack([x, y])
    kde = gaussian_kde(xy)

    x_min, x_max = x.min() - 0.05, x.max() + 0.05
    y_min, y_max = y.min() - 0.05, y.max() + 0.05

    x_min = max(x_min, -0.01)
    x_max = min(x_max, 1.01)
    y_min = max(y_min, -0.01)
    y_max = min(y_max, 1.01)

    xx, yy = np.mgrid[x_min:x_max:200j, y_min:y_max:200j]
    positions = np.vstack([xx.ravel(), yy.ravel()])
    z = np.reshape(kde(positions).T, xx.shape)

    fig, ax = plt.subplots(figsize=(5.9, 4.8))

    colors_white_blue = ["white", "lightblue", "blue", "darkblue"]
    cmap_white_blue = LinearSegmentedColormap.from_list(
        "white_blue", colors_white_blue, N=256
    )

    z_min, z_max = z.min(), z.max()
    density_vmax = min(z_max, z_min + 3 * (z_max - z_min) / 4)

    ax.plot([0, 1], [0, 1], "k--", alpha=0.7, linewidth=2, zorder=4)

    ax.set_xlabel(x_label, fontsize=18)
    ax.set_ylabel(y_label, fontsize=18)
    ax.set_xlim(-0.01, 1.01)
    ax.set_ylim(-0.01, 1.01)
    ax.set_aspect("equal", adjustable="box")
    ticks = np.round(np.linspace(0, 1, 6), 2)
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)

    ax.text(
        0.02,
        0.98,
        f"{above_percentage:.1f}%",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=18,
        fontweight="bold",
        color="darkblue",
    )

    ax.text(
        0.98,
        0.02,
        f"{below_percentage:.1f}%",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=18,
        fontweight="bold",
        color="darkblue",
    )

    # cax = fig.add_axes([0.88, 0.15, 0.02, 0.2])
    cax = fig.add_axes([0.88, 0.65, 0.02, 0.25])
    norm_density = mcolors.Normalize(vmin=z_min, vmax=density_vmax)
    sm = plt.cm.ScalarMappable(cmap=cmap_white_blue, norm=norm_density)
    sm.set_array([])

    cbar = plt.colorbar(sm, cax=cax)

    if density_vmax <= 5:
        tick_locs = np.linspace(z_min, density_vmax, 3)
        tick_labels = [f"{tick:.1f}" for tick in tick_locs]
    else:
        tick_locs = [z_min, density_vmax]
        tick_labels = [f"{z_min:.1f}", f"{density_vmax:.1f}"]

    cbar.set_ticks(tick_locs)
    cbar.set_ticklabels(tick_labels)
    cbar.ax.tick_params(labelsize=18)

    plt.tight_layout(rect=[0, 0, 0.9, 1])
    plt.show()


def thresh_highlight_genes(
    adata: AnnData,
    avg_thresh: float = 0.12,
    peak_thresh: float = 0.13,
    save_path: str = None,
    figsize: Tuple[int, int] = (5, 5),
) -> np.ndarray:
    if "avg_contribution" not in adata.var.columns:
        raise ValueError("Column 'avg_contribution' not found in adata.var")
    if "peak_contribution" not in adata.var.columns:
        raise ValueError("Column 'peak_contribution' not found in adata.var")

    plt.rcParams["xtick.labelsize"] = 12
    plt.rcParams["ytick.labelsize"] = 12
    mask = (adata.var["avg_contribution"] > avg_thresh) & (
        adata.var["peak_contribution"] > peak_thresh
    )

    fig, ax = plt.subplots(figsize=figsize)

    ax.scatter(
        adata.var["avg_contribution"],
        adata.var["peak_contribution"],
        c="gray",
        alpha=0.6,
        s=15,
    )

    ax.scatter(
        adata.var.loc[mask, "avg_contribution"],
        adata.var.loc[mask, "peak_contribution"],
        c="red",
        s=15,
    )

    ax.axvline(x=avg_thresh, color="black", linestyle="--", linewidth=1.5)
    ax.axhline(y=peak_thresh, color="black", linestyle="--", linewidth=1.5)
    ax.tick_params(axis="both", labelsize=12)

    ax.set_xlabel("Average contribution", fontsize=14)
    ax.set_ylabel("Peak contribution", fontsize=14)
    # ax.set_title("Gene Contribution Metrics",fontsize=13)

    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches="tight", pad_inches=0)
    plt.show()

    return mask


def plot_fateprobabilities(
    adata: AnnData,
    fate_key: str = "merged_fate_probs",
    embedding: str = "tsne",
    figsize: Tuple[int, int] = (6, 4),
    colors: Optional[List[str]] = None,
    legend_loc: str = "lower right",
    point_size: int = 150,
) -> None:
    embedding_key = f"X_{embedding}" if not embedding.startswith("X_") else embedding

    if embedding_key not in adata.obsm:
        raise ValueError(
            f"Embedding coordinates '{embedding_key}' not found in adata.obsm"
        )

    if fate_key not in adata.obsm:
        raise ValueError(f"Fate probabilities '{fate_key}' not found in adata.obsm")

    if colors is None:
        colors = ["#5a6b98", "#6b886d"]

    fate_probs = adata.obsm[fate_key]

    scv.pl.scatter(
        adata,
        basis=embedding,
        color_gradients=fate_probs,
        palette=colors,
        title="",
        show=False,
        figsize=figsize,
        size=point_size,
        legend_loc=legend_loc,
    )

    plt.tight_layout()
    plt.show()


def plot_lineage_fateprobabilities(
    adata: AnnData,
    lineage_col: str = "reprogramming",
    success_col: str = "successful_state_prob",
    dead_end_col: str = "dead_end_state_prob",
    figsize: Tuple[int, int] = (4, 3.5),
    colors: List[str] = None,
    bar_width: float = 0.3,
    inner_gap: float = 0.4,
    group_gap: float = 0.8,
    show_pvalues: bool = True,
) -> None:
    required_cols = [lineage_col, success_col, dead_end_col]
    for col in required_cols:
        if col not in adata.obs.columns:
            raise ValueError(f"Column '{col}' not found in adata.obs")

    if lineage_col == "reprogramming":
        if "Reprogramming Lineage" not in adata.obs[lineage_col].unique():
            raise ValueError("'Reprogramming Lineage' not found in lineage column")
        if "Dead-end Lineage" not in adata.obs[lineage_col].unique():
            raise ValueError("'Dead-end Lineage' not found in lineage column")

    if colors is None:
        colors = ["#ADD8E6", "#FFDAB9"]

    if len(colors) < 2:
        raise ValueError("colors list must contain at least 2 colors")

    adata_reprogramming = adata[
        adata.obs[lineage_col] == "Reprogramming Lineage"
    ].copy()
    adata_dead_end = adata[adata.obs[lineage_col] == "Dead-end Lineage"].copy()

    print(f"Reprogramming lineage data shape: {adata_reprogramming.shape}")
    print(f"Dead-end lineage data shape: {adata_dead_end.shape}")

    fig, ax = plt.subplots(figsize=figsize)

    data = [
        adata_reprogramming.obs[success_col],
        adata_dead_end.obs[success_col],
        adata_reprogramming.obs[dead_end_col],
        adata_dead_end.obs[dead_end_col],
    ]

    left_margin = 0.8
    positions = [
        left_margin + bar_width / 2,
        left_margin + bar_width / 2 + bar_width + inner_gap,
        left_margin + bar_width / 2 + bar_width + inner_gap + bar_width + group_gap,
        left_margin
        + bar_width / 2
        + bar_width
        + inner_gap
        + bar_width
        + group_gap
        + bar_width
        + inner_gap,
    ]

    box = ax.boxplot(
        data, positions=positions, patch_artist=True, widths=bar_width, showfliers=False
    )

    for i, box_item in enumerate(box["boxes"]):
        box_item.set_facecolor(colors[i % 2])
        box_item.set_edgecolor(colors[i % 2])
        box_item.set_linewidth(1)

    for median in box["medians"]:
        median.set_color("black")
        median.set_linewidth(1.5)

    for whisker in box["whiskers"]:
        whisker.set_color("black")
        whisker.set_linewidth(1)

    for cap in box["caps"]:
        cap.set_color("black")
        cap.set_linewidth(1)

    separator_x = (positions[1] + positions[2]) / 2
    ax.axvline(x=separator_x, color="gray", linestyle="--", linewidth=1.5, alpha=1)

    ax.text(
        (positions[0] + positions[1]) / 2,
        1.02,
        "Towards \n cluster 1",
        ha="center",
        va="bottom",
        transform=ax.get_xaxis_transform(),
        fontweight="bold",
        fontsize=11,
        color="#5a6b98",
    )
    ax.text(
        (positions[2] + positions[3]) / 2,
        1.02,
        "Towards \n cluster 3",
        ha="center",
        va="bottom",
        transform=ax.get_xaxis_transform(),
        fontweight="bold",
        fontsize=11,
        color="#6b886d",
    )
    ax.tick_params(axis="both", labelsize=11)

    from matplotlib.transforms import blended_transform_factory

    if show_pvalues:
        # ---- 左侧 p ----
        if len(data[0]) and len(data[1]):
            stat1, pval1 = stats.mannwhitneyu(data[0], data[1], alternative="greater")
            p_text = (
                r"$p < 10^{-16}$"
                if pval1 == 0 or pval1 < 1e-16
                else rf"$p = 10^{{{int(np.floor(np.log10(pval1)))}}}$"
                if pval1 < 1e-3
                else r"$p = 1.00$"
                if pval1 >= 0.999
                else rf"$p = {pval1:.3f}$"
                if pval1 < 0.1
                else rf"$p = {pval1:.2f}$"
            )
            ax.text(
                0.05,
                0.95,
                p_text,
                transform=ax.transAxes,
                fontsize=11,
                ha="left",
                va="top",
            )

        # ---- 右侧 p ----
        if len(data[2]) and len(data[3]):
            stat2, pval2 = stats.mannwhitneyu(data[2], data[3], alternative="less")
            p_text = (
                r"$p < 10^{-16}$"
                if pval2 == 0 or pval2 < 1e-16
                else rf"$p = 10^{{{int(np.floor(np.log10(pval2)))}}}$"
                if pval2 < 1e-3
                else r"$p = 1.00$"
                if pval2 >= 0.999
                else rf"$p = {pval2:.3f}$"
                if pval2 < 0.1
                else rf"$p = {pval2:.2f}$"
            )
            trans = blended_transform_factory(ax.transData, ax.transAxes)
            ax.text(
                separator_x + 0.06 * (ax.get_xlim()[1] - ax.get_xlim()[0]),
                0.95,
                p_text,
                transform=trans,
                fontsize=11,
                ha="left",
                va="top",
            )

    ax.set_ylabel("Probability", fontsize=11)
    ax.set_ylim(-0.05, 1.3)

    x_min = positions[0] - bar_width / 2 - 0.3
    x_max = positions[3] + bar_width / 2 + 0.3
    ax.set_xlim(x_min, x_max)

    ax.grid(False)
    ax.set_xticklabels([])
    ax.set_xticks([])

    plt.tight_layout()
    plt.show()


def plot_submacrostates(
    adata: AnnData,
    gpcca,
    show_initial: bool = True,
    show_terminal: bool = True,
    embedding: str = "tsne",
    figsize: Tuple[int, int] = (8, 5.5),
) -> None:
    embedding_key = f"X_{embedding}" if not embedding.startswith("X_") else embedding

    if embedding_key not in adata.obsm:
        raise ValueError(
            f"Embedding coordinates '{embedding_key}' not found in adata.obsm"
        )

    if not hasattr(gpcca, "initial_states") or not hasattr(gpcca, "terminal_states"):
        raise ValueError(
            "GPCCA object must have 'initial_states' and 'terminal_states' attributes"
        )

    coords = adata.obsm[embedding_key]

    initial_indices = []
    terminal_indices = []

    if show_initial:
        initial_cells = gpcca.initial_states.dropna().index
        initial_indices = [adata.obs.index.get_loc(cell) for cell in initial_cells]

    if show_terminal:
        terminal_cells = gpcca.terminal_states.dropna().index
        terminal_indices = [adata.obs.index.get_loc(cell) for cell in terminal_cells]

    all_highlighted = initial_indices + terminal_indices
    other_indices = np.setdiff1d(range(adata.n_obs), all_highlighted)

    fig, ax = plt.subplots(figsize=figsize)
    ax.axis("off")

    ax.scatter(
        coords[other_indices, 0],
        coords[other_indices, 1],
        c="gray",
        s=30,
        alpha=0.3,
        edgecolors="none",
        rasterized=True,
    )

    if show_initial and len(initial_indices) > 0:
        ax.scatter(
            coords[initial_indices, 0],
            coords[initial_indices, 1],
            c="red",
            s=60,
            alpha=0.9,
            edgecolors="black",
            linewidth=1,
            rasterized=True,
            label="Initial",
        )

    if show_terminal and len(terminal_indices) > 0:
        ax.scatter(
            coords[terminal_indices, 0],
            coords[terminal_indices, 1],
            c="blue",
            s=60,
            alpha=0.9,
            edgecolors="black",
            linewidth=1,
            rasterized=True,
            label="Terminal",
        )

    if (show_initial and len(initial_indices) > 0) or (
        show_terminal and len(terminal_indices) > 0
    ):
        ax.legend(
            loc="lower right",
            frameon=False,
            markerscale=1,
            fontsize=13,
            bbox_to_anchor=(1.1, -0.05),
        )
        # ax.legend(loc='upper left',
        #   bbox_to_anchor=(0.75, 1),
        #   frameon=False, markerscale=1, fontsize=13)
    # plt.tight_layout()

    plt.show()


def macrostates_violin(
    adata: AnnData,
    gpcca=None,
    clock_key: str = "getime",
    figsize: Tuple[int, int] = (2, 4),
    show_initial: bool = True,
    show_terminal: bool = True,
    jitter_std: float = 0.08,
    violin_width: float = 0.32,
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Plot violin plot with initial and/or terminal macrostates highlighted.

    """

    if clock_key not in adata.obs.columns:
        raise ValueError(f"Column '{clock_key}' not found in adata.obs")

    if gpcca is not None:
        if show_initial and not hasattr(gpcca, "initial_states"):
            raise ValueError(
                "GPCCA object must have 'initial_states' attribute when show_initial=True"
            )

        if show_terminal and not hasattr(gpcca, "terminal_states"):
            raise ValueError(
                "GPCCA object must have 'terminal_states' attribute when show_terminal=True"
            )

        if not show_initial and not show_terminal:
            print(
                "Warning: Both show_initial and show_terminal are False, no states will be highlighted"
            )

    fig, ax = plt.subplots(figsize=figsize)

    fig.patch.set_alpha(0)
    ax.patch.set_alpha(0)

    parts = ax.violinplot(
        [adata.obs[clock_key].values],
        positions=[0],
        widths=violin_width,
        showmeans=True,
        showmedians=False,
    )

    pc = parts["bodies"][0]
    pc.set_facecolor("gray")
    pc.set_alpha(0.6)
    pc.set_edgecolor("black")
    pc.set_linewidth(0.5)

    parts["cmeans"].set_color("black")
    parts["cmeans"].set_linewidth(1.5)
    parts["cmaxes"].set_color("black")
    parts["cmaxes"].set_linewidth(1)
    parts["cmins"].set_color("black")
    parts["cmins"].set_linewidth(1)
    parts["cbars"].set_color("black")
    parts["cbars"].set_linewidth(1)

    initial_times = []
    terminal_times = []

    if gpcca is not None:
        if show_initial:
            initial_cells = gpcca.initial_states.dropna().index.intersection(
                adata.obs.index
            )
            initial_times = adata.obs.loc[initial_cells, clock_key]

        if show_terminal:
            terminal_cells = gpcca.terminal_states.dropna().index.intersection(
                adata.obs.index
            )
            terminal_times = adata.obs.loc[terminal_cells, clock_key]

    np.random.seed(42)

    def get_jitter(data):
        return np.random.normal(0, jitter_std, size=len(data))

    if show_initial and len(initial_times) > 0:
        ax.scatter(
            get_jitter(initial_times),
            initial_times,
            color="red",
            alpha=0.9,
            s=30,
            zorder=3,
            edgecolors="black",
            linewidth=0.5,
        )

    if show_terminal and len(terminal_times) > 0:
        ax.scatter(
            get_jitter(terminal_times),
            terminal_times,
            color="blue",
            alpha=0.9,
            s=30,
            zorder=3,
            edgecolors="black",
            linewidth=0.5,
        )

    ax.set_xlim(-0.5, 0.5)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel("")

    for spine in ax.spines.values():
        spine.set_visible(False)

    # arrow_start_y = 0.1
    # arrow_end_y = 0.35
    # original_dy = (arrow_start_y - arrow_end_y)
    # arrow_length = abs(original_dy * 3)

    # arrow_tip_y = arrow_length
    # arrow_tail_y = 0.0

    # ax.annotate(
    #     "",
    #     xy=(0.85, arrow_tip_y),      # Arrow tip points to this position (Top)
    #     xytext=(0.85, arrow_tail_y), # Arrow starts from this position (Bottom)
    #     xycoords="axes fraction",
    #     arrowprops=dict(arrowstyle="->", color="black",
    #                     lw=2.0,
    #                     shrinkA=0, shrinkB=0),
    # )

    # ax.text(
    #     0.95,
    #     arrow_tail_y,              # Align text with the arrow tail (bottom)
    #     "Gene-embedded time",
    #     rotation=90,
    #     transform=ax.transAxes,
    #     fontsize=20,
    #     va="bottom",                # Anchor text at the bottom so it grows upwards
    #     ha="center",
    # )

    plt.tight_layout()
    plt.show()

    return fig, ax


def plot_getime_correlation(
    adata: AnnData,
    time_key: str = "getime",
    remake_key: str = "remake_getime",
    color_key: str = "celltype",
    figsize: Tuple[int, int] = (4, 4),
) -> float:
    if time_key not in adata.obs.columns:
        raise ValueError(f"Column '{time_key}' not found in adata.obs")
    if remake_key not in adata.obs.columns:
        raise ValueError(f"Column '{remake_key}' not found in adata.obs")

    correlation, p_value = stats.pearsonr(adata.obs[time_key], adata.obs[remake_key])

    fig, ax = plt.subplots(figsize=figsize)

    palette = None
    if color_key in adata.uns and f"{color_key}_colors" in adata.uns:
        palette = adata.uns[f"{color_key}_colors"]

    sc.pl.scatter(
        adata,
        x=time_key,
        y=remake_key,
        color=color_key,
        title="",
        show=False,
        legend_loc="none",
        palette=palette,
        ax=ax,
    )

    if p_value == 0.0 or p_value < 1e-16:
        p_txt = r"$< 10^{-16}$"
    elif p_value < 1e-3:
        p_txt = rf"$= 10^{{{int(np.floor(np.log10(p_value)))}}}$"
    elif p_value >= 0.999:
        p_txt = r"$= 1.00$"
    elif p_value < 0.1:
        p_txt = rf"$= {p_value:.3f}$"
    else:
        p_txt = rf"$= {p_value:.2f}$"

    ax.text(
        0.05,
        0.95,
        rf"Pearson $r={correlation:.2f}$" + "\n" + rf"$p$-value {p_txt}",
        transform=ax.transAxes,
        fontsize=17,
        verticalalignment="top",
    )
    ax.tick_params(axis="both", labelsize=17)

    plt.ylabel("Reconstructed \n gene-embedded time", fontsize=18)
    plt.xlabel("Gene-embedded time", fontsize=18)
    plt.tight_layout()
    plt.show()

    return correlation


def batch_velocity_dynamics(
    adata: AnnData,
    gene_list: List[str],
    color_key: str = "celltype",
    magnitude_scale: float = 1.0,
    save_directory: str = "./celldyc_output",
    show_figure: bool = False,
) -> None:
    """
    Batch process and save phase portrait plots for multiple genes.

    Parameters
    ----------
    adata : AnnData
        AnnData object with velocity data.
    gene_list : list of str
        List of gene names to plot.
    color_key : str, optional
        Column name for cell type coloring. Default is 'celltype'.
    magnitude_scale : float, optional
        Scaling factor for velocity magnitude. Default is 1.0.
    show_figure : bool, optional
        Whether to display each figure. Default is False.
    save_directory : str, optional
        Directory to save the output figures. Default is './celldyc_output'.

    Returns
    -------
    None
        Saves phase portrait plots to the specified directory.

    Raises
    ------
    ValueError
        If adata is missing required layers or columns.
    """

    if save_directory and not os.path.exists(save_directory):
        os.makedirs(save_directory, exist_ok=True)

    successful_genes = []
    failed_genes = []

    for gene_name in gene_list:
        try:
            if gene_name not in adata.var_names:
                print(f"Warning: Gene '{gene_name}' not found in adata.var_names")
                failed_genes.append(gene_name)
                continue

            save_path = f"{save_directory}/{gene_name}_phase_portrait.svg"

            velocity_dynamics(
                adata,
                gene_name=gene_name,
                color_key=color_key,
                save_path=save_path,
                show_figure=show_figure,
                magnitude_scale=magnitude_scale,
                fontsize=25,
            )

            successful_genes.append(gene_name)

        except Exception as e:
            print(f"Error processing gene '{gene_name}': {e}")
            failed_genes.append(gene_name)

    print("\nBatch processing complete.")
    print(f"Successfully processed: {len(successful_genes)} genes")
    if failed_genes:
        print(f"Failed genes: {len(failed_genes)}")
        if len(failed_genes) <= 10:
            print(f"  {failed_genes}")

    if save_directory:
        print(f"Figures saved to: {os.path.abspath(save_directory)}")


def batch_velocity_embedding(
    adata: AnnData,
    gene_list: List[str],
    layer: str = "velocity",
    basis: str = "umap",
    show_plot: bool = False,
    dpi: int = 300,
    save_format: str = "png",
    save_directory: str = "./celldyc_output",
) -> None:
    """
    Generates velocity embedding plots for a list of genes
    and saves them to a specified directory.

    Parameters
    ----------
    adata : AnnData
        AnnData object with embedding coordinates and velocity data.
    gene_list : list of str
        List of gene names to plot.
    layer : str, optional
        Layer containing gene velocity data. Default is "velocity".
    basis : str, optional
        Embedding basis to use (e.g., 'umap', 'tsne'). Default is 'umap'.
    dpi : int, optional
        Resolution for saved figures. Default is 300.
    save_format : str, optional
        File format for saving figures. Default is 'png'.
    show_plot : bool, optional
        Whether to display each figure. Default is False.
    save_directory : str, optional
        Directory to save the output figures. Default is './celldyc_output'.

    Returns
    -------
    None
        Saves embedding plots to the specified directory.

    Raises
    ------
    ValueError
        If adata is missing required layers or embeddings.
    """
    import os

    if layer not in adata.layers:
        raise ValueError(f"Layer '{layer}' not found in adata.layers")

    embedding_key = f"X_{basis}" if not basis.startswith("X_") else basis
    if embedding_key not in adata.obsm:
        raise ValueError(
            f"Embedding coordinates '{embedding_key}' not found in adata.obsm"
        )

    if save_directory and not os.path.exists(save_directory):
        os.makedirs(save_directory, exist_ok=True)

    successful_genes = []
    failed_genes = []

    for gene_name in gene_list:
        try:
            if gene_name not in adata.var_names:
                print(f"Warning: Gene '{gene_name}' not found in adata.var_names")
                failed_genes.append(gene_name)
                continue

            save_path = f"{save_directory}/{gene_name}_dynamic_{basis}.{save_format}"

            velocity_hot(
                adata=adata,
                gene=gene_name,
                basis=basis,
                layer=layer,
                save_path=save_path,
                save_format=save_format,
                dpi=dpi,
                show=show_plot,
                fontsize=23,
            )

            successful_genes.append(gene_name)

        except Exception as e:
            print(f"Error processing gene '{gene_name}': {e}")
            failed_genes.append(gene_name)

    print("\nBatch processing complete.")
    print(f"Successfully processed: {len(successful_genes)} genes")
    if failed_genes:
        print(f"Failed genes: {len(failed_genes)}")
        if len(failed_genes) <= 10:
            print(f"  {failed_genes}")

    if save_directory:
        print(f"Figures saved to: {os.path.abspath(save_directory)}")


def remake_getime(
    adata: AnnData,
    mask: np.ndarray,
) -> AnnData:
    """
    Reconstructs gene-embedded time from a pre-selected gene set.

    Parameters
    ----------
    adata : AnnData
        Single-cell data object with calculated gene-embedded time weights, and expression matrix in layers['X'].
    genes : np.ndarray
        Boolean mask for gene selection. Genes marked True will be included.

    Returns
    -------
    AnnData
        Updated AnnData object with 'remake_getime' in obs.

    Raises
    ------
    ValueError
        If required columns are not found in adata.var.
        If required layer 'X' is not found in adata.layers.
    """
    if "getime_weights" not in adata.var.columns:
        raise ValueError("Column 'getime_weights' not found in adata.var")

    if "X" not in adata.layers:
        raise ValueError("Layer 'X' not found in adata.layers")

    adata = adata[:, mask].copy()
    print(f"Selected data shape: {adata.shape}")

    getime_weights = adata.var["getime_weights"].values
    origin_getime_w = np.dot(adata.layers["X"], getime_weights)

    origin_getime_w = (origin_getime_w - origin_getime_w.min()) / (
        origin_getime_w.max() - origin_getime_w.min()
    )
    adata.obs["remake_getime"] = origin_getime_w

    return adata


def calculate_lineage_velocity(
    adata: AnnData, lineage_key: str = "lineage"
) -> np.ndarray:
    """
    Computes velocities based on parent-child relationships defined by lineage annotations.

    Parameters
    ----------
    adata : AnnData
        Input AnnData object with lineage annotations.
    lineage_key : str, optional
        Column name in obs containing lineage information. Default is 'lineage'.

    Returns
    -------
    np.ndarray
        Velocity matrix with shape (n_cells, n_genes).

    Raises
    ------
    ValueError
        If lineage_key is not found in adata.obs.
        If lineage data is not in expected format.
    """

    if lineage_key not in adata.obs.columns:
        raise ValueError(f"Column '{lineage_key}' not found in adata.obs")

    if adata.obs[lineage_key].isnull().any():
        raise ValueError(f"Column '{lineage_key}' contains missing values")

    lineage_to_idx = {}
    for idx, lineage in enumerate(adata.obs[lineage_key]):
        base_lineage = (
            str(lineage).split("/")[0] if isinstance(lineage, str) else str(lineage)
        )
        lineage_to_idx[base_lineage] = idx

    anc_lineages = []
    for lineage in adata.obs[lineage_key]:
        lineage_str = str(lineage)

        if lineage_str[:-1] in lineage_to_idx:
            anc_lineages.append(lineage_str[:-1])
        elif len(lineage_str) >= 2 and lineage_str[:-2] in lineage_to_idx:
            anc_lineages.append(lineage_str[:-2])
        else:
            anc_lineages.append(lineage_str.split("/")[0])

    child_idx = np.arange(len(adata))
    anc_idx = np.array(
        [lineage_to_idx.get(parent, i) for i, parent in enumerate(anc_lineages)]
    )

    if hasattr(adata.X, "toarray"):
        X = adata.X.toarray()
    else:
        X = adata.X.copy()

    velocity = X[anc_idx, :] - X[child_idx, :]

    return velocity


def reduce_timepoint(
    adata: AnnData,
    time_key: str = "time_point",
    num: int = 6,
    color: Optional[str] = None,
    basis: str = "umap",
    color_map: str = "gnuplot2",
) -> AnnData:
    """
    Downsamples single-cell data by retaining a specified number of time points.

    Parameters
    ----------
    adata : AnnData
        Input single-cell data.
    time_key : str, optional
        Column name in `adata.obs` containing time information. Default is 'time_point'.
    num : int, optional
        Number of time points to keep.
        If num is 3, 6 or 9, evenly spaced points are selected.
        Otherwise, random sampling is used. Default is 6.
    color : str, optional
        Column name in `adata.obs` for visualization color. If None, uses `time_key`.
        Default is None.
    basis : str, optional
        Embedding basis for visualization (e.g., 'umap', 'tsne').
        Default is 'umap'.
    color_map : str, optional
        Color map for visualization. Default is 'gnuplot2'.

    Returns
    -------
    AnnData
        Downsampled data.

    Raises
    ------
    ValueError
        If `time_key` is not found in `adata.obs`.
        If `num` exceeds the total number of time points.
    """

    if time_key not in adata.obs.columns:
        raise ValueError(f"Time key '{time_key}' not found in adata.obs")

    sc.pp.filter_genes(adata, min_cells=10)
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    sc.pp.highly_variable_genes(adata, n_top_genes=2000)
    adata = adata[:, adata.var.highly_variable].copy()

    categories = adata.obs[time_key].cat.categories

    if num in {3, 6, 9}:
        if num == 6:
            keep_indices = np.arange(0, len(categories), 2)
        elif num == 3:
            keep_indices = np.arange(0, len(categories), 4)
        else:  # 9
            keep_indices = np.linspace(0, len(categories) - 1, 9, dtype=int)
    else:
        if num > len(categories):
            raise ValueError(
                f"num={num} exceeds total time points ({len(categories)})."
            )
        keep_indices = np.random.choice(len(categories), size=num, replace=False)

    keep_categories = categories[keep_indices]

    mask = adata.obs[time_key].isin(keep_categories)
    adata_filtered = adata[mask].copy()

    plot_color = color or time_key
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    sc.pl.embedding(
        adata,
        color=plot_color,
        basis=basis,
        color_map=color_map,
        show=False,
        ax=ax1,
        title=f"Original Data\n{adata.shape[0]} cells",
    )

    sc.pl.embedding(
        adata_filtered,
        color=plot_color,
        basis=basis,
        color_map=color_map,
        show=False,
        ax=ax2,
        title=f"Downsampled Data (n={num})\n{adata_filtered.shape[0]} cells",
    )

    for ax in [ax1, ax2]:
        ax.set_xlabel("")
        ax.set_ylabel("")
        for spine in ax.spines.values():
            spine.set_visible(False)

    plt.tight_layout()
    plt.show()

    print(f"Downsampled data (n={num}).")

    sc.tl.pca(adata_filtered, svd_solver="arpack")
    sc.pp.neighbors(adata_filtered, n_neighbors=30, n_pcs=30)

    return adata_filtered
