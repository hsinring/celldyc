import scanpy as sc
import numpy as np
import pandas as pd
import scipy.stats as st
import seaborn as sns
import matplotlib.pyplot as plt
import scvelo as scv
import os
from typing import List, Tuple, Optional
from anndata import AnnData

from celldyc.tools.core import velocity_graph


def plot_velocity_projection(
    adata: AnnData,
    velocity_key: str = "velocity",
    basis: str = "umap",
    xkey: Optional[str] = None,
    color: Optional[str] = None,
    legend_loc: str = "on data",
    title: str = "",
    show: bool = True,
    figsize: Tuple[int, int] = (8, 6),
    size: Optional[int] = None,
    cmap: Optional[str] = None,
    alpha: float = 0.3,
    colorbar: bool = True,
    palette: Optional[str] = None,
    n_jobs: int = 10,
    graph_T: bool = False,
) -> None:
    """
    Computes a velocity graph from the calculated velocity vectors and projects it as a stream plot on the embedding.

    Parameters
    ----------
    adata : AnnData
        Input AnnData object with velocities data and embedding coordinates.
    velocity_key : str, optional
        Layer containing velocities data. Default is 'velocity'.
    basis : str, optional
        Embedding basis to use (e.g., 'umap', 'tsne'). Default is 'umap'.
    xkey : str, optional
        Layer to use as expression data.
        If None, uses adata.X converted to dense array. Default is None.
    color : str, optional
        Column name in obs for coloring points. If None, no coloring is applied.
        Default is None.
    legend_loc : str, optional
        Location of legend. Default is 'on data'.
    title : str, optional
        Plot title. Default is empty string.
    show : bool, optional
        Whether to display the plot. Default is True.
    figsize : tuple of int, optional
        Figure size (width, height) in inches. Default is (8, 6).
    size : int, optional
        Size of points in scatter plot. If None, uses default. Default is None.
    cmap : str, optional
        Colormap for coloring. If None, uses default. Default is None.
    alpha : float, optional
        Transparency of points. Default is 0.3.
    colorbar : bool, optional
        Whether to show colorbar. Default is True.
    palette : str, optional
        Color palette for categorical coloring. If None, uses default.
        Default is None.
    n_jobs : int, optional
        Number of jobs for parallel computation. Default is 10.
    graph_T : bool, optional
        If True, transpose the adjacency matrix. Default is False.

    Returns
    -------
    None
        Shows the velocity projection plot.

    Raises
    ------
    ValueError
        If required layers or embeddings are not found.
    """

    if velocity_key not in adata.layers:
        raise ValueError(f"Layer '{velocity_key}' not found in adata.layers")

    embedding_key = f"X_{basis}" if not basis.startswith("X_") else basis
    if embedding_key not in adata.obsm:
        raise ValueError(
            f"Embedding coordinates '{embedding_key}' not found in adata.obsm"
        )

    if xkey is None:
        xkey = "X"
        if hasattr(adata.X, "toarray"):
            adata.layers["X"] = adata.X.toarray()
    elif xkey not in adata.layers:
        raise ValueError(f"Layer '{xkey}' not found in adata.layers")

    if color is not None and color not in adata.obs.columns:
        raise ValueError(f"Column '{color}' not found in adata.obs")

    velocity_graph(adata, vkey=velocity_key, xkey=xkey, n_jobs=n_jobs)

    graph_key = f"{velocity_key}_graph"
    if graph_key not in adata.uns:
        raise KeyError(graph_key)
    if graph_T is True:
        adata.uns[graph_key] = adata.uns[graph_key].T

    velocity_embedding_stream(
        adata,
        basis=basis,
        vkey=velocity_key,
        color=color,
        legend_loc=legend_loc,
        title=title,
        show=show,
        figsize=figsize,
        size=size,
        cmap=cmap,
        alpha=alpha,
        colorbar=colorbar,
        palette=palette,
    )


def getime_violin(
    adata: AnnData,
    key_cont: str,
    key_cat: str,
    rotation: float = 45,
    palette: Optional[str] = "tab10",
    stripplot: bool = False,
    xlabel: str = "Time point",
    ylabel: Optional[str] = None,
    fontsize: int = 12,
    remove_ticks: bool = True,
    figsize: Tuple[int, int] = (3, 3),
    ax: Optional[plt.Axes] = None,
    show: bool = True,
    arrow_size: float = 15,
    arrow_style: str = "->",
    **kwargs,
) -> plt.Axes:
    """
    Creates violin plots for a variable across different categories.

    Parameters
    ----------
    adata : AnnData
        Input AnnData object.
    key_cont : str
        Variable column name in adata.obs for y-axis values.
    key_cat : str
        Categorical column name in adata.obs for grouping.
    rotation : float, optional
        Rotation angle for x-axis labels. Default is 45.
    palette : str, optional
        Color palette for violin plots. Default is tab10.
    stripplot : bool, optional
        Whether to overlay strip plot on violins. Default is False.
    xlabel : str, optional
        Label for x-axis. Default is 'Time point'.
    ylabel : str, optional
        Label for y-axis. If None, uses default labels based on key_cont.
        Default is None.
    fontsize: int, optional
        Font size for label in the plot.
    remove_ticks : bool, optional
        Whether to remove axis ticks. Default is True.
    figsize : tuple of int, optional
        Figure size (width, height) in inches. Default is (3, 3).
    ax : matplotlib.axes.Axes, optional
        Axes object to plot on. If None, creates new figure and axes.
    show : bool, optional
        Whether to display the plot. Default is True.
    arrow_size : float, optional
        Size of the arrow heads in points. Default is 15.
    arrow_style : str, optional
        Style of the arrow heads. Default is '->'.
    **kwargs : dict
        Additional arguments for sc.pl.violin.

    Returns
    -------
    matplotlib.axes.Axes
        Returns the axes object by default.

    Raises
    ------
    ValueError
        If required columns are not found in adata.obs.
    """

    if key_cont not in adata.obs.columns:
        raise ValueError(f"Column '{key_cont}' not found in adata.obs")

    if key_cat not in adata.obs.columns:
        raise ValueError(f"Column '{key_cat}' not found in adata.obs")

    adata.obs[key_cont] = pd.to_numeric(adata.obs[key_cont], errors="coerce")

    if not pd.api.types.is_categorical_dtype(adata.obs[key_cat]):
        adata.obs[key_cat] = adata.obs[key_cat].astype("category")

    if ax is None:
        w, h = figsize
        fig, ax = plt.subplots(figsize=(w + 0.2, h - 0.2))
        own_figure = True
    else:
        own_figure = False

    g = sc.pl.violin(
        adata,
        keys=key_cont,
        groupby=key_cat,
        rotation=rotation,
        stripplot=stripplot,
        palette=palette,
        show=False,
        ax=ax,
        **kwargs,
    )

    ax = g if hasattr(g, "axes") else plt.gca()

    if ylabel is None:
        if key_cont == "ct_pseudotime":
            ylabel_text = "CytoTRACE time"
        elif key_cont == "getime":
            ylabel_text = "Gene-embedded time"
        elif key_cont == "remake_getime":
            ylabel_text = "Rebuilt gene-embedded time"
        else:
            ylabel_text = key_cont
    else:
        ylabel_text = ylabel

    ax.set_ylabel(ylabel_text, fontsize=fontsize)
    ax.set_xlabel(xlabel, fontsize=fontsize)

    if remove_ticks:
        ax.set_xticks([])
        ax.set_xticklabels([])
    else:
        ax.tick_params(axis="x", labelrotation=rotation)

    u = np.unique(adata.obs[key_cont])
    y_ticks = (
        [0.0, 0.5, 1.0]
        if (u.min() >= 0 and u.max() <= 1)
        else [u[0], u[len(u) // 2], u[-1]]
    )
    ax.set_yticks(y_ticks)
    ax.set_yticklabels([f"{v}" for v in y_ticks])
    ax.tick_params(axis="y", labelsize=fontsize - 1)
    for spine in ax.spines.values():
        spine.set_visible(False)

    ax.grid(False)

    x_min, x_max = ax.get_xlim()
    y_min, y_max = ax.get_ylim()
    x_c, y_c = (x_min + x_max) / 2, (y_min + y_max) / 2
    x_rng, y_rng = (x_max - x_min) / 2, (y_max - y_min) / 2
    ax.set_xlim(x_c - x_rng * 1.2, x_c + x_rng * 1.2)
    ax.set_ylim(y_c - y_rng * 1.2, y_c + y_rng * 1.2)
    x_min, x_max = ax.get_xlim()
    y_min, y_max = ax.get_ylim()

    x_offset = (x_max - x_min) * 0.01
    y_offset = (y_max - y_min) * 0.01
    ax.set_xlim(x_min - x_offset, x_max)
    ax.set_ylim(y_min - y_offset, y_max)

    x_min, x_max = ax.get_xlim()
    y_min, y_max = ax.get_ylim()

    x_arrow_start = (x_min, y_min)
    x_arrow_end = (x_max, y_min)
    y_arrow_start = (x_min, y_min)
    y_arrow_end = (x_min, y_max)

    ax.annotate(
        "",
        xy=x_arrow_end,
        xytext=x_arrow_start,
        arrowprops=dict(
            arrowstyle=arrow_style,
            color="black",
            linewidth=1.5,
            shrinkA=0,
            shrinkB=0,
            mutation_scale=arrow_size,
        ),
    )

    ax.annotate(
        "",
        xy=y_arrow_end,
        xytext=y_arrow_start,
        arrowprops=dict(
            arrowstyle=arrow_style,
            color="black",
            linewidth=1.5,
            shrinkA=0,
            shrinkB=0,
            mutation_scale=arrow_size,
        ),
    )
    ax.set_facecolor("white")

    if own_figure:
        fig.patch.set_facecolor("white")
        if show:
            plt.tight_layout()
            plt.show()

    return ax


def getime_correlation(
    adata: AnnData,
    x_obs: str,
    y_obs: str,
    color_by: Optional[str] = None,
    palette: Optional[List[str]] = "tab10",
    figsize: Tuple[int, int] = (4, 4),
    xlabel: Optional[str] = None,
    ylabel: Optional[str] = None,
    fontsize: int = 12,
    legend: bool = False,
    dot_size: int = 10,
    add_reg: bool = True,
    xticks: bool = False,
    ax: Optional[plt.Axes] = None,
    arrow_size: float = 15,
    arrow_style: str = "->",
) -> None:
    """
    Creates a scatter plot with correlation analysis for two numeric variables.

    Parameters
    ----------
    adata : AnnData
        Input AnnData object containing observation data.
    x_obs : str
        Column name in adata.obs for the x-axis variable.
    y_obs : str
        Column name in adata.obs for the y-axis variable.
    color_by : str, optional
        Categorical column name for coloring points by group.
        If None, all points are the same color. Default is None.
    palette : List[str], optional
        Color palette for different groups when color_by is specified.
        If None, uses seaborn's "tab10" palette. Default is None.
    figsize : Tuple[int, int], optional
        Figure size (width, height) in inches. Default is (4, 4).
    xlabel : str, optional
        Custom label for x-axis. If None, uses x_obs. Default is None.
    ylabel : str, optional
        Custom label for y-axis. If None, uses y_obs. Default is None.
    fontsize: int, optional
        Font size for label in the plot.
    legend : bool, optional
        Whether to show legend when color_by is specified.
        Legend is always placed at lower right. Default is False.
    dot_size : int, optional
        Size of scatter plot markers. Default is 10.
    add_reg : bool, optional
        Whether to add a linear regression line. Default is True.
    xticks : bool, False
        Whether to show the ticks on the x-axis.
    ax : matplotlib.axes.Axes, optional
        Axes object to plot on. If None, creates new figure and axes.
    arrow_size : float, optional
        Size of arrow heads for coordinate axes in points. Default is 15.
    arrow_style : str, optional
        Style of arrow heads. Default is "->".

    Returns
    -------
    None
        Displays the plot directly.

    Raises
    ------
    ValueError
        If any of the specified columns (x_obs, y_obs, or color_by)
        are not found in adata.obs.
    """

    if x_obs not in adata.obs.columns:
        raise ValueError(f"Column '{x_obs}' not found in adata.obs")
    if y_obs not in adata.obs.columns:
        raise ValueError(f"Column '{y_obs}' not found in adata.obs")
    if color_by is not None and color_by not in adata.obs.columns:
        raise ValueError(f"Column '{color_by}' not found in adata.obs")

    if palette is None:
        palette = sns.color_palette("tab10")

    use_cols = [x_obs, y_obs]
    if color_by is not None:
        use_cols.append(color_by)

    df = adata.obs[use_cols].copy()
    df = df.dropna(subset=[x_obs, y_obs])

    if color_by is not None:
        df[color_by] = df[color_by].astype("category")

    w, h = figsize
    if ax is None:
        _, ax = plt.subplots(figsize=(w + 0.2, h))

    ax = sns.scatterplot(
        data=df,
        x=x_obs,
        y=y_obs,
        hue=color_by if color_by else None,
        s=dot_size,
        edgecolors=(0, 0, 0, 0),
        linewidth=0.2,
        palette=palette if color_by else None,
        ax=ax,
        legend=legend,
    )

    if add_reg:
        sns.regplot(
            data=df,
            x=x_obs,
            y=y_obs,
            scatter=False,
            color="grey",
            line_kws={"linewidth": 1.5},
            ax=ax,
        )

    r, _ = st.pearsonr(df[x_obs], df[y_obs])
    ax.text(
        0.05,
        0.95,
        f"Pearson r = {r:.2f}\n",
        transform=ax.transAxes,
        verticalalignment="top",
        fontweight="bold",
        bbox=dict(boxstyle="round", facecolor="w", alpha=0.8, lw=0),
        fontsize=fontsize - 1,
    )

    x_min, x_max = ax.get_xlim()
    y_min, y_max = ax.get_ylim()
    x_c, y_c = (x_min + x_max) / 2, (y_min + y_max) / 2
    x_rng, y_rng = (x_max - x_min) / 2, (y_max - y_min) / 2
    ax.set_xlim(x_c - x_rng * 1.2, x_c + x_rng * 1.2)
    ax.set_ylim(y_c - y_rng * 1.2, y_c + y_rng * 1.2)
    x_min, x_max = ax.get_xlim()
    y_min, y_max = ax.get_ylim()

    x_offset = (x_max - x_min) * 0
    y_offset = (y_max - y_min) * 0

    ax.set_xlim(x_min - x_offset, x_max)
    ax.set_ylim(y_min - y_offset, y_max)

    x_arrow_start = (x_min, y_min)
    x_arrow_end = (x_max, y_min)

    y_arrow_start = (x_min, y_min)
    y_arrow_end = (x_min, y_max)

    ax.annotate(
        "",
        xy=x_arrow_end,
        xytext=x_arrow_start,
        arrowprops=dict(
            arrowstyle=arrow_style,
            color="black",
            linewidth=1.5,
            shrinkA=0,
            shrinkB=0,
            mutation_scale=arrow_size,
        ),
    )
    ax.annotate(
        "",
        xy=y_arrow_end,
        xytext=y_arrow_start,
        arrowprops=dict(
            arrowstyle=arrow_style,
            color="black",
            linewidth=1.5,
            shrinkA=0,
            shrinkB=0,
            mutation_scale=arrow_size,
        ),
    )

    ax.set_xlabel(xlabel or x_obs, fontsize=fontsize)
    ax.set_ylabel(ylabel or y_obs, fontsize=fontsize)
    x_min, x_max = ax.get_xlim()
    y_min, y_max = ax.get_ylim()
    ax.tick_params(axis="y", labelsize=fontsize - 1)
    ax.set_yticks([0, 0.5, 1])
    ax.set_yticklabels(["0.0", "0.5", "1.0"])
    if xticks:
        x_min, x_max = df[x_obs].min(), df[x_obs].max()
        ticks = np.linspace(x_min, x_max, 3)
        ax.tick_params(axis="x", labelsize=fontsize - 1)
        ax.set_xticks(ticks)
        ax.set_xticklabels([f"{t:.1f}" for t in ticks])
    else:
        ax.set_xticks([])
    for spine in ["left", "bottom"]:
        ax.spines[spine].set_position(("outward", 0))
    for spine in ax.spines.values():
        spine.set_visible(False)
    plt.show()


def velocity_hot(
    adata: AnnData,
    gene: str,
    layer: str = "velocity",
    basis: str = "umap",
    cmap: str = "RdBu_r",
    figsize: Tuple[int, int] = (6, 6),
    fontsize: int = 10,
    show: bool = True,
    dpi: int = 300,
    save_format: str = "png",
    save_path: Optional[str] = None,
    **kwargs,
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Plot the velocity of a single gene on the embedding space.

    Parameters
    ----------
    adata : AnnData
        AnnData object with embedding coordinates and gene velocity data.
    gene : str
        Gene name to plot.
    layer : str, optional
        Layer containing gene velocity data. Default is "velocity".
    basis : str, optional
        Embedding basis to use (e.g., 'umap', 'tsne'). Default is 'umap'.
    cmap : str, optional
        Colormap for gene expression values. Default is 'RdBu_r'.
    figsize : tuple of int, optional
        Figure size (width, height) in inches. Default is (6, 6).
    fontsize: int, optional
        Font size for title in the plot.
    show : bool, optional
        Whether to display the figure. Default is True.
    dpi : int, optional
        Resolution for saved figure. Default is 300.
    save_format : str, optional
        File format for saving figure. Default is 'png'.
    save_path : str, optional
        Path to save the figure. If None, figure is not saved. Default is None.
    **kwargs : dict
        Additional arguments for sc.pl.embedding.

    Returns
    -------
    tuple
        Figure and axes objects of the embedding plot.

    Raises
    ------
    ValueError
        If gene is not found in adata.var_names.
        If embedding coordinates are not found in adata.obsm.
        If specified layer is not found.
    """

    if gene not in adata.var_names:
        raise ValueError(f"Gene '{gene}' not found in adata.var_names")
    embedding_key = f"X_{basis}" if not basis.startswith("X_") else basis
    if embedding_key not in adata.obsm:
        raise ValueError(
            f"Embedding coordinates '{embedding_key}' not found in adata.obsm"
        )
    if layer not in adata.layers:
        raise ValueError(f"Layer '{layer}' not found in adata.layers")

    fig, ax = plt.subplots(figsize=figsize)
    sc.pl.embedding(
        adata,
        basis=basis,
        color=gene,
        layer=layer,
        ax=ax,
        show=False,
        title=gene,
        colorbar_loc=None,
        cmap=cmap,
        vcenter=0,
        **kwargs,
    )
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_title(gene, fontsize=fontsize)  # 22
    for spine in ax.spines.values():
        spine.set_visible(False)

    if save_path is None:
        from mpl_toolkits.axes_grid1.inset_locator import inset_axes

        bar_ax = inset_axes(
            ax, width="3%", height="35%", loc="center right", borderpad=-0.8
        )
        gradient = np.linspace(-1, 1, 256).reshape(-1, 1)
        bar_ax.imshow(gradient, aspect="auto", cmap="RdBu", vmin=-1, vmax=1)
        bar_ax.set_axis_off()
        bar_ax.text(
            1.5,
            0.0,
            "Down",
            transform=bar_ax.transAxes,
            fontsize=12,
            va="center",
            ha="left",
        )  # 12
        bar_ax.text(
            1.5,
            0.5,
            "0",
            transform=bar_ax.transAxes,
            fontsize=12,
            va="center",
            ha="left",
        )
        bar_ax.text(
            1.5,
            1.0,
            "Up",
            transform=bar_ax.transAxes,
            fontsize=12,
            va="center",
            ha="left",
        )
        if show:
            plt.show()
        else:
            plt.close(fig)
    else:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        if not save_path.lower().endswith(
            (".png", ".pdf", ".svg", ".jpg", ".jpeg", ".tiff")
        ):
            base, _ = os.path.splitext(save_path)
            save_path = f"{base}.{save_format}"

        fig.savefig(save_path, dpi=dpi, bbox_inches="tight", format=save_format)

        bar_fig = plt.figure(figsize=(1, 1.25))
        cax = bar_fig.add_axes([0.45, 0.1, 0.1, 0.8])
        gradient = np.linspace(-1, 1, 256).reshape(-1, 1)
        cax.imshow(gradient, aspect="auto", cmap="RdBu", vmin=-1, vmax=1)
        cax.set_axis_off()
        cax.text(
            1.2,
            0.0,
            "Down",
            transform=cax.transAxes,
            fontsize=15,
            va="center",
            ha="left",
        )
        cax.text(
            1.2, 0.5, "0", transform=cax.transAxes, fontsize=15, va="center", ha="left"
        )
        cax.text(
            1.2, 1.0, "Up", transform=cax.transAxes, fontsize=15, va="center", ha="left"
        )
        bar_path = f"{os.path.splitext(save_path)[0]}_bar.{save_format}"
        bar_fig.savefig(
            bar_path, dpi=dpi, bbox_inches="tight", format=save_format, pad_inches=0
        )
        plt.close(bar_fig)
        plt.close(fig)

    return fig, ax


def velocity_embedding_stream(*a, **kw):
    """
    Wrapper function for scvelo's velocity_embedding_stream visualization.
    """
    import warnings

    warnings.filterwarnings("ignore")

    import builtins

    _real_print = builtins.print
    builtins.print = lambda *_, **__: None
    try:
        return scv.pl.velocity_embedding_stream(*a, **kw)
    finally:
        builtins.print = _real_print
        print("computing velocity embedding\nfinished.")
