import scanpy as sc
import numpy as np
import scvelo as scv
from anndata import AnnData
from typing import Optional

from celldyc.tools.semi_model import Model
from celldyc.tools.utils import get_magnitude


def preprocess(
    adata: AnnData,
    min_cells: int = 10,
    target_sum: float = 1e4,
    n_top_genes: int = 2000,
    n_neighbors: int = 30,
    n_pcs: int = 30,
    svd_solver: str = "arpack",
    copy: bool = True,
) -> AnnData:
    """
    Standard preprocessing steps for single-cell RNA-seq data.

    Parameters
    ----------
    adata : AnnData
        Input single-cell data.
    min_cells : int, optional
        Minimum cells required to keep a gene. Default is 10.
    target_sum : float, optional
        Normalization target per cell. Default is 10000.
    n_top_genes : int, optional
        Number of highly variable genes to select. Default is 2000.
    n_neighbors : int, optional
        Number of neighbors for graph construction. Default is 30.
    n_pcs : int, optional
        Number of principal components. Default is 30.
    svd_solver : str, optional
        SVD solver for PCA. Default is 'arpack'.
    copy : bool, optional
        Whether to return a copy of the data. Default is True.

    Returns
    -------
    AnnData
        Preprocessed AnnData object.
    """
    if copy:
        adata = adata.copy()

    sc.pp.filter_genes(adata, min_cells=min_cells)
    sc.pp.normalize_total(adata, target_sum=target_sum)
    sc.pp.log1p(adata)
    sc.pp.highly_variable_genes(adata, n_top_genes=n_top_genes)
    adata = adata[:, adata.var.highly_variable].copy()
    sc.tl.pca(adata, svd_solver=svd_solver)
    sc.pp.neighbors(adata, n_neighbors=n_neighbors, n_pcs=n_pcs)

    return adata


def recover_dyc(
    adata: AnnData,
    time_key: str = "time_point",
    time_weight: float = 0.1,
    model_path: Optional[str] = None,
    save_path: Optional[str] = None,
    n_epochs=None,
) -> AnnData:
    """
    Predicts transcriptomic velocities and gene-embedded time using a semi-supervised model.

    Parameters
    ----------
    adata : AnnData
        Input single-cell data with temporal labels.
    time_key : str, optional
        Key in `adata.obs` containing temporal labels. Default is 'time_point'.
    time_weight : float, optional
        Control the relative weighting of the two loss components in the dual-loss training architecture. Default is 0.1.
    model_path : str, optional
        Path to a pre-trained model file. If None, train a new model.
        Default is None.
    save_path : str, optional
        Path to save the trained model. If None, model is not saved.
        Default is None.
    n_epochs : int, optional
        Number of training epochs. If None, the model automatically determines the iteration count.

    Returns
    -------
    AnnData
        Modified AnnData object with the following additions:
        - `adata.obs['getime']`: Predicted gene-embedded time
        - `adata.layers['velocity']`: Predicted velocities matrix
    """
    if model_path is None:
        model = Model(adata, n_latent=10, time_key=time_key)
        model.train(max_epochs=500, time_weight=time_weight, n_epochs=n_epochs)
        if save_path is not None:
            model.save_model(f"{save_path}")
    else:
        model = Model(adata, time_key=time_key, model_path=model_path)

    adata.var["getime_weights"] = model.clock_weights()
    adata.obs["getime"] = model.get_time()
    trends = model.get_trends()
    magnitude = get_magnitude(adata)
    adata.layers["velocity"] = trends * magnitude

    return adata


def velocity_graph(*a, **kw):
    """
    Wrapper function for scvelo's velocity_graph computation.
    """
    import warnings

    warnings.filterwarnings("ignore")

    import builtins

    _real_print = builtins.print
    builtins.print = lambda *_, **__: None
    try:
        return scv.tl.velocity_graph(*a, **kw)
    finally:
        builtins.print = _real_print
        print("computing velocity graph\nfinished.")
