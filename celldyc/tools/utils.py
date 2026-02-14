import numpy as np
from scipy import sparse
import pandas as pd
from anndata import AnnData


def neighbors_direction(adata, time_key, input_layer=None, return_in_adata=True):
    connect = adata.obsp["connectivities"].copy()
    if not isinstance(connect, sparse.coo_matrix):
        if not sparse.issparse(connect):
            connect = sparse.coo_matrix(connect)
        else:
            connect = connect.tocoo()
    connect.setdiag(1)
    connect = connect.tocsr()

    pt = adata.obs[time_key].copy().to_numpy()

    if input_layer is None:
        SX = adata.X.copy()
    else:
        SX = adata.layers[input_layer].copy()
    if sparse.issparse(SX):
        SX = SX.toarray()
    SX = np.asarray(SX)

    expected_change = np.zeros(SX.shape, dtype=float)
    reliability = np.zeros(SX.shape[0], dtype=float)
    for obs_id in range(SX.shape[0]):
        start, end = connect.indptr[obs_id], connect.indptr[obs_id + 1]
        neigh = connect.indices[start:end]
        if neigh.size == 0:
            continue
        delta_t = pt[neigh] - pt[obs_id]
        weights = np.sign(delta_t).astype(float)
        delta_SX = np.sign(SX[neigh] - SX[obs_id])
        weights_sum = np.sum(np.abs(weights))
        expected_change[obs_id] = np.where(
            weights_sum != 0, (weights[:, None] * delta_SX).sum(axis=0) / weights_sum, 0
        )
        reliability[obs_id] = np.mean(np.abs(delta_t))

    if return_in_adata:
        adata.layers["raw_label"] = expected_change
        adata.obs["reliability"] = (reliability - reliability.min()) / (
            reliability.max() - reliability.min()
        )
    else:
        return expected_change


def make_label(adata_in, time_key="day", input_layer=None):
    import pandas.api.types as ptypes

    adata = AnnData(
        X=adata_in.X,
        obs=adata_in.obs.copy(),
        var=adata_in.var,
        obsm=adata_in.obsm,
        obsp=adata_in.obsp,
        varm=adata_in.varm,
        uns=adata_in.uns,
    )

    if ptypes.is_numeric_dtype(adata.obs[time_key]):
        adata.obs[time_key] = adata.obs[time_key].astype("category")

    time_stage = adata.obs[time_key].cat.categories

    time_num = pd.to_numeric(time_stage.to_numpy())
    time_ss = pd.Series(time_num, index=time_stage)
    time_ss = time_ss.sort_values(ascending=True)
    time_num = time_ss.values
    time_d = time_num[1:] - time_num[:-1]
    time_insert = np.insert(time_d, 0, 0) / 2
    time_ss = (time_ss.max() - time_ss + time_insert) / (time_ss.max() - time_ss.min())

    new_cats = time_ss.reindex(adata.obs[time_key].cat.categories).values
    adata.obs["time_curve"] = (
        adata.obs[time_key].cat.rename_categories(new_cats).astype(float)
    )
    adata.obs["time_num"] = adata.obs[time_key].cat.codes

    connect = adata.obsp["connectivities"].copy()

    if not isinstance(connect, sparse.coo_matrix):
        if not sparse.issparse(connect):
            connect = sparse.coo_matrix(connect)
        else:
            connect = connect.tocoo()
    connect.setdiag(1)
    connect = connect.tocsr()

    neighbor_loc = connect.copy()
    neighbor_loc.data = np.ones(connect.data.shape)
    neigh_nums = neighbor_loc.indptr[1:] - neighbor_loc.indptr[:-1]
    neigh_nums[neigh_nums == 0] = 1.0

    time_curve_vals = adata.obs["time_curve"].values
    cum_time_sum = sparse.csr_matrix.dot(neighbor_loc, time_curve_vals).astype(
        np.float32
    )
    adata.obs["cum_time"] = 1.0 - (cum_time_sum / neigh_nums)

    neighbors_direction(adata, time_key="cum_time", input_layer=input_layer)

    return adata


def get_magnitude(adata, input_layer=None, return_elements=False):
    connect = adata.obsp["connectivities"].copy()
    if not isinstance(connect, sparse.coo_matrix):
        if not sparse.issparse(connect):
            connect = sparse.coo_matrix(connect)
        else:
            connect = connect.tocoo()
    connect.setdiag(1)
    connect = connect.tocsr()

    if input_layer is None:
        SX = adata.X.copy()
    else:
        SX = adata.layers[input_layer].copy()
    if sparse.issparse(SX):
        SX = SX.toarray()
    SX = np.asarray(SX)

    ct = adata.obs["getime"].copy().to_numpy()

    gene_std = np.zeros(SX.shape, dtype=float)
    time_std = np.zeros(SX.shape[0], dtype=float)

    for obs_id in range(SX.shape[0]):
        start, end = connect.indptr[obs_id], connect.indptr[obs_id + 1]
        neigh = connect.indices[start:end]
        if neigh.size == 0:
            continue
        gene_std[obs_id] = np.std(SX[neigh], axis=0)
        time_std[obs_id] = np.std(ct[neigh])

    time_std = time_std.reshape(-1, 1)
    mag = np.where(time_std != 0, gene_std / time_std, 0)

    if return_elements:
        return time_std, gene_std
    else:
        return mag


def calculate_gene_avg(adata_subset):
    clock_weights = adata_subset.var["getime_weights"].values
    origin_clock = np.dot(adata_subset.layers["X"], clock_weights)
    contributions = np.multiply(adata_subset.layers["X"], clock_weights.reshape(1, -1))

    pert_clock = origin_clock.reshape(-1, 1) - contributions
    origin_clock = (origin_clock - origin_clock.min()) / (
        origin_clock.max() - origin_clock.min()
    )
    pert_clock = (pert_clock - pert_clock.min()) / (pert_clock.max() - pert_clock.min())

    gene_avg = np.linalg.norm(origin_clock[:, np.newaxis] - pert_clock, axis=0)
    gene_avg = (gene_avg - gene_avg.min()) / (gene_avg.max() - gene_avg.min())

    return gene_avg
