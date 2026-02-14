import numpy as np


def calculate_ovl(data1, data2, bins=100):
    data1 = np.asarray(data1).astype(float)
    data2 = np.asarray(data2).astype(float)

    min_val = min(np.min(data1), np.min(data2))
    max_val = max(np.max(data1), np.max(data2))

    hist1, bins_edges = np.histogram(
        data1, bins=bins, range=(min_val, max_val), density=True
    )
    hist2, _ = np.histogram(data2, bins=bins, range=(min_val, max_val), density=True)

    ovl_area = np.sum(np.minimum(hist1, hist2)) * (bins_edges[1] - bins_edges[0])
    return ovl_area
