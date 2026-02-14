import os
from pathlib import Path
from typing import Optional
import requests
import scanpy as sc
from anndata import AnnData
cache_dir = Path("data")

def download_zenodo_file(record_id: int, file_index: int, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True) 
    url = f"https://zenodo.org/api/records/{record_id}"
    r = requests.get(url)
    r.raise_for_status()
    data = r.json()
    files = data.get("files", [])
    if not files:
        raise RuntimeError(f"No files found in Zenodo record {record_id}.")
    if not (0 <= file_index < len(files)):
        raise IndexError(
            f"file_index {file_index} out of range for record {record_id} "
            f"(has {len(files)} files)."
        )
    f_meta = files[file_index]
    file_name = f_meta["key"]
    download_url = f_meta["links"]["self"]
    out_path = out_dir / file_name
    print(f"Downloading {file_name} from {download_url} ...")
    with requests.get(download_url, stream=True) as resp:
        resp.raise_for_status()
        with open(out_path, "wb") as fh:
            for chunk in resp.iter_content(chunk_size=8192):
                if chunk:
                    fh.write(chunk)
    print(f"Saved to: {out_path}")
    return out_path

def load_local_or_remote(
    rel_path: Path,
    record_id: Optional[int] = None,
    file_index: Optional[int] = None,
) -> AnnData:
    local_path = cache_dir / rel_path
    if local_path.is_file():
        return sc.read_h5ad(local_path)
    if record_id is None or file_index is None:
        raise FileNotFoundError(f"{local_path} not found and no remote source provided.")
    downloaded_path = download_zenodo_file(record_id, file_index, cache_dir)
    if downloaded_path != local_path:
        downloaded_path.rename(local_path)
    return sc.read_h5ad(local_path)


def mono2tam() -> AnnData:
    """
    **GBM Monocyte Differentiation**

    In vivo timestamping of monocyte infiltration and differentiation into Arg1⁺/Acp5⁺ TAMs in glioblastoma models via Zman-seq.

    **Data from**: Kirschenbaum, D., Xie, K., Ingelfinger, F., Katzenelenbogen, Y., Abadie, K., Look, T., ... & Amit, I. (2023). Time-resolved single-cell transcriptomics defines immune trajectories in glioblastoma. Cell. https://doi.org/10.1016/j.cell.2023.11.032 
    
    .. figure:: /_static/index_1.png
       :width: 500
       :align: center
       :alt: mono2tam dataset

    """
    record_id = 18639013  
    file_index = 5        
    return load_local_or_remote(
        Path("gbm_mono2tam.h5ad"),
        record_id=record_id,
        file_index=file_index,
    )


def celegans() -> AnnData:
    """
    *C. elegans* **Embryogenesis**

    A curated subset of the C. elegans AB lineage (333 cells) spanning three experimental time points, featuring single-cell representatives for each lineage node sampled to preserve original temporal distributions.

    **Data from**: Jonathan S. Packer et al. ,A lineage-resolved molecular atlas of C. elegans embryogenesis at single-cell resolution.Science365,eaax1971(2019). https://doi.org/10.1126/science.aax1971

    .. figure:: /_static/celegans.svg
       :width: 500
       :align: center
       :alt: celegans dataset

    """
    record_id = 18639013  
    file_index = 0        
    return load_local_or_remote(
        Path("GSE126954_ab_unique.h5ad"),
        record_id=record_id,
        file_index=file_index,
    )


def simudata() -> AnnData:
    """
    **Simulated Time-series Dataset**

    A synthetic time-series dataset of 5,000 cells and 500 genes generated via scDesign3, containing ground-truth time and transcriptomic velocities derived from time-dependent GAMLSS models.

    **Data from**: ScDesign3(Song, D., Wang, Q., Yan, G. et al. scDesign3 generates realistic in silico data for multimodal single-cell and spatial omics. Nat Biotechnol 42, 247–252 (2024). https://doi.org/10.1038/s41587-023-01772-1) and CellDyc(method).

    .. figure:: /_static/simu.svg
       :width: 500
       :align: center
       :alt: simudata

    """
    record_id = 18639013  
    file_index = 1        
    return load_local_or_remote(
        Path("scdesign3_5000c500g.h5ad"),
        record_id=record_id,
        file_index=file_index,
    )


def gastrulation() -> AnnData:
    """
    **Mouse Erythroid Maturation**

    An scRNA-seq dataset capturing the complete erythroid lineage specification during mouse gastrulation across eight experimental time points, ranging from hematoendothelial progenitors to Erythroid 3 cells.

    **Data from**: Pijuan-Sala, B., Griffiths, J.A., Guibentif, C. et al. A single-cell molecular map of mouse gastrulation and early organogenesis. Nature 566, 490–495 (2019). https://doi.org/10.1038/s41586-019-0933-9

    .. figure:: /_static/gas.svg
       :width: 500
       :align: center
       :alt: gastrulation erythropoiesis dataset
    """
    record_id = 18639013  
    file_index = 6        
    return load_local_or_remote(
        Path("gastrulation_erythropoiesis.h5ad"),
        record_id=record_id,
        file_index=file_index,
    )


def zebrafish() -> AnnData:
    """
    **Zebrafish Embryogenesis**

    A curated subset of 2,341 cells from a zebrafish embryogenesis atlas, capturing the axial mesoderm lineage (notochord and prechordal plate) across 12 experimental time points (3.3–12 hpf).

    **Data from**: Jeffrey A. Farrell et al. ,Single-cell reconstruction of developmental trajectories during zebrafish embryogenesis.Science360,eaar3131(2018). https://doi.org/10.1126/science.aar3131

    .. figure:: /_static/zebrafish.svg
       :width: 500
       :align: center
       :alt: zebrafish dataset

    """
    record_id = 18639013  
    file_index = 4        
    return load_local_or_remote(
        Path("zebrafish.h5ad"),
        record_id=record_id,
        file_index=file_index,
    )


def reprogramming_lineage() -> AnnData:
    """
    **Mouse Reprogramming Lineage**

    A CellTagging lineage-traced dataset of 3,049 cells across six time points, capturing the direct reprogramming of MEFs into iEPs and distinguishing between successful and dead-end trajectories.

    **Data from**: Biddy, B.A., Kong, W., Kamimoto, K. et al. Single-cell mapping of lineage and identity in direct reprogramming. Nature 564, 219–224 (2018). https://doi.org/10.1038/s41586-018-0744-4.

    .. figure:: /_static/repro.svg
       :width: 500
       :align: center
       :alt: reprogramming dataset

    """
    record_id = 18639013  
    file_index = 2        
    return load_local_or_remote(
        Path("reprogramming_lineage.h5ad"),
        record_id=record_id,
        file_index=file_index,
    )