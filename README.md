# XRD-VEC DB

Code and examples for **XRD-VEC_DB: A simulated crystal structure and X-ray
diffraction dataset for materials research**.

The canonical dataset release is archived on Zenodo:

- DOI: https://doi.org/10.5281/zenodo.20389824
- Version: v1
- Publication date: 2026-05-26
- Resource type: Dataset
- Data license: CC BY 4.0

Large database files are not stored in this GitHub repository. Download the
Zenodo release to obtain the ASE structure databases, ChromaDB vector index,
local embedding model, and the packaged query-system files.

## Repository contents

```text
.
├── noise/             # Parametric XRD perturbation utilities
├── mlmd/              # CHGNet-based MD perturbation example for Si
├── analysis/          # Reproducible Si augmentation examples and plots
├── si_structures/     # Example Si snapshots and spectra used by analysis
├── query_system/      # Lightweight query-system source and usage notes
├── DATA_RECORDS.md    # Dataset file structure and identifier contract
├── CITATION.cff       # Citation metadata
├── LICENSE            # Apache-2.0 license for code in this repository
└── requirements.txt   # Python dependencies
```

## Dataset layout after extraction

After downloading and extracting the Zenodo package, the reusable dataset is
organized as a self-contained query and data package. The main data-bearing
directories are:

```text
data/
├── ase_databases/     # ASE SQLite crystal-structure databases
├── vector_db/         # ChromaDB vector index
└── model/             # Local all-mpnet-base-v2 embedding model
```

The ChromaDB collection is named `merged_xrdselect`. Each indexed XRD record
stores a 180-point normalized powder XRD vector and metadata keys:

- `source_db`: source database label, such as `result1` or `resultoqmd`
- `id`: row identifier inside the corresponding ASE SQLite database

See [DATA_RECORDS.md](DATA_RECORDS.md) for the file-level description and
source-database mapping.

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Some workflows are optional. The MD example requires `chgnet` and a working
PyTorch installation; GPU acceleration is recommended but not required by the
dataset itself.

## Query workflow

The full Zenodo package includes shell wrappers and a pre-built local data
layout. A typical workflow is:

```bash
cd XRD-VEC_DB
python query_system/scripts/query_xrd.py \
  --xrd "0.1,0.2,0.3,..." \
  --top-k 5 \
  --output output
```

The query script validates the 180-point input vector, embeds it with the local
`all-mpnet-base-v2` model, searches the ChromaDB collection, retrieves matching
structures from ASE databases using `source_db` and `id`, and exports CIF plus
JSON metadata for each hit.

## Augmentation examples

Parametric perturbation:

```python
from noise.xrd_augmentor import AugmentConfig, XRDAugmentor

config = AugmentConfig(noise_std=0.25, max_shift=0.5)
augmentor = XRDAugmentor(config)
augmented = augmentor.augment(angles, intensities, hkls=miller_indices)
```

CHGNet-based molecular-dynamics example:

```bash
python mlmd/md_xrd_augmentation.py
```

The included Si examples demonstrate NVT-Langevin, NVT-Berendsen, and
NPT-Berendsen sampling at 300 K.

## Citation

If you use the dataset, cite the Zenodo record and the accompanying Scientific
Data descriptor:

```bibtex
@dataset{tianhao_su_2026_xrd_vec_db,
  author       = {Tianhao, Su},
  title        = {{XRD-VEC_DB: A simulated crystal structure and X-ray diffraction dataset for materials research}},
  month        = may,
  year         = 2026,
  publisher    = {Zenodo},
  version      = {v1},
  doi          = {10.5281/zenodo.20389824},
  url          = {https://doi.org/10.5281/zenodo.20389824}
}
```

## Licenses

- Data archived on Zenodo: CC BY 4.0
- Code in this repository: Apache License 2.0
