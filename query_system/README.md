# Query system

This directory contains lightweight source code and usage notes for querying the
XRD-VEC DB vector index. The large data-bearing directories are provided in the
Zenodo release, not in GitHub:

```text
data/
├── ase_databases/
├── vector_db/
└── model/
```

Place this `query_system/` directory at the root of the extracted Zenodo package,
or run it from the root if the package already contains the same layout.

## Expected data layout

```text
XRD-VEC_DB/
├── data/
│   ├── ase_databases/
│   ├── vector_db/
│   │   └── merged_result/
│   └── model/
│       └── all-mpnet-base-v2/
└── query_system/
    ├── config.py
    └── scripts/
        └── query_xrd.py
```

## Example

```bash
python query_system/scripts/query_xrd.py \
  --xrd "0.1,0.2,0.3,..." \
  --top-k 5 \
  --output output
```

The query vector must contain exactly 180 finite numeric values. Results are
exported as one CIF file and one JSON metadata file per retrieved structure.

## Identifier contract

The ChromaDB metadata pair (`source_db`, `id`) links each spectrum record to one
ASE structure row. `source_db` selects the ASE database file and `id` selects the
row in that file.
