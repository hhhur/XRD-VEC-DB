# XRD-VEC DB data records

This file describes the data layout used by the Zenodo v1 release:
https://doi.org/10.5281/zenodo.20389824.

The Zenodo archive stores the large data files. This GitHub repository stores
code, examples, and documentation.

## Top-level layout after extraction

```text
XRD-VEC_DB/
├── README.md
├── QUICKSTART.md
├── SERVER_MODE.md
├── SYSTEM_SUMMARY.txt
├── config.py
├── query.sh
├── start_server.sh
├── stop_server.sh
├── verify.sh
├── scripts/
│   └── query_xrd.py
└── data/
    ├── ase_databases/
    ├── vector_db/
    └── model/
```

## Structure records

Crystal structures are stored as ASE SQLite databases in
`data/ase_databases/`. Each row can be read with `ase.db.connect()` and
contains atomic species, Cartesian coordinates, unit-cell vectors,
periodic-boundary flags, and ASE row metadata.

The source label stored in ChromaDB maps to an ASE database file as follows.

| source_db | ASE database file | ASE rows |
| --- | --- | ---: |
| result1 | allfile_1.db | 137425 |
| result2 | allfile_2.db | 100858 |
| result3 | allfile_3.db | 89970 |
| result5 | allfile_5.db | 1061305 |
| result6 | allfile_6.db | 1011288 |
| result7 | allfile_7_1.db | 1402228 |
| result8 | allfile_8.db | 1792945 |
| resultaflow | aflow.db | 59968 |
| resultoqmd | oqmd.db | 913045 |

## Spectrum and vector-index records

The vector index is stored in `data/vector_db/merged_result/` as a ChromaDB
persistent database. The collection name is `merged_xrdselect`.

Each indexed record contains:

- a 180-point normalized powder XRD vector stored as a comma-separated document
- a 768-dimensional embedding produced with `all-mpnet-base-v2`
- metadata key `source_db`, identifying the source ASE database
- metadata key `id`, identifying the ASE row inside that database

The standard XRD vector spans 10 to 90 degrees in 2theta and is normalized to a
maximum finite intensity of 1.

## Local embedding model

The local embedding model is stored in:

```text
data/model/all-mpnet-base-v2/
```

Keeping the model in the release allows users to run the same query workflow
without relying on a remote model download.

## Identifier contract

The pair (`source_db`, `id`) is the stable link between a spectrum record and a
structure record. A query result should therefore always report both fields.
To retrieve the associated crystal structure:

1. map `source_db` to the corresponding ASE database file;
2. open that file with `ase.db.connect()`;
3. retrieve the row with `db.get(id=<id>)`;
4. export the resulting `Atoms` object as CIF or another structure format.

The query script in `query_system/scripts/query_xrd.py` implements this
contract.
