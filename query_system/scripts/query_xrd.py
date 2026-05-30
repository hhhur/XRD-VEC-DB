#!/usr/bin/env python3
"""Query XRD-VEC DB spectra and export linked crystal structures."""

import argparse
import csv
import json
import math
import sys
from pathlib import Path

import chromadb
import numpy as np
from ase.db import connect
from sentence_transformers import SentenceTransformer

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import (  # noqa: E402
    ASE_DB_MAPPING,
    COLLECTION_NAME,
    DB_PATH,
    DEFAULT_OUTPUT_DIR,
    DEFAULT_TOP_K,
    MODEL_PATH,
)


def parse_xrd_vector(value):
    values = [float(x.strip()) for x in value.split(",") if x.strip()]
    if len(values) != 180:
        raise ValueError(f"Expected 180 XRD values, received {len(values)}")
    if not all(math.isfinite(x) for x in values):
        raise ValueError("XRD vector contains non-finite values")
    return values


def read_xrd_from_csv(path, row_number, column):
    if row_number is None:
        raise ValueError("--row is required when --csv is used")
    with open(path, newline="") as handle:
        reader = csv.DictReader(handle)
        for idx, row in enumerate(reader, start=1):
            if idx == row_number:
                if column not in row:
                    raise ValueError(f"Column {column!r} not found in CSV")
                return parse_xrd_vector(row[column])
    raise ValueError(f"CSV row {row_number} not found")


class XRDQuerySystem:
    def __init__(self):
        self.client = chromadb.PersistentClient(path=str(DB_PATH))
        self.collection = self.client.get_collection(name=COLLECTION_NAME)
        self.model = SentenceTransformer(str(MODEL_PATH))
        self.ase_cache = {}

    def _ase_db(self, source_db):
        if source_db not in ASE_DB_MAPPING:
            raise ValueError(f"Unknown source_db: {source_db}")
        if source_db not in self.ase_cache:
            self.ase_cache[source_db] = connect(str(ASE_DB_MAPPING[source_db]))
        return self.ase_cache[source_db]

    def query(self, xrd_vector, top_k=DEFAULT_TOP_K, source=None):
        document = ",".join(str(x) for x in xrd_vector)
        embedding = self.model.encode([document], normalize_embeddings=True).tolist()
        where = {"source_db": source} if source else None
        result = self.collection.query(
            query_embeddings=embedding,
            n_results=top_k,
            where=where,
            include=["documents", "metadatas", "distances"],
        )

        records = []
        for metadata, distance, stored_vector in zip(
            result["metadatas"][0],
            result["distances"][0],
            result["documents"][0],
        ):
            source_db = metadata["source_db"]
            structure_id = metadata["id"]
            atoms = self._ase_db(source_db).get(id=structure_id).toatoms()
            records.append(
                {
                    "source_db": source_db,
                    "id": structure_id,
                    "distance": float(distance),
                    "xrd_vec": stored_vector,
                    "atoms": atoms,
                    "formula": atoms.get_chemical_formula(),
                }
            )
        return records


def save_records(records, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    saved = []
    for record in records:
        stem = f"{record['source_db']}_id{record['id']}"
        cif_path = output_dir / f"{stem}.cif"
        json_path = output_dir / f"{stem}.json"
        record["atoms"].write(str(cif_path))
        cell = record["atoms"].cell.cellpar().tolist()
        metadata = {
            "source_db": record["source_db"],
            "id": record["id"],
            "distance": record["distance"],
            "formula": record["formula"],
            "n_atoms": len(record["atoms"]),
            "cell_parameters": cell,
        }
        json_path.write_text(json.dumps(metadata, indent=2))
        saved.extend([cif_path, json_path])
    return saved


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--xrd", help="Comma-separated 180-point XRD vector")
    group.add_argument("--csv", help="CSV file containing an xrd_vec column")
    parser.add_argument("--row", type=int, help="1-based CSV row number")
    parser.add_argument("--column", default="xrd_vec", help="CSV column name")
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    parser.add_argument("--source", help="Optional source_db filter")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT_DIR))
    args = parser.parse_args()

    if args.xrd:
        xrd_vector = parse_xrd_vector(args.xrd)
    else:
        xrd_vector = read_xrd_from_csv(args.csv, args.row, args.column)

    system = XRDQuerySystem()
    records = system.query(np.asarray(xrd_vector), top_k=args.top_k, source=args.source)
    saved = save_records(records, args.output)

    for rank, record in enumerate(records, start=1):
        print(
            f"{rank}\t{record['distance']:.6f}\t{record['source_db']}\t"
            f"{record['id']}\t{record['formula']}"
        )
    print(f"Saved {len(saved)} files to {args.output}")


if __name__ == "__main__":
    main()
