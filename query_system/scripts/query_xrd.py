#!/usr/bin/env python3
"""
XRD Query System - Main Query Script
Complete tool for querying similar structures from XRD spectra
"""
import sys
import argparse
import sqlite3
from pathlib import Path
import numpy as np
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
from ase.db import connect
from ase import Atoms
import json

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import *

EXTERNAL_SOURCE_INFO = {
    'resultoqmd': {
        'name': 'OQMD',
        'url': 'http://oqmd.org/'
    },
    'resultaflow': {
        'name': 'AFLOW',
        'url': 'http://aflowlib.org/'
    }
}


class XRDQuerySystem:
    """XRD Query System Core Class"""

    def __init__(self, use_server=None, verbose=True):
        """
        Initialize query system

        Args:
            use_server: Whether to use server mode (None uses config file setting)
            verbose: Whether to show detailed information
        """
        self.verbose = verbose
        use_server = USE_SERVER if use_server is None else use_server

        if self.verbose:
            print("=" * 70)
            print("XRD Query System Initialization")
            print("=" * 70)

        # Connect to vector database
        self._connect_database(use_server)

        # Load embedding model
        self._load_model()

        # Database connection caches
        self.ase_db_cache = {}
        self.id_only_db_cache = {}
        self.db_mode_cache = {}

        if self.verbose:
            print("✓ System initialization complete\n")

    def _connect_database(self, use_server):
        """Connect to vector database"""
        if self.verbose:
            print(f"Connecting to database: {DB_PATH}")

        if use_server:
            if self.verbose:
                print(f"Using server mode: {SERVER_HOST}:{SERVER_PORT}")
            self.client = chromadb.HttpClient(host=SERVER_HOST, port=SERVER_PORT)
        else:
            if self.verbose:
                print("Using local mode (first load is slow)")
            self.client = chromadb.PersistentClient(
                path=str(DB_PATH),
                settings=Settings(anonymized_telemetry=False, allow_reset=False)
            )

        self.collection = self.client.get_collection(name=COLLECTION_NAME)
        if self.verbose:
            print("✓ Database connection successful")

    def _load_model(self):
        """Load embedding model"""
        if self.verbose:
            print(f"Loading embedding model: {MODEL_PATH}")
        self.model = SentenceTransformer(str(MODEL_PATH))
        if self.verbose:
            print("✓ Model loaded successfully")

    def _get_db_mode(self, source_db):
        """Detect whether a database file is an ID-only index or a full ASE structure database."""
        if source_db not in self.db_mode_cache:
            db_path = ASE_DB_MAPPING.get(source_db)
            if db_path is None:
                raise ValueError(f"Unknown database source: {source_db}")
            if not Path(db_path).exists():
                raise FileNotFoundError(f"Database file not found: {db_path}")

            conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
            try:
                tables = {
                    row[0]
                    for row in conn.execute(
                        "SELECT name FROM sqlite_master WHERE type='table'"
                    ).fetchall()
                }
            finally:
                conn.close()

            if 'structures_index' in tables and 'systems' not in tables:
                self.db_mode_cache[source_db] = 'id_only'
            else:
                self.db_mode_cache[source_db] = 'ase'

        return self.db_mode_cache[source_db]

    def _get_ase_db(self, source_db):
        """Get ASE database connection (with caching)"""
        if source_db not in self.ase_db_cache:
            db_path = ASE_DB_MAPPING.get(source_db)
            if db_path is None:
                raise ValueError(f"Unknown database source: {source_db}")
            self.ase_db_cache[source_db] = connect(db_path)
        return self.ase_db_cache[source_db]

    def _get_id_only_record(self, structure_id, source_db):
        """Get ID/formula metadata from an ID-only SQLite index database."""
        if source_db not in self.id_only_db_cache:
            db_path = ASE_DB_MAPPING.get(source_db)
            conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
            conn.row_factory = sqlite3.Row
            self.id_only_db_cache[source_db] = conn

        conn = self.id_only_db_cache[source_db]
        row = conn.execute(
            "SELECT * FROM structures_index WHERE id = ?",
            (int(structure_id),)
        ).fetchone()
        if row is None:
            raise KeyError(f"ID {structure_id} not found in {source_db}")

        keys = row.keys()
        return {
            'formula': row['formula'],
            'external_id': row['material_id'] if 'material_id' in keys else None
        }

    def _xrd_to_text(self, xrd_vec):
        """Convert XRD vector to text format"""
        if isinstance(xrd_vec, str):
            return xrd_vec
        elif isinstance(xrd_vec, (list, np.ndarray)):
            return ','.join([str(x) for x in xrd_vec])
        else:
            raise ValueError(f"Unsupported XRD vector type: {type(xrd_vec)}")

    def query(self, xrd_vec, top_k=None, source_filter=None):
        """
        Query similar XRD spectra and retrieve corresponding structures or ID metadata

        Args:
            xrd_vec: XRD spectrum vector (string, list, or numpy array)
            top_k: Return top-K results (None uses default value)
            source_filter: Optional source filter (e.g., 'result1')

        Returns:
            List of results, each containing structure or ID-only index information
        """
        top_k = DEFAULT_TOP_K if top_k is None else top_k

        if self.verbose:
            print(f"\nQuerying top-{top_k} similar structures...")
            if source_filter:
                print(f"Filter condition: source_db={source_filter}")

        # Convert XRD vector and compute embedding
        xrd_text = self._xrd_to_text(xrd_vec)
        query_embedding = self.model.encode([xrd_text], normalize_embeddings=True).tolist()

        # Build filter condition
        where_filter = {"source_db": source_filter} if source_filter else None

        # Query vector database
        results = self.collection.query(
            query_embeddings=query_embedding,
            n_results=top_k,
            where=where_filter,
            include=["documents", "metadatas", "distances"]
        )

        # Retrieve corresponding atomic structures or ID-only metadata
        structures = []
        for i in range(len(results['ids'][0])):
            structure_id = results['metadatas'][0][i]['id']
            source_db = results['metadatas'][0][i]['source_db']
            distance = results['distances'][0][i]
            xrd_vec_str = results['documents'][0][i]

            try:
                mode = self._get_db_mode(source_db)
                if mode == 'id_only':
                    record = self._get_id_only_record(structure_id, source_db)
                    ext_info = EXTERNAL_SOURCE_INFO.get(source_db, {})
                    structures.append({
                        'id': structure_id,
                        'external_id': record['external_id'],
                        'source_db': source_db,
                        'distance': distance,
                        'xrd_vec': xrd_vec_str,
                        'atoms': None,
                        'formula': record['formula'],
                        'structure_available': False,
                        'source_name': ext_info.get('name', source_db),
                        'source_url': ext_info.get('url')
                    })
                else:
                    atoms = self._get_structure(structure_id, source_db)
                    structures.append({
                        'id': structure_id,
                        'external_id': None,
                        'source_db': source_db,
                        'distance': distance,
                        'xrd_vec': xrd_vec_str,
                        'atoms': atoms,
                        'formula': atoms.get_chemical_formula(),
                        'structure_available': True,
                        'source_name': source_db,
                        'source_url': None
                    })
            except Exception as e:
                if self.verbose:
                    print(f"Warning: Unable to retrieve record ID={structure_id}, source={source_db}: {e}")
                continue

        if self.verbose:
            print(f"✓ Found {len(structures)} matching records\n")

        return structures

    def _get_structure(self, structure_id, source_db):
        """Get atomic structure by ID and source database"""
        db = self._get_ase_db(source_db)
        row = db.get(id=structure_id)
        return row.toatoms()

    def print_results(self, structures, show_details=True):
        """Format and print query results"""
        if not structures:
            print("No matching structures found")
            return

        print("=" * 70)
        print(f"Query Results ({len(structures)} records)")
        print("=" * 70)

        for i, struct in enumerate(structures, 1):
            print(f"\n[{i}] Similarity distance: {struct['distance']:.4f}")
            print(f"    Local ID: {struct['id']}")
            if struct.get('external_id'):
                print(f"    External Material ID: {struct['external_id']}")
            print(f"    Source database: {struct['source_db']}")
            print(f"    Formula: {struct['formula']}")

            if show_details:
                atoms = struct.get('atoms')
                if atoms is not None:
                    print(f"    Number of atoms: {len(atoms)}")
                    cell_params = atoms.cell.cellpar()
                    print(f"    Cell parameters: a={cell_params[0]:.3f} b={cell_params[1]:.3f} c={cell_params[2]:.3f}")
                    print(f"    Cell angles: alpha={cell_params[3]:.2f} beta={cell_params[4]:.2f} gamma={cell_params[5]:.2f}")
                else:
                    src_name = struct.get('source_name', struct['source_db'])
                    src_url = struct.get('source_url')
                    hint = f" ({src_url})" if src_url else ""
                    print(f"    Crystal structure: Not bundled due to {src_name} licensing restrictions.")
                    if struct.get('external_id'):
                        print(f"    Download hint: Retrieve '{struct['external_id']}' directly from {src_name}{hint}.")
                    else:
                        print(f"    Download hint: Obtain original crystal structure directly from {src_name}{hint}.")

            print("-" * 70)

    def save_results(self, structures, output_dir=None, save_cif=True, save_json=True):
        """Save query results to files"""
        output_dir = DEFAULT_OUTPUT_DIR if output_dir is None else Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        saved_files = []

        for i, struct in enumerate(structures, 1):
            base_name = f"{struct['source_db']}_id{struct['id']}"
            atoms = struct.get('atoms')

            # Save CIF file only when crystal structure is available
            if save_cif and atoms is not None:
                cif_file = output_dir / f"{base_name}.cif"
                atoms.write(str(cif_file))
                saved_files.append(str(cif_file))

            # Save JSON metadata
            if save_json:
                json_file = output_dir / f"{base_name}.json"
                metadata = {
                    'id': struct['id'],
                    'external_id': struct.get('external_id'),
                    'source_db': struct['source_db'],
                    'distance': struct['distance'],
                    'formula': struct['formula'],
                    'structure_available': atoms is not None
                }
                if atoms is not None:
                    metadata['n_atoms'] = len(atoms)
                    metadata['cell_parameters'] = atoms.cell.cellpar().tolist()
                else:
                    src_name = struct.get('source_name', struct['source_db'])
                    src_url = struct.get('source_url')
                    metadata['source_url'] = src_url
                    metadata['note'] = (
                        f"Crystal structure not redistributed due to {src_name} licensing restrictions. "
                        f"Please obtain the structure directly from {src_name}"
                        + (f" ({src_url})." if src_url else ".")
                    )
                with open(json_file, 'w') as f:
                    json.dump(metadata, f, indent=2)
                saved_files.append(str(json_file))

        if self.verbose:
            print(f"\n✓ Saved {len(saved_files)} files to: {output_dir}")

        return saved_files


def main():
    """Command line interface"""
    parser = argparse.ArgumentParser(
        description='XRD Query System - Query similar crystal structures from XRD spectra',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example usage:
  # Query from CSV file
  python query_xrd.py --csv result_1.csv --row 100 --top-k 5

  # Query from XRD vector string
  python query_xrd.py --xrd "0.1,0.2,0.3,..." --top-k 10

  # Filter specific database
  python query_xrd.py --csv result_1.csv --row 100 --source result8

  # Save results to specified directory
  python query_xrd.py --csv result_1.csv --row 100 --output ./my_results
        """
    )

    # Input arguments
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument('--csv', type=str, help='CSV file path')
    input_group.add_argument('--xrd', type=str, help='XRD vector string (comma-separated)')

    parser.add_argument('--row', type=int, help='Row number in CSV file (starting from 1)')
    parser.add_argument('--top-k', type=int, default=DEFAULT_TOP_K, help=f'Return top-K results (default {DEFAULT_TOP_K})')
    parser.add_argument('--source', type=str, help='Filter specific source database (e.g., result1, result8)')

    # Output arguments
    parser.add_argument('--output', type=str, help='Output directory (default: output/)')
    parser.add_argument('--no-save', action='store_true', help='Do not save result files')
    parser.add_argument('--quiet', action='store_true', help='Quiet mode (do not show detailed information)')

    args = parser.parse_args()

    # Read XRD vector
    if args.csv:
        if args.row is None:
            parser.error("--row argument is required when using --csv")

        import pandas as pd
        print(f"Reading from CSV file: {args.csv}, row: {args.row}")
        df = pd.read_csv(args.csv)
        if args.row < 1 or args.row > len(df):
            parser.error(f"Row number out of range (1-{len(df)})")
        xrd_vec = df.iloc[args.row - 1]['xrd_vec']
    else:
        xrd_vec = args.xrd

    # Initialize query system
    system = XRDQuerySystem(verbose=not args.quiet)

    # Execute query
    structures = system.query(xrd_vec, top_k=args.top_k, source_filter=args.source)

    # Display results
    system.print_results(structures, show_details=not args.quiet)

    # Save results
    if not args.no_save and structures:
        system.save_results(structures, output_dir=args.output)

    print(f"\n✓ Query complete! Found {len(structures)} matching records")


if __name__ == "__main__":
    main()
