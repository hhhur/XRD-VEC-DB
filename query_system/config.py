from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"

DB_PATH = DATA_DIR / "vector_db" / "merged_result"
COLLECTION_NAME = "merged_xrdselect"

MODEL_PATH = DATA_DIR / "model" / "all-mpnet-base-v2"

ASE_DB_DIR = DATA_DIR / "ase_databases"
ASE_DB_MAPPING = {
    "result1": ASE_DB_DIR / "allfile_1.db",
    "result2": ASE_DB_DIR / "allfile_2.db",
    "result3": ASE_DB_DIR / "allfile_3.db",
    "result5": ASE_DB_DIR / "allfile_5.db",
    "result6": ASE_DB_DIR / "allfile_6.db",
    "result7": ASE_DB_DIR / "allfile_7_1.db",
    "result8": ASE_DB_DIR / "allfile_8.db",
    "resultaflow": ASE_DB_DIR / "aflow.db",
    "resultoqmd": ASE_DB_DIR / "oqmd.db",
}

DEFAULT_TOP_K = 5
DEFAULT_OUTPUT_DIR = BASE_DIR / "output"
