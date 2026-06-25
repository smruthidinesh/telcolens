import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.vector_store import store
from app.ingestion import ingest_path, ingest_sample_dir

if __name__ == "__main__":
    store.load()
    if len(sys.argv) > 1:
        for path in sys.argv[1:]:
            print(f"ingested {ingest_path(path)} chunks from {path}")
    else:
        print(f"ingested {ingest_sample_dir()} sample chunks")
    print(f"total indexed: {store.size}")
