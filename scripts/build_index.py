"""
CLI entry point to build FAISS + Qdrant indexes from the IFC PDF.

Usage:
    python scripts/build_index.py
    python scripts/build_index.py --pdf path/to/report.pdf
"""

import argparse
from vectorstore.index_builder import build_index


def main() -> None:
    parser = argparse.ArgumentParser(description="Build FAISS and Qdrant vector indexes.")
    parser.add_argument(
        "--pdf",
        default="resources/documents/ifc-annual-report-2024-financials.pdf",
        help="Path to the PDF file to index.",
    )
    args = parser.parse_args()

    faiss_store, qdrant_store = build_index(args.pdf)
    print("Index built. FAISS and Qdrant stores ready.")


if __name__ == "__main__":
    main()
