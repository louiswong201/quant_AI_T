#!/usr/bin/env python3
"""
Prepare a clean project folder for GitHub upload.

Excludes:
  - ZIP archives, __pycache__, .pytest_cache
  - data/*.csv (large; use examples/download_*.py to fetch)
  - paper_trading.db, *.db-shm, *.db-wal
  - .spyproject, catboost_info, output files
  - IDE/editor temp files
"""

from pathlib import Path
import shutil

SRC = Path(__file__).resolve().parent
DST = SRC.parent / "quant_project_AI_github"

EXCLUDE_DIRS = {
    "__pycache__",
    ".pytest_cache",
    ".git",
    ".spyproject",
    "catboost_info",
    ".idea",
    ".vscode",
    "venv",
    ".venv",
    "env",
}

EXCLUDE_FILES = {
    "prepare_github.py",  # don't include the script itself
    "paper_trading.db",
    "paper_trading.db-shm",
    "paper_trading.db-wal",
}

EXCLUDE_SUFFIXES = {
    ".zip",
    ".pyc",
    ".pyo",
    ".pyd",
    ".db",
    ".db-shm",
    ".db-wal",
    ".log",
    ".xlsx",
}

# data/ subdirs: copy structure but replace CSVs with .gitkeep + README
DATA_EXCLUDE_SUFFIXES = {".csv", ".parquet", ".arrow", ".bin"}


def should_exclude(path: Path, is_in_data: bool = False) -> bool:
    if path.name in EXCLUDE_FILES:
        return True
    if path.suffix.lower() in EXCLUDE_SUFFIXES:
        return True
    if is_in_data and path.suffix.lower() in DATA_EXCLUDE_SUFFIXES:
        return True
    for part in path.parts:
        if part in EXCLUDE_DIRS:
            return True
    return False


def main():
    if DST.exists():
        shutil.rmtree(DST)
    DST.mkdir(parents=True)

    copied = 0
    skipped = 0

    for src_path in SRC.rglob("*"):
        if not src_path.is_file():
            continue

        rel = src_path.relative_to(SRC)
        dst_path = DST / rel
        is_in_data = "data" in rel.parts and rel.name != "README.md"

        if should_exclude(src_path, is_in_data):
            skipped += 1
            continue

        dst_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            shutil.copy2(src_path, dst_path)
            copied += 1
        except Exception as e:
            print(f"  Skip {rel}: {e}")
            skipped += 1

    # Create data subdir placeholders (empty dirs need .gitkeep for git)
    for sub in ["1m", "5m", "15m", "1h", "4h", "1d"]:
        (DST / "data" / sub).mkdir(parents=True, exist_ok=True)
        (DST / "data" / sub / ".gitkeep").touch()

    # Ensure .gitignore and LICENSE exist (copy from SRC if present)
    for f in [".gitignore", "LICENSE"]:
        src_f = SRC / f
        if src_f.exists() and not (DST / f).exists():
            shutil.copy2(src_f, DST / f)
            copied += 1

    print(f"Copied {copied} files, skipped {skipped}")
    print(f"Output: {DST}")
    print("\nNext steps:")
    print(f"  cd {DST}")
    print("  git init")
    print("  git add .")
    print("  git commit -m \"Initial commit\"")
    print("  git remote add origin <your-github-repo-url>")
    print("  git push -u origin main")


if __name__ == "__main__":
    main()
