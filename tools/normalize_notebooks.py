#!/usr/bin/env python3
# tools/normalize_notebooks.py
from pathlib import Path
import json, uuid, re

# Recursively fix all notebooks that Sphinx might touch.
GLOBS = ["docs/**/*.ipynb"]
ID_RX = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,63}$")

def new_id() -> str:
    return uuid.uuid4().hex  # 32 hex chars; valid per spec

def fix_notebook(p: Path) -> int:
    with p.open("r", encoding="utf-8") as f:
        nb = json.load(f)

    # ensure nbformat v4 and minor >=5 (cell IDs introduced in 4.5)
    nb["nbformat"] = 4
    nb["nbformat_minor"] = max(int(nb.get("nbformat_minor", 0) or 0), 5)

    fixed = 0
    cells = nb.get("cells") or []
    seen = set()
    for c in cells:
        cid = c.get("id")
        ok = isinstance(cid, str) and ID_RX.match(cid) and cid not in seen
        if not ok:
            cid = new_id()
            c["id"] = cid
            fixed += 1
        seen.add(cid)

    if fixed:
        with p.open("w", encoding="utf-8") as f:
            json.dump(nb, f, ensure_ascii=False, indent=1)
            f.write("\n")
    return fixed

def main():
    skip_parts = {"_build", ".jupyter_cache", ".ipynb_checkpoints"}
    total_fixed = 0
    for pat in GLOBS:
        for p in sorted(Path().glob(pat)):
            if any(part in skip_parts for part in p.parts):
                continue
            fixed = fix_notebook(p)
            total_fixed += fixed
            print(("fixed" if fixed else "ok"), fixed, p)
    print("total fixed cells:", total_fixed)

if __name__ == "__main__":
    main()
