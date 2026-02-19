# python3 ir_dir_to_whucad.py --ir_root ./IR --out_root ./RECON_ALL --dump_vec_json

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import h5py
import numpy as np


DEFAULT_ROW_LEN = 33
EOS_OPID = 5


def read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_h5_vec(out_h5: Path, vec: np.ndarray, dataset_name: str = "vec") -> None:
    out_h5.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(str(out_h5), "w") as h5:
        h5.create_dataset(dataset_name, data=vec, dtype=vec.dtype)


def write_vec_dump_json(out_json: Path, vec: np.ndarray, extra_meta: Dict[str, Any] | None = None) -> None:
    out_json.parent.mkdir(parents=True, exist_ok=True)
    payload: Dict[str, Any] = {
        "shape": [int(vec.shape[0]), int(vec.shape[1])],
        "dtype": str(vec.dtype),
        "vec": vec.astype(int).tolist(),
    }
    if extra_meta:
        payload["meta"] = extra_meta
    out_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def ensure_row_len(row: List[int], row_len: int) -> List[int]:
    if len(row) == row_len:
        return row
    if len(row) > row_len:
        return row[:row_len]
    return row + [-1] * (row_len - len(row))


def flatten_rows(ir: Dict[str, Any], row_len: int) -> Tuple[List[List[int]], Dict[str, Any]]:
    seq = ir.get("sequence", [])
    if not isinstance(seq, list):
        raise ValueError("IR: 'sequence' must be a list")

    rows: List[List[int]] = []
    stats = {"features": 0, "rows": 0, "missing_whucad_rows": 0}

    for feat in seq:
        stats["features"] += 1
        rws = feat.get("_whucad_rows", None)
        if not (isinstance(rws, list) and rws and all(isinstance(r, list) for r in rws)):
            stats["missing_whucad_rows"] += 1
            raise ValueError(
                f"missing _whucad_rows for feature id={feat.get('id')} type={feat.get('type')}"
            )

        for r in rws:
            rr = ensure_row_len([int(x) for x in r], row_len)
            rows.append(rr)

    if not rows:
        raise ValueError("No rows produced from neutral")

    if int(rows[-1][0]) != EOS_OPID:
        eos = [-1] * row_len
        eos[0] = EOS_OPID
        rows.append(eos)

    stats["rows"] = len(rows)
    return rows, stats


def iter_ir_files(root: Path) -> List[Path]:
    files = sorted(root.rglob("*.ir.json"))
    return [p for p in files if p.is_file()]


def derive_out_paths(ir_path: Path, in_root: Path, out_root: Path) -> Tuple[Path, Path]:
    rel = ir_path.relative_to(in_root)
    name = rel.name
    base = name[:-len(".ir.json")] if name.endswith(".ir.json") else ir_path.stem
    out_h5 = out_root / rel.parent / f"{base}.h5"
    out_vec_json = out_root / rel.parent / f"{base}.vec.json"
    return out_h5, out_vec_json


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--neutral_root", required=True, help="Input folder containing *.ir.json")
    ap.add_argument("--out_root", required=True, help="Output folder to write *.h5 (+ optional *.vec.json)")
    ap.add_argument("--dataset", default="vec", help="H5 dataset name (default vec)")
    ap.add_argument("--row_len", type=int, default=DEFAULT_ROW_LEN)
    ap.add_argument("--dtype", default="float64", choices=["float64", "int16"])
    ap.add_argument("--dump_vec_json", action="store_true", help="Also write *.vec.json next to *.h5")
    ap.add_argument("--limit", type=int, default=0, help="Process only first N files (0=all)")
    args = ap.parse_args()

    ir_root = Path(args.ir_root).expanduser().resolve()
    out_root = Path(args.out_root).expanduser().resolve()

    ir_files = iter_ir_files(ir_root)
    if not ir_files:
        raise SystemExit(f"[ERR] no *.ir.json under: {ir_root}")

    if args.limit and args.limit > 0:
        ir_files = ir_files[: args.limit]

    ok = 0
    failed = 0

    for ir_path in ir_files:
        out_h5, out_vec_json = derive_out_paths(ir_path, ir_root, out_root)
        try:
            ir = read_json(ir_path)
            rows, stats = flatten_rows(ir, row_len=args.row_len)

            if args.dtype == "float64":
                vec = np.asarray(rows, dtype=np.float64)
            else:
                vec = np.asarray(rows, dtype=np.int16)

            write_h5_vec(out_h5, vec, dataset_name=args.dataset)

            if args.dump_vec_json:
                extra = {
                    "from_ir": str(ir_path),
                    "h5_dataset": args.dataset,
                    "row_len": args.row_len,
                    "stats": stats,
                }
                write_vec_dump_json(out_vec_json, vec, extra_meta=extra)

            ok += 1
        except Exception as e:
            failed += 1
            print(f"[FAIL] {ir_path} -> {out_h5} : {e}")

    print(f"[DONE] ok={ok} failed={failed} out_root={out_root}")


if __name__ == "__main__":
    main()
