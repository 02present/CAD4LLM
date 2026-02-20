# dir
#   python3 whucad_vec_to_neutral.py \
#     --in_root /home/hj2/Desktop/WHUCAD/data/vec \
#     --out_root /home/hj2/Desktop/WHUCAD/ir_min \
#     --glob "**/*.h5"
#

# single
#   python3 whucad_vec_to_neutral.py \
#     --in_h5 /home/hj2/Desktop/WHUCAD/data/vec/0002/00002839.h5 \
#     --out_json /tmp/00002839.neutral.json

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import h5py
import numpy as np


OPID_TO_NAME: Dict[int, str] = {
    0: "Line",
    1: "Arc",
    2: "Circle",
    5: "EOS",
    6: "SOL",
    7: "Extrude",
    8: "Revolve",
    12: "Chamfer",
    17: "Topo",
    18: "Select",
}

SKETCH_OPS = {"Line", "Arc", "Circle"}
FEATURE_OPS = {"Extrude", "Revolve", "Chamfer"}


def op_name(op_id: int) -> str:
    return OPID_TO_NAME.get(int(op_id), f"OP_{int(op_id)}")


def to_int_row(row: np.ndarray) -> List[int]:
    return [int(round(float(x))) for x in row.tolist()]


def prune(obj: Any) -> Any:
    """Remove None/empty recursively."""
    if isinstance(obj, dict):
        out = {}
        for k, v in obj.items():
            pv = prune(v)
            if pv is None or pv == {} or pv == []:
                continue
            out[k] = pv
        return out if out else None
    if isinstance(obj, list):
        out_list = []
        for v in obj:
            pv = prune(v)
            if pv is None or pv == {} or pv == []:
                continue
            out_list.append(pv)
        return out_list if out_list else None
    return obj

def dequant_xy(q: int) -> Optional[float]:
    if q == -1:
        return None
    return float(q - 128)


def dequant_pos(q: int) -> Optional[float]:
    if q == -1:
        return None
    return float(q)


def first_valid(vals: List[int], invalid: Tuple[int, ...] = (-1, 0)) -> Optional[int]:
    for v in vals:
        if v not in invalid:
            return v
    return None


def decode_circle(params: List[int]) -> Dict[str, Any]:
    cxq = params[0] if len(params) > 0 else -1
    cyq = params[1] if len(params) > 1 else -1
    cx = dequant_xy(cxq) if cxq != -1 else None
    cy = dequant_xy(cyq) if cyq != -1 else None
    center = [cx, cy] if cx is not None and cy is not None else None

    rq = first_valid(params[2:10], invalid=(-1, 0))
    radius = dequant_pos(rq) if rq is not None else None

    out: Dict[str, Any] = {}
    if center is not None:
        out["center"] = center
    if radius is not None:
        out["radius"] = radius
    return out


def decode_line_p1(params: List[int]) -> Optional[List[float]]:
    xq = params[0] if len(params) > 0 else -1
    yq = params[1] if len(params) > 1 else -1
    x = dequant_xy(xq) if xq != -1 else None
    y = dequant_xy(yq) if yq != -1 else None
    if x is None or y is None:
        return None
    return [x, y]


def decode_arc(params: List[int]) -> Dict[str, Any]:
    p1 = decode_line_p1(params)
    sweep_q = params[2] if len(params) > 2 else -1
    ccw_q = params[3] if len(params) > 3 else -1
    out: Dict[str, Any] = {}
    if p1 is not None:
        out["p1"] = p1
    if sweep_q != -1:
        out["sweep_q"] = int(sweep_q)
    if ccw_q != -1:
        out["ccw"] = bool(ccw_q != 0)
    return out


def decode_extrude_distance(params: List[int]) -> Optional[float]:
    for idx in [12, 11, 13, 8, 9, 10, 6, 5, 4, 0]:
        if idx < len(params):
            v = params[idx]
            if v not in (-1, 0, 128):
                return float(v - 128) if v >= 128 and (v - 128) != 0 else float(v)
    return None


def decode_chamfer(params: List[int]) -> Tuple[Optional[float], Optional[float]]:
    cand = [v for v in params if v not in (-1, 0)]
    d1q = cand[0] if len(cand) > 0 else None
    d2q = cand[1] if len(cand) > 1 else None

    def conv(v: Optional[int]) -> Optional[float]:
        if v is None:
            return None
        return float(v - 128) if v >= 128 else float(v)

    return conv(d1q), conv(d2q)


def decode_select_tail4(row: List[int]) -> Dict[str, Any]:
    tail = row[-4:]
    if all(v == -1 for v in tail):
        return {}
    return {"tail4": tail}


def bbox_from_entities(entities: List[Dict[str, Any]]) -> Optional[List[float]]:
    xs: List[float] = []
    ys: List[float] = []
    for e in entities:
        k = e.get("kind")
        if k == "Circle":
            c = e.get("center")
            r = e.get("radius")
            if isinstance(c, list) and r is not None:
                cx, cy, rr = float(c[0]), float(c[1]), float(r)
                xs += [cx - rr, cx + rr]
                ys += [cy - rr, cy + rr]
        elif k == "Line" or k == "Arc":
            p0 = e.get("p0")
            p1 = e.get("p1")
            if isinstance(p0, list) and isinstance(p1, list):
                xs += [float(p0[0]), float(p1[0])]
                ys += [float(p0[1]), float(p1[1])]
    if not xs or not ys:
        return None
    return [min(xs), min(ys), max(xs), max(ys)]


def read_vec_h5(h5_path: Path, dataset: str = "vec") -> np.ndarray:
    with h5py.File(h5_path, "r") as h5:
        if dataset not in h5:
            raise KeyError(f"dataset '{dataset}' not found in {h5_path}")
        arr = np.array(h5[dataset])
    if arr.ndim != 2 or arr.shape[1] != 33:
        raise ValueError(f"expected [N,33], got {arr.shape} in {h5_path}")
    return arr


def build_ir(
    vec: np.ndarray,
    sample_id: str,
    input_relpath: Optional[str],
    units: str,
    eos_opid: int,
    keep_raw_ops: bool,
    keep_convert_logs: bool,
    keep_entity_raw: bool,
    keep_feature_raw: bool,
    keep_summary: bool,
) -> Dict[str, Any]:
    logs: List[str] = []
    logs.append(f"read_h5: vec shape={tuple(vec.shape)}")

    ops: List[Dict[str, Any]] = []
    eos_i: Optional[int] = None
    for i in range(vec.shape[0]):
        row = to_int_row(vec[i])
        oid = int(row[0])
        name = op_name(oid)
        params = row[1:]

        args: Dict[str, Any] = {}
        if name == "Circle":
            args = decode_circle(params)
        elif name == "Line":
            p1 = decode_line_p1(params)
            if p1 is not None:
                args = {"p1": p1}
        elif name == "Arc":
            args = decode_arc(params)
        elif name == "Extrude":
            dist = decode_extrude_distance(params)
            if dist is not None:
                args["distance_mm"] = dist
            args["op"] = "NewBody"
            args["sign"] = 1
        elif name == "Chamfer":
            d1, d2 = decode_chamfer(params)
            if d1 is not None:
                args["distance1_mm"] = d1
            if d2 is not None:
                args["distance2_mm"] = d2
        elif name == "Select":
            args = decode_select_tail4(row)

        ops.append({"i": i, "op_name": name, "args": args})

        if oid == eos_opid:
            eos_i = i
            break

    logs.append(f"stop_at_eos: token={eos_i if eos_i is not None else 'not_found'}")

    sequence: List[Dict[str, Any]] = []
    feat_id = 0
    sketch_count = 0
    extrude_count = 0
    revolve_count = 0
    chamfer_count = 0

    in_sketch = False
    cur_pt: List[float] = [0.0, 0.0]
    entities: List[Dict[str, Any]] = []
    ent_counter = 0
    sol_i_for_this_sketch: Optional[int] = None

    pending_select: List[Dict[str, Any]] = []

    def last_sketch_id() -> Optional[str]:
        for f in reversed(sequence):
            if f.get("type") == "Sketch":
                return f.get("id")
        return None

    def flush_sketch():
        nonlocal feat_id, sketch_count, entities, ent_counter, in_sketch, sol_i_for_this_sketch
        if not in_sketch or not entities:
            entities = []
            ent_counter = 0
            in_sketch = False
            sol_i_for_this_sketch = None
            return

        feat_id += 1
        sketch_count += 1
        sk_id = f"F{feat_id}"

        feat: Dict[str, Any] = {
            "id": sk_id,
            "type": "Sketch",
            "name": f"Sketch {sketch_count}",
            "plane": {"kind": "XY"},
            "entities": entities,
        }

        if keep_summary:
            bbox = bbox_from_entities(entities)
            feat["summary"] = {
                "n_lines": sum(1 for e in entities if e["kind"] == "Line"),
                "n_arcs": sum(1 for e in entities if e["kind"] == "Arc"),
                "n_circles": sum(1 for e in entities if e["kind"] == "Circle"),
                "n_entities": len(entities),
                "bbox": bbox,
            }

        if keep_feature_raw:
            feat["raw"] = {"ops": [{"i": sol_i_for_this_sketch if sol_i_for_this_sketch is not None else 0, "op": "SOL"}]}

        sequence.append(feat)

        entities = []
        ent_counter = 0
        in_sketch = False
        sol_i_for_this_sketch = None

    for op in ops:
        name = op["op_name"]

        if name == "SOL":
            flush_sketch()
            in_sketch = True
            cur_pt = [0.0, 0.0]
            sol_i_for_this_sketch = op["i"]
            continue

        if in_sketch and name in SKETCH_OPS:
            if name == "Circle":
                c = op["args"].get("center")
                r = op["args"].get("radius")
                if c is None or r is None:
                    continue
                ent_counter += 1
                e: Dict[str, Any] = {
                    "id": f"E{ent_counter}",
                    "kind": "Circle",
                    "is_construction": False,
                    "center": c,
                    "radius": r,
                }
                if keep_entity_raw:
                    e["raw"] = {"args": {"center": c, "radius": r}}
                entities.append(e)
                continue

            if name == "Line":
                p1 = op["args"].get("p1")
                if p1 is None:
                    continue
                p0 = [cur_pt[0], cur_pt[1]]
                ent_counter += 1
                e = {
                    "id": f"E{ent_counter}",
                    "kind": "Line",
                    "is_construction": False,
                    "p0": p0,
                    "p1": p1,
                }
                if keep_entity_raw:
                    e["raw"] = {"args": {"p0": p0, "p1": p1}}
                entities.append(e)
                cur_pt = [p1[0], p1[1]]
                continue

            if name == "Arc":
                p1 = op["args"].get("p1")
                if p1 is None:
                    continue
                p0 = [cur_pt[0], cur_pt[1]]
                ent_counter += 1
                e = {
                    "id": f"E{ent_counter}",
                    "kind": "Arc",
                    "is_construction": False,
                    "p0": p0,
                    "p1": p1,
                }
                if keep_entity_raw:
                    raw_args = {"p0": p0, "p1": p1}
                    if "sweep_q" in op["args"]:
                        raw_args["sweep_q"] = op["args"]["sweep_q"]
                    if "ccw" in op["args"]:
                        raw_args["ccw"] = op["args"]["ccw"]
                    e["raw"] = {"args": raw_args}
                entities.append(e)
                cur_pt = [p1[0], p1[1]]
                continue

        if name in ("Topo", "Select"):
            pending_select.append(op)
            continue

        if name in FEATURE_OPS:
            flush_sketch()

            if name == "Extrude":
                feat_id += 1
                extrude_count += 1
                dist = op["args"].get("distance_mm")
                f: Dict[str, Any] = {
                    "id": f"F{feat_id}",
                    "type": "Extrude",
                    "name": f"Extrude {extrude_count}",
                    "sketch_ref": last_sketch_id(),
                    "distance_mm": dist,
                    "direction": {"kind": "Normal", "sign": 1},
                    "operation": "NewBody",
                    "end_condition": "Blind",
                }
                if keep_feature_raw:
                    f["raw"] = {"op_name": "Extrude", "args": {"distance_mm": dist, "op": "NewBody", "sign": 1}}
                sequence.append(f)
                pending_select = []
                continue

            if name == "Revolve":
                feat_id += 1
                revolve_count += 1
                f = {
                    "id": f"F{feat_id}",
                    "type": "Revolve",
                    "name": f"Revolve {revolve_count}",
                    "sketch_ref": last_sketch_id(),
                }
                if keep_feature_raw and pending_select:
                    f["raw"] = {"op_name": "Revolve", "selection": [{"i": s["i"], "op_name": s["op_name"], "args": s.get("args", {})} for s in pending_select]}
                sequence.append(f)
                pending_select = []
                continue

            if name == "Chamfer":
                feat_id += 1
                chamfer_count += 1
                d1 = op["args"].get("distance1_mm")
                d2 = op["args"].get("distance2_mm")
                f = {
                    "id": f"F{feat_id}",
                    "type": "Chamfer",
                    "name": f"Chamfer {chamfer_count}",
                    "distance1_mm": d1,
                    "distance2_mm": d2,
                }
                if keep_feature_raw:
                    raw = {"op_name": "Chamfer", "args": {"distance1_mm": d1, "distance2_mm": d2}}
                    if pending_select:
                        raw["selection"] = [{"i": s["i"], "op_name": s["op_name"], "args": s.get("args", {})} for s in pending_select]
                    f["raw"] = raw
                sequence.append(f)
                pending_select = []
                continue

    flush_sketch()
    logs.append(f"built_features: sketch={sketch_count} extrude={extrude_count} revolve={revolve_count} chamfer={chamfer_count}")

    meta: Dict[str, Any] = {
        "source": "WHUCAD",
        "sample_id": sample_id,
        "units": units,
        "input_relpath": input_relpath,
    }
    if keep_convert_logs:
        meta["convert_logs"] = logs

    out: Dict[str, Any] = {
        "schema": "cad_neutral.v0.1",
        "meta": meta,
        "sequence": sequence,
    }
    if keep_raw_ops:
        out["raw_ops"] = ops

    pruned = prune(out)
    return pruned if pruned is not None else out


def dump_json(path: Path, obj: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_h5", type=str, default="", help="single input .h5")
    ap.add_argument("--out_json", type=str, default="", help="single output .neutral.json")
    ap.add_argument("--in_root", type=str, default="", help="batch input root")
    ap.add_argument("--out_root", type=str, default="", help="batch output root")
    ap.add_argument("--glob", type=str, default="**/*.h5", help="glob under in_root")
    ap.add_argument("--dataset", type=str, default="vec", help="h5 dataset name")
    ap.add_argument("--units", type=str, default="mm")
    ap.add_argument("--eos_opid", type=int, default=5)

    ap.add_argument("--keep_raw_ops", type=int, default=0)
    ap.add_argument("--keep_convert_logs", type=int, default=0)
    ap.add_argument("--keep_entity_raw", type=int, default=0)
    ap.add_argument("--keep_feature_raw", type=int, default=0)
    ap.add_argument("--keep_summary", type=int, default=1)
    args = ap.parse_args()

    if args.in_h5:
        in_h5 = Path(args.in_h5)
        vec = read_vec_h5(in_h5, dataset=args.dataset)
        out_json = Path(args.out_json) if args.out_json else in_h5.with_suffix(".neutral.json")

        sample_id = f"{in_h5.parent.name}/{in_h5.stem}"

        ir = build_ir(
            vec=vec,
            sample_id=sample_id,
            input_relpath=None,
            units=args.units,
            eos_opid=args.eos_opid,
            keep_raw_ops=bool(args.keep_raw_ops),
            keep_convert_logs=bool(args.keep_convert_logs),
            keep_entity_raw=bool(args.keep_entity_raw),
            keep_feature_raw=bool(args.keep_feature_raw),
            keep_summary=bool(args.keep_summary),
        )
        dump_json(out_json, ir)
        print(f"[OK] wrote: {out_json}")
        return

    if args.in_root and args.out_root:
        in_root = Path(args.in_root)
        out_root = Path(args.out_root)
        paths = sorted(in_root.glob(args.glob))
        if not paths:
            raise SystemExit(f"[ERR] no files matched: {in_root}/{args.glob}")

        n_ok, n_fail = 0, 0
        for p in paths:
            try:
                rel = p.relative_to(in_root)
                out_path = (out_root / rel).with_suffix(".neutral.json")
                vec = read_vec_h5(p, dataset=args.dataset)
                sample_id = f"{rel.parent.as_posix()}/{p.stem}"

                ir = build_ir(
                    vec=vec,
                    sample_id=sample_id,
                    input_relpath=rel.as_posix(),
                    units=args.units,
                    eos_opid=args.eos_opid,
                    keep_raw_ops=bool(args.keep_raw_ops),
                    keep_convert_logs=bool(args.keep_convert_logs),
                    keep_entity_raw=bool(args.keep_entity_raw),
                    keep_feature_raw=bool(args.keep_feature_raw),
                    keep_summary=bool(args.keep_summary),
                )
                dump_json(out_path, ir)
                n_ok += 1
            except Exception as e:
                n_fail += 1
                print(f"[FAIL] {p}: {e}")

        print(f"[DONE] ok={n_ok} fail={n_fail} out_root={out_root}")
        return

    raise SystemExit("Use --in_h5 for single or --in_root + --out_root for batch.")


if __name__ == "__main__":
    main()
