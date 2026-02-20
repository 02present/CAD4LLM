#python3 deepcad_json_to_ir_batch.py --root <cad_json_root> --out_dir <out>
#python3 deepcad_json_to_ir_batch.py --keep_raw

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional, Set

from cad_ir_builder import IRBuilder, DecodedToken, ir_to_json_dict

def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def iter_target_json_files(root: Path) -> List[Path]:
    files: List[Path] = []
    for i in range(100):
        sub = root / f"{i:04d}"
        if not sub.exists():
            continue
        files.extend(sorted(sub.rglob("*.json")))
    return files

def normalize_op_name(op: str) -> str:
    s = (op or "").strip()
    low = s.lower()
    mapping = {
        "sol": "SOL",
        "sketchbegin": "SOL",
        "sketch_begin": "SOL",
        "sketchstart": "SOL",
        "sketch_start": "SOL",
        "eos": "EOS",
        "sketchend": "EOS",
        "sketch_end": "EOS",
        "line": "Line",
        "arc": "Arc",
        "circle": "Circle",
        "extrude": "Extrude",
    }
    return mapping.get(low, s if s else "UnknownOp")


def pick_first(d: Dict[str, Any], keys: List[str], default=None):
    for k in keys:
        if k in d:
            return d[k]
    return default


def to_xy_pair(v: Any) -> Optional[Tuple[float, float]]:
    if v is None:
        return None
    if isinstance(v, (list, tuple)) and len(v) >= 2:
        try:
            return (float(v[0]), float(v[1]))
        except Exception:
            return None
    if isinstance(v, dict) and "x" in v and "y" in v:
        try:
            return (float(v["x"]), float(v["y"]))
        except Exception:
            return None
    return None


def map_extrude_op(v: Any) -> str:
    if v is None:
        return "Unknown"
    if isinstance(v, str):
        low = v.lower()
        if "newbody" in low or "new_body" in low or "newbodyfeatureoperation" in low:
            return "NewBody"
        if "join" in low or "add" in low or "union" in low or "merge" in low:
            return "Join"
        if "cut" in low or "remove" in low or "subtract" in low or "difference" in low:
            return "Cut"
        if "intersect" in low or "common" in low:
            return "Intersect"
        return "Unknown"
    if isinstance(v, (int, float)):
        code = int(v)
        return {0: "NewBody", 1: "Join", 2: "Cut", 3: "Intersect"}.get(code, "Unknown")
    return "Unknown"


def get_nested(d: Any, path: List[str]) -> Any:
    cur = d
    for p in path:
        if not isinstance(cur, dict) or p not in cur:
            return None
        cur = cur[p]
    return cur


def find_entities_dict(o: Any) -> Optional[Dict[str, Any]]:
    if isinstance(o, dict):
        for k in ["entities", "entityMap", "entity_dict", "entityTable", "entity_table"]:
            v = o.get(k)
            if isinstance(v, dict) and all(isinstance(kk, str) for kk in v.keys()):
                return v
    return None


def recursive_collect_entities(o: Any) -> Dict[str, Any]:
    ent = find_entities_dict(o)
    if ent is not None:
        return ent

    best: Dict[str, Any] = {}

    def walk(x: Any):
        nonlocal best
        if isinstance(x, dict):
            str_keys = [k for k in x.keys() if isinstance(k, str)]
            dict_vals = sum(1 for v in x.values() if isinstance(v, dict))
            if len(str_keys) >= 10 and dict_vals >= 5:
                if len(x) > len(best):
                    best = x
            for v in x.values():
                walk(v)
        elif isinstance(x, list):
            for v in x:
                walk(v)

    walk(o)
    return best


def strip_keys_inplace(obj: Any, keys_to_strip: Set[str]) -> Any:
    if isinstance(obj, dict):
        for k in list(obj.keys()):
            if k in keys_to_strip:
                obj.pop(k, None)
        for v in obj.values():
            strip_keys_inplace(v, keys_to_strip)
    elif isinstance(obj, list):
        for v in obj:
            strip_keys_inplace(v, keys_to_strip)
    return obj

def deepcad_json_to_tokens(
    obj: Any,
    coord_scale: float = 1.0,
    keep_raw: bool = False,
) -> Tuple[List[DecodedToken], List[str]]:
    logs: List[str] = []
    tokens: List[DecodedToken] = []

    def add_tok(op_name: str, args: Optional[Dict[str, Any]] = None, raw: Any = None):
        if keep_raw:
            tokens.append(DecodedToken(op_name=normalize_op_name(op_name), args=args or {}, raw=raw))
        else:
            tokens.append(DecodedToken(op_name=normalize_op_name(op_name), args=args or {}, raw=None))

    entity_lookup = recursive_collect_entities(obj)
    if entity_lookup:
        logs.append(f"entity_lookup: found {len(entity_lookup)} entities")
    else:
        logs.append("entity_lookup: NOT FOUND")

    def parse_sketch_entity(ent_data: Dict[str, Any], raw_item: Dict[str, Any]):
        add_tok("SOL", raw={"seq_item": raw_item, "ent": ent_data} if keep_raw else None)

        curves = None
        for k in ["curves", "geometry", "entities", "segments"]:
            if isinstance(ent_data.get(k), list):
                curves = ent_data[k]
                break

        if isinstance(curves, list) and curves:
            for c in curves:
                if not isinstance(c, dict):
                    continue
                ctype = str(pick_first(c, ["type", "kind", "op", "name"], "Unknown")).lower()

                if "line" in ctype:
                    p0 = to_xy_pair(pick_first(c, ["p0", "start", "from"], None))
                    p1 = to_xy_pair(pick_first(c, ["p1", "end", "to"], None))
                    args = {}
                    if p0 and p1:
                        args = {"p0": [p0[0] * coord_scale, p0[1] * coord_scale],
                                "p1": [p1[0] * coord_scale, p1[1] * coord_scale]}
                    add_tok("Line", args=args, raw=c if keep_raw else None)
                    continue

                if "circle" in ctype:
                    center = to_xy_pair(pick_first(c, ["center", "c"], None))
                    r = pick_first(c, ["radius", "r"], None)
                    args = {}
                    if center:
                        args["center"] = [center[0] * coord_scale, center[1] * coord_scale]
                    if r is not None:
                        args["radius"] = float(r) * coord_scale
                    add_tok("Circle", args=args, raw=c if keep_raw else None)
                    continue

                if "arc" in ctype:
                    center = to_xy_pair(pick_first(c, ["center", "c"], None))
                    r = pick_first(c, ["radius", "r"], None)
                    a0 = pick_first(c, ["a0", "start_angle", "theta0"], None)
                    a1 = pick_first(c, ["a1", "end_angle", "theta1"], None)
                    args = {}
                    if center:
                        args["center"] = [center[0] * coord_scale, center[1] * coord_scale]
                    if r is not None:
                        args["radius"] = float(r) * coord_scale
                    if a0 is not None:
                        args["a0"] = float(a0)
                    if a1 is not None:
                        args["a1"] = float(a1)
                    add_tok("Arc", args=args, raw=c if keep_raw else None)
                    continue

                add_tok("UnknownOp", args={"curve_type": ctype}, raw=c if keep_raw else None)
            return

        profiles = ent_data.get("profiles")
        if not isinstance(profiles, dict):
            return

        for prof_name, prof in profiles.items():
            if not isinstance(prof, dict):
                continue
            loops = prof.get("loops")
            if not isinstance(loops, list):
                continue

            for loop in loops:
                if not isinstance(loop, dict):
                    continue
                pcs = loop.get("profile_curves")
                if not isinstance(pcs, list):
                    continue

                for pc in pcs:
                    if not isinstance(pc, dict):
                        continue
                    pctype = str(pc.get("type", "Unknown")).lower()

                    if "circle" in pctype:
                        center = pc.get("center_point")
                        r = pc.get("radius", None)
                        if isinstance(center, dict) and "x" in center and "y" in center and r is not None:
                            cxy = (float(center["x"]) * coord_scale, float(center["y"]) * coord_scale)
                            add_tok(
                                "Circle",
                                args={"center": [cxy[0], cxy[1]], "radius": float(r) * coord_scale},
                                raw=pc if keep_raw else None,
                            )
                        else:
                            add_tok("UnknownOp", args={"curve_type": pc.get("type")}, raw=pc if keep_raw else None)
                        continue

                    if "line" in pctype:
                        sp = pc.get("start_point")
                        ep = pc.get("end_point")
                        if (
                            isinstance(sp, dict) and isinstance(ep, dict)
                            and "x" in sp and "y" in sp and "x" in ep and "y" in ep
                        ):
                            p0 = [float(sp["x"]) * coord_scale, float(sp["y"]) * coord_scale]
                            p1 = [float(ep["x"]) * coord_scale, float(ep["y"]) * coord_scale]
                            add_tok("Line", args={"p0": p0, "p1": p1}, raw=pc if keep_raw else None)
                        else:
                            add_tok("UnknownOp", args={"curve_type": pc.get("type")}, raw=pc if keep_raw else None)
                        continue

                    if "arc" in pctype:
                        add_tok("UnknownOp", args={"curve_type": pc.get("type")}, raw=pc if keep_raw else None)
                        continue

                    add_tok("UnknownOp", args={"curve_type": pc.get("type")}, raw=pc if keep_raw else None)

    def parse_extrude_entity(ent_data: Dict[str, Any], raw_item: Dict[str, Any]):
        dist = pick_first(ent_data, ["distance_mm", "distance", "depth", "extent", "d"], None)

        if dist is None:
            dist = get_nested(ent_data, ["extent_one", "distance", "value"])
        if dist is None:
            dist = get_nested(ent_data, ["extent_two", "distance", "value"])

        if dist is None:
            d1 = get_nested(ent_data, ["extent_one", "distance"])
            if isinstance(d1, dict):
                dist = d1.get("value")

        args: Dict[str, Any] = {}
        if dist is not None:
            args["distance_mm"] = float(dist) * coord_scale

        opv = pick_first(ent_data, ["operation", "op", "boolean", "mode"], None)
        args["op"] = map_extrude_op(opv)

        sign = pick_first(ent_data, ["sign", "dir", "direction_sign"], 1)
        try:
            args["sign"] = 1 if int(sign) >= 0 else -1
        except Exception:
            args["sign"] = 1

        add_tok("Extrude", args=args, raw={"seq_item": raw_item, "ent": ent_data} if keep_raw else None)

    if not isinstance(obj, dict):
        logs.append("fallback: non-dict top-level -> UnknownOp")
        add_tok("UnknownOp", args={"note": "unparsed json"}, raw=obj if keep_raw else None)
        return tokens, logs

    seq = obj.get("sequence")
    if not isinstance(seq, list):
        logs.append("fallback: no dict['sequence'] list -> UnknownOp")
        add_tok("UnknownOp", args={"note": "no sequence"}, raw=obj if keep_raw else None)
        return tokens, logs

    logs.append("detected: dict['sequence'] list")

    for item in seq:
        if not isinstance(item, dict):
            add_tok("UnknownOp", args={"value": item}, raw=item if keep_raw else None)
            continue

        op_type = str(item.get("type", item.get("op", "UnknownOp")))
        ent_id = item.get("entity", None)

        ent_data = None
        if isinstance(ent_id, str) and entity_lookup:
            ent_data = entity_lookup.get(ent_id)

        low = op_type.lower()

        if low == "sketch":
            if isinstance(ent_data, dict):
                parse_sketch_entity(ent_data, item)
            else:
                add_tok("SOL", raw=item if keep_raw else None)
            continue

        if low in ("extrudefeature", "extrude"):
            if isinstance(ent_data, dict):
                parse_extrude_entity(ent_data, item)
            else:
                add_tok("Extrude", args={}, raw=item if keep_raw else None)
            continue

        add_tok(op_type, args={}, raw={"seq_item": item, "ent": ent_data} if keep_raw else None)

    return tokens, logs

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--root",
        type=str,
        default="../deepcad_download_1217_R3/deepcad_download/deepcad_download/cad_json",
        help="root dir containing 0000~0099 folders with DeepCAD json",
    )
    ap.add_argument(
        "--out_dir",
        type=str,
        default="../deepcad_download_1217_R3/deepcad_download/deepcad_download/cad_json_ir",
        help="output dir for IR json (mirrors structure)",
    )
    ap.add_argument("--dry_run", action="store_true", help="no files written")
    ap.add_argument(
        "--coord_scale",
        type=float,
        default=1000,
        help="multiply all coordinates/distances by this scale (DeepCAD often uses meters -> use 1000 for mm)",
    )
    ap.add_argument(
        "--keep_raw",
        action="store_true",
        help="keep raw blocks (default: strip all raw + remove raw_ops)",
    )
    args = ap.parse_args()

    root = Path(args.root).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()

    if not root.exists():
        raise FileNotFoundError(f"root not found: {root}")

    targets = iter_target_json_files(root)
    print(f"[INFO] found {len(targets)} json files under 0000~0099 in: {root}")

    ok = 0
    fail = 0

    for idx, path in enumerate(targets, start=1):
        rel = path.relative_to(root)
        parent = rel.parts[0] if len(rel.parts) >= 1 else "0000"
        sample_id = f"{parent}/{path.stem}"

        try:
            obj = load_json(path)
            tokens, logs = deepcad_json_to_tokens(obj, coord_scale=args.coord_scale, keep_raw=args.keep_raw)

            builder = IRBuilder(source="DeepCAD", sample_id=sample_id, units="mm")
            ir = builder.consume(tokens, keep_unknown_as_feature=False)
            ir_dict = ir_to_json_dict(ir)

            ir_dict["schema"] = "cad_ir.v0.1"
            ir_dict.setdefault("meta", {})
            ir_dict["meta"].setdefault("source", "DeepCAD")
            ir_dict["meta"].setdefault("sample_id", sample_id)
            ir_dict["meta"].setdefault("units", "mm")
            ir_dict["meta"]["convert_logs"] = logs
            ir_dict["meta"]["input_relpath"] = rel.as_posix()

            if not args.keep_raw:
                strip_keys_inplace(ir_dict, keys_to_strip={"raw"})
                ir_dict.pop("raw_ops", None)

            out_path = (out_dir / rel).with_suffix(".neutral.json")

            if not args.dry_run:
                out_path.parent.mkdir(parents=True, exist_ok=True)
                with out_path.open("w", encoding="utf-8") as f:
                    json.dump(ir_dict, f, ensure_ascii=False, indent=2)

            ok += 1
            if idx == 1 or idx % 500 == 0:
                print(f"[{idx}/{len(targets)}] ok -> {out_path}")

        except Exception as e:
            fail += 1
            print(f"[{idx}/{len(targets)}] FAIL: {path} | {type(e).__name__}: {e}")

    print("\n[SUMMARY]")
    print(f"  total: {len(targets)}")
    print(f"  ok:    {ok}")
    print(f"  fail:  {fail}")
    print(f"  mode:  {'DRY_RUN' if args.dry_run else 'WRITE'}")
    print(f"  out:   {out_dir}")
    print(f"  raw:   {'KEEP' if args.keep_raw else 'STRIP'}")


if __name__ == "__main__":
    main()