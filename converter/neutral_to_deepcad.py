#   python3 ir_to_deepcad_json_rich.py --root input_ir_dir --out_dir out_json_dir
#   python3 ir_to_deepcad_json_rich.py --root one.ir.json --out_dir out_json_dir


from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

BASE62 = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"


def b62_from_bytes(b: bytes, length: int = 15) -> str:
    n = int.from_bytes(b, "big")
    out = []
    while n > 0 and len(out) < length:
        n, r = divmod(n, 62)
        out.append(BASE62[r])
    if not out:
        out = ["0"]
    s = "".join(reversed(out))
    if len(s) < length:
        s = (s + ("0" * length))[:length]
    return s[:length]


def pseudo_id(seed: str, prefix: str = "F", length: int = 15) -> str:
    h = hashlib.sha1(seed.encode("utf-8")).digest()
    return prefix + b62_from_bytes(h, length=length)


def mm_to_user(v_mm: float, coord_inv: float) -> float:
    return float(v_mm) * float(coord_inv)


def xy_mm_to_user(xy: List[float], coord_inv: float) -> List[float]:
    return [mm_to_user(float(xy[0]), coord_inv), mm_to_user(float(xy[1]), coord_inv)]


def xyz_obj(x: float, y: float, z: float) -> Dict[str, float]:
    return {"y": float(y), "x": float(x), "z": float(z)}


def identity_transform() -> Dict[str, Any]:
    return {
        "origin": {"y": 0.0, "x": 0.0, "z": 0.0},
        "y_axis": {"y": 1.0, "x": 0.0, "z": 0.0},
        "x_axis": {"y": 0.0, "x": 1.0, "z": 0.0},
        "z_axis": {"y": 0.0, "x": 0.0, "z": 1.0},
    }


def op_to_deepcad_operation(op: str) -> str:
    low = (op or "").lower()
    if "newbody" in low:
        return "NewBodyFeatureOperation"
    if "join" in low or "add" in low or "union" in low:
        return "JoinFeatureOperation"
    if "cut" in low or "remove" in low:
        return "CutFeatureOperation"
    if "intersect" in low or "common" in low:
        return "IntersectFeatureOperation"
    return "UnknownFeatureOperation"


def build_circle3d(center_xy_user: List[float], radius_user: float, curve_id: str) -> Dict[str, Any]:
    cx, cy = center_xy_user
    return {
        "center_point": xyz_obj(cx, cy, 0.0),
        "type": "Circle3D",
        "radius": float(radius_user),
        "curve": curve_id,
        "normal": {"y": 0.0, "x": 0.0, "z": 1.0},
    }


def build_line3d(p0_user: List[float], p1_user: List[float], curve_id: str) -> Dict[str, Any]:
    x0, y0 = p0_user
    x1, y1 = p1_user

    return {
        "type": "Line3D",
        "curve": curve_id,
        "start_point": xyz_obj(x0, y0, 0.0),
        "end_point": xyz_obj(x1, y1, 0.0),
    }


def build_arc3d_stub(center_xy_user: List[float], radius_user: float, a0: Optional[float], a1: Optional[float], curve_id: str) -> Dict[str, Any]:
    cx, cy = center_xy_user
    d = {
        "center_point": xyz_obj(cx, cy, 0.0),
        "type": "Arc3D",
        "radius": float(radius_user),
        "curve": curve_id,
        "normal": {"y": 0.0, "x": 0.0, "z": 1.0},
    }
    if a0 is not None:
        d["a0"] = float(a0)
    if a1 is not None:
        d["a1"] = float(a1)
    return d

def bbox_from_sketch_and_extrude(sketch_bbox_xy: Optional[List[float]], z0: float, z1: float) -> Optional[Dict[str, Any]]:
    if not sketch_bbox_xy or len(sketch_bbox_xy) != 4:
        return None
    xmin, ymin, xmax, ymax = map(float, sketch_bbox_xy)
    return {
        "type": "BoundingBox3D",
        "min_point": {"y": ymin, "x": xmin, "z": float(min(z0, z1))},
        "max_point": {"y": ymax, "x": xmax, "z": float(max(z0, z1))},
    }

def ir_to_deepcad_json_rich(ir: Dict[str, Any], coord_inv: float = 0.001) -> Dict[str, Any]:
    seq = ir.get("sequence", [])
    if not isinstance(seq, list):
        raise ValueError("IR 'sequence' must be a list")

    entities: Dict[str, Any] = {}
    out_sequence: List[Dict[str, Any]] = []

    id_map: Dict[str, str] = {}

    sketch_profile_map: Dict[str, Tuple[str, str]] = {}  # IR_sketch_fid -> (sketch_entity_id, profile_id)

    last_sketch_bbox_user: Optional[List[float]] = None
    last_extrude_z1_user: Optional[float] = None

    for idx, feat in enumerate(seq):
        if not isinstance(feat, dict):
            continue
        ftype = str(feat.get("type", "Unknown"))
        fid = str(feat.get("id", f"F{idx+1}"))
        name = feat.get("name", f"{ftype} {idx+1}")

        if ftype.lower() == "sketch":
            sketch_eid = pseudo_id(f"{ir.get('meta', {}).get('sample_id','')}-{fid}-Sketch", prefix="F", length=15)
            id_map[fid] = sketch_eid

            profile_id = "JGC"
            curve_prefix = "JGB"

            profile_curves: List[Dict[str, Any]] = []
            ents = feat.get("entities", [])
            if not isinstance(ents, list):
                ents = []

            curve_counter = 0
            for e in ents:
                if not isinstance(e, dict):
                    continue
                kind = str(e.get("kind", "Unknown"))
                is_const = bool(e.get("is_construction", False))

                curve_counter += 1
                curve_id = curve_prefix
                if curve_counter > 1:
                    base = ord("A") + (curve_counter % 26)
                    curve_id = "JG" + chr(base)

                if kind.lower() == "circle":
                    center = e.get("center")
                    radius = e.get("radius")
                    if isinstance(center, list) and len(center) >= 2 and radius is not None:
                        c = xy_mm_to_user([float(center[0]), float(center[1])], coord_inv)
                        r = mm_to_user(float(radius), coord_inv)
                        profile_curves.append(build_circle3d(c, r, curve_id))
                    continue

                if kind.lower() == "line":
                    p0 = e.get("p0")
                    p1 = e.get("p1")
                    if isinstance(p0, list) and isinstance(p1, list) and len(p0) >= 2 and len(p1) >= 2:
                        p0u = xy_mm_to_user([float(p0[0]), float(p0[1])], coord_inv)
                        p1u = xy_mm_to_user([float(p1[0]), float(p1[1])], coord_inv)
                        line3d = build_line3d(p0u, p1u, curve_id)
                        line3d["isConstruction"] = is_const
                        profile_curves.append(line3d)
                    continue

                if kind.lower() == "arc":
                    center = e.get("center")
                    radius = e.get("radius")
                    a0 = e.get("a0")
                    a1 = e.get("a1")
                    if isinstance(center, list) and len(center) >= 2 and radius is not None:
                        c = xy_mm_to_user([float(center[0]), float(center[1])], coord_inv)
                        r = mm_to_user(float(radius), coord_inv)
                        arc3d = build_arc3d_stub(c, r, a0 if a0 is not None else None, a1 if a1 is not None else None, curve_id)
                        arc3d["isConstruction"] = is_const
                        profile_curves.append(arc3d)
                    continue
                continue

            bbox = None
            summ = feat.get("summary")
            if isinstance(summ, dict) and summ.get("bbox") and isinstance(summ["bbox"], list) and len(summ["bbox"]) == 4:
                b = [mm_to_user(float(summ["bbox"][0]), coord_inv),
                     mm_to_user(float(summ["bbox"][1]), coord_inv),
                     mm_to_user(float(summ["bbox"][2]), coord_inv),
                     mm_to_user(float(summ["bbox"][3]), coord_inv)]
                last_sketch_bbox_user = b
            else:
                last_sketch_bbox_user = None

            sketch_ent = {
                "transform": identity_transform(),
                "type": "Sketch",
                "name": name,
                "profiles": {
                    profile_id: {
                        "loops": [
                            {
                                "is_outer": True,
                                "profile_curves": profile_curves
                            }
                        ],
                        "properties": {}
                    }
                },
                "reference_plane": {}, 
            }

            entities[sketch_eid] = sketch_ent
            out_sequence.append({"index": len(out_sequence), "type": "Sketch", "entity": sketch_eid})
            sketch_profile_map[fid] = (sketch_eid, profile_id)
            continue

        if ftype.lower() == "extrude":
            extrude_eid = pseudo_id(f"{ir.get('meta', {}).get('sample_id','')}-{fid}-Extrude", prefix="F", length=15)
            id_map[fid] = extrude_eid

            sketch_ref = feat.get("sketch_ref")
            sketch_eid, profile_id = (None, None)
            if isinstance(sketch_ref, str) and sketch_ref in sketch_profile_map:
                sketch_eid, profile_id = sketch_profile_map[sketch_ref]

            dist_mm = float(feat.get("distance_mm", 0.0))
            dist_user = mm_to_user(dist_mm, coord_inv)
            last_extrude_z1_user = dist_user

            op = op_to_deepcad_operation(str(feat.get("operation", "Unknown")))
            sign = 1
            direction = feat.get("direction")
            if isinstance(direction, dict) and "sign" in direction:
                try:
                    sign = 1 if int(direction["sign"]) >= 0 else -1
                except Exception:
                    sign = 1

            extrude_ent = {
                "name": name,
                "type": "ExtrudeFeature",
                "profiles": ([] if (profile_id is None or sketch_eid is None) else [{"profile": profile_id, "sketch": sketch_eid}]),
                "extent_two": {
                    "distance": {"type": "ModelParameter", "role": "AgainstDistance", "name": "none", "value": 0.0},
                    "type": "DistanceExtentDefinition",
                    "taper_angle": {"type": "ModelParameter", "role": "Side2TaperAngle", "name": "none", "value": 0.0},
                },
                "extent_one": {
                    "distance": {"type": "ModelParameter", "role": "AlongDistance", "name": "none", "value": float(dist_user)},
                    "type": "DistanceExtentDefinition",
                    "taper_angle": {"type": "ModelParameter", "role": "TaperAngle", "name": "none", "value": 0.0},
                },
                "operation": op,
                "start_extent": {"type": "ProfilePlaneStartDefinition"},
                "extent_type": "OneSideFeatureExtentType",
                "sign": int(sign),
            }

            entities[extrude_eid] = extrude_ent
            out_sequence.append({"index": len(out_sequence), "type": "ExtrudeFeature", "entity": extrude_eid})
            continue

        other_eid = pseudo_id(f"{ir.get('meta', {}).get('sample_id','')}-{fid}-{ftype}", prefix="F", length=15)
        id_map[fid] = other_eid
        entities[other_eid] = {"type": ftype, "name": name}
        out_sequence.append({"index": len(out_sequence), "type": ftype, "entity": other_eid})

    bbox3d = None
    if last_sketch_bbox_user is not None and last_extrude_z1_user is not None:
        bbox3d = bbox_from_sketch_and_extrude(last_sketch_bbox_user, 0.0, float(last_extrude_z1_user))

    out = {
        "entities": entities,
        "properties": ({} if bbox3d is None else {"bounding_box": bbox3d}),
        "sequence": out_sequence,
    }
    return out

def load_json(p: Path) -> Any:
    return json.loads(p.read_text(encoding="utf-8"))


def dump_json(p: Path, obj: Any):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def iter_ir_files(root: Path) -> List[Path]:
    if root.is_file():
        return [root]
    return sorted(root.rglob("*.ir.json"))


def make_out_path(root: Path, out_dir: Path, in_path: Path) -> Path:
    if root.is_file():
        name = in_path.name
        if name.endswith(".ir.json"):
            name = name[:-7] + ".json"
        else:
            name = in_path.stem + ".json"
        return out_dir / name

    rel = in_path.relative_to(root)
    name = rel.name
    if name.endswith(".ir.json"):
        name = name[:-7] + ".json"
    else:
        name = rel.stem + ".json"
    return out_dir / rel.parent / name


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="IR file (*.ir.json) or directory containing them")
    ap.add_argument("--out_dir", required=True, help="output directory")
    ap.add_argument("--coord_inv", type=float, default=0.001, help="IR(mm)*coord_inv => DeepCAD units (default mm->m)")
    ap.add_argument("--dry_run", action="store_true")
    args = ap.parse_args()

    root = Path(args.root).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()

    if not root.exists():
        raise FileNotFoundError(root)

    targets = iter_ir_files(root)
    print(f"[INFO] found {len(targets)} IR files")

    ok = 0
    fail = 0
    for i, p in enumerate(targets, start=1):
        try:
            ir = load_json(p)
            out = ir_to_deepcad_json_rich(ir, coord_inv=args.coord_inv)
            out_path = make_out_path(root, out_dir, p)

            if not args.dry_run:
                dump_json(out_path, out)

            ok += 1
            if i == 1 or i % 500 == 0 or i == len(targets):
                print(f"[{i}/{len(targets)}] ok -> {out_path}")
        except Exception as e:
            fail += 1
            print(f"[{i}/{len(targets)}] FAIL: {p} | {type(e).__name__}: {e}")

    print("\n[SUMMARY]")
    print(f"  total: {len(targets)}")
    print(f"  ok:    {ok}")
    print(f"  fail:  {fail}")
    print(f"  mode:  {'DRY_RUN' if args.dry_run else 'WRITE'}")
    print(f"  out:   {out_dir}")


if __name__ == "__main__":
    main()