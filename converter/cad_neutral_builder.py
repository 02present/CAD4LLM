
from __future__ import annotations

from dataclasses import dataclass, asdict, field
from typing import Any, Dict, List, Optional, Literal, Tuple
import json
import math


@dataclass
class DecodedToken:
    op_name: str                 
    args: Dict[str, Any] = field(default_factory=dict)
    raw: Any = None              


OpType = Literal["NewBody", "Join", "Cut", "Intersect", "Unknown"]
PlaneKind = Literal["XY", "XZ", "YZ", "FaceRef", "Unknown"]
DirKind = Literal["Normal", "Flip", "Unknown"]
EntityKind = Literal["Line", "Arc", "Circle", "Spline", "Unknown"]


@dataclass
class PlaneRef:
    kind: PlaneKind
    ref: Optional[str] = None


@dataclass
class DirectionRef:
    kind: DirKind = "Normal"
    sign: int = 1


@dataclass
class SketchEntity:
    id: str
    kind: EntityKind
    is_construction: bool = False

    p0: Optional[Tuple[float, float]] = None
    p1: Optional[Tuple[float, float]] = None
    center: Optional[Tuple[float, float]] = None
    radius: Optional[float] = None
    a0: Optional[float] = None
    a1: Optional[float] = None

    raw: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SketchFeature:
    id: str
    type: Literal["Sketch"] = "Sketch"
    name: str = ""
    plane: PlaneRef = field(default_factory=lambda: PlaneRef(kind="XY"))
    entities: List[SketchEntity] = field(default_factory=list)
    profiles: Dict[str, Any] = field(default_factory=lambda: {"loops": None})
    summary: Dict[str, Any] = field(default_factory=dict)
    raw: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ExtrudeFeature:
    id: str
    type: Literal["Extrude"] = "Extrude"
    name: str = ""
    sketch_ref: str = ""
    distance_mm: float = 0.0
    direction: DirectionRef = field(default_factory=DirectionRef)
    operation: OpType = "Unknown"
    end_condition: str = "Blind"
    raw: Dict[str, Any] = field(default_factory=dict)


@dataclass
class GenericFeature:
    """Fallback for operations we don't model yet."""
    id: str
    type: str
    name: str
    params: Dict[str, Any] = field(default_factory=dict)
    refs: Dict[str, Any] = field(default_factory=dict)
    raw: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CADIR:
    schema: str = "cad_neutral.v0.1"
    meta: Dict[str, Any] = field(default_factory=dict)
    sequence: List[Dict[str, Any]] = field(default_factory=list)
    raw_ops: List[Dict[str, Any]] = field(default_factory=list)

def _next_id(prefix: str, n: int) -> str:
    return f"{prefix}{n}"


def _safe_float(x: Any, default: float = 0.0) -> float:
    try:
        if x is None:
            return default
        return float(x)
    except Exception:
        return default


def _bbox_update(b: Optional[List[float]], x: float, y: float) -> List[float]:
    if b is None:
        return [x, y, x, y]
    b[0] = min(b[0], x)
    b[1] = min(b[1], y)
    b[2] = max(b[2], x)
    b[3] = max(b[3], y)
    return b


def _entity_bbox(ent: SketchEntity) -> Optional[List[float]]:
    b = None
    if ent.kind == "Line" and ent.p0 and ent.p1:
        b = _bbox_update(b, ent.p0[0], ent.p0[1])
        b = _bbox_update(b, ent.p1[0], ent.p1[1])
        return b

    if ent.kind == "Circle" and ent.center and ent.radius is not None:
        cx, cy = ent.center
        r = ent.radius
        b = _bbox_update(b, cx - r, cy - r)
        b = _bbox_update(b, cx + r, cy + r)
        return b

    if ent.kind == "Arc" and ent.center and ent.radius is not None:
        cx, cy = ent.center
        r = ent.radius
        a0 = ent.a0 if ent.a0 is not None else 0.0
        a1 = ent.a1 if ent.a1 is not None else a0

        lo, hi = (a0, a1) if a0 <= a1 else (a1, a0)

        def pt(a: float) -> Tuple[float, float]:
            return (cx + r * math.cos(a), cy + r * math.sin(a))

        for a in [lo, hi]:
            x, y = pt(a)
            b = _bbox_update(b, x, y)

        for a in [0.0, math.pi / 2, math.pi, 3 * math.pi / 2, 2 * math.pi]:
            if lo <= a <= hi:
                x, y = pt(a)
                b = _bbox_update(b, x, y)

        return b

    return None


def _sketch_summary(sk: SketchFeature) -> Dict[str, Any]:
    n_lines = sum(1 for e in sk.entities if e.kind == "Line")
    n_arcs = sum(1 for e in sk.entities if e.kind == "Arc")
    n_circles = sum(1 for e in sk.entities if e.kind == "Circle")

    bbox = None
    for e in sk.entities:
        eb = _entity_bbox(e)
        if eb is None:
            continue
        if bbox is None:
            bbox = eb
        else:
            bbox[0] = min(bbox[0], eb[0])
            bbox[1] = min(bbox[1], eb[1])
            bbox[2] = max(bbox[2], eb[2])
            bbox[3] = max(bbox[3], eb[3])

    return {
        "n_lines": n_lines,
        "n_arcs": n_arcs,
        "n_circles": n_circles,
        "n_entities": len(sk.entities),
        "bbox": bbox,
        "loops": sk.profiles.get("loops", None),
    }


class IRBuilder:
    def __init__(self, source: str = "DeepCAD", sample_id: str = "", units: str = "mm"):
        self.source = source
        self.sample_id = sample_id
        self.units = units

        self.feature_idx = 0
        self.entity_idx = 0
        self.extrude_idx = 0

        self.sequence: List[Dict[str, Any]] = []
        self.raw_ops: List[Dict[str, Any]] = []

        self._current_sketch: Optional[SketchFeature] = None
        self._last_extrude_feature_id: Optional[str] = None

    def _start_sketch(self):
        self.feature_idx += 1
        sk_id = _next_id("F", self.feature_idx)
        name = f"Sketch {sum(1 for f in self.sequence if f.get('type') == 'Sketch') + 1}"

        if self._last_extrude_feature_id is None:
            plane = PlaneRef(kind="XY", ref=None)
        else:
            plane = PlaneRef(kind="FaceRef", ref=f"FaceRef:{self._last_extrude_feature_id}:end")

        self._current_sketch = SketchFeature(id=sk_id, name=name, plane=plane)

    def _ensure_sketch(self):
        if self._current_sketch is None:
            self._start_sketch()

    def _append_entity(self, ent: SketchEntity):
        self._ensure_sketch()
        self._current_sketch.entities.append(ent)

    def _finalize_sketch_if_any(self) -> Optional[str]:
        if self._current_sketch is None:
            return None
        self._current_sketch.summary = _sketch_summary(self._current_sketch)
        self.sequence.append(asdict(self._current_sketch))
        sk_id = self._current_sketch.id
        self._current_sketch = None
        return sk_id

    def _emit_extrude(self, sketch_id: str, tok: DecodedToken):
        self.feature_idx += 1
        self.extrude_idx += 1
        ex_id = _next_id("F", self.feature_idx)
        name = f"Extrude {self.extrude_idx}"

        dist = _safe_float(tok.args.get("distance_mm", tok.args.get("distance", 0.0)), 0.0)

        op = tok.args.get("operation", tok.args.get("op", "Unknown"))
        if op not in ("NewBody", "Join", "Cut", "Intersect"):
            op = "Unknown"

        sign = int(tok.args.get("sign", 1))
        if sign not in (-1, 1):
            sign = 1

        ex = ExtrudeFeature(
            id=ex_id,
            name=name,
            sketch_ref=sketch_id,
            distance_mm=dist,
            direction=DirectionRef(kind="Normal", sign=sign),
            operation=op,
            raw={"op_name": tok.op_name, "args": tok.args, "raw": tok.raw},
        )
        self.sequence.append(asdict(ex))
        self._last_extrude_feature_id = ex_id

    def consume(self, tokens: List[DecodedToken], keep_unknown_as_feature: bool = False) -> CADIR:
        for i, tok in enumerate(tokens):
            self.raw_ops.append({"i": i, "op_name": tok.op_name, "args": tok.args, "raw": tok.raw})
            op = tok.op_name

            if op in ("SOL", "SketchBegin"):
                if self._current_sketch and self._current_sketch.entities:
                    self._finalize_sketch_if_any()
                self._start_sketch()
                self._current_sketch.raw.setdefault("ops", []).append({"i": i, "op": op, "args": tok.args})
                continue

            if op in ("EOS", "SketchEnd"):
                if self._current_sketch:
                    self._current_sketch.raw.setdefault("ops", []).append({"i": i, "op": op, "args": tok.args})
                    self._finalize_sketch_if_any()
                continue

            if op == "Line":
                self.entity_idx += 1
                ent_id = _next_id("E", self.entity_idx)
                if "p0" in tok.args and "p1" in tok.args:
                    p0 = tuple(tok.args["p0"])
                    p1 = tuple(tok.args["p1"])
                else:
                    p0 = (_safe_float(tok.args.get("x0")), _safe_float(tok.args.get("y0")))
                    p1 = (_safe_float(tok.args.get("x1")), _safe_float(tok.args.get("y1")))
                ent = SketchEntity(id=ent_id, kind="Line", p0=p0, p1=p1, raw={"args": tok.args})
                self._append_entity(ent)
                continue

            if op == "Circle":
                self.entity_idx += 1
                ent_id = _next_id("E", self.entity_idx)
                if "center" in tok.args:
                    c = tuple(tok.args["center"])
                else:
                    c = (_safe_float(tok.args.get("cx")), _safe_float(tok.args.get("cy")))
                r = _safe_float(tok.args.get("radius", tok.args.get("r")))
                ent = SketchEntity(id=ent_id, kind="Circle", center=c, radius=r, raw={"args": tok.args})
                self._append_entity(ent)
                continue

            if op == "Arc":
                self.entity_idx += 1
                ent_id = _next_id("E", self.entity_idx)
                if "center" in tok.args:
                    c = tuple(tok.args["center"])
                else:
                    c = (_safe_float(tok.args.get("cx")), _safe_float(tok.args.get("cy")))
                r = _safe_float(tok.args.get("radius", tok.args.get("r")))
                a0 = tok.args.get("a0", tok.args.get("start_angle"))
                a1 = tok.args.get("a1", tok.args.get("end_angle"))
                ent = SketchEntity(
                    id=ent_id,
                    kind="Arc",
                    center=c,
                    radius=r,
                    a0=_safe_float(a0) if a0 is not None else None,
                    a1=_safe_float(a1) if a1 is not None else None,
                    raw={"args": tok.args},
                )
                self._append_entity(ent)
                continue

            if op == "Extrude":
                sk_id = self._finalize_sketch_if_any()
                if sk_id is None:
                    self._start_sketch()
                    sk_id = self._finalize_sketch_if_any()
                self._emit_extrude(sk_id, tok)
                continue

            if keep_unknown_as_feature:
                self.feature_idx += 1
                fid = _next_id("F", self.feature_idx)
                gf = GenericFeature(id=fid, type=op, name=f"{op} 1", params=tok.args, raw={"raw": tok.raw})
                self.sequence.append(asdict(gf))

        if self._current_sketch and self._current_sketch.entities:
            self._finalize_sketch_if_any()

        return CADIR(
            meta={"source": self.source, "sample_id": self.sample_id, "units": self.units},
            sequence=self.sequence,
            raw_ops=self.raw_ops,
        )

def ir_to_json_dict(ir: CADIR) -> Dict[str, Any]:
    return asdict(ir)


def save_ir_json(ir: CADIR, out_path: str, indent: int = 2):
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(ir_to_json_dict(ir), f, ensure_ascii=False, indent=indent)


def example_make_tokens() -> List[DecodedToken]:
    return [
        DecodedToken("SOL"),
        DecodedToken("Line", {"x0": 0, "y0": 0, "x1": 40, "y1": 0}),
        DecodedToken("Line", {"x0": 40, "y0": 0, "x1": 40, "y1": 20}),
        DecodedToken("Line", {"x0": 40, "y0": 20, "x1": 0, "y1": 20}),
        DecodedToken("Line", {"x0": 0, "y0": 20, "x1": 0, "y1": 0}),
        DecodedToken("Extrude", {"distance_mm": 15.0, "op": "NewBody", "sign": 1}),
        DecodedToken("SOL"),
        DecodedToken("Circle", {"cx": 10, "cy": 10, "r": 3}),
        DecodedToken("Circle", {"cx": 30, "cy": 10, "r": 3}),
        DecodedToken("Extrude", {"distance_mm": 15.0, "op": "Cut", "sign": 1}),
    ]


if __name__ == "__main__":
    toks = example_make_tokens()
    builder = IRBuilder(source="DeepCAD", sample_id="demo/00000001", units="mm")
    ir = builder.consume(toks, keep_unknown_as_feature=False)
    print(json.dumps(ir_to_json_dict(ir), ensure_ascii=False, indent=2))
