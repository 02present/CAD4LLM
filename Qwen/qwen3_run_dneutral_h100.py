from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


DEFAULT_MODEL = "Qwen/Qwen3-8B"
DEFAULT_GLOB = "**/*.neutral.json"
DEFAULT_MAX_NEW_TOKENS = 180
DEFAULT_TEMPERATURE = 0.35
DEFAULT_TOP_P = 0.9
DEFAULT_TOP_K = 60
DEFAULT_REP_PENALTY = 1.05
DEFAULT_BATCH_SIZE = 12

LANG = "English" #Korean
MAX_PTS = 8000
MAX_ENTS_SCAN = 4000

ALWAYS_MENTION_EXTRUDE_IF_UNKNOWN = True

TREAT_INNER_CIRCLES_AS_HOLES = True


def read_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def rank_info() -> Tuple[int, int, int]:
    rank = int(os.environ.get("RANK", "0"))
    world = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    return rank, world, local_rank


def list_ir_files(neutral_root: Path, glob_pat: str) -> List[Path]:
    files = sorted(neutral_root.glob(glob_pat))
    return [p for p in files if p.is_file()]


def shard_files(files: List[Path], rank: int, world: int) -> List[Path]:
    if world <= 1:
        return files
    return [p for i, p in enumerate(files) if (i % world) == rank]


def _get_first(d: Dict[str, Any], keys: List[str]) -> Any:
    for k in keys:
        if k in d:
            return d[k]
    return None


def _as_point(x: Any) -> Optional[Tuple[float, float]]:
    if isinstance(x, (list, tuple)) and len(x) >= 2:
        try:
            return float(x[0]), float(x[1])
        except Exception:
            return None
    return None


def _round(x: Optional[float], nd: int = 2) -> Optional[float]:
    if x is None:
        return None
    try:
        return round(float(x), nd)
    except Exception:
        return None


def _bbox_from_points(pts: List[Tuple[float, float]]) -> Optional[Tuple[float, float, float, float]]:
    if not pts:
        return None
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    return min(xs), min(ys), max(xs), max(ys)


def _infer_extrude_depth(op: Dict[str, Any]) -> Optional[float]:
    v = _get_first(op, ["distance", "depth", "height", "length", "extent", "extrude_distance", "extrudeDepth"])
    if isinstance(v, (int, float)):
        return float(v)
    if isinstance(v, dict):
        vv = _get_first(v, ["distance", "depth", "height", "length", "value"])
        if isinstance(vv, (int, float)):
            return float(vv)
    return _find_numeric_recursive(op, include_keys=(
        "depth", "distance", "height", "thickness", "extent", "length",
        "extrude", "pad",
    ))


def _infer_extrude_operation(op: Dict[str, Any]) -> Optional[str]:
    t = str(op.get("type") or op.get("op") or op.get("kind") or op.get("name") or "").lower()
    if any(k in t for k in ("pocket", "cut", "subtract", "remove")):
        return "cut"
    if any(k in t for k in ("pad", "boss", "add", "join", "union", "extrude")):
        if "extrude" in t and not any(k in t for k in ("add", "join", "union")):
            pass
        else:
            return "add"

    for key in ("operation", "boolean", "mode", "op", "combine", "merge", "result"):
        v = op.get(key)
        if isinstance(v, str):
            vs = v.lower()
            if any(k in vs for k in ("cut", "subtract", "remove", "difference")):
                return "cut"
            if any(k in vs for k in ("add", "join", "union", "merge")):
                return "add"
        if isinstance(v, dict):
            vv = v.get("value")
            if isinstance(vv, str):
                vs = vv.lower()
                if any(k in vs for k in ("cut", "subtract", "remove", "difference")):
                    return "cut"
                if any(k in vs for k in ("add", "join", "union", "merge")):
                    return "add"

    s = _find_string_recursive(op, include_keys=(
        "operation", "boolean", "mode", "combine", "merge", "result", "type"
    ))
    if s:
        ss = s.lower()
        if any(k in ss for k in ("cut", "subtract", "remove", "difference")):
            return "cut"
        if any(k in ss for k in ("add", "join", "union", "merge")):
            return "add"

    return None


def _find_string_recursive(
    obj: Any,
    include_keys: Tuple[str, ...],
    *,
    max_depth: int = 5,
    _depth: int = 0,
) -> Optional[str]:
    if _depth > max_depth:
        return None
    if isinstance(obj, str):
        return obj
    if isinstance(obj, dict):
        for k, v in obj.items():
            ks = str(k).lower()
            if any(s in ks for s in include_keys):
                if isinstance(v, str):
                    return v
                if isinstance(v, dict) and isinstance(v.get("value"), str):
                    return v.get("value")
                found = _find_string_recursive(v, include_keys, max_depth=max_depth, _depth=_depth + 1)
                if found is not None:
                    return found
            found = _find_string_recursive(v, include_keys, max_depth=max_depth, _depth=_depth + 1)
            if found is not None:
                return found
    if isinstance(obj, (list, tuple)):
        for it in obj:
            found = _find_string_recursive(it, include_keys, max_depth=max_depth, _depth=_depth + 1)
            if found is not None:
                return found
    return None


def _find_numeric_recursive(
    obj: Any,
    include_keys: Tuple[str, ...],
    *,
    max_depth: int = 6,
    _depth: int = 0,
) -> Optional[float]:
    if _depth > max_depth:
        return None

    if isinstance(obj, (int, float)):
        return float(obj)

    if isinstance(obj, dict):
        for k, v in obj.items():
            ks = str(k).lower()
            if ks in {"p0", "p1", "p2", "center", "origin", "normal", "axis", "vector", "point", "points", "x", "y", "z"}:
                continue

            if any(s in ks for s in include_keys):
                if isinstance(v, (int, float)):
                    return float(v)
                if isinstance(v, dict):
                    vv = v.get("value")
                    if isinstance(vv, (int, float)):
                        return float(vv)
                found = _find_numeric_recursive(v, include_keys, max_depth=max_depth, _depth=_depth + 1)
                if found is not None:
                    return found

            found = _find_numeric_recursive(v, include_keys, max_depth=max_depth, _depth=_depth + 1)
            if found is not None:
                return found

    if isinstance(obj, (list, tuple)):
        for it in obj:
            found = _find_numeric_recursive(it, include_keys, max_depth=max_depth, _depth=_depth + 1)
            if found is not None:
                return found

    return None


def _infer_fillet_radius(op: Dict[str, Any]) -> Optional[float]:
    v = _get_first(op, ["radius", "r", "fillet_radius", "filletRadius"])
    if isinstance(v, (int, float)):
        return float(v)
    if isinstance(v, dict):
        vv = _get_first(v, ["radius", "r", "value"])
        if isinstance(vv, (int, float)):
            return float(vv)
    return None


def _infer_chamfer(op: Dict[str, Any]) -> Optional[Tuple[Optional[float], Optional[float]]]:
    d1 = _get_first(op, ["distance", "d", "length", "chamfer_distance", "chamferDistance"])
    d2 = _get_first(op, ["distance2", "d2", "length2", "chamfer_distance2", "chamferDistance2"])

    def tofloat(x):
        if isinstance(x, (int, float)):
            return float(x)
        if isinstance(x, dict):
            vv = _get_first(x, ["value", "distance", "length"])
            if isinstance(vv, (int, float)):
                return float(vv)
        return None

    return tofloat(d1), tofloat(d2)


def _plane_name_from_sketch(op: Dict[str, Any]) -> Optional[str]:
    pl = op.get("plane")
    if isinstance(pl, dict):
        k = _get_first(pl, ["kind", "name", "type"])
        if isinstance(k, str):
            return k
    k2 = _get_first(op, ["sketch_plane", "sketchPlane", "plane_kind"])
    if isinstance(k2, str):
        return k2
    return None


def summarize_ir(ir: Dict[str, Any]) -> Dict[str, Any]:
    meta = ir.get("meta", {}) or {}
    seq = ir.get("sequence", []) or []

    units = meta.get("units") or meta.get("unit") or "mm"
    out: Dict[str, Any] = {
        "source": meta.get("source"),
        "sample_id": meta.get("sample_id"),
        "units": units,
        "sketch": None,
        "extrude_depth": None,
        "extrude_operation": None,
        "fillet_radius": None,
        "chamfer": None,
        "has_extrude": False,
    }

    for op in seq:
        t = (op.get("type") or op.get("op") or op.get("kind") or "").lower()
        is_extrude_like = any(s in t for s in ("extrude", "extrusion", "pad", "pocket", "pull"))
        if is_extrude_like:
            out["has_extrude"] = True
            if out["extrude_operation"] is None:
                out["extrude_operation"] = _infer_extrude_operation(op)
            if out["extrude_depth"] is None:
                out["extrude_depth"] = _round(_infer_extrude_depth(op))
        if "fillet" in t and out["fillet_radius"] is None:
            out["fillet_radius"] = _round(_infer_fillet_radius(op), 3)
        if "chamfer" in t and out["chamfer"] is None:
            d1, d2 = _infer_chamfer(op)
            if d1 is not None:
                out["chamfer"] = {"d1": _round(d1, 3), "d2": _round(d2, 3) if d2 is not None else None}

    sketch_op = None
    for op in seq:
        t = (op.get("type") or op.get("op") or op.get("kind") or "").lower()
        if t == "sketch" or "sketch" in t:
            sketch_op = op
            break

    if sketch_op and isinstance(sketch_op, dict):
        ents = sketch_op.get("entities", []) or []
        plane = _plane_name_from_sketch(sketch_op)

        line_count = 0
        arc_count = 0
        circle_radii: List[float] = []
        circle_count = 0
        pts: List[Tuple[float, float]] = []

        for e in ents[:MAX_ENTS_SCAN]:
            k = (e.get("kind") or e.get("type") or "").lower()
            for key in ("p0", "p1", "center"):
                p = _as_point(e.get(key))
                if p:
                    pts.append(p)
                    if len(pts) >= MAX_PTS:
                        break
            if len(pts) >= MAX_PTS:
                break

            if "line" in k:
                line_count += 1
            elif "arc" in k:
                arc_count += 1
            elif "circle" in k:
                circle_count += 1
                r = e.get("radius")
                if isinstance(r, (int, float)):
                    circle_radii.append(float(r))
                elif isinstance(r, dict):
                    rv = r.get("value")
                    if isinstance(rv, (int, float)):
                        circle_radii.append(float(rv))

        bbox = _bbox_from_points(pts)
        bbox_wh = None
        if bbox:
            xmin, ymin, xmax, ymax = bbox
            bbox_wh = {"w": _round(xmax - xmin), "h": _round(ymax - ymin)}

        rr = None
        if circle_radii:
            circle_radii.sort()
            rr = {"min": _round(circle_radii[0], 3), "max": _round(circle_radii[-1], 3)}

        has_outer_profile = (line_count >= 4) and (bbox_wh is not None)
        circles_role = "circles"
        if TREAT_INNER_CIRCLES_AS_HOLES and has_outer_profile and circle_count >= 1:
            circles_role = "holes"
        if out.get("extrude_operation") == "cut" and circle_count >= 1 and line_count < 2:
            circles_role = "holes"

        out["sketch"] = {
            "plane": plane,
            "bbox": bbox_wh,
            "entities": {"lines": line_count, "arcs": arc_count, "circles": circle_count},
            "circles_role": circles_role,
            "circles": {"count": circle_count, "radius_range": rr} if circle_count else {"count": 0},
        }

    return out


def build_messages(summary: Dict[str, Any]) -> List[Dict[str, str]]:
    units = summary.get("units") or "mm"
    sk = summary.get("sketch") or {}
    plane = sk.get("plane") or "XY"
    bbox = (sk.get("bbox") or {})
    ents = (sk.get("entities") or {})
    circles = (sk.get("circles") or {})
    circles_role = (sk.get("circles_role") or "circles")

    w = bbox.get("w")
    h = bbox.get("h")

    l_count = ents.get("lines")
    a_count = ents.get("arcs")

    c_count = circles.get("count", 0)
    rr = circles.get("radius_range") or {}
    rmin = rr.get("min")
    rmax = rr.get("max")

    ex_depth = summary.get("extrude_depth")
    ex_op = summary.get("extrude_operation")
    fil_r = summary.get("fillet_radius")
    cham = summary.get("chamfer")

    profile_hint = None
    if c_count == 1 and (l_count or 0) < 2 and rmin is not None:
        profile_hint = f"single circular hole (radius {rmin} {units})" if circles_role == "holes" else f"single circle (radius {rmin} {units})"
    elif c_count >= 2 and (l_count or 0) < 2 and rmin is not None:
        profile_hint = f"{c_count} circles (radius about {rmin} {units})"
    elif (l_count or 0) >= 4 and w is not None and h is not None:
        if circles_role == "holes" and c_count:
            profile_hint = f"rectangle about {w} x {h} {units} with {c_count} circular holes"
        else:
            profile_hint = f"closed polygon (likely rectangle) about {w} x {h} {units}"
    elif w is not None and h is not None:
        profile_hint = f"closed profile within about {w} x {h} {units}"

    facts = []
    facts.append(f"Units: {units}")
    facts.append(f"Sketch plane: {plane}")
    if profile_hint is not None:
        facts.append(f"Profile hint: {profile_hint}")
    else:
        if w is not None and h is not None:
            facts.append(f"Overall sketch size approx: {w} x {h} {units}")
        if l_count is not None or a_count is not None:
            facts.append(f"Entity counts: lines={l_count}, arcs={a_count}, circles={c_count}")
    if c_count:
        label = "Holes" if circles_role == "holes" else "Circles"
        if rmin is not None and rmax is not None and rmax != rmin:
            facts.append(f"{label}: {c_count}, radius approx {rmin}~{rmax} {units}")
        elif rmin is not None:
            facts.append(f"{label}: {c_count}, radius approx {rmin} {units}")
        else:
            facts.append(f"{label}: {c_count}")
    should_extrude = bool(summary.get("has_extrude")) or ALWAYS_MENTION_EXTRUDE_IF_UNKNOWN
    if should_extrude:
        if ex_op in ("add", "cut"):
            facts.append(f"Extrude operation: {ex_op}")
        if ex_depth is not None:
            facts.append(f"Extrude depth approx: {ex_depth} {units}")
        else:
            facts.append("Extrude: yes (depth not provided)")
    if fil_r is not None:
        facts.append(f"Fillet radius approx: {fil_r} {units}")
    if cham is not None and cham.get("d1") is not None:
        d1 = cham.get("d1")
        d2 = cham.get("d2")
        facts.append(f"Chamfer approx: {d1} {units}" + (f" and {d2} {units}" if d2 is not None else ""))

    facts_block = "\n".join(f"- {x}" for x in facts)

    if LANG.lower() == "korean":
        sys = (
            "너는 CAD 모델을 만들기 위한 '사용자 지시문'을 작성한다.\n"
            "규칙:\n"
            "1) 출력은 1~2문장.\n"
            "2) FACTS의 숫자(치수/반지름/깊이)를 최대한 포함.\n"
            "3) 'Human:', 'Assistant:', '#', 목록, JSON, 코드, 설명 금지.\n"
            "4) FACTS에 Holes가 있으면 circles가 아니라 '홀/구멍'이라고 명시.\n"
            "5) FACTS에 Extrude operation이 cut이면 '컷 익스트루드/제거', add이면 '익스트루드로 생성'처럼 맞춰서 표현.\n"
            "6) 가능한 자연스러운 CAD 문장(예: 스케치→익스트루드→홀/필렛/모따기).\n"
        )
        user = f"FACTS:\n{facts_block}\n\n위 FACTS를 반영해 사용자 지시문 1~2문장으로 작성:"
    else:
        sys = (
            "You write the exact user instruction someone would type to recreate a CAD model.\n"
            "Rules:\n"
            "1) Output 1–2 sentences only.\n"
            "2) Describe the geometry explicitly (e.g., rectangle/circle/holes), not just 'a sketch with dimensions'.\n"
            "3) Use numeric values from FACTS when available (dimensions/radius/depth).\n"
            "4) MUST include an extrusion step if FACTS says 'Extrude: yes' or an extrude depth is provided.\n"
            "5) If FACTS mentions Holes, call them holes/through-holes (not just circles).\n"
            "6) If FACTS says 'Extrude operation: cut', use 'cut-extrude'/'remove material'. If 'add', use 'extrude to create a solid'.\n"
            "7) No speaker tags (Human/Assistant), no '#', no JSON, no bullet points, no explanations, no code.\n"
        )
        user = f"FACTS:\n{facts_block}\n\nWrite the user CAD instruction (1–2 sentences):"

    return [{"role": "system", "content": sys}, {"role": "user", "content": user}]


_SPEAKER_PREFIX = re.compile(r"(?im)^\s*(human|assistant|system|user)\s*[:：]\s*")
_LEADING_HASH = re.compile(r"(?m)^\s*#\s*")
_LEADING_YOU_ARE = re.compile(r"(?i)^\s*(you are to|you should|please)\s+")


def extract_clean(text: str) -> str:
    t = (text or "").strip()

    t = re.sub(r"<think>.*?</think>", "", t, flags=re.DOTALL | re.IGNORECASE)
    t = t.replace("<think>", "").replace("</think>", "").strip()

    t = re.sub(r"```.*?```", "", t, flags=re.DOTALL).strip()

    for _ in range(4):
        t2 = _SPEAKER_PREFIX.sub("", t).strip()
        t2 = _LEADING_HASH.sub("", t2).strip()
        t2 = _LEADING_YOU_ARE.sub("", t2).strip()
        if t2 == t:
            break
        t = t2

    t = re.sub(r"\s+", " ", t).strip()
    t = t.strip("“”\"' ").strip()
    return t


def load_model_tokenizer(model_name: str, local_rank: int, use_flash_attn: bool = True):
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        device_map = {"": local_rank}
    else:
        device_map = None

    tok = AutoTokenizer.from_pretrained(model_name)

    attn_impl = "sdpa"
    if use_flash_attn:
        try:
            import flash_attn
            attn_impl = "flash_attention_2"
        except Exception:
            attn_impl = "sdpa"

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype="auto",
        device_map=device_map,
        attn_implementation=attn_impl,
    )
    model.eval()
    return model, tok


def _apply_chat_template_safe(tok, messages):
    try:
        return tok.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
    except TypeError:
        return tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


@torch.no_grad()
def generate_batch(
    model,
    tok,
    messages_list: List[List[Dict[str, str]]],
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    top_k: int,
    repetition_penalty: float,
) -> List[str]:
    texts = [_apply_chat_template_safe(tok, m) for m in messages_list]
    batch = tok(texts, return_tensors="pt", padding=True, truncation=True)

    device = next(model.parameters()).device
    batch = {k: v.to(device) for k, v in batch.items()}

    gen = model.generate(
        **batch,
        max_new_tokens=max_new_tokens,
        do_sample=True if temperature > 0 else False,
        temperature=temperature if temperature > 0 else None,
        top_p=top_p,
        top_k=top_k,
        repetition_penalty=repetition_penalty,
        pad_token_id=tok.eos_token_id,
    )

    outs: List[str] = []
    in_len = batch["input_ids"].shape[1]
    for i in range(len(texts)):
        out_ids = gen[i][in_len:]
        outs.append(tok.decode(out_ids, skip_special_tokens=True))
    return outs


def make_out_path(neutral_root: Path, out_root: Path, f: Path) -> Path:
    rel = f.relative_to(neutral_root)
    name = rel.name
    if name.endswith(".neutral.json"):
        out_name = name[:-len(".neutral.json")] + ".neutral.prompt.txt"
    else:
        out_name = rel.with_suffix(".prompt.txt").name
    return out_root / rel.parent / out_name


def make_fallback_instruction(summary: Dict[str, Any]) -> str:
    units = summary.get("units") or "mm"
    sk = summary.get("sketch") or {}
    plane = (sk.get("plane") or "XY")
    bbox = (sk.get("bbox") or {})
    ents = (sk.get("entities") or {})
    circles = (sk.get("circles") or {})
    circles_role = (sk.get("circles_role") or "circles")

    w = bbox.get("w")
    h = bbox.get("h")
    l_count = ents.get("lines") or 0
    c_count = circles.get("count") or 0
    rr = circles.get("radius_range") or {}
    rmin = rr.get("min")

    parts: List[str] = []

    if l_count >= 4 and w is not None and h is not None and circles_role == "holes" and c_count >= 1 and rmin is not None:
        parts.append(
            f"Sketch a closed rectangular profile about {w} x {h} {units} on the {plane} plane and add {c_count} circular through-holes of radius about {rmin} {units}."
        )
    elif c_count == 1 and l_count < 2 and rmin is not None:
        parts.append(f"Sketch a circle of radius {rmin} {units} on the {plane} plane.")
    elif c_count >= 2 and rmin is not None:
        label = "holes" if circles_role == "holes" else "circles"
        parts.append(
            f"Sketch {c_count} {label} (radius about {rmin} {units}) on the {plane} plane." +
            (f" Keep the overall profile within about {w} x {h} {units}." if w is not None and h is not None else "")
        )
    elif l_count >= 4 and w is not None and h is not None:
        parts.append(f"Sketch a closed rectangular profile about {w} x {h} {units} on the {plane} plane.")
    elif w is not None and h is not None:
        parts.append(f"Sketch a closed profile within about {w} x {h} {units} on the {plane} plane.")
    else:
        parts.append(f"Create a sketch on the {plane} plane to define the profile.")

    ex_depth = summary.get("extrude_depth")
    ex_op = summary.get("extrude_operation")
    should_extrude = bool(summary.get("has_extrude")) or ALWAYS_MENTION_EXTRUDE_IF_UNKNOWN
    if should_extrude:
        if ex_op == "cut":
            if ex_depth is not None:
                parts.append(f"Cut-extrude the sketch by {ex_depth} {units} to remove material.")
            else:
                parts.append("Cut-extrude the sketch to remove material.")
        else:
            if ex_depth is not None:
                if circles_role == "holes" and c_count >= 1:
                    parts.append(f"Extrude the sketch by {ex_depth} {units} to form a solid with through-holes.")
                else:
                    parts.append(f"Extrude the sketch by {ex_depth} {units} to form a solid.")
            else:
                if circles_role == "holes" and c_count >= 1:
                    parts.append("Extrude the sketch to a reasonable thickness to form a solid with through-holes.")
                else:
                    parts.append("Extrude the sketch to a reasonable thickness to form a solid.")

    fil_r = summary.get("fillet_radius")
    if fil_r is not None:
        parts.append(f"Apply fillets with radius about {fil_r} {units} where appropriate.")
    cham = summary.get("chamfer") or {}
    if cham.get("d1") is not None:
        d1 = cham.get("d1")
        d2 = cham.get("d2")
        parts.append(f"Add a chamfer of about {d1} {units}" + (f" by {d2} {units}" if d2 is not None else "") + ".")

    if len(parts) >= 2:
        return " ".join(parts[:2])
    return " ".join(parts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--neutral_root", type=str, required=True)
    ap.add_argument("--out_root", type=str, required=True)
    ap.add_argument("--glob", type=str, default=DEFAULT_GLOB)
    ap.add_argument("--model", type=str, default=DEFAULT_MODEL)
    ap.add_argument("--max_new_tokens", type=int, default=DEFAULT_MAX_NEW_TOKENS)
    ap.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE)
    ap.add_argument("--top_p", type=float, default=DEFAULT_TOP_P)
    ap.add_argument("--top_k", type=int, default=DEFAULT_TOP_K)
    ap.add_argument("--repetition_penalty", type=float, default=DEFAULT_REP_PENALTY)
    ap.add_argument("--batch_size", type=int, default=DEFAULT_BATCH_SIZE)
    ap.add_argument("--no_flash_attn", action="store_true")
    args = ap.parse_args()

    neutral_root = Path(args.neutral_root).resolve()
    out_root = Path(args.out_root).resolve()
    ensure_dir(out_root)

    rank, world, local_rank = rank_info()

    files = list_ir_files(neutral_root, args.glob)
    shard = shard_files(files, rank, world)

    if rank == 0:
        print(f"[INFO] neutral_root={neutral_root}")
        print(f"[INFO] out_root={out_root}")
        print(f"[INFO] total_files={len(files)} world_size={world}")
    print(f"[RANK {rank}] shard_files={len(shard)} local_rank={local_rank}")

    model, tok = load_model_tokenizer(args.model, local_rank=local_rank, use_flash_attn=not args.no_flash_attn)

    bs = max(1, args.batch_size)
    for b0 in range(0, len(shard), bs):
        batch_files = shard[b0:b0 + bs]

        messages_list: List[List[Dict[str, str]]] = []
        out_paths: List[Path] = []
        summaries: List[Dict[str, Any]] = []

        for f in batch_files:
            ir = read_json(f)
            summary = summarize_ir(ir)
            msgs = build_messages(summary)
            messages_list.append(msgs)
            summaries.append(summary)

            out_txt = make_out_path(neutral_root, out_root, f)
            ensure_dir(out_txt.parent)
            out_paths.append(out_txt)

        raw_outs = generate_batch(
            model, tok, messages_list,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
            repetition_penalty=args.repetition_penalty,
        )

        for p, raw, summary in zip(out_paths, raw_outs, summaries):
            t = extract_clean(raw)
            low = t.lower()

            too_generic = (
                (not t)
                or low.startswith("create a new sketch")
                or ("sketch" in low and "dimension" in low and not any(k in low for k in ("rectangle", "circle", "hole", "arc", "line", "polygon")))
            )
            if too_generic:
                t = make_fallback_instruction(summary)

            should_extrude = bool(summary.get("has_extrude")) or ALWAYS_MENTION_EXTRUDE_IF_UNKNOWN
            if should_extrude and not re.search(r"(?i)\bextrude\b", t):
                t = make_fallback_instruction(summary)

            if not t:
                t = "Sketch the profile and extrude it to form a solid."

            p.write_text(t + "\n", encoding="utf-8")

        if (b0 // bs) % 25 == 0:
            print(f"[RANK {rank}] processed {b0 + len(batch_files)}/{len(shard)}")

    if rank == 0:
        print("[DONE]")


if __name__ == "__main__":
    main()