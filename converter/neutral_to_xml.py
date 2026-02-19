from __future__ import annotations

import argparse
import base64
import hashlib
import json
import re
import sys
import uuid
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import xml.etree.ElementTree as ET
from xml.dom import minidom


QIF_NS = "http://qifstandards.org/xsd/qif3"
NEUTRAL_NS = "urn:cadneutral:neutral" 

def _now_iso_kst() -> str:
    kst = timezone(timedelta(hours=9))
    return datetime.now(tz=kst).isoformat(timespec="seconds")


def _b64_utf8(s: str) -> str:
    return base64.b64encode(s.encode("utf-8")).decode("ascii")


def _sha256_utf8(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def _pretty_xml(elem: ET.Element) -> str:
    raw = ET.tostring(elem, encoding="utf-8")
    dom = minidom.parseString(raw)
    pretty = dom.toprettyxml(indent="  ", encoding="utf-8").decode("utf-8")
    lines = [ln for ln in pretty.splitlines() if ln.strip() != ""]
    return "\n".join(lines) + "\n"


def _load_json(p: Path) -> Dict[str, Any]:
    with p.open("r", encoding="utf-8") as f:
        return json.load(f)


def _ensure_dict(x: Any) -> Dict[str, Any]:
    return x if isinstance(x, dict) else {}


def _ensure_list(x: Any) -> List[Any]:
    return x if isinstance(x, list) else []


def _qif(tag: str) -> str:
    return f"{{{QIF_NS}}}{tag}"


def _cn(tag: str) -> str:
    return f"{{{NEUTRAL_NS}}}{tag}"


def _add_text(parent: ET.Element, tag: str, text: str) -> ET.Element:
    e = ET.SubElement(parent, tag)
    e.text = text
    return e


_tag_re = re.compile(r"[^A-Za-z0-9_.-]")
def _sanitize_tag(name: str) -> str:
    name = str(name)
    name = name.replace(":", "_")
    name = _tag_re.sub("_", name)
    if not name:
        name = "field"
    if not re.match(r"[A-Za-z_]", name[0]):
        name = "_" + name
    return name


def _to_text_value(v: Any) -> str:
    if v is None:
        return ""
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, (int, float)):
        return str(v)
    return str(v)


def _norm_length_unit(u: str) -> Tuple[str, Optional[str]]:
    key = str(u).strip().lower()
    if key in {"m", "meter", "metre"}:
        return ("meter", None)
    if key in {"mm", "millimeter", "millimetre"}:
        return ("millimeter", "0.001")
    if key in {"cm", "centimeter", "centimetre"}:
        return ("centimeter", "0.01")
    if key in {"in", "inch", "inches"}:
        return ("inch", "0.0254")
    if key in {"ft", "foot", "feet"}:
        return ("foot", "0.3048")
    return (u if u else "millimeter", "0.001")


def _norm_angle_unit(u: str) -> Tuple[str, Optional[str]]:
    key = str(u).strip().lower()
    if key in {"rad", "radian", "radians"}:
        return ("radian", None)
    if key in {"deg", "degree", "degrees"}:
        return ("degree", "0.017453292519943295") 
    return (u if u else "degree", "0.017453292519943295")

def _extract_meta(neutral: Dict[str, Any]) -> Dict[str, str]:
    meta = _ensure_dict(neutral.get("meta"))
    return {
        "source": str(meta.get("source", neutral.get("source", "CadNeutral"))),
        "sample_id": str(meta.get("sample_id", neutral.get("sample_id", ""))),
        "input_relpath": str(meta.get("input_relpath", neutral.get("input_relpath", ""))),
        "schema": str(neutral.get("schema", "")),
    }


def _extract_units(neutral: Dict[str, Any]) -> Dict[str, str]:
    units = _ensure_dict(neutral.get("units"))
    meta = _ensure_dict(neutral.get("meta"))
    meta_units = str(meta.get("units", "")).strip()

    length_unit = str(units.get("length_unit", "")).strip()
    angle_unit = str(units.get("angle_unit", "")).strip()
    if not length_unit and meta_units:
        length_unit = meta_units

    return {
        "length_unit": length_unit or "mm",
        "angle_unit": angle_unit or "deg",
        "angle_zero_axis": str(units.get("angle_zero_axis", "+X")),
        "angle_positive": str(units.get("angle_positive", "CCW")),
    }


def _extract_sequence(neutral: Dict[str, Any]) -> List[Dict[str, Any]]:
    seq = neutral.get("sequence", None)
    if isinstance(seq, list):
        return [x if isinstance(x, dict) else {"value": x} for x in seq]

    feats = neutral.get("features", None)
    if isinstance(feats, list):
        return [x if isinstance(x, dict) else {"value": x} for x in feats]

    return []

def _expand_value_to_xml(
    parent: ET.Element,
    key: str,
    value: Any,
    max_list_items: int,
) -> None:
    tag = _cn(_sanitize_tag(key))
    if isinstance(value, dict):
        node = ET.SubElement(parent, tag)
        for k, v in value.items():
            _expand_value_to_xml(node, k, v, max_list_items)
    elif isinstance(value, list):
        node = ET.SubElement(parent, tag)
        n = len(value)
        node.set("n", str(n))
        limit = min(n, max_list_items)
        for i in range(limit):
            item = value[i]
            item_el = ET.SubElement(node, _cn("Item"), attrib={"idx": str(i)})
            if isinstance(item, dict):
                if "id" in item:
                    item_el.set("id", _to_text_value(item.get("id")))
                if "kind" in item:
                    item_el.set("kind", _to_text_value(item.get("kind")))
                for k, v in item.items():
                    _expand_value_to_xml(item_el, k, v, max_list_items)
            elif isinstance(item, list):
                _expand_value_to_xml(item_el, "list", item, max_list_items)
            else:
                item_el.text = _to_text_value(item)
        if n > limit:
            more = ET.SubElement(node, _cn("Truncated"))
            more.text = f"true (showing first {limit} of {n})"
    else:
        node = ET.SubElement(parent, tag)
        node.text = _to_text_value(value)


def _feature_type(feat: Dict[str, Any]) -> str:
    return str(feat.get("type", feat.get("op", feat.get("kind", "Unknown"))))


def _feature_id(feat: Dict[str, Any], fallback: int) -> str:
    return str(feat.get("id", fallback))


def _feature_name(feat: Dict[str, Any]) -> str:
    return str(feat.get("name", ""))


def _expand_feature_visible(
    feature_el: ET.Element,
    feat: Dict[str, Any],
    max_list_items: int,
    embed_base64_payload: bool,
) -> None:
    """
    Make feature readable:
    - cn:params expanded XML (no base64 required)
    - optionally cn:payload(json+base64) for exact original
    """
    ftype = _feature_type(feat)

    params = ET.SubElement(feature_el, _cn("params"))

    if ftype.lower() == "sketch":
        if "plane" in feat:
            _expand_value_to_xml(params, "plane", feat.get("plane"), max_list_items)
        if "summary" in feat:
            _expand_value_to_xml(params, "summary", feat.get("summary"), max_list_items)
        if "entities" in feat:
            _expand_value_to_xml(params, "entities", feat.get("entities"), max_list_items)

        for k, v in feat.items():
            if k in {"id", "type", "name", "plane", "summary", "entities"}:
                continue
            _expand_value_to_xml(params, k, v, max_list_items)

    elif ftype.lower() == "extrude":
        preferred = ["sketch_ref", "operation", "end_condition", "distance_mm", "direction"]
        for k in preferred:
            if k in feat:
                _expand_value_to_xml(params, k, feat.get(k), max_list_items)
        for k, v in feat.items():
            if k in {"id", "type", "name"} or k in preferred:
                continue
            _expand_value_to_xml(params, k, v, max_list_items)

    elif ftype.lower() == "chamfer":
        preferred = ["distance1_mm", "distance2_mm", "edges_ref", "mode"]
        for k in preferred:
            if k in feat:
                _expand_value_to_xml(params, k, feat.get(k), max_list_items)
        for k, v in feat.items():
            if k in {"id", "type", "name"} or k in preferred:
                continue
            _expand_value_to_xml(params, k, v, max_list_items)

    else:
        for k, v in feat.items():
            if k in {"id", "type", "name"}:
                continue
            _expand_value_to_xml(params, k, v, max_list_items)

    if embed_base64_payload:
        payload_json = json.dumps(feat, ensure_ascii=False, separators=(",", ":"), sort_keys=True)
        pay = ET.SubElement(feature_el, _cn("payload"), attrib={
            "format": "json+base64",
            "sha256": _sha256_utf8(payload_json),
        })
        pay.text = _b64_utf8(payload_json)

def build_qif_document_from_neutral_visible(
    neutral: Dict[str, Any],
    qpid: Optional[str],
    embed_original_json: bool,
    embed_base64_payloads: bool,
    max_list_items: int,
) -> ET.Element:
    meta = _extract_meta(neutral)
    units = _extract_units(neutral)
    seq = _extract_sequence(neutral)

    qpid_val = (qpid or str(uuid.uuid4())).lower()

    ET.register_namespace("", QIF_NS)
    ET.register_namespace("cn", NEUTRAL_NS)

    root = ET.Element(_qif("QIFDocument"), attrib={
        "versionQIF": "3.0.0",
        "idMax": "0",
    })

    _add_text(root, _qif("QPId"), qpid_val)

    ver = ET.SubElement(root, _qif("Version"))
    _add_text(ver, _qif("TimeCreated"), _now_iso_kst())

    hdr = ET.SubElement(root, _qif("Header"))
    if meta["source"]:
        _add_text(hdr, _qif("Application"), meta["source"])
    _add_text(hdr, _qif("Author"), "unknown")
    if meta["input_relpath"]:
        _add_text(hdr, _qif("ApplicationSource"), meta["input_relpath"])

    desc_parts = []
    if meta["schema"]:
        desc_parts.append(f"schema={meta['schema']}")
    if meta["sample_id"]:
        desc_parts.append(f"sample_id={meta['sample_id']}")
    if desc_parts:
        _add_text(hdr, _qif("Description"), " | ".join(desc_parts))

    fu = ET.SubElement(root, _qif("FileUnits"))
    pu = ET.SubElement(fu, _qif("PrimaryUnits"))

    lin_name, lin_factor = _norm_length_unit(units["length_unit"])
    lin = ET.SubElement(pu, _qif("LinearUnit"))
    _add_text(lin, _qif("SIUnitName"), "meter")
    _add_text(lin, _qif("UnitName"), lin_name)
    if lin_factor is not None:
        conv = ET.SubElement(lin, _qif("UnitConversion"))
        _add_text(conv, _qif("Factor"), lin_factor)

    ang_name, ang_factor = _norm_angle_unit(units["angle_unit"])
    ang = ET.SubElement(pu, _qif("AngularUnit"))
    _add_text(ang, _qif("SIUnitName"), "radian")
    _add_text(ang, _qif("UnitName"), ang_name)
    if ang_factor is not None:
        conv = ET.SubElement(ang, _qif("UnitConversion"))
        _add_text(conv, _qif("Factor"), ang_factor)

    ud = ET.SubElement(root, _qif("UserDataXML"))
    cn_root = ET.SubElement(ud, _cn("CadNeutral"), attrib={
        "schema": meta["schema"] or "",
        "embeddedAt": _now_iso_kst(),
    })

    cn_meta = ET.SubElement(cn_root, _cn("Meta"))
    for k in ["source", "sample_id", "input_relpath", "schema"]:
        v = meta.get(k, "")
        if v:
            _add_text(cn_meta, _cn(k), v)

    cn_units = ET.SubElement(cn_root, _cn("Units"))
    for k in ["length_unit", "angle_unit", "angle_zero_axis", "angle_positive"]:
        v = units.get(k, "")
        if v:
            _add_text(cn_units, _cn(k), v)

    cn_seq = ET.SubElement(cn_root, _cn("Sequence"), attrib={"n": str(len(seq))})
    for i, feat in enumerate(seq):
        f_id = _feature_id(feat, i)
        f_type = _feature_type(feat)
        f_name = _feature_name(feat)

        feat_el = ET.SubElement(cn_seq, _cn("Feature"), attrib={
            "idx": str(i),
            "id": f_id,
            "type": f_type,
        })
        if f_name:
            _add_text(feat_el, _cn("name"), f_name)

        _expand_feature_visible(
            feature_el=feat_el,
            feat=feat,
            max_list_items=max_list_items,
            embed_base64_payload=embed_base64_payloads,
        )

    if embed_original_json:
        original = json.dumps(neutral, ensure_ascii=False, separators=(",", ":"), sort_keys=True)
        orig_el = ET.SubElement(cn_root, _cn("OriginalNeutralJson"), attrib={
            "format": "json+base64",
            "sha256": _sha256_utf8(original),
        })
        orig_el.text = _b64_utf8(original)

    return root


def convert_one(in_path: Path, out_path: Path, args: argparse.Namespace) -> None:
    neutral = _load_json(in_path)
    root = build_qif_document_from_neutral_visible(
        neutral=neutral,
        qpid=args.qpid,
        embed_original_json=not args.no_embed_original,
        embed_base64_payloads=args.embed_base64_payloads,
        max_list_items=args.max_list_items,
    )
    xml_text = _pretty_xml(root)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(xml_text, encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("input", help="Neutral JSON file or directory")
    ap.add_argument("output", help="Output QIF XML file or directory")
    ap.add_argument("--glob", default="**/*.json",
                    help="When input is a directory, which files to convert (default: **/*.json)")
    ap.add_argument("--qpid", default=None,
                    help="Optional QPId UUID for the QIF document. If omitted, a random UUID is used.")
    ap.add_argument("--no-embed-original", action="store_true",
                    help="Do NOT embed full original neutral JSON(base64) into UserDataXML")
    ap.add_argument("--embed-base64-payloads", action="store_true",
                    help="Also embed per-feature payload(json+base64). Default is XML-visible only.")
    ap.add_argument("--max-list-items", type=int, default=200,
                    help="Max list items to expand (prevent huge XML). Default: 200")
    args = ap.parse_args()

    in_path = Path(args.input)
    out_path = Path(args.output)

    if in_path.is_file():
        if out_path.is_dir() or str(args.output).endswith(("/", "\\")):
            out_file = out_path / (in_path.stem + ".qif.xml")
        else:
            out_file = out_path
        convert_one(in_path, out_file, args)
        return 0

    if in_path.is_dir():
        out_dir = out_path
        out_dir.mkdir(parents=True, exist_ok=True)
        files = sorted(in_path.glob(args.glob))
        if not files:
            print(f"[WARN] No files matched: {in_path} / {args.glob}", file=sys.stderr)
            return 0

        for fp in files:
            rel = fp.relative_to(in_path)
            out_fp = out_dir / rel.with_suffix("")
            out_fp = out_fp.with_name(out_fp.name + ".qif.xml")
            convert_one(fp, out_fp, args)
        return 0

    print(f"[ERROR] Input not found: {in_path}", file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
