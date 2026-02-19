from __future__ import annotations

import argparse
import json
import os
import re
import sys
import traceback
from datetime import timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.distributed as dist
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed

NON_ASCII_RE = re.compile(r"[^\x00-\x7F]")
NUM_RE = re.compile(r"[-+]?\d*\.\d+|[-+]?\d+")
BAD_WORDS_RE = [
    re.compile(r"\bjson\b", re.IGNORECASE),
    re.compile(r"\bir\b", re.IGNORECASE),
    re.compile(r"cad_ir", re.IGNORECASE),
    re.compile(r"\bschema\b", re.IGNORECASE),
    re.compile(r"\btoken\b", re.IGNORECASE),
    re.compile(r"\beos\b", re.IGNORECASE),
    re.compile(r"\bassistant\b", re.IGNORECASE),
    re.compile(r"<think>", re.IGNORECASE),
    re.compile(r"</think>", re.IGNORECASE),
]
STEPY_RE = re.compile(r"^\s*(step\s*\d+[:\)\.\-]|\-\s+|\*\s+)", re.IGNORECASE)

def get_dist_env() -> Tuple[int, int, int]:
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", os.environ.get("SLURM_LOCALID", "0")))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    return rank, local_rank, world_size


def dist_init_if_needed() -> Tuple[int, int, int]:
    rank, local_rank, world_size = get_dist_env()
    if world_size > 1 and not dist.is_initialized():
        dist.init_process_group(
            backend="nccl",
            timeout=timedelta(minutes=120),
        )
    return rank, local_rank, world_size


def dist_barrier_if_needed() -> None:
    if dist.is_available() and dist.is_initialized():
        dist.barrier()


def dist_destroy_if_needed() -> None:
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def enable_max_determinism(seed: int) -> None:
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    os.environ["PYTHONHASHSEED"] = str(seed)

    set_seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    try:
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    except Exception:
        pass
    try:
        torch.use_deterministic_algorithms(True)
    except Exception:
        pass


def reseed_for_call(seed: int) -> None:
    set_seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))

def safe_get(d: Dict[str, Any], keys: List[str], default=None):
    cur = d
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


def collect_numbers_from_ir(ir: Dict[str, Any], max_n: int = 200) -> List[float]:
    nums: List[float] = []

    seq = ir.get("sequence", [])
    if isinstance(seq, list):
        for feat in seq:
            if not isinstance(feat, dict):
                continue
            for k in ["distance_mm", "radius_mm", "diameter_mm", "width_mm", "height_mm", "depth_mm",
                      "fillet_radius_mm", "chamfer_d1_mm", "chamfer_d2_mm", "angle_deg"]:
                v = feat.get(k)
                if isinstance(v, (int, float)) and 0.0001 <= float(v) <= 500000:
                    nums.append(float(v))

            params = feat.get("params")
            if isinstance(params, dict):
                for _, v in params.items():
                    if isinstance(v, (int, float)) and 0.0001 <= float(v) <= 500000:
                        nums.append(float(v))

    out = []
    seen = set()
    for v in nums:
        key = round(v, 6)
        if key in seen:
            continue
        seen.add(key)
        out.append(v)
    return out[:max_n]


def feature_histogram(seq: List[Dict[str, Any]]) -> Dict[str, int]:
    hist: Dict[str, int] = {}
    for f in seq:
        t = f.get("type") or f.get("op_name") or "Unknown"
        if not isinstance(t, str):
            t = "Unknown"
        hist[t] = hist.get(t, 0) + 1
    return hist


def summarize_feature(feat: Dict[str, Any], idx: int, entity_sample: int = 6) -> str:
    t = feat.get("type") or feat.get("op_name") or "Unknown"
    name = feat.get("name") or f"{t} {idx}"
    parts = [f"{idx}) {t} ({name})"]

    if str(t).lower() == "sketch":
        plane_kind = safe_get(feat, ["plane", "kind"], None) or feat.get("plane") or "unspecified"
        summ = feat.get("summary") if isinstance(feat.get("summary"), dict) else {}
        n_ent = summ.get("n_entities", None)
        bbox = summ.get("bbox", None)
        if plane_kind:
            parts.append(f"plane={plane_kind}")
        if isinstance(n_ent, int):
            parts.append(f"entities={n_ent}")
        if isinstance(bbox, list) and len(bbox) == 4:
            parts.append(f"bbox={bbox}")

        ents = feat.get("entities")
        if isinstance(ents, list) and ents:
            kinds = []
            for e in ents[:entity_sample]:
                if isinstance(e, dict):
                    k = e.get("kind")
                    if isinstance(k, str):
                        kinds.append(k)
            if kinds:
                parts.append("entity_kinds=" + ",".join(kinds))

    if str(t).lower() in ["extrude", "extrudefeature", "cutextrude", "extrude_cut", "extrude_cutfeature"]:
        dist_mm = feat.get("distance_mm")
        op = feat.get("operation")
        endc = feat.get("end_condition")
        if isinstance(dist_mm, (int, float)):
            parts.append(f"distance_mm={dist_mm}")
        if isinstance(op, str):
            parts.append(f"op={op}")
        if isinstance(endc, str):
            parts.append(f"end={endc}")
        sk = feat.get("sketch_ref")
        if isinstance(sk, str):
            parts.append(f"sketch_ref={sk}")

    if str(t).lower() == "revolve":
        ang = feat.get("angle_deg")
        if isinstance(ang, (int, float)):
            parts.append(f"angle_deg={ang}")

    if str(t).lower() == "fillet":
        r = feat.get("radius_mm") or feat.get("fillet_radius_mm")
        if isinstance(r, (int, float)):
            parts.append(f"radius_mm={r}")

    if str(t).lower() == "chamfer":
        d1 = feat.get("chamfer_d1_mm") or feat.get("d1_mm")
        d2 = feat.get("chamfer_d2_mm") or feat.get("d2_mm")
        if isinstance(d1, (int, float)):
            parts.append(f"d1_mm={d1}")
        if isinstance(d2, (int, float)):
            parts.append(f"d2_mm={d2}")

    if str(t).lower() in ["pattern", "linearpattern", "circularpattern"]:
        cnt = feat.get("count")
        if isinstance(cnt, int):
            parts.append(f"count={cnt}")

    if str(t).lower() == "mirror":
        plane = feat.get("plane")
        if isinstance(plane, str):
            parts.append(f"mirror_plane={plane}")

    if str(t).lower() in ["hole", "drill", "holefeature"]:
        dia = feat.get("diameter_mm")
        if isinstance(dia, (int, float)):
            parts.append(f"diameter_mm={dia}")

    return "; ".join(parts)


def ir_to_text_summary(ir: Dict[str, Any], max_features: int = 32) -> str:
    meta = ir.get("meta", {}) if isinstance(ir.get("meta"), dict) else {}
    units = meta.get("units") or ir.get("units") or "mm"
    sample_id = meta.get("sample_id") or ir.get("sample_id") or "unspecified"

    seq = ir.get("sequence", [])
    if not isinstance(seq, list):
        seq = []

    seq2: List[Dict[str, Any]] = [x for x in seq if isinstance(x, dict)]
    hist = feature_histogram(seq2)

    lines: List[str] = []
    lines.append(f"sample_id: {sample_id}")
    lines.append(f"units: {units}")
    lines.append(f"sequence_len: {len(seq2)}")
    lines.append("feature_histogram: " + ", ".join([f"{k}={v}" for k, v in sorted(hist.items())]))

    lines.append("sequence:")
    for i, feat in enumerate(seq2[:max_features], start=1):
        lines.append("  " + summarize_feature(feat, i))

    nums = collect_numbers_from_ir(ir, max_n=120)
    if nums:
        nums_str = ", ".join([str(int(x)) if abs(x-round(x)) < 1e-6 else f"{x:.6f}".rstrip("0").rstrip(".") for x in nums[:80]])
        lines.append("numbers_seen: " + nums_str)

    return "\n".join(lines)

def build_messages(summary_text: str, lang: str) -> List[Dict[str, str]]:
    # System message: force output style
    if lang == "en":
        sys_msg = (
            "You are a CAD design assistant.\n"
            "Your job is to write ONE natural English instruction that a human would say to a parametric CAD system.\n\n"
            "Hard rules:\n"
            "- Output ONLY plain English prose (no headings, no bullet lists, no 'Step 1', no JSON, no code).\n"
            "- Do NOT mention IR, JSON, schema, tokens, or that you were given data.\n"
            "- Use the operations that appear in the data (Sketch/Extrude/Cut/Fillet/Chamfer/Revolve/Pattern/Mirror/Hole, etc.).\n"
            "- If a numeric dimension is clearly present, you may mention a few; otherwise describe proportions.\n"
            "- Keep it short: 2 to 6 sentences.\n"
            "- English only.\n"
        )
        usr_msg = (
            "Here is the CAD model information:\n"
            f"{summary_text}\n\n"
            "Now write the user instruction."
        )
    else:
        sys_msg = (
            "너는 CAD 설계 보조자다.\n"
            "사람이 CAD에게 말하듯이 '한 문단' 지시문을 작성해라.\n"
            "규칙: 제목/불릿/Step/JSON/코드 금지, IR/JSON/schema/token 언급 금지.\n"
        )
        usr_msg = f"CAD 정보:\n{summary_text}\n\n지시문을 작성해라."

    messages = [
        {"role": "system", "content": sys_msg + "\n/no_think"},
        {"role": "user", "content": usr_msg},
    ]
    return messages


def apply_chat_template_safe(tokenizer: AutoTokenizer, messages: List[Dict[str, str]]) -> str:
    try:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
    except TypeError:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )


def is_banned(text: str) -> Tuple[bool, str]:
    t = (text or "").strip()
    for r in BAD_WORDS_RE:
        if r.search(t):
            return True, f"banned: {r.pattern}"
    if re.search(r"\bstep\s*\d+\b", t, flags=re.IGNORECASE):
        return True, "banned: step-format"
    return False, "ok"


def clean_output(text: str) -> str:
    t = (text or "").replace("\r\n", "\n").replace("\r", "\n").strip()
    t = re.sub(r"<think>.*?</think>", "", t, flags=re.DOTALL | re.IGNORECASE)
    t = t.replace("<think>", "").replace("</think>", "").strip()

    lines: List[str] = []
    for line in t.split("\n"):
        line = line.strip()
        if not line:
            continue
        if NON_ASCII_RE.search(line):
            continue
        if STEPY_RE.match(line):
            line = re.sub(r"^\s*(step\s*\d+[:\)\.\-]\s*)", "", line, flags=re.IGNORECASE).strip()
            line = line.lstrip("-* ").strip()
            if not line:
                continue
        lines.append(line)

    out = " ".join(lines)
    out = re.sub(r"\s+", " ", out).strip()
    if not out:
        out = "Create the shape using the sketches and features shown in the model, then apply the same extrusions, cuts, and edge treatments to match the overall form."
    return out


@torch.inference_mode()
def qwen_generate(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    messages: List[Dict[str, str]],
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    seed: int,
) -> str:
    do_sample = True
    if temperature <= 0:
        do_sample = False
        temperature = 1.0
        top_p = 1.0

    reseed_for_call(seed)

    text = apply_chat_template_safe(tokenizer, messages)
    inputs = tokenizer(
        text,
        return_tensors="pt",
        padding=True,
        truncation=True,
    ).to(model.device)

    out_ids = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=do_sample,
        temperature=float(temperature),
        top_p=float(top_p),
        repetition_penalty=1.06,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
        use_cache=True,
    )

    gen = out_ids[0, inputs["input_ids"].shape[1] :]
    raw = tokenizer.decode(gen, skip_special_tokens=True).strip()
    return raw


def make_out_path(out_root: Path, ir_root: Path, ir_path: Path, out_suffix: str) -> Path:
    rel = ir_path.relative_to(ir_root)
    out_name = rel.name.replace(".neutral.json", out_suffix)
    return (out_root / rel.parent / out_name).resolve()


def run_one_file(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    ir_path: Path,
    out_path: Path,
    args: argparse.Namespace,
) -> Tuple[bool, str]:
    try:
        if out_path.exists() and not args.overwrite:
            return True, "SKIP_EXISTS"

        ir = load_json(ir_path)
        summary_text = ir_to_text_summary(ir, max_features=args.max_features)
        messages = build_messages(summary_text, lang=args.lang)

        final_text = ""
        last_raw = ""

        for attempt in range(max(1, args.retries + 1)):
            raw = qwen_generate(
                model=model,
                tokenizer=tokenizer,
                messages=messages,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
                seed=args.seed + attempt,
            )
            last_raw = raw

            cleaned = clean_output(raw)
            banned, _ = is_banned(cleaned)
            if banned:
                continue

            final_text = cleaned
            break

        if not final_text:
            final_text = clean_output(last_raw)

        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(final_text.rstrip() + "\n", encoding="utf-8")
        return True, "OK"

    except Exception:
        return False, traceback.format_exc()


def main() -> None:
    ap = argparse.ArgumentParser()

    ap.add_argument("--model", type=str, default="Qwen/Qwen3-8B")

    ap.add_argument("--neutral_json", type=str, default="")
    ap.add_argument("--out_txt", type=str, default="")

    ap.add_argument("--ir_root", type=str, default="")
    ap.add_argument("--out_root", type=str, default="")
    ap.add_argument("--glob", type=str, default="**/*.ir.json")
    ap.add_argument("--out_suffix", type=str, default="_prompt.txt")

    ap.add_argument("--overwrite", type=int, default=1)

    ap.add_argument("--lang", type=str, default="en", choices=["en", "ko"])
    ap.add_argument("--max_features", type=int, default=32)

    ap.add_argument("--max_new_tokens", type=int, default=220)
    ap.add_argument("--temperature", type=float, default=0.35, help="0=greedy, >0=sampling")
    ap.add_argument("--top_p", type=float, default=0.9)
    ap.add_argument("--retries", type=int, default=2)

    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--deterministic", type=int, default=1)
    ap.add_argument("--dtype", choices=["auto", "float16", "bfloat16"], default="auto")
    ap.add_argument("--attn", choices=["sdpa", "flash_attention_2", "eager"], default="sdpa")

    args = ap.parse_args()

    rank, local_rank, world_size = dist_init_if_needed()

    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
    else:
        device = torch.device("cpu")

    base_seed = int(args.seed) + rank * 100000
    if args.deterministic == 1:
        enable_max_determinism(base_seed)
    else:
        set_seed(base_seed)

    if args.dtype == "auto":
        dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    elif args.dtype == "float16":
        dtype = torch.float16
    else:
        dtype = torch.bfloat16

    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    try:
        model = AutoModelForCausalLM.from_pretrained(
            args.model,
            torch_dtype=dtype,
            attn_implementation=args.attn,
            low_cpu_mem_usage=True,
        )
    except TypeError:
        model = AutoModelForCausalLM.from_pretrained(
            args.model,
            torch_dtype=dtype,
            low_cpu_mem_usage=True,
        )

    model.to(device)
    model.eval()

    if rank == 0:
        print(f"[INFO] model={args.model} dtype={dtype} attn={args.attn} world_size={world_size}")
        print(f"[INFO] gen: max_new_tokens={args.max_new_tokens} temp={args.temperature} top_p={args.top_p} retries={args.retries}")

    if args.ir_json:
        ir_path = Path(args.ir_json).resolve()
        if not ir_path.exists():
            raise SystemExit(f"[ERR] IR not found: {ir_path}")
        if not args.out_txt:
            raise SystemExit("[ERR] --out_txt is required with --ir_json")
        out_path = Path(args.out_txt).resolve()

        if rank == 0:
            ok, msg = run_one_file(model, tokenizer, ir_path, out_path, args)
            if ok:
                print(f"[OK] {msg} -> {out_path}")
            else:
                print(f"[ERR] failed: {ir_path}", file=sys.stderr)
                print(msg, file=sys.stderr)

        dist_barrier_if_needed()
        dist_destroy_if_needed()
        return

    if not args.ir_root or not args.out_root:
        raise SystemExit("[ERR] Dir mode requires --ir_root and --out_root (or use --ir_json + --out_txt).")

    ir_root = Path(args.ir_root).resolve()
    out_root = Path(args.out_root).resolve()
    if not ir_root.exists():
        raise SystemExit(f"[ERR] ir_root not found: {ir_root}")
    out_root.mkdir(parents=True, exist_ok=True)

    all_files = sorted([p for p in ir_root.glob(args.glob) if p.is_file()])
    if rank == 0:
        print(f"[INFO] found {len(all_files)} files under {ir_root} (glob={args.glob})")
        print("[INFO] sharding: files[rank::world_size]")

    my_files = all_files[rank::world_size]

    ok_cnt = 0
    fail_cnt = 0
    skip_cnt = 0

    for i, ir_path in enumerate(my_files, 1):
        out_path = make_out_path(out_root, ir_root, ir_path, args.out_suffix)

        ok, msg = run_one_file(model, tokenizer, ir_path, out_path, args)
        if ok:
            if msg == "SKIP_EXISTS":
                skip_cnt += 1
            else:
                ok_cnt += 1
        else:
            fail_cnt += 1
            print(f"[R{rank}] [FAIL] {ir_path}", file=sys.stderr)
            print(msg, file=sys.stderr)

        if i % 50 == 0 or i == len(my_files):
            print(f"[R{rank}] progress {i}/{len(my_files)} (ok={ok_cnt}, skip={skip_cnt}, fail={fail_cnt})")

    dist_barrier_if_needed()
    if rank == 0:
        print("[INFO] all ranks finished.")
    dist_destroy_if_needed()


if __name__ == "__main__":
    main()
