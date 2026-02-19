import json
from pathlib import Path

BASE_DIR = Path("../FT/data")
OUTPUT_JSONL = BASE_DIR / "train.jsonl"

DATASETS = [
    {
        "name": "DeepCAD",
        "neutral": BASE_DIR / "deepcad" / "deepcad",
        "prompt":  BASE_DIR / "deepcad" / "deepcad_prompt_qwen3"
    },
    {
        "name": "WHUCAD",
        "neutral": BASE_DIR / "WHUCAD" / "WHUCAD",
        "prompt":  BASE_DIR / "WHUCAD" / "WHUCAD_prompt_qwen3_v2"
    }
]

records = []
stats = {}

def find_prompt(prompt_dir: Path, neutral_path: Path) -> Path | None:
    """
    Neutral: xxx.neutral.json
    Prompt candidates:
      1) xxx.neutral.prompt.txt   (DeepCAD)
      2) xxx_prompt.txt           (WHUCAD old)
    """
    stem = neutral_path.stem

    candidates = [
        prompt_dir / f"{stem}.prompt.txt",
        prompt_dir / f"{stem.replace('.neutral', '')}_prompt.txt"
    ]

    for c in candidates:
        if c.exists():
            return c
    return None

for ds in DATASETS:
    print(f"\n=== Processing {ds['name']} ===")
    stats[ds["name"]] = {"ok": 0, "skip": 0}

    for subdir in sorted(ds["neutral"].iterdir()):
        if not subdir.is_dir():
            continue

        prompt_subdir = ds["prompt"] / subdir.name
        if not prompt_subdir.exists():
            print(f"[SKIP] prompt 폴더 없음: {prompt_subdir}")
            continue

        for neutral_path in sorted(subdir.glob("*.neutral.json")):
            prompt_path = find_prompt(prompt_subdir, neutral_path)

            if prompt_path is None:
                print(f"[SKIP] {ds['name']} / {subdir.name} / {neutral_path.name}")
                stats[ds["name"]]["skip"] += 1
                continue

            with open(prompt_path, "r", encoding="utf-8") as f:
                prompt_text = f.read().strip()

            with open(neutral_path, "r", encoding="utf-8") as f:
                neutral_json = json.load(f)

            records.append({
                "instruction": prompt_text,
                "input": "",
                "output": neutral_json,
                "meta": {
                    "dataset": ds["name"],
                    "group": subdir.name,
                    "stem": neutral_path.stem
                }
            })

            stats[ds["name"]]["ok"] += 1

with open(OUTPUT_JSONL, "w", encoding="utf-8") as f:
    for r in records:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")

print("\n===================================")
print(f"JSONL 생성 완료: {OUTPUT_JSONL}")
print(f"총 샘플 수: {len(records)}")
for k, v in stats.items():
    print(f"{k}: 성공 {v['ok']} / 스킵 {v['skip']}")
