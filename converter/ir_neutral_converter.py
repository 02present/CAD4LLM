import json
from pathlib import Path

root = Path("WHUCAD")

for p in root.rglob("*.json"):
    try:
        with open(p, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        continue

    if isinstance(data, dict) and data.get("schema") == "cad_ir.v0.1":
        data["schema"] = "cad_neutral.v0.1"

        with open(p, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

        print("updated:", p)
