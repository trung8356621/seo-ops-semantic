from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SEED = ROOT / "tests" / "fixtures" / "topic_vi_keywords.json"
OUT = ROOT / "artifacts" / "local" / "topic_vi_keywords_expanded.json"

SUFFIXES = [
    "",
    " giá rẻ",
    " cao cấp",
    " đẹp",
    " tốt nhất",
    " 2024",
    " 2025",
    " Hà Nội",
    " TPHCM",
    " bán buôn",
    " số lượng lớn",
    " theo yêu cầu",
    " chống nước",
    " chính hãng",
]


def main() -> None:
    seed = json.loads(SEED.read_text(encoding="utf-8"))
    keywords = []
    seen: set[str] = set()
    idx = 1
    for item in seed["keywords"]:
        base = item["text"]
        for suffix in SUFFIXES:
            text = f"{base}{suffix}".strip()
            key = text.casefold()
            if key in seen:
                continue
            seen.add(key)
            keywords.append({"ref": f"kw-{idx:04d}", "text": text})
            idx += 1
            if len(keywords) >= 900:
                break
        if len(keywords) >= 900:
            break

    payload = {
        "site_ref": "fixture-vi-expanded",
        "language": "vi",
        "keywords": keywords,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"wrote {OUT} keywords={len(keywords)}")


if __name__ == "__main__":
    main()
