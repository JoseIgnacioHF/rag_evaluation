#!/usr/bin/env python3
"""Inspect FinanceBench evaluation questions (JSONL).

Reads: data/FinanceBench/data/financebench_open_source.jsonl
"""

import json
import sys
from pathlib import Path


def main() -> int:
    root = Path.cwd()
    path = root / "data" / "FinanceBench" / "data" / "financebench_open_source.jsonl"
    if not path.exists():
        print(f"ERROR: No se encontró archivo en {path}", file=sys.stderr)
        return 1

    records = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue

    print(f"Archivo: {path}")
    print(f"Nº total de preguntas: {len(records)}\n")

    if not records:
        return 0

    ex = records[0]
    print("Ejemplo (primer registro) - campos (sin contenido completo de evidence_text):")
    for k, v in ex.items():
        if k == "evidence":
            if isinstance(v, list):
                s = f"list[{len(v)}] con keys: {list(v[0].keys()) if v else '[]'}"
                print(f"- {k}: {s}")
            else:
                print(f"- {k}: {v}")
        else:
            s = str(v) if v is not None else "None"
            if len(s) > 200:
                s = s[:197] + "..."
            print(f"- {k}: {s}")

    print("\nEvidencias de la pregunta de ejemplo:")
    ev = ex.get("evidence")
    if isinstance(ev, list) and ev:
        e0 = ev[0]
        et = e0.get("evidence_text")
        print(f"- evidence_text (fragmento): {str(et)[:400]}..." if et else "- evidence_text: None")
        print(f"- evidence_page_num: {e0.get('evidence_page_num')}")
        print(f"- doc_name (dentro de evidence): {e0.get('doc_name')}")
        print(f"- evidence_text_full_page (presente): {'sí' if 'evidence_text_full_page' in e0 else 'no'}")
    else:
        print("- Sin evidencia estructurada")

    with_e = 0
    total_e = 0
    for rec in records:
        ev = rec.get("evidence")
        if isinstance(ev, list) and ev:
            with_e += 1
            total_e += len(ev)

    print("\nConteo de evidencias:")
    print(f"- Preguntas con evidencia: {with_e}")
    print(f"- Nº total de objetos de evidencia: {total_e}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
