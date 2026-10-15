#!/usr/bin/env python3
"""Evaluador experimental del pipeline RAG para una única pregunta de Amazon 2017.

Lee la pregunta ``financebench_id_08135`` del dataset FinanceBench, ejecuta el
pipeline (extracción + retrieval + generación con Qwen) y compara la respuesta
generada con la referencia mediante igualdad exacta estricta.

Este script es temporal y específico para el PDF de Amazon 2017.
"""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from generation.ollama_generator import GENERATION_MODEL, generate_answer  # noqa: E402
from ingestion.pdf import extract_pdf_pages  # noqa: E402
from retrieval.retriever import retrieve  # noqa: E402


FINANCEBENCH_ID = "financebench_id_08135"
DATASET_PATH = ROOT / "data" / "FinanceBench" / "data" / "financebench_open_source.jsonl"
PDF_PATH = ROOT / "data" / "FinanceBench" / "pdfs" / "AMAZON_2017_10K.pdf"
RESULTS_DIR = ROOT / "data" / "FinanceBench" / "results"
RESULTS_FILENAME = "amazon_rag_evaluation.jsonl"
TOP_K = 5


def load_question(financebench_id: str) -> dict:
    """Devuelve el registro del dataset cuyo ``financebench_id`` coincide."""
    with DATASET_PATH.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            if record.get("financebench_id") == financebench_id:
                return record
    raise LookupError(f"No se encontró {financebench_id} en {DATASET_PATH}")


def resolve_output_path(financebench_id: str) -> Path:
    """Elige un fichero de salida sin sobrescribir resultados existentes.

    - Si no existe, usa el nombre base.
    - Si existe y no contiene ya esta pregunta, se añadirá al mismo fichero.
    - Si ya contiene esta pregunta, usa un nombre alternativo numerado.
    """
    base = RESULTS_DIR / RESULTS_FILENAME
    if not base.exists():
        return base

    with base.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                if json.loads(line).get("financebench_id") == financebench_id:
                    break
            except json.JSONDecodeError:
                continue
        else:
            return base

    stem = base.stem
    suffix = base.suffix
    counter = 2
    while True:
        candidate = RESULTS_DIR / f"{stem}_{counter}{suffix}"
        if not candidate.exists():
            return candidate
        counter += 1


def save_result(path: Path, record: dict) -> None:
    """Añade el registro de evaluación al fichero JSONL indicado."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def main() -> int:
    if not DATASET_PATH.exists():
        print(f"ERROR: dataset no encontrado: {DATASET_PATH}", file=sys.stderr)
        return 1
    if not PDF_PATH.exists():
        print(f"ERROR: PDF no encontrado: {PDF_PATH}", file=sys.stderr)
        return 1

    try:
        dataset_record = load_question(FINANCEBENCH_ID)
    except (LookupError, json.JSONDecodeError) as exc:
        print(f"ERROR leyendo el dataset: {exc}", file=sys.stderr)
        return 1

    question = dataset_record["question"]
    gold_answer = dataset_record["answer"]
    doc_name = dataset_record["doc_name"]

    print(f"financebench_id: {FINANCEBENCH_ID}")
    print(f"Pregunta: {question}\n")
    print(f"Respuesta de referencia (gold): {gold_answer!r}\n")

    chunks = extract_pdf_pages(PDF_PATH, doc_name)
    if not chunks:
        print("ERROR: no se extrajo ninguna página del PDF.", file=sys.stderr)
        return 1

    try:
        results = retrieve(question, chunks, top_k=TOP_K)
    except Exception as exc:
        print(f"ERROR durante el retrieval: {exc}", file=sys.stderr)
        return 1

    if not results:
        print("ERROR: el retrieval no devolvió resultados.", file=sys.stderr)
        return 1

    try:
        model_answer = generate_answer(question, results, model=GENERATION_MODEL)
    except Exception as exc:
        print(f"ERROR durante la generación: {exc}", file=sys.stderr)
        return 1

    print(f"Respuesta generada ({GENERATION_MODEL}):")
    print("-" * 60)
    print(model_answer)
    print("-" * 60)

    print("\nPáginas recuperadas (top-%d):" % len(results))
    for rank, result in enumerate(results, start=1):
        print(
            f"{rank}. {result['doc_name']} - pág. {result['page']} "
            f"- similitud coseno: {result['score']:.4f}"
        )

    label = (
        "Correct Answer"
        if model_answer.strip() == gold_answer.strip()
        else "Incorrect Answer"
    )

    print(f"\nEvaluación (igualdad exacta tras eliminar espacios exteriores): {label}")
    print(
        "Nota: esta primera evaluación es estricta; puede marcar como incorrectas "
        "respuestas numéricamente equivalentes expresadas con otro formato."
    )

    output_path = resolve_output_path(FINANCEBENCH_ID)
    eval_record = {
        "financebench_id": FINANCEBENCH_ID,
        "model_name": GENERATION_MODEL,
        "eval_mode": "rag",
        "temp": None,
        "question": question,
        "gold_answer": gold_answer,
        "model_answer": model_answer,
        "label": label,
    }
    save_result(output_path, eval_record)
    print(f"\nResultado guardado en: {output_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
