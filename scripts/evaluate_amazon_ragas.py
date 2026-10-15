#!/usr/bin/env python3
"""Evaluación con Ragas del pipeline RAG para una única pregunta de Amazon 2017.

Evalúa de forma independiente cuatro dimensiones para ``financebench_id_08135``:

1. Retrieval: métricas Ragas deterministas (no-LLM) de precisión y recall de
   contexto, comparando los chunks recuperados con la evidencia de referencia.
2. Faithfulness (Ragas).
3. Answer relevance: Response Relevancy (Ragas).
4. Answer correctness: Answer Correctness (Ragas) + comprobación numérica
   determinista propia con tolerancia configurable.

Reutiliza el pipeline existente:
``extract_pdf_pages``, ``retrieve`` y ``generate_answer``. No reimplementa esas
fases. La configuración del modelo generativo y la del modelo juez están
separadas.
"""

import json
import math
import re
import sys
from importlib import metadata as importlib_metadata
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from generation.ollama_generator import GENERATION_MODEL, generate_answer  # noqa: E402
from ingestion.pdf import extract_pdf_pages  # noqa: E402
from retrieval.retriever import retrieve  # noqa: E402

from ragas import EvaluationDataset, evaluate  # noqa: E402
from ragas.dataset_schema import SingleTurnSample  # noqa: E402
from ragas.embeddings import LangchainEmbeddingsWrapper  # noqa: E402
from ragas.llms import LangchainLLMWrapper  # noqa: E402
from ragas.metrics import (  # noqa: E402
    AnswerCorrectness,
    Faithfulness,
    NonLLMContextPrecisionWithReference,
    NonLLMContextRecall,
    ResponseRelevancy,
)
from langchain_ollama import ChatOllama, OllamaEmbeddings  # noqa: E402


FINANCEBENCH_ID = "financebench_id_08135"
DATASET_PATH = ROOT / "data" / "FinanceBench" / "data" / "financebench_open_source.jsonl"
PDF_PATH = ROOT / "data" / "FinanceBench" / "pdfs" / "AMAZON_2017_10K.pdf"
RESULTS_DIR = ROOT / "data" / "FinanceBench" / "results"
RESULTS_FILENAME = "amazon_ragas_evaluation.jsonl"
TOP_K = 5

OLLAMA_BASE_URL = "http://localhost:11434"
JUDGE_MODEL = "qwen2.5:1.5b"
JUDGE_TEMPERATURE = 0
EMBEDDINGS_MODEL = "nomic-embed-text"

# Tolerancia de la comprobación numérica determinista, en puntos porcentuales.
NUMERIC_TOLERANCE_PCT_POINTS = 0.1

_PERCENT_RE = re.compile(r"([-+]?\d+(?:[.,]\d+)?)\s*%")
_PERCENT_WORD_RE = re.compile(
    r"([-+]?\d+(?:[.,]\d+)?)\s*(?:percent|per cent|pct)\b", re.IGNORECASE
)


def load_record(financebench_id: str) -> dict:
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


def reference_evidence(record: dict) -> tuple[str, int | None]:
    """Devuelve el texto de evidencia de referencia y su página (0-based)."""
    evidence = record.get("evidence") or []
    if not evidence:
        raise ValueError("El registro no contiene evidencias.")
    first = evidence[0]
    text = first.get("evidence_text_full_page") or first.get("evidence_text")
    if not text:
        raise ValueError("La evidencia no contiene texto.")
    return text, first.get("evidence_page_num")


def extract_percentages(text: str) -> list[float]:
    """Extrae todos los valores porcentuales de un texto.

    Reconoce tanto ``30.8%`` como ``30.8 percent`` y elimina duplicados
    conservando el orden de aparición.
    """
    matches: list[tuple[int, float]] = []
    for pattern in (_PERCENT_RE, _PERCENT_WORD_RE):
        for match in pattern.finditer(text):
            matches.append((match.start(), float(match.group(1).replace(",", "."))))
    matches.sort(key=lambda item: item[0])

    values: list[float] = []
    for _, value in matches:
        if value not in values:
            values.append(value)
    return values


def deterministic_numeric_check(
    model_answer: str, gold_answer: str, tolerance: float
) -> dict:
    """Compara el porcentaje generado con el de referencia.

    Devuelve un diccionario independiente de las puntuaciones de Ragas. No
    codifica ningún valor de referencia: lo extrae de ``gold_answer``.
    """
    reference_values = extract_percentages(gold_answer)
    generated_values = extract_percentages(model_answer)

    result = {
        "reference_values": reference_values,
        "generated_values": generated_values,
        "tolerance_pct_points": tolerance,
        "correct": None,
        "reason": None,
    }

    if not reference_values:
        result["reason"] = "No se pudo extraer un valor porcentual de la referencia."
        return result
    if not generated_values:
        result["correct"] = False
        result["reason"] = (
            "No se pudo extraer ningún valor porcentual de la respuesta generada."
        )
        return result

    reference = reference_values[0]
    matched = [v for v in generated_values if abs(v - reference) <= tolerance]
    result["reference"] = reference
    result["matched_values"] = matched
    result["correct"] = bool(matched)
    if not matched:
        result["reason"] = (
            f"Ningún valor generado está a menos de {tolerance} puntos "
            f"porcentuales de {reference}."
        )
    return result


def build_judge() -> tuple[LangchainLLMWrapper, LangchainEmbeddingsWrapper]:
    """Construye el modelo juez y los embeddings, separados del generador."""
    judge_llm = LangchainLLMWrapper(
        ChatOllama(
            model=JUDGE_MODEL,
            base_url=OLLAMA_BASE_URL,
            temperature=JUDGE_TEMPERATURE,
        )
    )
    embeddings = LangchainEmbeddingsWrapper(
        OllamaEmbeddings(model=EMBEDDINGS_MODEL, base_url=OLLAMA_BASE_URL)
    )
    return judge_llm, embeddings


def run_metric(
    metric,
    dataset: EvaluationDataset,
    judge_llm: LangchainLLMWrapper,
    embeddings: LangchainEmbeddingsWrapper,
) -> dict:
    """Ejecuta una única métrica Ragas y devuelve nombre, puntuación o error."""
    entry = {"name": metric.name, "score": None, "error": None}
    try:
        result = evaluate(
            dataset,
            metrics=[metric],
            llm=judge_llm,
            embeddings=embeddings,
            show_progress=False,
            raise_exceptions=True,
        )
        frame = result.to_pandas()
        score = float(frame[metric.name].iloc[0])
        if math.isnan(score):
            entry["error"] = "La métrica devolvió NaN."
        else:
            entry["score"] = score
    except Exception as exc:  # noqa: BLE001 - se reporta explícitamente
        entry["error"] = f"{type(exc).__name__}: {exc}"
    return entry


def collected_versions() -> dict:
    """Recoge las versiones de las piezas relevantes del entorno."""
    packages = [
        "ragas",
        "langchain",
        "langchain-core",
        "langchain-community",
        "langchain-ollama",
        "langchain-openai",
        "rapidfuzz",
        "numpy",
        "pandas",
        "pymupdf",
    ]
    versions = {}
    for package in packages:
        try:
            versions[package] = importlib_metadata.version(package)
        except importlib_metadata.PackageNotFoundError:
            versions[package] = None
    return versions


def resolve_output_path(financebench_id: str) -> Path:
    """Elige un fichero de salida sin sobrescribir evaluaciones previas."""
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

    counter = 2
    while True:
        candidate = RESULTS_DIR / f"{base.stem}_{counter}{base.suffix}"
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
        record = load_record(FINANCEBENCH_ID)
        evidence_text, evidence_page = reference_evidence(record)
    except (LookupError, ValueError, json.JSONDecodeError) as exc:
        print(f"ERROR leyendo el dataset: {exc}", file=sys.stderr)
        return 1

    question = record["question"]
    gold_answer = record["answer"]
    doc_name = record["doc_name"]

    print(f"financebench_id: {FINANCEBENCH_ID}")
    print(f"Modelo generador: {GENERATION_MODEL}")
    print(f"Modelo juez: {JUDGE_MODEL} (temp={JUDGE_TEMPERATURE})")
    print(f"Modelo de embeddings: {EMBEDDINGS_MODEL}")
    print(f"Pregunta: {question}\n")

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
        model_answer = generate_answer(question, results)
    except Exception as exc:
        print(f"ERROR durante la generación: {exc}", file=sys.stderr)
        return 1

    print(f"Respuesta de referencia: {gold_answer!r}")
    print(f"Respuesta generada:\n{model_answer}\n")

    print(f"Páginas recuperadas (top-{len(results)}):")
    for rank, item in enumerate(results, start=1):
        print(
            f"{rank}. {item['doc_name']} - pág. {item['page']} "
            f"- similitud coseno: {item['score']:.4f}"
        )
    if evidence_page is not None:
        print(
            f"\nEvidencia de referencia: pág. {evidence_page} (índice 0-based) "
            f"= pág. {evidence_page + 1} (1-based, como en los chunks)."
        )

    retrieved_contexts = [item["text"] for item in results]
    sample = SingleTurnSample(
        user_input=question,
        retrieved_contexts=retrieved_contexts,
        reference_contexts=[evidence_text],
        response=model_answer,
        reference=gold_answer,
    )
    dataset = EvaluationDataset(samples=[sample])

    print("\nConstruyendo modelo juez y embeddings (Ollama)...")
    try:
        judge_llm, embeddings = build_judge()
    except Exception as exc:
        print(f"ERROR configurando Ollama: {exc}", file=sys.stderr)
        return 1

    # Retrieval: métricas deterministas (no usan el modelo juez).
    retrieval_metrics = [
        NonLLMContextPrecisionWithReference(),
        NonLLMContextRecall(),
    ]
    llm_metrics = [
        Faithfulness(llm=judge_llm),
        ResponseRelevancy(llm=judge_llm, embeddings=embeddings),
        AnswerCorrectness(llm=judge_llm, embeddings=embeddings),
    ]

    print("\nEvaluando métricas de retrieval (deterministas)...")
    metric_entries = []
    for metric in retrieval_metrics:
        entry = run_metric(metric, dataset, judge_llm, embeddings)
        metric_entries.append(entry)
        _print_metric(entry)

    print("\nEvaluando métricas con Ragas + Ollama (puede tardar)...")
    for metric in llm_metrics:
        entry = run_metric(metric, dataset, judge_llm, embeddings)
        metric_entries.append(entry)
        _print_metric(entry)

    numeric_check = deterministic_numeric_check(
        model_answer, gold_answer, NUMERIC_TOLERANCE_PCT_POINTS
    )
    print("\nComprobación numérica determinista (independiente de Ragas):")
    print(f"  Valores de referencia extraídos: {numeric_check['reference_values']}")
    print(f"  Valores generados extraídos: {numeric_check['generated_values']}")
    print(f"  Tolerancia: {numeric_check['tolerance_pct_points']} puntos porcentuales")
    print(f"  Resultado: {numeric_check['correct']}")
    if numeric_check.get("reason"):
        print(f"  Motivo: {numeric_check['reason']}")

    print(
        "\nLimitaciones: qwen2.5:1.5b es un modelo pequeño; las métricas basadas "
        "en el juez (faithfulness, answer relevancy, answer correctness) pueden "
        "ser poco fiables. Las métricas de retrieval son deterministas (similitud "
        "de cadenas) y no dependen del juez."
    )

    eval_record = {
        "financebench_id": FINANCEBENCH_ID,
        "question": question,
        "gold_answer": gold_answer,
        "model_answer": model_answer,
        "generator_model": GENERATION_MODEL,
        "judge_model": JUDGE_MODEL,
        "judge_temperature": JUDGE_TEMPERATURE,
        "embeddings_model": EMBEDDINGS_MODEL,
        "top_k": TOP_K,
        "numeric_tolerance_pct_points": NUMERIC_TOLERANCE_PCT_POINTS,
        "retrieved_pages": [
            {"doc_name": item["doc_name"], "page": item["page"], "score": item["score"]}
            for item in results
        ],
        "reference_evidence_page_num_zero_based": evidence_page,
        "ragas_metrics": {
            entry["name"]: {"score": entry["score"], "error": entry["error"]}
            for entry in metric_entries
        },
        "deterministic_numeric_check": numeric_check,
        "versions": collected_versions(),
    }

    output_path = resolve_output_path(FINANCEBENCH_ID)
    save_result(output_path, eval_record)
    print(f"\nResultado guardado en: {output_path}")

    return 0


def _print_metric(entry: dict) -> None:
    if entry["error"]:
        print(f"  - {entry['name']}: ERROR ({entry['error']})")
    else:
        print(f"  - {entry['name']}: {entry['score']}")


if __name__ == "__main__":
    raise SystemExit(main())
