"""
SCRIPT PRINCIPAL DE RECUPERACIÓN (FASE 5) — CODEFEST AD ASTRA 2026.

Este script orquesta el flujo completo de inferencia:
  1. Carga del índice FAISS (IndexFlatIP) y almacén de metadatos (metadata.jsonl).
  2. Carga del encoder BGE-M3 (con aceleración GPU si está disponible).
  3. Procesamiento de consultas en lenguaje natural (ES, EN, PT).
  4. Búsqueda vectorial Top-K en FAISS.
  5. Agregación a documentos mediante Max Pooling (Top-3 documentos).
  6. Recorte lingüístico de fragmentos a máximo 250 palabras (respetando oraciones completas).
  7. Exportación del archivo oficial de resultados a entrega/resultados.jsonl.

Ejecución:
    python entrega/generador.py [--consultas RUTA_CONSULTAS] [--salida RUTA_SALIDA]
"""

from __future__ import annotations

import argparse
import json
import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import faiss
import numpy as np
import torch
from sentence_transformers import SentenceTransformer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
# CONFIGURACIÓN Y RUTAS POR DEFECTO
# --------------------------------------------------------------------------- #
BASE_DIR: Path = Path(__file__).resolve().parent.parent
ENTREGA_DIR: Path = BASE_DIR / "entrega"
VECTOR_DIR: Path = ENTREGA_DIR / "base_vectorial" / "encoder_bge_m3"
INDEX_PATH: Path = VECTOR_DIR / "index.faiss"
METADATA_PATH: Path = VECTOR_DIR / "metadata.jsonl"
RESULTADOS_PATH: Path = ENTREGA_DIR / "resultados.jsonl"

MODEL_NAME: str = "BAAI/bge-m3"
LOCAL_MODEL_DIR: Path = BASE_DIR / "modelo_bge_m3"
ESPERADO_DIM: int = 1024


# --------------------------------------------------------------------------- #
# 1. CARGA DE MODELO Y RECURSOS
# --------------------------------------------------------------------------- #
def cargar_encoder() -> SentenceTransformer:
    """Carga el encoder BGE-M3 con aceleración GPU (CUDA) si se encuentra disponible."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info("Dispositivo para inferencia de consultas: %s", device)
    if LOCAL_MODEL_DIR.exists():
        logger.info("Cargando modelo local desde: %s", LOCAL_MODEL_DIR)
        return SentenceTransformer(str(LOCAL_MODEL_DIR), device=device)
    logger.info("Cargando modelo '%s' desde HuggingFace.", MODEL_NAME)
    return SentenceTransformer(MODEL_NAME, device=device)


def cargar_almacen_metadata(ruta_metadata: Path = METADATA_PATH) -> List[Dict[str, Any]]:
    """Carga metadata.jsonl en una lista indexada (línea N = ID N de FAISS)."""
    if not ruta_metadata.exists():
        raise FileNotFoundError(f"No se encontró el almacén de metadata en: {ruta_metadata}")
    store: List[Dict[str, Any]] = []
    with open(ruta_metadata, "r", encoding="utf-8") as fh:
        for num_linea, linea in enumerate(fh, 1):
            linea = linea.strip()
            if not linea:
                continue
            store.append(json.loads(linea))
    logger.info("Almacén de metadata cargado: %d registros.", len(store))
    return store


# --------------------------------------------------------------------------- #
# 2. FILTRO LINGÜÍSTICO (RECORTE A MÁXIMO 250 PALABRAS POR ORACIONES COMPLETAS)
# --------------------------------------------------------------------------- #
def recortar_texto_250_palabras(texto: str, max_palabras: int = 250) -> str:
    """
    Si el texto supera max_palabras (250), corta el fragmento en el límite
    de la última oración completa antes de alcanzar la palabra 250.
    Regla de Oro: No corta oraciones por la mitad.
    """
    palabras = texto.split()
    if len(palabras) <= max_palabras:
        return texto

    # Dividir el texto en oraciones usando expresión regular (delimitadores . ! ?)
    patron_oraciones = re.compile(r"(?<=[.!?])\s+")
    oraciones = patron_oraciones.split(texto)

    oraciones_seleccionadas: List[str] = []
    conteo_palabras = 0

    for oracion in oraciones:
        palabras_oracion = len(oracion.split())
        if conteo_palabras + palabras_oracion > max_palabras:
            break
        oraciones_seleccionadas.append(oracion)
        conteo_palabras += palabras_oracion

    # Si ninguna oración completa cabe antes del límite, tomar las primeras 250 palabras
    if not oraciones_seleccionadas:
        return " ".join(palabras[:max_palabras])

    return " ".join(oraciones_seleccionadas)


# --------------------------------------------------------------------------- #
# 3. RECUPERACIÓN Y MAX POOLING A DOCUMENTOS
# --------------------------------------------------------------------------- #
def ejecutar_recuperacion(
    query_texto: str,
    modelo: SentenceTransformer,
    indice: faiss.Index,
    metadata_store: List[Dict[str, Any]],
    top_k_chunks: int = 10,
    top_k_docs: int = 3,
    candidatos_faiss: int = 50,
) -> Dict[str, Any]:
    """
    Ejecuta la búsqueda semántica para una consulta:
      1. Codifica y normaliza L2 la consulta.
      2. Busca candidatos en el índice FAISS.
      3. Agrupa por doc_id aplicando Max Pooling para extraer el Top-3 de documentos.
      4. Aplica el filtro de 250 palabras a los Top-10 fragmentos.
    """
    # 1. Vectorizar y normalizar L2 el query
    query_emb = modelo.encode([query_texto], convert_to_numpy=True, normalize_embeddings=False)
    query_emb = np.ascontiguousarray(query_emb, dtype=np.float32)
    faiss.normalize_L2(query_emb)

    # 2. Búsqueda en FAISS
    scores, indices = indice.search(query_emb, candidatos_faiss)

    raw_scores = scores[0]
    raw_ids = indices[0]

    fragmentos_candidatos: List[Dict[str, Any]] = []
    doc_scores: Dict[str, float] = {}

    for faiss_id, score in zip(raw_ids, raw_scores):
        if faiss_id < 0 or faiss_id >= len(metadata_store):
            continue
        rec = metadata_store[faiss_id]
        doc_id = rec.get("doc_id")
        score_val = float(score)

        # Max Pooling por doc_id
        if doc_id:
            if doc_id not in doc_scores or score_val > doc_scores[doc_id]:
                doc_scores[doc_id] = score_val

        fragmentos_candidatos.append({
            "faiss_id": int(faiss_id),
            "chunk_id": rec.get("chunk_id"),
            "doc_id": doc_id,
            "score": round(score_val, 6),
            "texto_original": rec.get("texto_original", ""),
            "metadata": rec.get("metadata", {}),
        })

    # 3. Top-3 Documentos por Max Pooling
    docs_ordenados = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)
    top_3_docs = [doc_id for doc_id, _ in docs_ordenados[:top_k_docs]]

    # 4. Top-10 Fragmentos con recorte de 250 palabras
    top_10_fragmentos: List[Dict[str, Any]] = []
    for frag in fragmentos_candidatos[:top_k_chunks]:
        texto_recortado = recortar_texto_250_palabras(frag["texto_original"], max_palabras=250)
        top_10_fragmentos.append({
            "chunk_id": frag["chunk_id"],
            "doc_id": frag["doc_id"],
            "score": frag["score"],
            "texto": texto_recortado,
        })

    return {
        "documentos": top_3_docs,
        "fragmentos": top_10_fragmentos,
    }


# --------------------------------------------------------------------------- #
# 4. PROCESAMIENTO BATCH DE CONSULTAS Y GENERACIÓN DE RESULTADOS
# --------------------------------------------------------------------------- #
def procesar_consultas(
    ruta_consultas: Optional[Path],
    ruta_salida: Path = RESULTADOS_PATH,
) -> None:
    """
    Procesa la lista de 50 consultas y genera resultados.jsonl en entrega/.
    """
    if not INDEX_PATH.exists():
        raise FileNotFoundError(f"No se encontró el índice FAISS en: {INDEX_PATH}")

    modelo = cargar_encoder()
    metadata_store = cargar_almacen_metadata()
    indice = faiss.read_index(str(INDEX_PATH))

    # Cargar consultas
    consultas: List[Dict[str, str]] = []
    if ruta_consultas and ruta_consultas.exists():
        logger.info("Cargando consultas desde: %s", ruta_consultas)
        with open(ruta_consultas, "r", encoding="utf-8") as fh:
            if ruta_consultas.suffix == ".jsonl":
                for linea in fh:
                    if linea.strip():
                        consultas.append(json.loads(linea))
            else:
                consultas = json.load(fh)
    else:
        logger.warning("No se proporcionó archivo de consultas válido. Generando ejecuciones de prueba sintéticas q001..q050.")
        # Ejemplo sintético para validación del pipeline
        for i in range(1, 51):
            q_id = f"q{i:03d}"
            consultas.append({
                "query_id": q_id,
                "query": f"Consulta de prueba {q_id} sobre inteligencia artificial en defensa y seguridad espacial.",
            })

    logger.info("Procesando %d consultas...", len(consultas))

    RESULTADOS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(ruta_salida, "w", encoding="utf-8") as fh_out:
        for item in consultas:
            query_id = item.get("query_id") or item.get("id") or "q_unknown"
            query_texto = item.get("query") or item.get("texto") or ""

            res = ejecutar_recuperacion(query_texto, modelo, indice, metadata_store)

            salida_item = {
                "query_id": query_id,
                "query": query_texto,
                "documentos": res["documentos"],
                "fragmentos": res["fragmentos"],
            }
            fh_out.write(json.dumps(salida_item, ensure_ascii=False) + "\n")

    logger.info("Resultados de inferencia guardados exitosamente en: %s", ruta_salida)
    print(f"\n>> Inferencia completada. Archivo generado: {ruta_salida}\n")


# --------------------------------------------------------------------------- #
# MAIN / CLI
# --------------------------------------------------------------------------- #
def main() -> None:
    parser = argparse.ArgumentParser(description="Script Generador de Resultados (Fase 5) - CODEFEST AD ASTRA 2026")
    parser.add_argument("--consultas", type=str, default=None, help="Ruta al archivo de consultas (JSON/JSONL)")
    parser.add_argument("--salida", type=str, default=str(RESULTADOS_PATH), help="Ruta del archivo resultados.jsonl de salida")
    args = parser.parse_args()

    ruta_c = Path(args.consultas) if args.consultas else None
    ruta_s = Path(args.salida)

    procesar_consultas(ruta_c, ruta_s)


if __name__ == "__main__":
    main()
