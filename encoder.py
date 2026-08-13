"""
PIPELINE DE EMBEDDINGS E INDIZACIÓN VECTORIAL — CODEFEST AD ASTRA 2026 (Fases 3 y 4).

Este script realiza la codificación semántica y la indización en FAISS a partir de los
chunks procesados en las Fases 1 y 2 (por defecto 'chunks.jsonl', 'data/chunks.jsonl' o 'datos.json').

Flujo de trabajo:
  1. Lectura y validación en streaming por lotes (JSONL).
  2. Codificación densa con BAAI/bge-m3 en lotes (con soporte GPU/CUDA automático).
  3. Normalización L2 (faiss.normalize_L2) -> Similitud Coseno por Producto Interno.
  4. Indización incremental en un índice FAISS IndexFlatIP (dimensión 1024).
  5. Persistencia oficial en entrega/:
     - entrega/base_vectorial/encoder_bge_m3/index.faiss
     - entrega/base_vectorial/encoder_bge_m3/metadata.jsonl

Ejecución:
    python encoder.py [--input RUTA_CHUNKS]
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional

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
# 1. CONFIGURACIÓN Y RUTAS DE ENTREGA
# --------------------------------------------------------------------------- #
BASE_DIR: Path = Path(__file__).resolve().parent

# Archivo de entrada con chunks de las Fases 1 y 2.
# Se busca preferentemente 'chunks.jsonl' (o 'data/chunks.jsonl'), con fallback a 'datos.json'.
RUTAS_INPUT_CANDIDATAS: List[Path] = [
    BASE_DIR / "chunks.jsonl",
    BASE_DIR / "data" / "chunks.jsonl",
    BASE_DIR / "chunks_procesados.jsonl",
    BASE_DIR / "datos.json",
]

# Rutas oficiales bajo entrega/
ENTREGA_DIR: Path = BASE_DIR / "entrega" / "base_vectorial" / "encoder_bge_m3"
SALIDA_DIR: Path = BASE_DIR / "salida"
INDEX_PATH: Path = ENTREGA_DIR / "index.faiss"
METADATA_PATH: Path = ENTREGA_DIR / "metadata.jsonl"

MODEL_NAME: str = "BAAI/bge-m3"
LOCAL_MODEL_DIR: Path = BASE_DIR / "modelo_bge_m3"

BATCH_SIZE: int = 16
ESPERADO_DIM: int = 1024  # dimensión de embeddings de BGE-M3

# Campos requeridos indispensables en el esquema de chunk.
CAMPO_TEXTO: str = "texto"
CAMPOS_OBLIGATORIOS = ["chunk_id", "doc_id", CAMPO_TEXTO]


def obtener_ruta_entrada_defecto() -> Path:
    """Devuelve la primera ruta existente entre las candidatas o 'chunks.jsonl'."""
    for ruta in RUTAS_INPUT_CANDIDATAS:
        if ruta.exists():
            return ruta
    return BASE_DIR / "chunks.jsonl"


# --------------------------------------------------------------------------- #
# 2. MODELO ENCODER
# --------------------------------------------------------------------------- #
def cargar_modelo(nombre: str = MODEL_NAME, local_dir: Path = LOCAL_MODEL_DIR) -> SentenceTransformer:
    """Carga el encoder BGE-M3 con aceleración GPU (CUDA) si está disponible."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info("Dispositivo seleccionado para inferencia: %s", device)

    if local_dir.exists():
        logger.info("Cargando modelo local desde: %s", local_dir)
        return SentenceTransformer(str(local_dir), device=device)
    logger.info("Cargando modelo '%s' desde HuggingFace.", nombre)
    return SentenceTransformer(nombre, device=device)


# --------------------------------------------------------------------------- #
# 3. VALIDACIÓN Y CARGA DE DATOS EN STREAMING
# --------------------------------------------------------------------------- #
def validar_chunk(chunk: Dict[str, Any], num_linea: int) -> bool:
    """
    Valida el esquema esencial de un chunk para asegurar compatibilidad.
    Requiere 'texto', 'chunk_id' y 'doc_id'.
    """
    for campo in CAMPOS_OBLIGATORIOS:
        if campo not in chunk:
            raise ValueError(f"Línea {num_linea}: Falta el campo obligatorio '{campo}'")

    texto = chunk.get(CAMPO_TEXTO)
    if not isinstance(texto, str) or not texto.strip():
        raise ValueError(f"Línea {num_linea}: El chunk no tiene contenido textual válido.")

    return True


def iterar_chunks_jsonl(
    ruta: Path, batch_size: int = BATCH_SIZE
) -> Generator[List[Dict[str, Any]], None, None]:
    """
    Lee los chunks procesados en streaming (línea por línea), valida cada chunk
    y devuelve lotes de registros completos preservando el orden secuencial.
    """
    batch: List[Dict[str, Any]] = []
    with open(ruta, "r", encoding="utf-8") as fh:
        for num_linea, linea in enumerate(fh, 1):
            linea = linea.strip()
            if not linea:
                continue
            try:
                chunk = json.loads(linea)
                validar_chunk(chunk, num_linea)
            except json.JSONDecodeError:
                logger.error("Línea %d: error de sintaxis JSON. Se omite.", num_linea)
                continue
            except ValueError as exc:
                logger.warning("%s", exc)
                continue
            batch.append(chunk)
            if len(batch) == batch_size:
                yield batch
                batch = []
    if batch:
        yield batch


# --------------------------------------------------------------------------- #
# 4. GENERACIÓN DE EMBEDDINGS
# --------------------------------------------------------------------------- #
def generar_embeddings(modelo: SentenceTransformer, textos: List[str]) -> np.ndarray:
    """
    Genera embeddings densos para una lista de textos y los normaliza a norma
    unitaria L2 (faiss.normalize_L2) para que el producto interno en
    IndexFlatIP coincida con la similitud coseno.
    """
    embeddings = modelo.encode(
        textos,
        batch_size=len(textos),
        convert_to_numpy=True,
        normalize_embeddings=False,
        show_progress_bar=True,
    )
    embeddings = np.ascontiguousarray(embeddings, dtype=np.float32)
    faiss.normalize_L2(embeddings)
    return embeddings


# --------------------------------------------------------------------------- #
# 5. INDEXACIÓN Y PERSISTENCIA (METADATA EN FORMATO JSONL)
# --------------------------------------------------------------------------- #
def construir_indice_y_almacenar(
    ruta_input: Path, modelo: SentenceTransformer
) -> faiss.Index:
    """
    Codifica en streaming por lotes, agrega incrementalmente a FAISS
    y guarda metadata.jsonl (línea N = ID N de FAISS).
    """
    ENTREGA_DIR.mkdir(parents=True, exist_ok=True)
    SALIDA_DIR.mkdir(parents=True, exist_ok=True)

    indice = faiss.IndexFlatIP(ESPERADO_DIM)
    faiss_id_actual = 0

    logger.info("Iniciando indización en streaming desde: %s", ruta_input)

    with open(METADATA_PATH, "w", encoding="utf-8") as meta_file:
        for lote in iterar_chunks_jsonl(ruta_input, batch_size=BATCH_SIZE):
            textos = [c[CAMPO_TEXTO] for c in lote]
            embeddings = generar_embeddings(modelo, textos)

            dim = embeddings.shape[1]
            if dim != ESPERADO_DIM:
                logger.warning(
                    "Dimensión esperada %d pero el encoder devolvió %d.", ESPERADO_DIM, dim
                )

            indice.add(embeddings)

            for chunk in lote:
                record = {
                    "faiss_id": faiss_id_actual,
                    "chunk_id": chunk.get("chunk_id"),
                    "doc_id": chunk.get("doc_id"),
                    "texto_original": chunk.get(CAMPO_TEXTO),
                    "metadata": {
                        k: v for k, v in chunk.items()
                        if k not in {"chunk_id", "doc_id", CAMPO_TEXTO}
                    },
                }
                meta_file.write(json.dumps(record, ensure_ascii=False) + "\n")
                faiss_id_actual += 1

    # --- Persistir el índice FAISS en disco ---
    faiss.write_index(indice, str(INDEX_PATH))
    logger.info("Índice FAISS guardado en: %s", INDEX_PATH)
    logger.info("Almacén de metadata guardado en: %s", METADATA_PATH)

    print(
        f"\n>> Indexados {indice.ntotal} vectores en el índice FAISS "
        f"(dimensiones = {indice.d}).\n"
    )
    return indice


# --------------------------------------------------------------------------- #
# 6. VERIFICACIÓN
# --------------------------------------------------------------------------- #
def verificar(
    ruta_input: Path, query: Optional[str] = None, k: int = 3
) -> None:
    """
    Demuestra el pipeline de recuperación leyendo metadata.jsonl.
    """
    if not INDEX_PATH.exists() or not METADATA_PATH.exists():
        raise FileNotFoundError(
            "Ejecute el pipeline previamente para generar index.faiss y metadata.jsonl."
        )

    modelo = cargar_modelo()

    if query is None:
        for primer_lote in iterar_chunks_jsonl(ruta_input, batch_size=1):
            query = primer_lote[0][CAMPO_TEXTO]
            break

    indice = faiss.read_index(str(INDEX_PATH))
    store: List[Dict[str, Any]] = []
    with open(METADATA_PATH, "r", encoding="utf-8") as fh:
        for linea in fh:
            if linea.strip():
                store.append(json.loads(linea))

    query_emb = generar_embeddings(modelo, [query])

    scores, indices = indice.search(query_emb, k)

    print("\n--- Verificación de recuperación ---")
    print(f"Query (primeros 90 chars): '{query[:90]}'")
    print(f"Índice cargado: ntotal = {indice.ntotal}, dim = {indice.d}")
    for rank, (faiss_id, score) in enumerate(zip(indices[0], scores[0]), 1):
        rec = store[int(faiss_id)] if 0 <= int(faiss_id) < len(store) else {}
        preview = rec.get("texto_original", "")[:80]
        print(
            f"  {rank}. faiss_id={faiss_id} | score={score:.4f} | "
            f"chunk_id={rec.get('chunk_id')} | doc_id={rec.get('doc_id')}"
        )
        print(f"     texto: {preview}...")


# --------------------------------------------------------------------------- #
# 7. ORQUESTACIÓN
# --------------------------------------------------------------------------- #
def main() -> None:
    parser = argparse.ArgumentParser(description="Encoder e Indización FAISS — CODEFEST AD ASTRA 2026")
    parser.add_argument(
        "--input",
        type=str,
        default=None,
        help="Ruta del archivo con los chunks procesados (ej. chunks.jsonl o datos.json)",
    )
    args = parser.parse_args()

    if args.input:
        ruta_input = Path(args.input)
    else:
        ruta_input = obtener_ruta_entrada_defecto()

    if not ruta_input.exists():
        raise FileNotFoundError(
            f"No se encontró el archivo de chunks de entrada en: {ruta_input}. "
            f"Por favor coloque 'chunks.jsonl' o especifique la ruta con --input."
        )

    logger.info("Cargando chunks procesados desde: %s", ruta_input)
    modelo = cargar_modelo()

    # Codificar en streaming → normalizar → indexar → persistir.
    construir_indice_y_almacenar(ruta_input, modelo)

    # Verificación
    verificar(ruta_input)


if __name__ == "__main__":
    main()
