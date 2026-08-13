"""
Fase 1B: Fragmentación Lingüística Determinista — CODEFEST AD ASTRA 2026

Lee `corpus_limpio.jsonl` y produce `chunks.jsonl` garantizando la COMPLETITUD LINGÜÍSTICA:
- Cortes únicamente en límites de oración completos (., !, ?).
- Tamaño máximo de 400 tokens (contados por espacios), avanzando ~350 tokens.
- Superposición de oraciones para preservar contexto entre fragmentos.
- División limpia de oraciones ultra-largas sin cortar palabras a la mitad.
- Inserción de metadatos obligatorios por fragmento.
"""

from __future__ import annotations

import argparse
import json
import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Tuple

try:
    import regex as _re
except Exception:
    _re = re

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("chunker")

BASE_DIR: Path = Path(__file__).resolve().parents[2]
INPUT_DEFAULT: Path = BASE_DIR / "corpus_limpio.jsonl"
OUTPUT_DEFAULT: Path = BASE_DIR / "chunks.jsonl"

MAX_TOKENS = 400      # Límite absoluto máximo de tokens por chunk
TARGET_TOKENS = 350   # Tamaño objetivo de ventana
OVERLAP_SENTENCES = 1 # Número de oraciones de solapamiento entre chunks consecutivos

# Abreviaturas comunes para evitar falsos cierres de oración
ABREVIATURAS = {
    "dr", "dra", "sr", "sra", "srta", "srs", "sres",
    "ing", "inga", "lic", "prof", "profa", "etc", "ej", "ejemplo",
    "ie", "eg", "et al", "et al.", "al", "uds", "ud", "vs", "v",
    "mm", "cm", "km", "kg", "hs", "h", "m", "s", "gr", "mol",
    "us", "ee uu", "jan", "feb", "mar", "abr", "may", "jun", "jul",
    "ago", "sep", "sept", "oct", "nov", "dic", "num", "ed", "rev",
    "univ", "co", "ltd", "corp", "s.a", "ph.d", "m.sc", "b.sc", "a.m", "p.m",
    "s.l", "s.c.", "c.v", "d.l",
}

RE_PUNTO_FINAL = _re.compile(r"[.!?…]+")
RE_TOKEN_PREVIO = _re.compile(r"([^\s.!?…]+)\s*$")
RE_NUM_LISTA = _re.compile(r"^\d{1,3}$")


def contar_tokens(texto: str) -> int:
    """Cuenta el número de tokens basándose en separación por espacios."""
    return len(texto.split())


def _es_corte_oracion(texto: str, m) -> bool:
    """Verifica si un signo de puntuación representa un final de oración válido."""
    pre = texto[:m.start()]
    fol = texto[m.end():]
    punct = m.group()

    # Evitar cortes en decimales (ej. 3.14)
    if pre and pre[-1].isdigit() and fol[:1].isdigit():
        return False

    tok_m = RE_TOKEN_PREVIO.search(pre)
    token = tok_m.group(1) if tok_m else ""
    core = token.lower().rstrip(".")

    # Evitar cortes en iniciales individuales (ej. J. K.)
    if len(core) == 1 and core.isalpha():
        return False
    # Evitar cortes en abreviaturas registradas
    if core in ABREVIATURAS:
        return False
    # Evitar cortes en listas numeradas (ej. 1. Introducción)
    if RE_NUM_LISTA.match(core) and fol[:1].isalpha() and fol[:1].isupper():
        return False

    if "!" in punct or "?" in punct:
        return True
    if not fol or fol.isspace():
        return True
    
    fol_first = fol.lstrip()[:1]
    if fol_first and (fol_first.isupper() or fol_first in "\"'([{¿¡"):
        return True

    return False


def split_unidades(texto: str) -> List[str]:
    """Divide el texto en oraciones/unidades respetando límites naturales."""
    if not texto:
        return []
    cortes = [0]
    for m in RE_PUNTO_FINAL.finditer(texto):
        if m.start() == 0:
            continue
        if _es_corte_oracion(texto, m):
            cortes.append(m.end())
    
    for m in _re.finditer(r"\n", texto):
        cortes.append(m.end())
        
    cortes = sorted(set(cortes))
    unidades: List[str] = []
    for i in range(len(cortes)):
        inicio = cortes[i]
        fin = cortes[i + 1] if i + 1 < len(cortes) else len(texto)
        seg = texto[inicio:fin].strip()
        if seg:
            unidades.append(seg)
    return unidades


def split_oracion_larga(texto: str, max_tokens: int) -> List[str]:
    """Divide oraciones que exceden los 400 tokens por comas/signos o palabras completas."""
    segs = _re.split(r"(?<=[,;:\u2013\u2014])\s+", texto)
    partes = [s for s in segs if s]
    
    salida: List[str] = []
    for parte in partes:
        if contar_tokens(parte) <= max_tokens:
            salida.append(parte)
        else:
            palabras = parte.split()
            cur: List[str] = []
            for w in palabras:
                if len(cur) + 1 > max_tokens and cur:
                    salida.append(" ".join(cur))
                    cur = []
                cur.append(w)
            if cur:
                salida.append(" ".join(cur))
    return salida


def chunk_text(texto: str) -> List[Tuple[str, bool]]:
    """Genera lista de tuplas (texto_chunk, fue_division_forzada)."""
    unidades = split_unidades(texto)
    if not unidades:
        return []

    chunks: List[Tuple[str, bool]] = []
    current: List[str] = []
    cur_tok = 0

    for unidad in unidades:
        tok = contar_tokens(unidad)

        if tok > MAX_TOKENS:
            if current:
                chunks.append(("\n".join(current).strip(), False))
                current = []
                cur_tok = 0
            for seg in split_oracion_larga(unidad, MAX_TOKENS):
                chunks.append((seg.strip(), True))
            continue

        if cur_tok + tok > MAX_TOKENS and current:
            chunks.append(("\n".join(current).strip(), False))
            current = current[-OVERLAP_SENTENCES:] if len(current) >= OVERLAP_SENTENCES else list(current)
            cur_tok = sum(contar_tokens(u) for u in current)

        current.append(unidad)
        cur_tok += tok

    if current:
        chunks.append(("\n".join(current).strip(), False))

    return chunks


def _termina_con_puntuacion(texto: str) -> bool:
    return bool(_re.search(r"[.!?…]\s*$", texto))


def chunk_documento(doc: Dict[str, Any]) -> List[Dict[str, Any]]:
    texto = doc.get("texto", "") or ""
    doc_id = doc.get("doc_id", "")
    fuente = doc.get("fuente", "")
    formato = doc.get("formato", "")
    fenomeno = doc.get("fenomeno", 0)

    resultados: List[Dict[str, Any]] = []
    for posicion, (texto_chunk, dividido) in enumerate(chunk_text(texto)):
        chunk_id = f"{doc_id}_chunk_{posicion:03d}"
        
        if not dividido and not _termina_con_puntuacion(texto_chunk):
            logger.warning(
                "Chunk %s (doc_id=%s) termina sin puntuación final válida.", chunk_id, doc_id
            )
            
        resultados.append({
            "doc_id": doc_id,
            "chunk_id": chunk_id,
            "fuente": fuente,
            "formato": formato,
            "fenomeno": fenomeno,
            "posicion": posicion,
            "num_tokens": contar_tokens(texto_chunk),
            "texto": texto_chunk,
        })
    return resultados


def construir_chunks(ruta_entrada: Path = INPUT_DEFAULT, ruta_salida: Path = OUTPUT_DEFAULT) -> int:
    if not ruta_entrada.exists():
        raise FileNotFoundError(f"No se encontró el corpus de entrada: {ruta_entrada}")
        
    ruta_salida.parent.mkdir(parents=True, exist_ok=True)
    total_chunks = 0
    total_docs = 0

    with open(ruta_entrada, "r", encoding="utf-8") as in_fh, \
         open(ruta_salida, "w", encoding="utf-8") as out_fh:
        for num, linea in enumerate(in_fh, 1):
            linea = linea.strip()
            if not linea:
                continue
            try:
                doc = json.loads(linea)
            except json.JSONDecodeError:
                logger.error("Línea %d: JSON inválido. Se omite.", num)
                continue

            total_docs += 1
            try:
                chunks = chunk_documento(doc)
            except Exception as exc:
                logger.error("Error procesando doc_id=%s (%s): %s",
                             doc.get("doc_id", "?"), doc.get("fuente", "?"), exc)
                continue

            for chunk in chunks:
                out_fh.write(json.dumps(chunk, ensure_ascii=False) + "\n")
            total_chunks += len(chunks)

            if total_docs % 100 == 0:
                logger.info("Procesados %d documentos (%d chunks)...", total_docs, total_chunks)

    logger.info("Chunks generados exitosamente: %s (%d fragmentos en %d documentos)",
                ruta_salida, total_chunks, total_docs)
    return total_chunks


def main() -> None:
    parser = argparse.ArgumentParser(description="Chunker lingüístico determinista — CODEFEST AD ASTRA 2026")
    parser.add_argument("--in", dest="entrada", type=str, default=str(INPUT_DEFAULT), help="Ruta a corpus_limpio.jsonl")
    parser.add_argument("--out", dest="salida", type=str, default=str(OUTPUT_DEFAULT), help="Ruta a chunks.jsonl")
    args = parser.parse_args()

    construir_chunks(Path(args.entrada), Path(args.salida))


if __name__ == "__main__":
    main()
