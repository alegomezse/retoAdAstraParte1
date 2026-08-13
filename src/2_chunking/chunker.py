"""
FUENTE 1-B: CHUNKER LINGÜÍSTICO — CODEFEST AD ASTRA 2026.

Lee `corpus_limpio.jsonl` (un documento por línea) y produce `chunks.jsonl`
donde cada línea es un fragmento que respeta la COMPLETITUD LINGÜÍSTICA:

    {
      "doc_id": "fen1_pdf_001",
      "chunk_id": "fen1_pdf_001_chunk_0003",
      "fuente": "nombre_archivo.ext",
      "formato": "pdf" | "html" | "json" | ...,
      "fenomeno": 1 | 2 | 3,
      "posicion": 0,            # índice ordinal dentro del documento
      "num_tokens": 342,        # tokens contados por espacios
      "texto": "<fragmento sin modificar>"
    }

Reglas (100% determinísticas, sin LLMs):
  * Tamaño objetivo: <= 400 tokens. Avanzar ~350 y retroceder al último
    punto final (., !, ?) —o salto de línea— que quepa dentro del límite.
  * Corte SOLO al final de oraciones/unidades; nunca cortar a la mitad de una
    oración ni de una palabra.
  * Superposición de 1 oración entre chunks consecutivos.
  * Oración > 400 tokens -> dividir primero por comas/puntos y coma, luego por
    palabras (nunca a mitad de palabra).
  * Detección de abreviaturas (Dr., Sr., p. ej., etc.) para no cortar falsamente.
  * Warning si un chunk termina sin puntuación final válida.

Ejecución:
    python src/2_chunking/chunker.py
    python src/2_chunking/chunker.py --in corpus_limpio.jsonl --out chunks.jsonl
"""

from __future__ import annotations

import argparse
import json
import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Tuple

try:
    import regex as _re  # unicode-aware
except Exception:  # pragma: no cover
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

MAX_TOKENS = 400      #tamaño objetivo máximo por chunk
TARGET_TOKENS = 350   #avanzar ~350 (soft)
OVERLAP = 1           #oraciones de superposición entre chunks consecutivos

# Abreviaturas comunes (es/en/pt) que NO deben cortar la oración.
ABREVIATURAS = {
    "dr", "dra", "sr", "sra", "srta", "srs", "sres",
    "ing", "inga", "lic", "licenciado", "licenciada", "prof", "profa",
    "etc",
    "ej", "ejemplo",
    "ie", "eg",
    "et al", "et al.", "al", "uds", "ud", "vs", "v",
    "mm", "cm", "km", "kg", "hs", "h", "m", "s", "gr", "mol",
    "us", "us.", "ee uu", "e.u", "e.u.",
    "jan", "feb", "mar", "abr", "may", "jun", "jul", "ago", "sep", "sept",
    "oct", "nov", "dic",
    "med", "ciertos", "num", "ed", "red", "rev", "fac", "univ", "co", "ltd",
    "corp", "s.a", "ph.d", "m.sc", "b.sc", "a.m", "p.m",
    "est", "sta", "pto", "cdad", "apdo", "tel", "fax", "www", "http",
    "incl", "comp", "cont", "edit", "trad", "revis", "publ",
    "s.l", "s.l.", "s.c.", "r.s", "c.v", "d.l", "d.l.",
}

# Puntuación que cierra una oración.
RE_PUNTO_FINAL = _re.compile(r"[.!?…]+")
# Un token final (palabra posiblemente con puntos internos) antes de un corte.
RE_TOKEN_PREVIO = _re.compile(r"([^\s.!?…]+)\s*$")
# Número de lista estilo "1." "12." antes de una mayúscula.
RE_NUM_LISTA = _re.compile(r"^\d{1,3}$")


# --------------------------------------------------------------------------- #
# 1. TOKENIZACIÓN DE ESPACIOS (aproximación a tokens)
# --------------------------------------------------------------------------- #
def contar_tokens(texto: str) -> int:
    """Cuenta tokens aproximando con split por espacios."""
    return len(texto.split())


# --------------------------------------------------------------------------- #
# 2. SPLIT DE ORACIONES / UNIDADES (con detección de abreviaturas)
# --------------------------------------------------------------------------- #
def _es_corte_oracion(texto: str, m) -> bool:
    """Indica si un match de puntuación [.!?…]+ es un verdadero cierre de oración."""
    pre = texto[:m.start()]
    fol = texto[m.end():]
    punct = m.group()

    # decimal: dÍgito.digITO -> "3.14" (el '.' entre dígitos NO es corte)
    if pre and pre[-1].isdigit() and fol[:1].isdigit():
        return False

    tok_m = RE_TOKEN_PREVIO.search(pre)
    token = tok_m.group(1) if tok_m else ""
    core = token.lower().rstrip(".")

    # inicial simple: "J." / "A." -> no corta
    if len(core) == 1 and core.isalpha():
        return False
    # abreviatura conocida: "Dr." / "etc." / "p.ej." -> no corta
    if core in ABREVIATURAS:
        return False
    # numeración de lista: "1." seguido de mayúscula -> no corta ("1. Introducción")
    if RE_NUM_LISTA.match(core) and fol[:1].isalpha() and fol[:1].isupper():
        return False

    # '!' / '?' son cortes contundentes (menos ambiguos que '.' en abreviaturas)
    if "!" in punct or "?" in punct:
        return True
    # cierre al final del texto
    if not fol or fol.isspace():
        return True
    fol_first = fol.lstrip()[:1]
    # siguiente oración empieza con mayúscula/cita
    if fol_first and (fol_first.isupper() or fol_first in "\"'([{¿¡"):
        return True
    return False


def split_unidades(texto: str) -> List[str]:
    """
    Divide el texto en unidades (oraciones o párrafos) preservando el orden de
    lectura. Los cortes ocurren SOLO al final de oraciones (.!?) o en saltos de
    línea (párrafos/líneas). Nunca corta a mitad de palabra.
    """
    if not texto:
        return []
    cortes = [0]
    for m in RE_PUNTO_FINAL.finditer(texto):
        if m.start() == 0:
            continue
        if _es_corte_oracion(texto, m):
            cortes.append(m.end())
    # cortar también en saltos de línea (para CSV / multi-párrafos)
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


# --------------------------------------------------------------------------- #
# 3. DIVISIÓN DE ORACIONES DEMASIADO LARGAS (>400 tokens)
# --------------------------------------------------------------------------- #
def _dividir_por_comas(texto: str) -> List[str]:
    """Primera pasada: cortar en comas / puntos y coma / dos puntos / guiones."""
    segs = _re.split(r"(?<=[,;:\u2013\u2014])\s+", texto)
    return [s for s in segs if s]


def _dividir_por_palabras(texto: str, max_tokens: int) -> List[str]:
    """Última pasada: cortar por palabras completas, sin partirlas."""
    palabras = texto.split()
    salida: List[str] = []
    cur: List[str] = []
    for w in palabras:
        if len(cur) + 1 > max_tokens and cur:
            salida.append(" ".join(cur))
            cur = []
        cur.append(w)
    if cur:
        salida.append(" ".join(cur))
    return salida


def split_oracion_larga(texto: str, max_tokens: int) -> List[str]:
    """Divide una oración larga primero por signos, luego por palabras."""
    partes = _dividir_por_comas(texto)
    salida: List[str] = []
    for parte in partes:
        if contar_tokens(parte) <= max_tokens:
            salida.append(parte)
            continue
        salida.extend(_dividir_por_palabras(parte, max_tokens))
    return salida


# --------------------------------------------------------------------------- #
# 4. ARMADO DE CHUNKS (ventana 350 -> retroceso al límite de oración <=400)
# --------------------------------------------------------------------------- #
def _unir_unidades(unidades: List[str]) -> str:
    """Une unidades preservando párrafos (salto doble entre párrafos)."""
    texto = "\n".join(unidades)
    texto = _re.sub(r"\n{3,}", "\n\n", texto)
    return texto.strip()


def chunk_text(texto: str) -> List[Tuple[str, bool]]:
    """
    Produce una lista de (texto_chunk, fue_division_forzada).
    Mantiene completitud lingüística, overlap de 1 oración y límite 400 tokens.
    """
    unidades = split_unidades(texto)
    if not unidades:
        return []

    chunks: List[Tuple[str, bool]] = []
    current: List[str] = []
    cur_tok = 0

    for unidad in unidades:
        tok = contar_tokens(unidad)

        # Oración individual demasiado grande -> dividirla (forzado)
        if tok > MAX_TOKENS:
            if current:
                chunks.append((_unir_unidades(current), False))
                current = []
                cur_tok = 0
            for seg in split_oracion_larga(unidad, MAX_TOKENS):
                chunks.append((seg.strip(), True))
            continue

        # Al agregar excede el límite -> cerrar chunk en el último corte válido
        if cur_tok + tok > MAX_TOKENS and current:
            chunks.append((_unir_unidades(current), False))
            # overlap: reutilizar la última oración del chunk cerrado
            current = current[-OVERLAP:] if len(current) >= OVERLAP else list(current)
            cur_tok = sum(contar_tokens(u) for u in current)

        current.append(unidad)
        cur_tok += tok

    if current:
        chunks.append((_unir_unidades(current), False))

    return chunks


def _termina_con_puntuacion(texto: str) -> bool:
    return bool(_re.search(r"[.!?…]\s*$", texto))


# --------------------------------------------------------------------------- #
# 5. PROCESO POR DOCUMENTO
# --------------------------------------------------------------------------- #
def chunk_documento(doc: Dict[str, Any]) -> List[Dict[str, Any]]:
    texto = doc.get("texto", "") or ""
    doc_id = doc.get("doc_id", "")
    fuente = doc.get("fuente", "")
    formato = doc.get("formato", "")
    fenomeno = doc.get("fenomeno", 0)

    resultados: List[Dict[str, Any]] = []
    for posicion, (texto_chunk, dividido) in enumerate(chunk_text(texto)):
        chunk_id = f"{doc_id}_chunk_{posicion:04d}"
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


# --------------------------------------------------------------------------- #
# 6. ORQUESTACIÓN
# --------------------------------------------------------------------------- #
def iterar_corpus(ruta: Path):
    with open(ruta, "r", encoding="utf-8") as fh:
        for num, linea in enumerate(fh, 1):
            linea = linea.strip()
            if not linea:
                continue
            try:
                yield json.loads(linea)
            except json.JSONDecodeError:
                logger.error("Línea %d: JSON inválido. Se omite.", num)


def construir_chunks(ruta_entrada: Path = INPUT_DEFAULT, ruta_salida: Path = OUTPUT_DEFAULT) -> int:
    if not ruta_entrada.exists():
        raise FileNotFoundError(f"No se encontró el corpus de entrada: {ruta_entrada}")
    ruta_salida.parent.mkdir(parents=True, exist_ok=True)
    total = 0
    docs = 0
    with open(ruta_salida, "w", encoding="utf-8") as out_fh:
        for doc in iterar_corpus(ruta_entrada):
            docs += 1
            try:
                chunks = chunk_documento(doc)
            except Exception as exc:
                logger.error("Error procesando doc_id=%s (%s): %s",
                             doc.get("doc_id", "?"), doc.get("fuente", "?"), exc)
                continue
            for chunk in chunks:
                out_fh.write(json.dumps(chunk, ensure_ascii=False) + "\n")
            total += len(chunks)
            if docs % 100 == 0:
                logger.info("Procesados %d documentos (%d chunks)...", docs, total)
    logger.info("Chunks generados: %s (%d fragmentos en %d documentos)", ruta_salida, total, docs)
    return total


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Chunker lingüístico determinista — CODEFEST AD ASTRA 2026"
    )
    parser.add_argument("--in", dest="entrada", type=str, default=str(INPUT_DEFAULT),
                        help="Ruta a corpus_limpio.jsonl")
    parser.add_argument("--out", dest="salida", type=str, default=str(OUTPUT_DEFAULT),
                        help="Ruta a chunks.jsonl")
    args = parser.parse_args()
    construir_chunks(Path(args.entrada), Path(args.salida))


if __name__ == "__main__":
    main()
