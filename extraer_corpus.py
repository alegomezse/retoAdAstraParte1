#!/usr/bin/env python3
"""
extraer_corpus.py â€” Fase 1 del pipeline RAG CODEFEST AD ASTRA.

Recorre data/, resuelve el campo `fuente` oficial vÃ­a Indice_Datos_Codefest.xlsx
y los catÃ¡logos por organizaciÃ³n, enruta cada archivo por tipo, extrae texto y
produce 3 JSONL (uno por fase F1/F2/F3) + tablas de referencia y logs.

USO:
    python extraer_corpus.py --data-dir /ruta/a/data --output-dir ./salida --workers 4

Requiere: pdfplumber, pandas, pypdf, tqdm
    pip install pdfplumber pandas pypdf tqdm --break-system-packages
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import multiprocessing as mp
import re
import sys
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Optional

# ---------------------------------------------------------------------------
# Dependencias externas â€” import perezoso con mensaje claro si faltan
# ---------------------------------------------------------------------------
import pdfplumber  # noqa: E402
import pandas as pd  # noqa: E402

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, **kwargs):
        return iterable

# ---------------------------------------------------------------------------
# Dependencias opcionales — import perezosos con fallback elegante
# ---------------------------------------------------------------------------
try:
    import pytesseract
    from PIL import Image
    _OCR_OK = True
except ImportError:  # pragma: no cover
    pytesseract = None
    Image = None
    _OCR_OK = False
    logging.warning("pytesseract/PIL no instalados; OCR de imágenes deshabilitado")

try:
    from bs4 import BeautifulSoup
    _BS4_OK = True
except ImportError:  # pragma: no cover
    BeautifulSoup = None
    _BS4_OK = False
    logging.warning("beautifulsoup4 no instalado; extracción HTML limitada")

try:
    import osmium as _osmium
    _OSM_OK = True
except ImportError:  # pragma: no cover
    _osmium = None
    _OSM_OK = False


# ---------------------------------------------------------------------------
# Mapeo organizaciÃ³n (nombre de carpeta) -> cÃ³digo de prefijo fÃ­sico
# ExtraÃ­do del Inventario de Archivos (Paso 0). Si el inventario real trae
# columnas distintas, este dict es el fallback y la fuente de verdad para
# reconocer prefijos al hacer matching contra catÃ¡logos.
# ---------------------------------------------------------------------------
ORG_FOLDER_TO_CODE = {
    "AI_Index_Stanford": "AIINDEX",
    "Atlantic_Council": "ATLCOUNCIL",
    "CENIA": "CENIA",
    "CSET_Georgetown": "CSET",
    "DAIO": "DAIO",
    "Defensa21_LatAm": "DEFENSA21",
    "ILIA_Latam": "ILIA",
    "RutaN_GEIAL": "RUTAN",
    "CSIS_Aerospace": "CSIS",
    "ESA_Space_Debris": "ESA",
    "INPE": "INPE",
    "SWF_Counterspace": "SWF",
    "UNOOSA": "UNOOSA",
    "Alertas_Tempranas": "ALERTAS",
    "Amazon_Underworld": "AMAZONUW",
    "CEEEP": "CEEEP",
    "CEOBS": "CEOBS",
    "MAPP_OEA": "MAPPOEA",
    "RESDAL": "RESDAL",
    "SIPRI": "SIPRI",
    "Wilson_Center": "WILSON",
}

# Extensiones/patrones que se excluyen del corpus textual sin intentar extraer.
# Imágenes (.jpg/.png) y PBF ahora se procesan (OCR y extracción de atributos).
# .avif se excluye por falta de soporte OCR fiable en pytesseract.
EXCLUDE_SUFFIXES = {".avif", ".idx", ".pack", ".rev", ".sample"}
EXCLUDE_NAMES = {".DS_Store"}
# Ãndices maestros de la raÃ­z: se leen aparte (Paso 0), no se indexan como corpus.
ROOT_INDEX_FILES = {
    "Indice_Datos_Codefest.xlsx",
    "FASE ORDENADA CODEFEST.xlsx",
    "Extracto_Preguntas_50_v2.pdf",
}
# Patrones de nombre que identifican archivos de catÃ¡logo/metadata (no contenido).
CATALOG_NAME_PATTERNS = re.compile(
    r"(catalog-2|catalogo|registro|mapp-catalog)\.(json|csv)$", re.IGNORECASE
)

MIN_CHARS_VALID = 50  # umbral para marcar texto_extraido como sospechoso


# ---------------------------------------------------------------------------
# Utilidades de matching / normalizaciÃ³n
# ---------------------------------------------------------------------------
def normalize_alnum(s: str) -> str:
    """MinÃºsculas + solo alfanumÃ©ricos. Uso para comparar nombres de archivo
    entre fuentes con formato distinto (guiones, espacios, mayÃºsculas)."""
    return re.sub(r"[^a-z0-9]", "", s.lower())


def strip_org_prefix(basename: str, org_code: str) -> str:
    """Quita el prefijo ORG_ del nombre fÃ­sico para compararlo contra el
    nombre original que aparece en los catÃ¡logos (que no llevan prefijo)."""
    if org_code:
        prefix = f"{org_code}_"
        if basename.upper().startswith(prefix.upper()):
            return basename[len(prefix):]
    return basename


def sha256_short(text: str, length: int = 16) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:length]


# ---------------------------------------------------------------------------
# Limpieza de texto (Fase 1 — requisito 7)
# ---------------------------------------------------------------------------
_RE_CONTROL_CHARS = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]")


def clean_control_chars(text: str) -> str:
    """Elimina caracteres de control no imprimibles (excepto \\n y \\t)."""
    return _RE_CONTROL_CHARS.sub("", text)


def clean_text(text: str) -> str:
    """Limpieza completa de texto extraído (Fase 1).

    1. Elimina caracteres de control no imprimibles.
    2. Normaliza espacios en blanco redundantes.
    3. Garantiza UTF-8 limpio (ya viene leyéndose como UTF-8).
    """
    text = clean_control_chars(text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = text.replace("\u00a0", " ")   # non-breaking space → space
    text = text.replace("\ufffd", "")    # replacement char
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"[ \t]*\n[ \t]*", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = text.replace("\ufeff", "")    # BOM
    return text.strip()


def detect_org_code(path: Path, data_root: Path) -> Optional[str]:
    """Determina el cÃ³digo de organizaciÃ³n a partir de la carpeta de segundo
    nivel bajo data/ (data/F1_.../AI_Index_Stanford/...)."""
    try:
        rel_parts = path.relative_to(data_root).parts
    except ValueError:
        return None
    for part in rel_parts:
        if part in ORG_FOLDER_TO_CODE:
            return ORG_FOLDER_TO_CODE[part]
    return None


def detect_fase(path: Path, data_root: Path) -> Optional[str]:
    try:
        rel_parts = path.relative_to(data_root).parts
    except ValueError:
        return None
    for part in rel_parts:
        m = re.match(r"^(F[123])_", part)
        if m:
            return m.group(1)
    return None


# ---------------------------------------------------------------------------
# Paso 0 â€” Carga de Ã­ndices maestros y catÃ¡logos
# ---------------------------------------------------------------------------
def load_inventario(indice_xlsx: Path) -> dict[str, dict]:
    """Lee la hoja 'Inventario de Archivos' del Ã­ndice maestro.
    Devuelve dict: basename_normalizado -> registro con fuente oficial.

    NOTA: los nombres exactos de columna pueden variar ligeramente respecto
    a lo documentado en Paso 0 ('DOC_ID', 'Nombre estandarizado', 'Carpeta',
    'Tipo', 'Fenomeno', 'Codigo Observatorio'). Se hace matching flexible por
    substring para tolerar variantes de acentos/mayÃºsculas.
    """
    if pd is None:
        raise RuntimeError("pandas no disponible â€” requerido para leer el Ã­ndice maestro")

    result: dict[str, dict] = {}
    if not indice_xlsx.exists():
        logging.warning("No se encontrÃ³ %s â€” se continuarÃ¡ sin fuente oficial", indice_xlsx)
        return result

    xl = pd.ExcelFile(indice_xlsx)
    sheet_name = next((s for s in xl.sheet_names if "inventario" in s.lower()), None)
    if sheet_name is None:
        logging.warning("Hoja 'Inventario de Archivos' no encontrada en %s (hojas: %s)",
                         indice_xlsx, xl.sheet_names)
        return result

    df = xl.parse(sheet_name)
    cols = {c.lower(): c for c in df.columns}

    def find_col(*keywords):
        for kw in keywords:
            for lc, orig in cols.items():
                if kw in lc:
                    return orig
        return None

    col_docid = find_col("doc_id", "doc id", "docid")
    col_nombre = find_col("estandarizado", "nombre estandarizado", "nombre")
    col_carpeta = find_col("carpeta")
    col_tipo = find_col("tipo")
    col_fenomeno = find_col("fenomeno", "fenÃ³meno")
    col_codigo = find_col("codigo observatorio", "cÃ³digo observatorio", "codigo")

    if col_nombre is None:
        logging.error("No se pudo identificar la columna de nombre estandarizado en %s", indice_xlsx)
        return result

    n_loaded = 0
    for _, row in df.iterrows():
        nombre_std = str(row[col_nombre]) if col_nombre else None
        if not nombre_std or nombre_std == "nan":
            continue
        basename_key = normalize_alnum(Path(nombre_std).name)
        record = {
            "doc_id_oficial": str(row[col_docid]) if col_docid else None,
            "fuente": nombre_std,
            "carpeta": str(row[col_carpeta]) if col_carpeta else None,
            "tipo": str(row[col_tipo]) if col_tipo else None,
            "fenomeno": str(row[col_fenomeno]) if col_fenomeno else None,
            "codigo_observatorio": str(row[col_codigo]) if col_codigo else None,
        }
        result[basename_key] = record
        n_loaded += 1

    logging.info("Inventario oficial cargado: %d registros", n_loaded)
    return result


def _extract_catalog_entries(obj: Any) -> list[dict]:
    """Aplana estructuras de catÃ¡logo heterogÃ©neas (listas de dicts, dicts
    con listas anidadas tipo 'articulos'/'items', o dicts planos tipo
    SIPRI_registro) en una lista uniforme de entradas candidatas."""
    entries: list[dict] = []
    if isinstance(obj, list):
        for item in obj:
            if isinstance(item, dict):
                entries.append(item)
    elif isinstance(obj, dict):
        for key in ("articulos", "items", "files", "records", "pdfs"):
            if key in obj and isinstance(obj[key], list):
                for item in obj[key]:
                    if isinstance(item, dict):
                        entries.append(item)
                    elif isinstance(item, str):
                        entries.append({"file": item})
                return entries
        # Esquema tipo SIPRI_registro: {urls: [...], hashes: [...]} sin mapeo
        # directo archivo->fuente fiable; se registra crudo.
        entries.append({"_raw_unparsed": True,
                        **{k: v for k, v in obj.items() if not isinstance(v, (list, dict))}})
    return entries


def load_catalogs(data_root: Path) -> dict[tuple[str, str], str]:
    """Recorre data/ buscando archivos de catÃ¡logo/registro y construye:
    (org_code, nombre_normalizado_sin_prefijo) -> valor de fuente.

    Prioridad de campo para 'fuente' dentro de cada entrada: 'fuente' > 'url'
    > 'titulo'/'title'. Prioridad de campo para el nombre de archivo a matchear:
    'nombre' > 'file' > 'dest' > 'path' > 'json'.
    """
    mapping: dict[tuple[str, str], str] = {}

    for path in data_root.rglob("*"):
        if not path.is_file():
            continue
        if not CATALOG_NAME_PATTERNS.search(path.name) and "catalog" not in path.name.lower():
            continue
        if path.suffix.lower() != ".json":
            continue  # los *.csv gemelos se ignoran: el json trae la misma info

        org_code = detect_org_code(path, data_root)
        if org_code is None:
            continue

        try:
            with open(path, "r", encoding="utf-8") as f:
                obj = json.load(f)
        except (json.JSONDecodeError, UnicodeDecodeError) as e:
            logging.warning("No se pudo parsear catÃ¡logo %s: %s", path, e)
            continue

        entries = _extract_catalog_entries(obj)
        n_added = 0
        for entry in entries:
            if entry.get("_raw_unparsed"):
                continue
            filename_field = next(
                (entry[k] for k in ("nombre", "file", "dest", "path", "json")
                 if k in entry and entry[k]),
                None,
            )
            fuente_field = next(
                (entry[k] for k in ("fuente", "url", "titulo", "title")
                 if k in entry and entry[k]),
                None,
            )
            if not filename_field or not fuente_field:
                continue
            if isinstance(filename_field, list):
                filename_field = filename_field[0] if filename_field else None
            if not filename_field:
                continue
            key_name = normalize_alnum(Path(str(filename_field)).name)
            mapping[(org_code, key_name)] = str(fuente_field)
            n_added += 1

        logging.info("CatÃ¡logo %s (%s): %d entradas mapeadas", path.name, org_code, n_added)

    return mapping


# ---------------------------------------------------------------------------
# ResoluciÃ³n del campo fuente para un archivo fÃ­sico
# ---------------------------------------------------------------------------
def resolve_fuente(
    path: Path,
    data_root: Path,
    inventario: dict[str, dict],
    catalog_map: dict[tuple[str, str], str],
) -> tuple[str, str, Optional[dict]]:
    """Devuelve (fuente, fuente_origen, registro_inventario_o_None).

    IMPORTANTE: tanto el Inventario oficial como los catÃ¡logos por organizaciÃ³n
    registran nombres que pueden coincidir con o sin el prefijo ORG_. Se prueba
    primero el basename sin prefijo y luego con prefijo, para robustez.
    """
    basename = path.name
    org_code = detect_org_code(path, data_root)
    stripped = strip_org_prefix(basename, org_code) if org_code else basename

    for candidate in (stripped, basename):
        key = normalize_alnum(candidate)
        if key in inventario:
            rec = inventario[key]
            return rec["fuente"], "indice_oficial", rec

    if org_code:
        catalog_key = (org_code, normalize_alnum(stripped))
        if catalog_key in catalog_map:
            return catalog_map[catalog_key], "catalog_metadata", None

    # Fallback: ruta relativa a data/
    try:
        rel = str(path.relative_to(data_root))
    except ValueError:
        rel = str(path)
    return rel, "nombre_archivo_fallback", None


# ---------------------------------------------------------------------------
# Extractores por tipo
# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# Detección de boilerplate dinámico (requerimiento 10)
# ---------------------------------------------------------------------------
def _normalize_line_for_compare(line: str) -> str:
    """Normaliza una línea para comparación de boilerplate:
    strip, collapse spaces, ignore case."""
    return re.sub(r"\s+", " ", line.strip().lower())


def _remove_dynamic_boilerplate(texto_paginas: list[str], threshold: float = 0.6) -> list[str]:
    """Elimina líneas que aparecen en >threshold (def 60%) de las páginas.
    Identifica encabezados/pies de página, numeración, branding repetitivo."""
    from collections import Counter

    n_pages = len(texto_paginas)
    if n_pages < 2:
        return texto_paginas

    line_counts: Counter = Counter()
    page_line_sets: list[set[str]] = []

    for page_text in texto_paginas:
        seen: set[str] = set()
        lines = page_text.split("\n")
        for raw_line in lines:
            norm = _normalize_line_for_compare(raw_line)
            if not norm:
                continue
            if len(norm) < 4:
                continue  # saltos de línea vacíos, numeración corta
            if norm in seen:
                continue  # contar una vez por página
            seen.add(norm)
            line_counts[norm] += 1
        page_line_sets.append(seen)

    # Líneas que aparecen en > threshold de páginas
    n_threshold = threshold * n_pages
    boilerplate_lines: set[str] = set()
    for norm, count in line_counts.items():
        if count > n_threshold:
            boilerplate_lines.add(norm)

    n_removed = len(boilerplate_lines)
    if boilerplate_lines:
        logging.info("Boilerplate dinámico: %d líneas recurrentes eliminadas", n_removed)

    # Reconstruir páginas eliminando esas líneas
    clean_pages: list[str] = []
    for page_text in texto_paginas:
        kept_lines = []
        for raw_line in page_text.split("\n"):
            norm = _normalize_line_for_compare(raw_line)
            if norm in boilerplate_lines:
                continue
            kept_lines.append(raw_line)  # conservar formatting original
        clean_pages.append("\n".join(kept_lines))

    return clean_pages


# ---------------------------------------------------------------------------
# Extractores por tipo
# ---------------------------------------------------------------------------
def extract_pdf(path: Path) -> tuple[str, list[dict], list[Any]]:
    """Extrae texto de un PDF preservando el orden de lectura.

    Estrategia (requisitos 1 y 10):
      a) Usa find_tables() para identificar bounding boxes de tablas y extrae
         cuerpo SIN superposición con celdas (evita duplicación).
      b) Si pdfplumber falla (PDF con solo imagen), devuelve ("", [], []) y
         el caller hará fallback a OCR.
      c) Aplica detección de boilerplate dinámico: líneas que aparecen en
         >60% de las páginas (cabecera/pie) se eliminan de todas.
    Devuelve (texto_completo, metadata_paginas, tablas).
    """
    if pdfplumber is None:
        raise RuntimeError("pdfplumber no está instalado")

    texto_paginas: list[str] = []
    paginas_meta: list[dict] = []
    tablas: list[Any] = []
    try:
        with pdfplumber.open(path) as pdf:
            total_pages = len(pdf.pages)
            for i, page in enumerate(pdf.pages):
                # --- Detectar tablas y extraer cuerpo sin solapar ---
                page_tables = page.find_tables() or []

                if page_tables:
                    table_bbox = None
                    for t in page_tables:
                        bbox = t.bbox
                        if table_bbox is None:
                            table_bbox = list(bbox)
                        else:
                            table_bbox[0] = min(table_bbox[0], bbox[0])
                            table_bbox[1] = min(table_bbox[1], bbox[1])
                            table_bbox[2] = max(table_bbox[2], bbox[2])
                            table_bbox[3] = max(table_bbox[3], bbox[3])

                    # Extraer texto EVITANDO la bounding box de las tablas
                    # (solo si hay suficiente texto restante)
                    safe_text = page.extract_text(
                        keep_blank_chars=False,
                        use_text_flow=False,
                        cropbbox=tuple(table_bbox) if table_bbox else None,
                    )
                    if not safe_text or len(safe_text) < (total_pages * 10):
                        # No hubo texto suficiente: usar extract_text simple (sin crop)
                        page_text = page.extract_text() or ""
                    else:
                        page_text = safe_text
                else:
                    page_text = page.extract_text() or ""

                texto_paginas.append(page_text)
                paginas_meta.append({"pagina": i + 1, "n_caracteres": len(page_text)})

                # Serializar tablas
                for t_obj in page_tables:
                    clean = [
                        [str(c) if c is not None else "" for c in row]
                        for row in (t_obj.extract() or [])
                    ]
                    tablas.append({"pagina": i + 1, "tablas": clean})

    except Exception as e:
        logging.error("Fallo extrayendo PDF %s: %s", path, e)
        return "", [], []

    # --- Eliminación de boilerplate dinámico ---
    if len(texto_paginas) >= 3:
        texto_paginas = _remove_dynamic_boilerplate(texto_paginas)

    return "\n\n".join(texto_paginas), paginas_meta, tablas


def extract_json_articulo(path: Path) -> tuple[Optional[str], bool, dict]:
    """Extrae texto del cuerpo de un JSON preservando metadata descriptiva.

    Prioridad de campos de cuerpo:
        body_text > body_paragraphs > abstract > excerpt > sections > content

    Devuelve (texto_extraido_o_None, es_catalogo, metadata_descriptiva).
    * Si el archivo encaja en CATALOG_NAME_PATTERNS, devuelve (None, True, {}).
    * metadata_descriptiva captura: title, titulo, author(s), date/fecha/publication_date.
    """
    if CATALOG_NAME_PATTERNS.search(path.name):
        return None, True, {}

    try:
        with open(path, "r", encoding="utf-8") as f:
            obj = json.load(f)
    except (json.JSONDecodeError, UnicodeDecodeError) as e:
        logging.warning("JSON inválido %s: %s", path, e)
        return None, False, {}

    def _extract_metadata_descriptiva(o: Any) -> dict:
        """Extrae campos descriptivos (title, author, date) recursivamente
        en los primeros niveles del JSON."""
        meta: dict = {}
        if not isinstance(o, dict):
            return meta
        for key_meta, key_orig in [
            ("title", ["title", "titulo", "name", "nombre"]),
            ("authors", ["author", "authors", "autor", "autores"]),
            ("date", ["date", "fecha", "publication_date", "fecha_publicacion"]),
        ]:
            for k_orig in key_orig:
                if k_orig in o and str(o[k_orig]).strip():
                    meta[key_meta] = str(o[k_orig])
                    break
        return meta

    # Estructuras tipo lista (Amazon tiles-index, etc.)
    if isinstance(obj, list):
        return None, False, {}

    if not isinstance(obj, dict):
        return None, False, {}

    # --- Extraer metadata descriptiva ---
    metadata_desc = _extract_metadata_descriptiva(obj)

    # --- Extraer cuerpo textual (prioridad de campos) ---
    if obj.get("body_text"):
        return str(obj["body_text"]), False, metadata_desc

    if obj.get("body_paragraphs"):
        parts = obj["body_paragraphs"]
        if isinstance(parts, list):
            return "\n\n".join(str(p) for p in parts), False, metadata_desc
        return str(parts), False, metadata_desc

    if obj.get("abstract"):
        return str(obj["abstract"]), False, metadata_desc

    if obj.get("excerpt"):
        return str(obj["excerpt"]), False, metadata_desc

    if obj.get("sections"):
        chunks = []
        for sec in obj["sections"]:
            if isinstance(sec, dict):
                heading = sec.get("heading", "")
                paragraphs = sec.get("paragraphs", [])
                body = "\n".join(str(p) for p in paragraphs) if isinstance(paragraphs, list) else str(paragraphs)
                chunks.append(f"{heading}\n{body}".strip())
        if chunks:
            return "\n\n".join(chunks), False, metadata_desc

    if obj.get("content"):
        return str(obj["content"]), False, metadata_desc

    # Solo título/catálogo, sin cuerpo indexable
    return None, False, metadata_desc


def extract_txt(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return path.read_text(encoding="latin-1")


def extract_tabular(path: Path) -> Optional[str]:
    """Serializa CSV/XLSX leyendo TODAS las columnas (incl. numéricas/fechas).
    Formatea cada registro como pares 'columna: valor' (requerimiento 4).
    Omite celdas vacías (NaN) — no aplica heurística de longitud."""
    try:
        if path.suffix.lower() == ".csv":
            df = pd.read_csv(path, nrows=5000, on_bad_lines="skip")
        else:
            df = pd.read_excel(path, nrows=5000)
    except Exception as e:
        logging.error("No se pudo leer tabular %s: %s", path, e)
        return None

    if df.empty:
        return None

    cols = df.columns.tolist()
    rows_text: list[str] = []
    for _, row in df.iterrows():
        parts = []
        for col in cols:
            val = row[col]
            if pd.isna(val):
                continue
            val_str = str(val).strip()
            if not val_str:
                continue
            parts.append(f"{col}: {val_str}")
        if parts:
            rows_text.append(", ".join(parts))

    return "\n".join(rows_text) if rows_text else None


# ---------------------------------------------------------------------------
# Extracción de imágenes (OCR) — requerimiento 5
# ---------------------------------------------------------------------------
def extract_image(path: Path) -> str:
    """Aplica OCR a imágenes (.jpg/.jpeg/.png) usando pytesseract + Pillow.
    Si pytesseract no está disponible, devuelve texto vacío."""
    if not _OCR_OK:
        return ""
    try:
        img = Image.open(path)
        # Convertir a grises para mejorar OCR
        img = img.convert("L")
        text = pytesseract.image_to_string(img, config="--psm 6")
        return clean_text(text)
    except Exception as e:
        logging.warning("OCR falló para imagen %s: %s", path, e)
        return ""


# ---------------------------------------------------------------------------
# Extracción de HTML — requerimiento 2
# ---------------------------------------------------------------------------
def extract_html(path: Path) -> str:
    """Extrae texto visible de archivos HTML/HTM.
    Elimina <script>, <style>, <nav>, <header>, <footer>, <noscript>, etc.
    Conserva estructura por párrafos."""
    if not _BS4_OK:
        return ""
    try:
        content = path.read_text(encoding="utf-8", errors="ignore")
        soup = BeautifulSoup(content, "lxml")
        # Remover elementos no deseados
        for element in soup(["script", "style", "nav", "header", "footer",
                             "noscript", "iframe", "svg", "canvas",
                             "meta", "link", "title"]):
            element.decompose()
        # text(separator) preserva estructura por bloques
        text = soup.get_text(separator="\n", strip=True)
        return clean_text(text)
    except Exception as e:
        logging.warning("Extracción HTML falló para %s: %s", path, e)
        return ""


# ---------------------------------------------------------------------------
# Extracción de archivos PBF — requerimiento 6
# ---------------------------------------------------------------------------
class _PbfAttributeHandler:
    """Handler osmium para extraer atributos clave:valor de nodos/ways/relations."""

    def __init__(self) -> None:
        self.results: list[str] = []

    def _add_tags(self, tags: dict, element_type: str, element_id: Any) -> None:
        """Convierte etiquetas OSM a pares 'atributo: valor'."""
        sorted_keys = sorted(tags.keys())
        for k in sorted_keys:
            v = tags[k]
            if v is None:
                continue
            self.results.append(f"{k}: {v}")

    def node(self, n: Any) -> None:
        self._add_tags(dict(n.tags), "node", n.id)

    def way(self, w: Any) -> None:
        self._add_tags(dict(w.tags), "way", w.id)

    def relation(self, r: Any) -> None:
        self._add_tags(dict(r.tags), "relation", r.id)


def extract_pbf(path: Path) -> str:
    """Extrae atributos clave:valor de capas/elementos OSM de un archivo .pbf.
    Usa pyrosm si está disponible; fallback a extracción de strings imprimibles
    si pyrosm/osmium falla."""
    if _OSM_OK:
        try:
            handler = _PbfAttributeHandler()
            _osmium.SimpleHandler(handler).apply_file(str(path))
            if handler.results:
                return "\n".join(handler.results)
        except Exception as e:
            logging.warning("pyrosm falló para %s: %s", path, e)

    # Fallback: extraer strings imprimibles del binario PBF
    # (útil para obtener nombres,descripciones textuales)
    try:
        raw = path.read_bytes()
        # Extraer strings de 4+ caracteres que parezcan texto
        text = raw.decode("utf-8", errors="ignore")
        # Filtrar strings imprimibles de longitud razonable
        strings = re.findall(r"[\x20-\x7e\u00a0-\uffff]{4,}", text)
        # Deduplicar manteniendo order
        seen = set()
        unique = []
        for s in strings:
            stripped = s.strip()
            if stripped and stripped not in seen:
                seen.add(stripped)
                unique.append(stripped)
        if unique:
            logging.info("PBF fallback: %d strings extraídos de %s", len(unique), path)
            return "\n".join(unique)
    except Exception as e:
        logging.warning("Extracción fallback falló para %s: %s", path, e)

    return ""


# ---------------------------------------------------------------------------
# ClasificaciÃ³n de exclusiÃ³n
# ---------------------------------------------------------------------------
def exclusion_reason(path: Path) -> Optional[str]:
    if path.name in EXCLUDE_NAMES:
        return "artefacto_sistema"
    if path.suffix.lower() in EXCLUDE_SUFFIXES:
        return "tipo_no_exclusion_explicita"
    if path.name in ROOT_INDEX_FILES:
        return "indice_maestro_no_corpus"
    return None


# ---------------------------------------------------------------------------
# Registro de salida
# ---------------------------------------------------------------------------
@dataclass
class Documento:
    doc_id: str
    fuente: str
    fuente_origen: str
    fase: Optional[str]
    organizacion: Optional[str]
    tipo_original: str
    ruta_relativa: str
    texto_extraido: str
    fenomeno: Optional[str] = None
    doc_id_oficial: Optional[str] = None
    tablas_extraidas: list = field(default_factory=list)
    metadata: dict = field(default_factory=dict)
    metadata_descriptiva: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Procesamiento de un archivo individual (usado por workers y hilo principal)
# ---------------------------------------------------------------------------
def process_file(
    path: Path,
    data_root: Path,
    inventario: dict[str, dict],
    catalog_map: dict[tuple[str, str], str],
) -> tuple[Optional[Documento], Optional[dict]]:
    """Devuelve (Documento_o_None, exclusion_dict_o_None)."""
    reason = exclusion_reason(path)
    if reason:
        return None, {"ruta": str(path), "razon": reason}

    suffix = path.suffix.lower()
    fase = detect_fase(path, data_root)
    org = detect_org_code(path, data_root)

    texto = ""
    tablas: list = []
    metadata: dict = {}
    metadata_desc: dict = {}

    if suffix == ".pdf":
        texto, paginas_meta, tablas = extract_pdf(path)
        metadata["paginas"] = paginas_meta
        tipo = "pdf"
    elif suffix == ".json":
        extracted, es_catalogo, metadata_desc = extract_json_articulo(path)
        if es_catalogo:
            return None, {"ruta": str(path), "razon": "catalogo_metadata_no_indexable"}
        if extracted is None:
            return None, {"ruta": str(path), "razon": "json_sin_contenido_indexable"}
        texto = extracted
        tipo = "json"
    elif suffix == ".txt":
        texto = extract_txt(path)
        tipo = "txt"
    elif suffix == ".html" or suffix == ".htm":
        texto = extract_html(path)
        tipo = "html"
    elif suffix == ".csv" or suffix == ".xlsx":
        extracted = extract_tabular(path)
        if extracted is None:
            return None, {"ruta": str(path), "razon": "tabular_sin_contenido_descriptivo"}
        texto = extracted
        tipo = "csv" if suffix == ".csv" else "xlsx"
    elif suffix in (".jpg", ".jpeg", ".png"):
        texto = extract_image(path)
        tipo = "image"
    elif suffix == ".pbf":
        texto = extract_pbf(path)
        tipo = "pbf"
    else:
        return None, {"ruta": str(path), "razon": f"extension_no_enrutada_{suffix}"}

    # --- Limpieza de texto (requisito 7: Fase 1) ---
    texto = clean_text(texto)
    if not texto:
        # Texto vacío después de limpieza
        return None, {"ruta": str(path), "razon": "texto_vacio_despues_limpieza"}

    # --- Resolver fuente oficial (prioridad Inventario > Catálogo > nombre archivo) ---
    fuente, fuente_origen, inv_rec = resolve_fuente(path, data_root, inventario, catalog_map)

    # --- Extraer fenómeno del Inventario Oficial ---
    fenomeno = None
    if inv_rec:
        fenomeno = inv_rec.get("fenomeno")

    metadata["n_caracteres"] = len(texto)

    doc = Documento(
        doc_id=sha256_short(str(path.resolve())),
        fuente=fuente,
        fuente_origen=fuente_origen,
        fase=fase,
        organizacion=org,
        tipo_original=tipo,
        ruta_relativa=str(path.relative_to(data_root)),
        texto_extraido=texto,
        fenomeno=fenomeno,
        doc_id_oficial=inv_rec["doc_id_oficial"] if inv_rec else None,
        tablas_extraidas=tablas,
        metadata=metadata,
        metadata_descriptiva=metadata_desc,
    )
    return doc, None


# Wrapper para multiprocessing (funciÃ³n de nivel de mÃ³dulo, picklable)
def _worker_pdf(args):
    path_str, data_root_str, inventario, catalog_map = args
    path = Path(path_str)
    data_root = Path(data_root_str)
    return process_file(path, data_root, inventario, catalog_map)


# ---------------------------------------------------------------------------
# OrquestaciÃ³n principal
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--workers", type=int, default=max(1, mp.cpu_count() - 1))
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    log_level = getattr(logging, str(args.log_level).upper(), logging.INFO)
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    data_root: Path = args.data_dir.resolve()
    out_dir: Path = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- Paso 0: Ã­ndices maestros y catÃ¡logos ---
    inventario = load_inventario(data_root / "Indice_Datos_Codefest.xlsx")
    catalog_map = load_catalogs(data_root)

    # Guardar fuente_mapping.jsonl como tabla de referencia
    fuente_mapping_path = out_dir / "fuente_mapping.jsonl"
    with open(fuente_mapping_path, "w", encoding="utf-8") as f:
        for (org_code, key), fuente in catalog_map.items():
            f.write(json.dumps({"organizacion": org_code, "nombre_normalizado": key, "fuente": fuente},
                                ensure_ascii=False) + "\n")
    logging.info("fuente_mapping.jsonl escrito con %d entradas", len(catalog_map))

    # --- Checkpointing ---
    procesados_path = out_dir / "procesados.log"
    procesados: set[str] = set()
    if procesados_path.exists():
        procesados = set(procesados_path.read_text(encoding="utf-8").splitlines())
        logging.info("Checkpoint cargado: %d documentos ya procesados", len(procesados))

    # --- Enumerar todos los archivos bajo data/ (recursivo) ---
    all_files = [p for p in data_root.rglob("*") if p.is_file()]
    logging.info("Total de archivos encontrados bajo %s: %d", data_root, len(all_files))

    pdf_files = [p for p in all_files if p.suffix.lower() == ".pdf" and p.name not in ROOT_INDEX_FILES]
    other_files = [p for p in all_files if p not in pdf_files]

    def already_done(p: Path) -> bool:
        return sha256_short(str(p.resolve())) in procesados

    pdf_files = [p for p in pdf_files if not already_done(p)]
    other_files = [p for p in other_files if not already_done(p)]

    fase_writers = {
        "F1": open(out_dir / "f1_extraido.jsonl", "a", encoding="utf-8"),
        "F2": open(out_dir / "f2_extraido.jsonl", "a", encoding="utf-8"),
        "F3": open(out_dir / "f3_extraido.jsonl", "a", encoding="utf-8"),
    }
    exclusiones_f = open(out_dir / "exclusiones.log", "w", encoding="utf-8")
    procesados_f = open(procesados_path, "a", encoding="utf-8")

    stats = {"ok": 0, "excluidos": 0, "sin_fase": 0, "vacios": 0}

    def write_doc(doc_dict: dict):
        nonlocal stats
        fase = doc_dict.get("fase")
        writer = fase_writers.get(fase)
        if writer is None:
            stats["sin_fase"] += 1
            logging.warning("Documento sin fase detectada (omitido): %s", doc_dict["ruta_relativa"])
            return
        writer.write(json.dumps(doc_dict, ensure_ascii=False) + "\n")
        writer.flush()
        procesados_f.write(doc_dict["doc_id"] + "\n")
        procesados_f.flush()
        stats["ok"] += 1
        if doc_dict["metadata"].get("n_caracteres", 0) < MIN_CHARS_VALID:
            stats["vacios"] += 1

    def write_exclusion(excl: dict):
        nonlocal stats
        exclusiones_f.write(json.dumps(excl, ensure_ascii=False) + "\n")
        stats["excluidos"] += 1

    # --- Secuencial: JSON / CSV / XLSX / TXT ---
    for path in tqdm(other_files, desc="Procesando JSON/CSV/XLSX/TXT"):
        doc, excl = process_file(path, data_root, inventario, catalog_map)
        if doc:
            write_doc(asdict(doc))
        elif excl:
            write_exclusion(excl)

    # --- Paralelo: PDFs (el tipo mÃ¡s costoso) ---
    if pdf_files:
        task_args = [(str(p), str(data_root), inventario, catalog_map) for p in pdf_files]
        num_workers = min(args.workers, len(task_args))
        with mp.Pool(processes=num_workers) as pool:
            for doc, excl in tqdm(pool.imap_unordered(_worker_pdf, task_args),
                                  total=len(task_args), desc="Procesando PDFs"):
                if doc:
                    write_doc(asdict(doc))
                elif excl:
                    write_exclusion(excl)

    for w in fase_writers.values():
        w.close()
    exclusiones_f.close()
    procesados_f.close()

    # --- Reporte final ---
    reporte_path = out_dir / "reporte_extraccion.md"
    with open(reporte_path, "w", encoding="utf-8") as f:
        f.write("# Reporte de extracciÃ³n â€” CODEFEST AD ASTRA\n\n")
        f.write(f"- Archivos totales encontrados: {len(all_files)}\n")
        f.write(f"- Documentos extraÃ­dos exitosamente: {stats['ok']}\n")
        f.write(f"- Documentos excluidos: {stats['excluidos']}\n")
        f.write(f"- Documentos sin fase detectada (omitidos): {stats['sin_fase']}\n")
        f.write(f"- Documentos con texto sospechosamente corto (<{MIN_CHARS_VALID} chars): {stats['vacios']}\n")
        f.write(f"- Entradas en fuente_mapping.jsonl (catÃ¡logos): {len(catalog_map)}\n")
        f.write(f"- Entradas en inventario oficial cargadas: {len(inventario)}\n")

    logging.info("ExtracciÃ³n completa. Reporte en %s", reporte_path)
    logging.info("Stats: %s", stats)


if __name__ == "__main__":
    main()
