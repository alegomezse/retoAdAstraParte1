"""
FUENTE 1-A: EXTRACTOR DE CORPUS LIMPIO — CODEFEST AD ASTRA 2026.

Lee todos los documentos crudos bajo data/ (F1, F2, F3) con formatos mixtos
(.pdf, .html, .json, .csv, .xlsx, imágenes .png/.jpg/.avif y .pbf) y genera
`corpus_limpio.jsonl`: un JSON por documento con el esquema:

    {
      "doc_id": "fen1_pdf_001",
      "fuente": "nombre_archivo_original.ext",
      "formato": "pdf" | "html" | "json" | "csv" | "xlsx" | "img" | "pbf",
      "fenomeno": 1 | 2 | 3,
      "idioma": "es" | "en" | "pt" | ... | "unknown",
      "texto": "<texto limpio, orden de lectura preservado>",
      "metadata_doc": { ...campos descriptivos... }
    }

Características:
  * Orden de lectura preservado (PDFs página a página; HTML por flujo DOM).
  * HTML con BeautifulSoup/lxml; headers <h1..h6> marcados con saltos dobles.
  * PDFs vía PyMuPDF (fitz) con remoción de headers/footers y numeración repetida.
  * JSON genérico: concatena title/body_text/body_paragraphs/etc. en orden;
    url/date/authors/... van a metadata (separados del cuerpo).
  * CSV/XLSX: "columna: valor | columna: valor" por fila (celdas vacías omitidas).
  * Imágenes: OCR con pytesseract si existe; si no, texto vacío + filename en metadata.
  * PBF: decoder Mapbox Vector Tile sin dependencias (pyrosm/osmium si existen).
  * Limpieza: UTF-8 (ftfy), eliminación de control, colapsado de espacios.
  * Detección de idioma con langdetect.
  * Skip por error por archivo + log. Nada de LLMs.

Ejecución:
    python src/1_extraccion/extractor.py
    python src/1_extraccion/extractor.py --data data --out corpus_limpio.jsonl
"""

from __future__ import annotations

import csv
import io
import json
import logging
import re
import struct
import unicodedata
import zlib
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

# --- Dependencias opcionales (importación defensiva) ---
try:
    import ftfy
except Exception:  # pragma: no cover
    ftfy = None
try:
    import fitz  # PyMuPDF
except Exception:  # pragma: no cover
    fitz = None
try:
    import pdfplumber
except Exception:  # pragma: no cover
    pdfplumber = None
try:
    from bs4 import BeautifulSoup, NavigableString, Tag
except Exception:  # pragma: no cover
    BeautifulSoup = None
    NavigableString = None
    Tag = None
try:
    from lxml import html as lxml_html
    from lxml import etree as lxml_etree
except Exception:  # pragma: no cover
    lxml_html = None
    lxml_etree = None
try:
    import pytesseract
    from PIL import Image as PILImage
except Exception:  # pragma: no cover
    pytesseract = None
    PILImage = None
try:
    from langdetect import DetectorFactory, detect
    from langdetect.lang_detect_exception import LangDetectException
except Exception:  # pragma: no cover
    DetectorFactory = None
    detect = None
    LangDetectException = Exception

import html as _html

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("extractor")
if DetectorFactory is not None:
    DetectorFactory.seed = 888

# --------------------------------------------------------------------------- #
# CONFIGURACIÓN
# --------------------------------------------------------------------------- #
BASE_DIR: Path = Path(__file__).resolve().parents[2]
DATA_DIR: Path = BASE_DIR / "data"
CORPUS_DEFAULT: Path = BASE_DIR / "corpus_limpio.jsonl"
EXCLUSIONES_LOG: Path = BASE_DIR / "salida" / "exclusiones_extractor.log"

FENOMENO_POR_CARPETA: Dict[str, int] = {
    "F1_IA_y_Capacidades_Estrategicas": 1,
    "F2_Seguridad_Entorno_Espacial": 2,
    "F3_Dinamicas_Territoriales": 3,
}
EXT_FORMATO: Dict[str, str] = {
    ".pdf": "pdf", ".html": "html", ".htm": "html",
    ".json": "json", ".csv": "csv",
    ".xls": "xlsx", ".xlsx": "xlsx",
    ".png": "img", ".jpg": "img", ".jpeg": "img",
    ".avif": "img", ".webp": "img", ".tif": "img", ".tiff": "img",
    ".pbf": "pbf",
}
NOMBRES_IGNORAR = {
    ".ds_store", "indice_datos_codefest.xlsx", "extracto_preguntas_50_v2.pdf",
}
MAX_PDF_OCR = 1


# --------------------------------------------------------------------------- #
# 1. LIMPIEZA Y UTILIDADES
# --------------------------------------------------------------------------- #
def _ftfy_fix(text: str) -> str:
    if ftfy is not None:
        try:
            return ftfy.fix_text(text, normalization="NFKC")
        except Exception:
            pass
    return text


def clean_text(text: str) -> str:
    """Normaliza UTF-8, elimina caracteres de control, colapsa espacios y líneas."""
    if not text:
        return ""
    text = _ftfy_fix(text)
    text = unicodedata.normalize("NFC", text)
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", text)  # control (preserva \t \n \r)
    text = text.replace("\u00a0", " ").replace("\u200b", "")
    text = re.sub(r"[ \t\r]+", " ", text)          # colapsar espacios/tabs
    text = re.sub(r"\n[ \t]*\n+", "\n\n", text)     # colapsar líneas en blanco
    text = re.sub(r" *[ \t]*\n", "\n", text)        # trailing spaces
    return text.strip()


def detectar_idioma(texto: str) -> str:
    if not texto or len(texto.split()) < 3:
        return "unknown"
    try:
        return detect(texto)
    except (LangDetectException, Exception):
        return "unknown"


def _doc_id(fenomeno: int, formato: str, contador: int) -> str:
    return f"fen{fenomeno}_{formato}_{contador:03d}"


# --------------------------------------------------------------------------- #
# 2. DETECCIÓN DE FENÓMENO Y FORMATO
# --------------------------------------------------------------------------- #
def fenomeno_de_path(path: Path) -> Optional[int]:
    try:
        rel = path.relative_to(DATA_DIR)
    except ValueError:
        return None
    return FENOMENO_POR_CARPETA.get(rel.parts[0])


def formato_de_ext(ext: str) -> Optional[str]:
    return EXT_FORMATO.get(ext.lower())


# --------------------------------------------------------------------------- #
# 3. CLASE EXTRACTOR
# --------------------------------------------------------------------------- #
class Extractor:
    """Extractor de corpus limpio para todos los formatos soportados."""

    # Campos textuales (concatenar en orden) vs descriptivos en JSON.
    JSON_TEXT_KEYS = {
        "title", "headline", "name", "subheading", "subtitle", "description",
        "resumen", "summary", "excerpt", "body_text", "body", "content",
        "texto", "article", "lead", "abstract", "introduction", "heading",
        "paragraph", "body",
    }
    JSON_PARAGRAPH_LIST_KEYS = {"body_paragraphs", "paragraphs", "parrafos", "paragraph_list"}
    JSON_META_KEYS = {
        "url", "date", "datetime", "created", "updated", "modified",
        "authors", "author", "tags", "topics", "keywords", "category",
        "categories", "source", "publisher", "license", "language", "lang",
        "year", "numero", "issue", "volume", "doi", "status", "scraped_at",
        "page_url", "pdf_url", "dest", "size_bytes", "nombre_normalizado",
        "fuente_origen", "tipo_original", "organizacion", "ruta_relativa",
    }

    def __init__(self, data_dir: Path = DATA_DIR):
        self.data_dir = data_dir

    # ---------------- PDF ----------------
    def extraer_pdf(self, path: Path) -> Tuple[str, Dict[str, Any]]:
        if fitz is None:
            raise RuntimeError("PyMuPDF (fitz) no está disponible")
        page_line_blocks: List[List[str]] = []
        n_paginas = 0
        with fitz.open(path) as doc:
            n_paginas = doc.page_count
            for page in doc:
                page_line_blocks.append(self._pdf_page_lines(page))
        page_line_blocks = self._strip_boilerplate(page_line_blocks)
        texto = "\n".join(l for lines in page_line_blocks for l in lines)

        motor = "pymupdf"
        if not texto.strip() and pdfplumber is not None:
            try:
                texto = self._pdfplumber_text(path)
                motor = "pdfplumber"
            except Exception as exc:
                logger.warning("pdfplumber falló en %s: %s", path.name, exc)
        if not texto.strip() and MAX_PDF_OCR > 0 and fitz is not None:
            texto = self._pdf_ocr(path)
            motor = "pymupdf+ocr" if texto else motor

        metadata_doc = {"paginas": n_paginas, "motor": motor}
        return texto, metadata_doc

    @staticmethod
    def _pdf_page_lines(page) -> List[str]:
        lineas: List[str] = []
        try:
            bloques = page.get_text("dict")["blocks"]
        except Exception:
            bloques = []
        for bloque in bloques:
            if bloque.get("type", 0) != 0:  # solo texto (no imagen)
                continue
            for linea in bloque.get("lines", []):
                txt = "".join(span["text"] for span in linea.get("spans", []))
                txt = txt.strip()
                if txt:
                    lineas.append(txt)
        return lineas

    @staticmethod
    def _pdfplumber_text(path: Path) -> str:
        partes: List[str] = []
        with pdfplumber.open(path) as pdf:
            for page in pdf.pages:
                txt = page.extract_text()
                if txt:
                    partes.append(txt.strip())
        return "\n".join(partes)

    @staticmethod
    def _pdf_ocr(path: Path) -> str:
        if pytesseract is None or PILImage is None or fitz is None:
            return ""
        partes: List[str] = []
        try:
            with fitz.open(path) as doc:
                for idx in range(min(MAX_PDF_OCR, doc.page_count)):
                    pix = doc[idx].get_pixmap(dpi=150)
                    with PILImage.open(io.BytesIO(pix.tobytes("png"))) as img:
                        try:
                            partes.append(pytesseract.image_to_string(img))
                        except Exception as exc:
                            logger.warning("OCR falló en %s p%d: %s", path.name, idx, exc)
                            break
        except Exception as exc:
            logger.warning("OCR (fitz) falló en %s: %s", path.name, exc)
        return "\n".join(partes)

    @staticmethod
    def _strip_boilerplate(page_line_blocks: List[List[str]]) -> List[List[str]]:
        """Elimina headers/footers repetitivos y numeración de página."""
        if not page_line_blocks:
            return page_line_blocks
        n = len(page_line_blocks)
        re_num = re.compile(
            r"^\s*[\w\s\-./]*\b(p[áa]g(?:ina)?|page|p[áa]g\.?)\b\.?\s*\d+\s*$", re.IGNORECASE
        )
        re_puro = re.compile(r"^\s*\d{1,4}\s*$")

        def es_bp(linea: str) -> bool:
            return bool(re_num.search(linea) or re_puro.match(linea))

        head_c: Counter = Counter()
        foot_c: Counter = Counter()
        for lineas in page_line_blocks:
            if not lineas:
                continue
            head_c[lineas[0]] += 1
            if len(lineas) >= 2:
                head_c[lineas[1]] += 1
            foot_c[lineas[-1]] += 1
            if len(lineas) >= 2:
                foot_c[lineas[-2]] += 1
        umbral = max(2, int(n * 0.5))
        bp = set()
        for linea, c in (*head_c.items(), *foot_c.items()):
            if len(linea.split()) >= 2 and c >= umbral:
                bp.add(linea)
        return [[l for l in lines if l not in bp and not es_bp(l)] for lines in page_line_blocks]

    # ---------------- HTML ----------------
    def extraer_html(self, path: Path) -> Tuple[str, Dict[str, Any]]:
        raw = path.read_bytes()
        motor = "regex"
        if BeautifulSoup is not None:
            soup = BeautifulSoup(raw, "lxml" if lxml_etree else "html.parser")
            for tag in ("script", "style", "noscript", "header", "footer", "nav",
                        "svg", "iframe", "img", "video", "audio", "picture"):
                for el in soup.find_all(tag):
                    el.decompose()
            texto = self._soup_texto(soup)
            motor = "bs4"
        elif lxml_html is not None:
            try:
                doc = lxml_html.fromstring(raw)
                for tag in ("script", "style", "noscript", "header", "footer", "nav"):
                    for el in doc.iter(tag):
                        el.drop_tree()
                texto = self._lxml_texto(doc)
                motor = "lxml"
            except Exception:
                texto = self._html_regex(raw)
                motor = "regex"
        else:
            texto = self._html_regex(raw)
            motor = "regex"
        return texto, {"motor": motor}

    @staticmethod
    def _soup_texto(soup) -> str:
        partes: List[str] = []
        HEADER_TAGS = {"h1", "h2", "h3", "h4", "h5", "h6"}
        BLOCK_TAGS = {"p", "div", "li", "tr", "td", "th", "br", "section", "article",
                      "header", "footer", "blockquote", "pre", "hr", "table", "span"}

        def rec(node):
            if isinstance(node, NavigableString):
                t = str(node)
                if t.strip():
                    partes.append(t)
                return
            if not isinstance(node, Tag):
                return
            tag = node.name
            if tag in ("script", "style", "noscript", "svg", "img", "video",
                       "audio", "picture", "iframe", "link", "meta", "title"):
                return
            for child in node.children:
                rec(child)
            if tag in HEADER_TAGS:
                partes.append("\n\n")
            elif tag in BLOCK_TAGS:
                partes.append("\n")

        root = soup.body if soup.body else soup
        rec(root)
        texto = "".join(partes)
        # Garantizar doble salto tras headers que quedaron como única línea
        texto = re.sub(r"(\n\n|^)([^\n]+?)(\n)$", r"\1\2\n\n", texto)
        return texto

    @staticmethod
    def _lxml_texto(doc) -> str:
        partes: List[str] = []

        def walk(node):
            tag = getattr(node, "tag", None)
            if isinstance(tag, str):
                if tag in ("script", "style", "noscript", "svg", "img", "iframe"):
                    return
            t = (node.text or "") if hasattr(node, "text") else ""
            if t.strip():
                partes.append(t)
            for child in node:
                walk(child)
            if getattr(node, "tail", None):
                partes.append(node.tail)

        walk(doc)
        texto = "".join(partes)
        for h in ("h1", "h2", "h3", "h4", "h5", "h6"):
            texto = re.sub(rf"<{h}>(.*?)</{h}>", r"\n\n\1\n\n", texto, flags=re.IGNORECASE | re.DOTALL) if False else texto
        return texto

    @staticmethod
    def _html_regex(raw: bytes) -> str:
        texto = raw.decode("utf-8", "ignore")
        texto = re.sub(r"(?is)<(script|style|noscript|header|footer|nav|svg|iframe)\b.*?</\1>", "", texto)
        for h in ("h1", "h2", "h3", "h4", "h5", "h6"):
            texto = re.sub(rf"<{h}(?:\s[^>]*)?>(.*?)</{h}>", r"\n\n\1\n\n", texto, flags=re.IGNORECASE | re.DOTALL)
        texto = re.sub(r"<br\s*/?>", "\n", texto, flags=re.IGNORECASE)
        texto = re.sub(r"<[^>]+>", "\n", texto)
        texto = _html.unescape(texto)
        return texto

    # ---------------- JSON ----------------
    def extraer_json(self, path: Path) -> Tuple[str, Dict[str, Any]]:
        with open(path, "r", encoding="utf-8-sig") as fh:
            datos = json.load(fh)
        body: List[str] = []
        meta: Dict[str, Any] = {}
        self._walk_json(datos, body, meta)
        texto = "\n".join(p for p in body if p)
        meta["campos_totales"] = self._contar_campos(datos)
        return texto, meta

    def _walk_json(self, nodo, body: List[str], meta: Dict[str, Any], clave: str = "") -> None:
        if isinstance(nodo, dict):
            for k, v in nodo.items():
                kl = k.lower()
                if isinstance(v, list) and kl in self.JSON_PARAGRAPH_LIST_KEYS:
                    for item in v:
                        if isinstance(item, str) and item.strip():
                            body.append(item.strip())
                        elif isinstance(item, dict):
                            self._walk_json(item, body, meta, clave=k)
                elif isinstance(v, str) and kl in self.JSON_TEXT_KEYS:
                    if v.strip():
                        body.append(v.strip())
                elif isinstance(v, dict):
                    self._walk_json(v, body, meta, clave=k)
                elif kl in self.JSON_META_KEYS:
                    meta[k] = v
                elif isinstance(v, list):
                    meta[k] = v if self._es_lista_escalar(v) else None
                else:
                    meta[k] = v
        elif isinstance(nodo, list):
            for item in nodo:
                self._walk_json(item, body, meta, clave=clave)

    @staticmethod
    def _es_lista_escalar(v: list) -> bool:
        return all(not isinstance(x, (dict, list)) for x in v)

    @staticmethod
    def _contar_campos(nodo) -> int:
        n = 0
        if isinstance(nodo, dict):
            n += len(nodo)
            for v in nodo.values():
                n += Extractor._contar_campos(v)
        elif isinstance(nodo, list):
            for v in nodo:
                n += Extractor._contar_campos(v)
        return n

    # ---------------- CSV ----------------
    def extraer_csv(self, path: Path) -> Tuple[str, Dict[str, Any]]:
        try:
            df = pd.read_csv(path, dtype=object, keep_default_na=False, low_memory=False)
        except pd.errors.ParserError:
            df = pd.read_csv(path, sep=None, dtype=object, keep_default_na=False, engine="python")
        except Exception:
            df = self._csv_fallback(path)
        columnas = list(df.columns) if not df.empty else []
        texto_filas: List[str] = []
        filas = 0
        for _, fila in df.iterrows():
            partes = [f"{col}: {str(val).strip()}" for col, val in fila.items() if str(val).strip()]
            if partes:
                texto_filas.append(" | ".join(partes))
                filas += 1
        meta = {"filas": filas, "columnas": columnas, "num_columnas": len(columnas)}
        return "\n".join(texto_filas), meta

    @staticmethod
    def _csv_fallback(path: Path) -> pd.DataFrame:
        filas = []
        with open(path, "r", encoding="utf-8-sig", newline="") as fh:
            lector = csv.reader(fh)
            encabezado = next(lector, None)
            for r in lector:
                filas.append(r)
        return pd.DataFrame(filas, columns=encabezado) if encabezado else pd.DataFrame(filas)

    # ---------------- XLSX ----------------
    def extraer_xlsx(self, path: Path) -> Tuple[str, Dict[str, Any]]:
        texto_filas: List[str] = []
        hojas: List[str] = []
        columnas: List[str] = []
        filas = 0
        try:
            import openpyxl  # noqa: F401
            xl = pd.ExcelFile(path, engine="openpyxl")
            hojas = xl.sheet_names
            for hoja in hojas:
                df = xl.parse(hoja, dtype=object)
                col = list(df.columns) if not df.columns.empty else []
                if hoja == hojas[0]:
                    columnas = col
                for _, fila in df.iterrows():
                    partes = [f"{c}: {str(v).strip()}" for c, v in fila.items() if str(v).strip()]
                    if partes:
                        texto_filas.append(" | ".join(partes))
                        filas += 1
        except Exception as exc:
            logger.warning("Error leyendo xlsx %s: %s", path.name, exc)
            raise
        return "\n".join(texto_filas), {"hojas": hojas, "filas": filas, "columnas": columnas}

    # ---------------- IMÁGENES ----------------
    def extraer_img(self, path: Path) -> Tuple[str, Dict[str, Any]]:
        ancho, alto = 0, 0
        ocr_realizado = False
        texto = ""
        if PILImage is not None:
            try:
                with PILImage.open(path) as img:
                    ancho, alto = img.size
                    if pytesseract is not None:
                        try:
                            texto = pytesseract.image_to_string(img)
                            ocr_realizado = True
                        except Exception as exc:
                            logger.warning("pytesseract falló en %s (%s)", path.name, exc)
            except Exception as exc:
                logger.warning("Pillow no pudo abrir %s: %s", path.name, exc)
        meta = {
            "ancho": ancho, "alto": alto,
            "ocr_realizado": ocr_realizado,
            "nombre_archivo": path.name,
        }
        return texto, meta

    # ---------------- PBF ----------------
    def extraer_pbf(self, path: Path) -> Tuple[str, Dict[str, Any]]:
        for mod in ("pyrosm", "osmium"):
            try:
                __import__(mod)
                if mod == "pyrosm":
                    return self._pbf_pyrosm(path)
            except Exception:
                continue
        return self._pbf_mvt(path)

    @staticmethod
    def _pbf_pyrosm(path: Path) -> Tuple[str, Dict[str, Any]]:
        import pyrosm
        fp = pyrosm.Feed(str(path))
        partes: List[str] = []
        total = 0
        for key in ("amenity", "highway", "place", "building", "landuse", "natural", "waterway", "name"):
            try:
                datos = fp.get_data_by_custom_query(f'["{key}"~".*"]', "osm")
                if datos is not None and not datos.empty:
                    for _, r in datos.iterrows():
                        fila = " | ".join(f"{c}: {v}" for c, v in r.items() if str(v))
                        if fila:
                            partes.append(fila)
                            total += 1
            except Exception:
                continue
        return "\n".join(partes), {"motor": "pyrosm", "elementos": total}

    @staticmethod
    def _pbf_mvt(path: Path) -> Tuple[str, Dict[str, Any]]:
        """Decoder Mapbox Vector Tile (OSM PBF) sin dependencias externas."""
        raw = path.read_bytes()
        z, x, y = Extractor._tile_coords(path)
        pares: set = set()
        num_features = 0
        fallback = False
        try:
            tile = _decode_protobuf(raw)          # {field_no: [(wt, payload)]}
            for _fno, payload in tile.get(3, []):  # Tile.layers = field 3 (repetida)
                capa_raw = _decode_protobuf(payload)
                layer = {
                    "name": _first_str(capa_raw.get(2, [])),
                    "extent": _first_int(capa_raw.get(3, [])),
                    "keys": [_first_str(v) for _, v in capa_raw.get(4, [])],
                    "values": [_decode_value_msg(_decode_protobuf(v)) for _, v in capa_raw.get(5, [])],
                    "features": [_mvt_feature(_decode_protobuf(v)) for _, v in capa_raw.get(6, [])],
                }
                for feat in layer["features"]:
                    num_features += 1
                    kidx = feat["keys"]
                    vidx = feat["values"]
                    tags = {}
                    for ki, vi in zip(kidx, vidx):
                        if ki < len(layer["keys"]) and vi < len(layer["values"]):
                            k = layer["keys"][ki]
                            v = layer["values"][vi]
                            if k and v is not None and str(v).strip():
                                tags[k] = str(v)
                    for k, v in tags.items():
                        pares.add(f"{k}: {v}")
        except Exception as exc:
            logger.warning("Decoder MVT falló en %s: %s", path.name, exc)
            fallback = True
            pares.update({s for s in re.findall(rb"[ -~]{4,}", raw) if s.strip()})
            pares = {p.decode("latin-1", "ignore") if isinstance(p, bytes) else p for p in pares}
        texto = "\n".join(sorted(pares))
        meta = {
            "motor": "strings" if fallback else "mvt_decoder",
            "tile_z": z, "tile_x": x, "tile_y": y,
            "elementos": num_features,
        }
        return texto, meta

    @staticmethod
    def _tile_coords(path: Path) -> Tuple[Optional[int], Optional[int], Optional[int]]:
        try:
            partes = path.parts
            idx = partes.index("tiles")
            z = int(partes[idx + 1])
            x = int(partes[idx + 2])
        except (ValueError, IndexError):
            z, x = None, None
        m = re.search(r"(\d+)\.pbf$", path.name)
        y = int(m.group(1)) if m else None
        return z, x, y

    # ---------------- DESPACHO ----------------
    def extraer(self, path: Path, formato: str) -> Tuple[str, Dict[str, Any]]:
        if formato == "pdf":
            return self.extraer_pdf(path)
        if formato == "html":
            return self.extraer_html(path)
        if formato == "json":
            return self.extraer_json(path)
        if formato == "csv":
            return self.extraer_csv(path)
        if formato == "xlsx":
            return self.extraer_xlsx(path)
        if formato == "img":
            return self.extraer_img(path)
        if formato == "pbf":
            return self.extraer_pbf(path)
        raise ValueError(f"Formato no soportado: {formato}")


# --------------------------------------------------------------------------- #
# 4. DECODIFICADOR PROTOBUF (para PBF/MVT) — wire-format sin .proto
# --------------------------------------------------------------------------- #
def _decode_protobuf(buf: bytes) -> Dict[int, List[Tuple[int, Any]]]:
    i = 0
    n = len(buf)
    out: Dict[int, List[Tuple[int, Any]]] = {}
    while i < n:
        key, i = _read_varint(buf, i)
        fno = key >> 3
        wt = key & 0x7
        if wt == 0:                                  # varint
            val, i = _read_varint(buf, i)
            out.setdefault(fno, []).append((0, val))
        elif wt == 2:                                # length-delimited
            ln, i = _read_varint(buf, i)
            if i + ln > n:
                break
            out.setdefault(fno, []).append((2, buf[i:i + ln]))
            i += ln
        elif wt == 5:                                # 32-bit
            out.setdefault(fno, []).append((5, int.from_bytes(buf[i:i + 4], "little")))
            i += 4
        elif wt == 1:                                # 64-bit
            out.setdefault(fno, []).append((1, int.from_bytes(buf[i:i + 8], "little")))
            i += 8
        else:
            break
    return out


def _read_varint(buf: bytes, i: int) -> Tuple[int, int]:
    result = 0
    shift = 0
    while i < len(buf):
        b = buf[i]; i += 1
        result |= (b & 0x7F) << shift
        if not (b & 0x80):
            return result, i
        shift += 7
        if shift > 63:
            raise ValueError("varint demasiado largo")
    raise ValueError("varint truncado")


def _first_int(items) -> int:
    return items[0][1] if items else 0


def _first_str(items) -> str:
    for _, payload in items:
        if isinstance(payload, (bytes, bytearray)):
            try:
                return payload.decode("utf-8")
            except Exception:
                return payload.decode("latin-1", "ignore")
    return ""


def _decode_value_msg(vmsg: Dict[int, List]) -> Any:
    """Value de MVT: 1=string, 4=int64, 5=uint64, 6=sint64(zigzag), 7=bool."""
    for wt, payload in vmsg.get(1, []):     # string_value
        if isinstance(payload, (bytes, bytearray)):
            try:
                return payload.decode("utf-8")
            except Exception:
                return payload.decode("latin-1", "ignore")
    for wt, payload in vmsg.get(4, []):     # int_value
        return _zigzag(payload)
    for wt, payload in vmsg.get(5, []):    # uint_value
        return payload
    for wt, payload in vmsg.get(6, []):    # sint_value (zigzag)
        return _zigzag(payload)
    for wt, payload in vmsg.get(7, []):    # bool_value
        return bool(payload)
    return None


def _zigzag(n: int) -> int:
    return (n >> 1) ^ -(n & 1)


def _mvt_feature(fmsg: Dict[int, List]) -> Dict[str, Any]:
    return {
        "id": _first_int(fmsg.get(1, [])),
        "keys": [_first_int(v) for _, v in fmsg.get(2, [])],
        "values": [_first_int(v) for _, v in fmsg.get(3, [])],
        "type": _first_int(fmsg.get(4, [])),
    }


# --------------------------------------------------------------------------- #
# 5. ORQUESTACIÓN
# --------------------------------------------------------------------------- #
def _ignorar(path: Path) -> bool:
    nombre = path.name.lower()
    if nombre.startswith(".") or nombre in NOMBRES_IGNORAR:
        return True
    partes = path.parts
    for p in partes:
        if p.startswith(".") and p != ".":
            return True
    return False


def descubrir_archivos(data_dir: Path) -> List[Path]:
    archivos: List[Path] = []
    for path in sorted(data_dir.rglob("*")):
        if not path.is_file():
            continue
        if _ignorar(path):
            continue
        ext = path.suffix.lower()
        if ext not in EXT_FORMATO:
            continue
        if fenomeno_de_path(path) is None:
            continue
        archivos.append(path)
    return archivos


def construir_corpus(
    data_dir: Path = DATA_DIR,
    out_path: Path = CORPUS_DEFAULT,
    solo_fenomeno: Optional[int] = None,
) -> None:
    extractor = Extractor(data_dir)
    EXCLUSIONES_LOG.parent.mkdir(parents=True, exist_ok=True)
    archivos = descubrir_archivos(data_dir)
    if solo_fenomeno is not None:
        archivos = [a for a in archivos if fenomeno_de_path(a) == solo_fenomeno]

    # Contadores por (fenomeno, formato) para doc_id legible y determinista.
    contadores: Dict[Tuple[int, str], int] = {}

    with open(out_path, "w", encoding="utf-8") as out_fh, \
         open(EXCLUSIONES_LOG, "w", encoding="utf-8") as exc_fh:
        procesados = 0
        for path in archivos:
            fen = fenomeno_de_path(path) or 0
            ext = path.suffix.lower()
            formato = formato_de_ext(ext)
            if formato is None:
                continue
            clave = (fen, formato)
            contadores[clave] = contadores.get(clave, 0) + 1
            doc_id = _doc_id(fen, formato, contadores[clave])
            fuente = path.name
            try:
                texto, meta = extractor.extraer(path, formato)
            except Exception as exc:
                razon = f"{type(exc).__name__}: {exc}"
                logger.error("SKIP %s (%s)", path, razon)
                exc_fh.write(json.dumps(
                    {"doc_id": doc_id, "fuente": fuente, "formato": formato,
                     "fenomeno": fen, "ruta": str(path), "razon": razon},
                    ensure_ascii=False) + "\n")
                continue

            texto = clean_text(texto)
            # Saltar si quedó vacío (excepto imágenes: pueden ser vacías por OCR ausente)
            if not texto.strip() and formato != "img":
                exc_fh.write(json.dumps(
                    {"doc_id": doc_id, "fuente": fuente, "formato": formato,
                     "fenomeno": fen, "ruta": str(path), "razon": "sin_contenido_indexable"},
                    ensure_ascii=False) + "\n")
                continue

            idioma = detectar_idioma(texto)
            meta["fuente_original"] = fuente
            meta["ruta_relativa"] = str(path.relative_to(BASE_DIR)) if path.is_relative_to(BASE_DIR) else str(path)
            meta["fenomeno"] = fen

            record = {
                "doc_id": doc_id,
                "fuente": fuente,
                "formato": formato,
                "fenomeno": fen,
                "idioma": idioma,
                "texto": texto,
                "metadata_doc": meta,
            }
            try:
                out_fh.write(json.dumps(record, ensure_ascii=False) + "\n")
            except (TypeError, ValueError) as exc:
                #Forzar serialización segura de metadata
                record["metadata_doc"] = json.loads(json.dumps(record["metadata_doc"], default=str))
                out_fh.write(json.dumps(record, ensure_ascii=False) + "\n")
            procesados += 1
            if procesados % 100 == 0:
                logger.info("Procesados %d documentos...", procesados)

    logger.info("Corpus generado: %s (%d documentos)", out_path, procesados)


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description="Extractor de corpus — CODEFEST AD ASTRA 2026")
    parser.add_argument("--data", type=str, default=str(DATA_DIR), help="Carpeta raíz de datos")
    parser.add_argument("--out", type=str, default=str(CORPUS_DEFAULT), help="Archivo corpus_limpio.jsonl de salida")
    parser.add_argument("--fenomeno", type=int, default=None, help="Procesar solo un fenómeno (1,2,3)")
    args = parser.parse_args()
    construir_corpus(Path(args.data), Path(args.out), args.fenomeno)


if __name__ == "__main__":
    main()
