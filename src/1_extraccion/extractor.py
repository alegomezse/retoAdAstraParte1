"""
Fase 1A: Lectura, Extracción y Limpieza de Corpus — CODEFEST AD ASTRA 2026

Módulo extractor modular para formatos mixtos (.pdf, .html, .json, .csv, .xlsx, .png, .jpg, .pbf).
Genera `corpus_limpio.jsonl` preservando el orden de lectura y metadata estructurada.
"""

from __future__ import annotations

import csv
import io
import json
import logging
import re
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

# Dependencias opcionales importadas defensivamente
try:
    import ftfy
except Exception:
    ftfy = None

try:
    import fitz  # PyMuPDF
except Exception:
    fitz = None

try:
    import pdfplumber
except Exception:
    pdfplumber = None

try:
    from bs4 import BeautifulSoup, NavigableString, Tag
except Exception:
    BeautifulSoup = None
    NavigableString = None
    Tag = None

try:
    import pytesseract
    from PIL import Image as PILImage
except Exception:
    pytesseract = None
    PILImage = None

try:
    from langdetect import DetectorFactory, detect
    from langdetect.lang_detect_exception import LangDetectException
    DetectorFactory.seed = 888
except Exception:
    detect = None
    LangDetectException = Exception

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("extractor")

BASE_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = BASE_DIR / "data"
CORPUS_DEFAULT = BASE_DIR / "corpus_limpio.jsonl"
EXCLUSIONES_LOG = BASE_DIR / "salida" / "exclusiones_extractor.log"

FENOMENO_POR_CARPETA: Dict[str, int] = {
    "F1_IA_y_Capacidades_Estrategicas": 1,
    "F2_Seguridad_Entorno_Espacial": 2,
    "F3_Dinamicas_Territoriales": 3,
    "fenomeno_1": 1,
    "fenomeno_2": 2,
    "fenomeno_3": 3,
    "fen1": 1,
    "fen2": 2,
    "fen3": 3,
}

EXT_FORMATO: Dict[str, str] = {
    ".pdf": "pdf",
    ".html": "html",
    ".htm": "html",
    ".json": "json",
    ".csv": "csv",
    ".xls": "xlsx",
    ".xlsx": "xlsx",
    ".png": "img",
    ".jpg": "img",
    ".jpeg": "img",
    ".avif": "img",
    ".webp": "img",
    ".tif": "img",
    ".tiff": "img",
    ".pbf": "pbf",
}

NOMBRES_IGNORAR = {
    ".ds_store",
    "indice_datos_codefest.xlsx",
    "extracto_preguntas_50_v2.pdf",
}

# --- Limpieza y Utilidades ---

def _ftfy_fix(text: str) -> str:
    if ftfy is not None:
        try:
            return ftfy.fix_text(text, normalization="NFKC")
        except Exception:
            pass
    return text


def clean_text(text: str) -> str:
    """
    Normaliza texto UTF-8, elimina caracteres de control, 
    elimina espacios redundantes y colapsa saltos de línea repetidos.
    """
    if not text:
        return ""
    text = _ftfy_fix(text)
    text = unicodedata.normalize("NFC", text)
    # Eliminar caracteres de control sin eliminar \n y \t
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", text)
    text = text.replace("\u00a0", " ").replace("\u200b", "")
    text = re.sub(r"[ \t\r]+", " ", text)
    text = re.sub(r"\n[ \t]*\n+", "\n\n", text)
    text = re.sub(r" *[ \t]*\n", "\n", text)
    return text.strip()


def detectar_idioma(texto: str) -> str:
    """Detecta el idioma predominante del texto usando langdetect."""
    if not texto or detect is None or len(texto.split()) < 3:
        return "unknown"
    try:
        return detect(texto)
    except (LangDetectException, Exception):
        return "unknown"


def fenomeno_de_path(path: Path, data_dir: Path = DATA_DIR) -> Optional[int]:
    """Determina el número de fenómeno (1, 2 o 3) a partir del path del archivo."""
    try:
        rel = path.relative_to(data_dir)
        first_part = rel.parts[0]
    except (ValueError, IndexError):
        first_part = path.parent.name

    if first_part in FENOMENO_POR_CARPETA:
        return FENOMENO_POR_CARPETA[first_part]

    m = re.search(r"f(?:enomeno)?[_\-\s]*([123])", first_part, re.IGNORECASE)
    if m:
        return int(m.group(1))

    return None


def formato_de_ext(ext: str) -> Optional[str]:
    return EXT_FORMATO.get(ext.lower())


def generar_doc_id(fenomeno: int, formato: str, contador: int) -> str:
    return f"fen{fenomeno}_{formato}_{contador:03d}"


# --- Clase Principal Extractor ---

class Extractor:
    """Clase modular de extracción por formato para el RAG Pipeline."""

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

    # 1. PDF
    def extraer_pdf(self, path: Path) -> Tuple[str, Dict[str, Any]]:
        """Extrae texto de PDF manteniendo el orden de lectura y filtrando encabezados/pies de página."""
        page_line_blocks: List[List[str]] = []
        n_paginas = 0
        motor = "pymupdf"

        if fitz is not None:
            with fitz.open(path) as doc:
                n_paginas = doc.page_count
                for page in doc:
                    lineas: List[str] = []
                    try:
                        bloques = page.get_text("dict")["blocks"]
                    except Exception:
                        bloques = []
                    for b in bloques:
                        if b.get("type", 0) == 0:  # texto
                            for line in b.get("lines", []):
                                txt = "".join(span["text"] for span in line.get("spans", [])).strip()
                                if txt:
                                    lineas.append(txt)
                    page_line_blocks.append(lineas)

            page_line_blocks = self._strip_boilerplate(page_line_blocks)
            texto = "\n".join(l for lines in page_line_blocks for l in lines)
        else:
            texto = ""

        # Fallback a pdfplumber si no hay texto extraído
        if not texto.strip() and pdfplumber is not None:
            try:
                partes = []
                with pdfplumber.open(path) as pdf:
                    n_paginas = len(pdf.pages)
                    for page in pdf.pages:
                        t = page.extract_text()
                        if t:
                            partes.append(t.strip())
                texto = "\n\n".join(partes)
                motor = "pdfplumber"
            except Exception as exc:
                logger.warning("pdfplumber falló en %s: %s", path.name, exc)

        metadata_doc = {"paginas": n_paginas, "motor": motor}
        return texto, metadata_doc

    @staticmethod
    def _strip_boilerplate(page_line_blocks: List[List[str]]) -> List[List[str]]:
        """Remueve headers, footers y números de página repetitivos."""
        if not page_line_blocks:
            return page_line_blocks
        n = len(page_line_blocks)
        re_num = re.compile(r"^\s*[\w\s\-./]*\b(p[áa]g(?:ina)?|page|p[áa]g\.?)\b\.?\s*\d+\s*$", re.IGNORECASE)
        re_puro = re.compile(r"^\s*\d{1,4}\s*$")

        def es_bp(linea: str) -> bool:
            return bool(re_num.search(linea) or re_puro.match(linea))

        head_c, foot_c = Counter(), Counter()
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
        for linea, count in (*head_c.items(), *foot_c.items()):
            if len(linea.split()) >= 2 and count >= umbral:
                bp.add(linea)

        return [[l for l in lines if l not in bp and not es_bp(l)] for lines in page_line_blocks]

    # 2. HTML
    def extraer_html(self, path: Path) -> Tuple[str, Dict[str, Any]]:
        """Extrae texto visible de HTML omitiendo tags no visibles y separando headers con doble salto."""
        raw = path.read_bytes()
        motor = "bs4"

        if BeautifulSoup is not None:
            soup = BeautifulSoup(raw, "html.parser")
            for tag in ("script", "style", "noscript", "header", "footer", "nav", "svg", "iframe", "img", "video", "audio"):
                for el in soup.find_all(tag):
                    el.decompose()
            
            partes: List[str] = []
            HEADER_TAGS = {"h1", "h2", "h3", "h4", "h5", "h6"}
            BLOCK_TAGS = {"p", "div", "li", "tr", "td", "th", "br", "section", "article", "blockquote", "pre"}

            def rec(node):
                if isinstance(node, NavigableString):
                    t = str(node)
                    if t.strip():
                        partes.append(t)
                    return
                if not isinstance(node, Tag):
                    return
                tag = node.name
                for child in node.children:
                    rec(child)
                if tag in HEADER_TAGS:
                    partes.append("\n\n")
                elif tag in BLOCK_TAGS:
                    partes.append("\n")

            root = soup.body if soup.body else soup
            rec(root)
            texto = "".join(partes)
        else:
            motor = "regex"
            texto = raw.decode("utf-8", "ignore")
            texto = re.sub(r"(?is)<(script|style|noscript|header|footer|nav|svg|iframe)\b.*?</\1>", "", texto)
            for h in ("h1", "h2", "h3", "h4", "h5", "h6"):
                texto = re.sub(rf"<{h}(?:\s[^>]*)?>(.*?)</{h}>", r"\n\n\1\n\n", texto, flags=re.IGNORECASE | re.DOTALL)
            texto = re.sub(r"<br\s*/?>", "\n", texto, flags=re.IGNORECASE)
            texto = re.sub(r"<[^>]+>", "\n", texto)

        return texto, {"motor": motor}

    # 3. JSON
    def extraer_json(self, path: Path) -> Tuple[str, Dict[str, Any]]:
        """Extrae texto concatenado en orden y separa metadata descriptiva."""
        with open(path, "r", encoding="utf-8-sig") as fh:
            datos = json.load(fh)

        body: List[str] = []
        meta: Dict[str, Any] = {}
        self._walk_json(datos, body, meta)
        texto = "\n\n".join(p for p in body if p)
        return texto, meta

    def _walk_json(self, nodo: Any, body: List[str], meta: Dict[str, Any], clave: str = "") -> None:
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
                else:
                    meta[k] = v
        elif isinstance(nodo, list):
            for item in nodo:
                self._walk_json(item, body, meta, clave=clave)

    # 4. CSV
    def extraer_csv(self, path: Path) -> Tuple[str, Dict[str, Any]]:
        """Extrae filas en formato 'columna: valor | columna: valor', omitiendo vacíos."""
        try:
            df = pd.read_csv(path, dtype=object, keep_default_na=False, low_memory=False)
        except Exception:
            df = pd.read_csv(path, sep=None, dtype=object, keep_default_na=False, engine="python")

        texto_filas: List[str] = []
        for _, fila in df.iterrows():
            partes = [f"{col}: {str(val).strip()}" for col, val in fila.items() if str(val).strip()]
            if partes:
                texto_filas.append(" | ".join(partes))

        meta = {"filas": len(texto_filas), "columnas": list(df.columns) if not df.empty else []}
        return "\n".join(texto_filas), meta

    # 5. XLSX
    def extraer_xlsx(self, path: Path) -> Tuple[str, Dict[str, Any]]:
        """Extrae hojas y filas de Excel en formato 'columna: valor | columna: valor'."""
        texto_filas: List[str] = []
        hojas: List[str] = []
        try:
            xl = pd.ExcelFile(path)
            hojas = xl.sheet_names
            for hoja in hojas:
                df = xl.parse(hoja, dtype=object)
                for _, fila in df.iterrows():
                    partes = [f"{c}: {str(v).strip()}" for c, v in fila.items() if str(v).strip()]
                    if partes:
                        texto_filas.append(" | ".join(partes))
        except Exception as exc:
            logger.warning("Error al leer Excel %s: %s", path.name, exc)
            raise

        return "\n".join(texto_filas), {"hojas": hojas, "filas": len(texto_filas)}

    # 6. IMÁGENES
    def extraer_img(self, path: Path) -> Tuple[str, Dict[str, Any]]:
        """Aplica OCR con pytesseract si está disponible. Si no hay texto, lo marca como vacío."""
        texto = ""
        ocr_exitoso = False

        if PILImage is not None and pytesseract is not None:
            try:
                with PILImage.open(path) as img:
                    texto = pytesseract.image_to_string(img)
                    ocr_exitoso = bool(texto.strip())
            except Exception as exc:
                logger.warning("OCR falló para imagen %s: %s", path.name, exc)

        metadata = {
            "nombre_archivo": path.name,
            "ocr_realizado": ocr_exitoso,
        }
        return texto, metadata

    # 7. PBF (OSM Protobuf)
    def extraer_pbf(self, path: Path) -> Tuple[str, Dict[str, Any]]:
        """Extrae pares atributo:valor de PBF usando pyosmium, geopandas o decodificador MVT nativo."""
        # 1. pyosmium
        try:
            import osmium
            class OSMHandler(osmium.SimpleHandler):
                def __init__(self):
                    super().__init__()
                    self.pairs = set()
                def node(self, n):
                    for tag in n.tags:
                        self.pairs.add(f"{tag.k}: {tag.v}")
                def way(self, w):
                    for tag in w.tags:
                        self.pairs.add(f"{tag.k}: {tag.v}")
                def relation(self, r):
                    for tag in r.tags:
                        self.pairs.add(f"{tag.k}: {tag.v}")

            h = OSMHandler()
            h.apply_file(str(path))
            if h.pairs:
                return "\n".join(sorted(h.pairs)), {"motor": "pyosmium", "elementos": len(h.pairs)}
        except Exception:
            pass

        # 2. geopandas
        try:
            import geopandas as gpd
            df = gpd.read_file(path)
            filas = []
            for _, r in df.iterrows():
                partes = [f"{col}: {str(val).strip()}" for col, val in r.items() if str(val).strip() and col != "geometry"]
                if partes:
                    filas.append(" | ".join(partes))
            unicos = list(dict.fromkeys(filas))
            return "\n".join(unicos), {"motor": "geopandas", "elementos": len(unicos)}
        except Exception:
            pass

        # 3. Fallback nativo
        return self._pbf_mvt_fallback(path)

    @staticmethod
    def _pbf_mvt_fallback(path: Path) -> Tuple[str, Dict[str, Any]]:
        raw = path.read_bytes()
        strings = re.findall(rb"[ -~]{4,}", raw)
        pares = sorted(list({s.decode("latin-1", "ignore").strip() for s in strings if s.strip()}))
        return "\n".join(pares), {"motor": "strings_fallback", "elementos": len(pares)}

    # DESPACHO CENTRAL
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


# --- Orquestación Principal ---

def descubrir_archivos(data_dir: Path) -> List[Path]:
    archivos: List[Path] = []
    for path in sorted(data_dir.rglob("*")):
        if not path.is_file():
            continue
        if path.name.lower() in NOMBRES_IGNORAR or path.name.startswith("."):
            continue
        ext = path.suffix.lower()
        if ext not in EXT_FORMATO:
            continue
        if fenomeno_de_path(path, data_dir) is None:
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
        archivos = [a for a in archivos if fenomeno_de_path(a, data_dir) == solo_fenomeno]

    contadores: Dict[Tuple[int, str], int] = {}
    procesados = 0

    with open(out_path, "w", encoding="utf-8") as out_fh, \
         open(EXCLUSIONES_LOG, "w", encoding="utf-8") as exc_fh:
        for path in archivos:
            fen = fenomeno_de_path(path, data_dir) or 0
            ext = path.suffix.lower()
            formato = formato_de_ext(ext)
            if formato is None:
                continue

            clave = (fen, formato)
            contadores[clave] = contadores.get(clave, 0) + 1
            doc_id = generar_doc_id(fen, formato, contadores[clave])
            fuente = path.name

            try:
                texto, meta = extractor.extraer(path, formato)
            except Exception as exc:
                razon = f"{type(exc).__name__}: {exc}"
                logger.error("SKIP por error en %s (%s)", path, razon)
                exc_fh.write(json.dumps(
                    {"doc_id": doc_id, "fuente": fuente, "formato": formato,
                     "fenomeno": fen, "ruta": str(path), "razon": razon},
                    ensure_ascii=False) + "\n")
                continue

            texto = clean_text(texto)

            # Para cualquier formato que no sea imagen, omitir si no hay texto extraído
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
            except (TypeError, ValueError):
                record["metadata_doc"] = json.loads(json.dumps(record["metadata_doc"], default=str))
                out_fh.write(json.dumps(record, ensure_ascii=False) + "\n")

            procesados += 1
            if procesados % 100 == 0:
                logger.info("Procesados %d documentos...", procesados)

    logger.info("Corpus generado exitosamente: %s (%d documentos procesados)", out_path, procesados)


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description="Extractor de corpus RAG — CODEFEST AD ASTRA 2026")
    parser.add_argument("--data", type=str, default=str(DATA_DIR), help="Carpeta raíz con los datos")
    parser.add_argument("--out", type=str, default=str(CORPUS_DEFAULT), help="Ruta para corpus_limpio.jsonl")
    parser.add_argument("--fenomeno", type=int, default=None, help="Filtrar por fenómeno (1, 2 o 3)")
    args = parser.parse_args()

    construir_corpus(Path(args.data), Path(args.out), args.fenomeno)


if __name__ == "__main__":
    main()
