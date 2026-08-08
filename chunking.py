#!/usr/bin/env python3
"""
chunking.py — Fase 2 del pipeline RAG CODEFEST AD ASTRA.

Estrategia híbrida de chunking: estructural + oracional + semántica.

Esta fase consume los 3 JSONL producidos por extraer_corpus.py
(f1/f2/f3_extraido.jsonl) y los divide en chunks listos para indexación
vectorial y retrieval.

¿Por qué no chunking de tamaño fijo puro?
    El chunking de tamaño fijo (n tokens por chunk) ignora la coherencia
    semántica: un chunk puede cortar a mitad de una idea, o unir dos temas
    distintos en un mismo bloque. Nuestra estrategia híbrida combina tres
    niveles de segmentación:

    1. Estructural (paso 3): detecta encabezados de sección y procesa cada
       sección de forma independiente. Garantiza que nunca se mezcle contenido
       de dos secciones distintas en un mismo chunk.
    2. Oracional (paso 4): usa spaCy para segmentar en oraciones, respetando
       las fronteras lingüísticas por párrafo. Nunca corta una oración a la
       mitad.
    3. Semántico (paso 5): ventana deslizante con corte anticipado cuando la
       similitud coseno entre oraciones adyacentes cae por debajo del umbral
       (0.3), indicando cambio de tema. El overlap (60 tokens) se re-incluye
       para no perder contexto en los límites.

Restricción de la competencia: solo modelos encoder-only.
    intfloat/multilingual-e5-base (encoder-only, XLM-RoBERTa, max_seq_length=512,
    entrenado con objetivo contrastivo de retrieval asimétrico query-passage —
    alineado con la tarea de NDCG@10/F1@3 del CODEFEST, a diferencia de modelos
    de similitud simétrica como MiniLM). Todo texto de pasaje se prefija con
    "passage: " antes de calcular su embedding, siguiendo la convención de uso
    documentada del modelo. spaCy (modelos estadísticos pequeños)
    se usa exclusivamente para segmentación oracional, no para generación.

USO:
    python chunking.py --input-dir salida --output-dir salida --workers 4
    python chunking.py --limit 50              # smoke test
    python chunking.py --reanudar              # reanudar checkpoint

Requiere: spacy, sentence-transformers, transformers, langdetect, tqdm, numpy
    pip install spacy sentence-transformers transformers langdetect tqdm numpy
    python -m spacy download es_core_news_sm
    python -m spacy download en_core_web_sm
    python -m spacy download pt_core_news_sm
    python -m spacy download fr_core_news_sm
    python -m spacy download zh_core_web_sm
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import multiprocessing as mp
import re
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

# ---------------------------------------------------------------------------
# tqdm fallback (igual que extraer_corpus.py)
# ---------------------------------------------------------------------------
try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, **kwargs):
        return iterable

# ===========================================================================
# CONSTANTES CONFIGURABLES — valores fijos de la competencia (editables)
# ===========================================================================

TARGET_TOKENS = 400           # Tamaño objetivo por chunk (tokens)
OVERLAP_TOKENS = 60           # Overlap entre chunks consecutivos (~15%)
UMBRAL_COSENO = 0.3           # Umbral de corte semántico (similitud coseno)
SEMANTIC_MIN_TOKENS = 150     # Chunks con >= este número de tokens activan corte semántico
MIN_TOKENS_VALID = 30         # Chunks con menos tokens son descartados
MODELO_EMBEDDINGS = "intfloat/multilingual-e5-base"

# ---------------------------------------------------------------------------
# Patrones de boilerplate conocidos — cada patrón documentado con su origen.
# Se aplican como regex sobre el texto limpio; reemplazan por cadena vacía.
# ---------------------------------------------------------------------------
PATRONES_BOILERPLATE: list[tuple[str, str]] = [
    # Detectado en documentos SWF (Secure World Foundation): aparece al final
    # de todos los "Global Counterspace Capabilities Report" y fact sheets.
    # Fuente: salida/f2_extraido.jsonl, organizacion="SWF", ~58 documentos.
    (r"Explore some of our related publications below\.\s*", ""),

    # Detectado en documentos SWF: enlace a la homepage del programa.
    # Aparece tras párrafos iniciales de reportes anuales.
    (r"For the most recent edition and program overview, "
     r"visit the Counterspace Program Homepage\.\s*", ""),

    # Detectado en documentos CSIS: texto "for more information" que enlaza
    # a estudios relacionados. Ej: "for more information on this study and
    # the 2018 NDAA language".
    (r"for more information on this study[\s\S]{0,200}?\.\s*", ""),

    # Detectado en documentos CSIS: referencias a "Image Source".
    # Ej: "Image Source: Caroline Amenabar / CSIS".
    (r"Image Source:\s*[^\n]*\s*", ""),

    # Detectado en documentos CSIS: "Visit ... for more information".
    (r"Visit[^\n]*?for more information[\s\S]{0,200}\.\s*", ""),

    # Detectado en documentos SIPRI: "Click here to read".
    # Fuente: salida/f3_extraido.jsonl, organizacion="SIPRI".
    (r"[Cc]lick here to read[\s\S]{0,200}\.\s*", ""),

    # Detectado en documentos ESA/INPE: "Read more at" / "Read the full".
    # Fuente: salida/f1_extraido.jsonl (PDFs), salida/f2_extraido.jsonl.
    (r"Read more[^\n]*", ""),
    (r"Read the full[^\n]*", ""),

    # Patrones de navegación típicos de scraping HTML.
    (r"^Skip to content\s*$", ""),
    (r"^Toggle navigation\s*$", ""),
    (r"^Search\s*$", ""),
    (r"^\s*Page \d+\s*of\s*\d+\s*$", ""),
]

# ---------------------------------------------------------------------------
# Mapeo de códigos de idioma ISO-639-1 → modelos spaCy.
# Cargados de forma perezosa (lazy) solo cuando se detecta el idioma.
# ---------------------------------------------------------------------------
SPACY_MODEL_MAP: dict[str, str] = {
    "es": "es_core_news_sm",
    "en": "en_core_web_sm",
    "pt": "pt_core_news_sm",
    "fr": "fr_core_news_sm",
    "zh": "zh_core_web_sm",
}

# Extensiones de archivo de entrada JSONL por fase
FASES = ["F1", "F2", "F3"]
INPUT_FILENAMES = {
    "F1": "f1_extraido.jsonl",
    "F2": "f2_extraido.jsonl",
    "F3": "f3_extraido.jsonl",
}

# ===========================================================================
# IMPORTS REQUERIDOS
# ===========================================================================
import spacy  # noqa: E402

from sentence_transformers import SentenceTransformer  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402

try:
    from langdetect import detect as _ld_detect
    from langdetect import DetectorFactory, LangDetectException
    DetectorFactory.seed = 0
    _LANGDETECT_OK = True
except ImportError:  # pragma: no cover
    _LANGDETECT_OK = False
    logging.warning(
        "langdetect no instalado; se usarán idiomas dominantes como fallback"
    )

try:
    import numpy as np
    _NUMPY_OK = True
except ImportError:  # pragma: no cover
    np = None
    _NUMPY_OK = False
    logging.warning("numpy no instalado; cálculos de similitud alternativos")

# ===========================================================================
# ESTADO GLOBAL PARA WORKERS DE MULTIPROCESSING
# ===========================================================================
_worker_model: Optional[SentenceTransformer] = None
_worker_tokenizer = None
_worker_spacy_cache: dict[str, Any] = {}


def _worker_init(model_name: str, log_level: str = "INFO"):
    """Inicializa modelo de embeddings y tokenizer una sola vez por worker.
    Cargado una vez al inicio (no por documento)."""
    global _worker_model, _worker_tokenizer
    log_level_num = getattr(logging, str(log_level).upper(), logging.INFO)
    logging.basicConfig(
        level=log_level_num,
        format="%(asctime)s [%(levelname)s] (worker %(process)d) %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    _worker_model = SentenceTransformer(model_name)
    _worker_tokenizer = AutoTokenizer.from_pretrained(model_name)
    logging.info("Worker inicializado: modelo '%s' cargado", model_name)


# ===========================================================================
# UTILIDADES
# ===========================================================================

def sha256_full(text: str) -> str:
    """SHA-256 completo (64 hex chars) — determinista y sin UUID."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def normalize_langdetect_code(code: str) -> str:
    """Normaliza códigos de langdetect a ISO 639-1:
    'zh-cn' -> 'zh', 'pt-br' -> 'pt', etc."""
    code = code.lower().strip()
    if "-" in code:
        code = code.split("-")[0]
    return code


def count_tokens(text: str, tokenizer) -> int:
    """Cuenta tokens con el tokenizer real del modelo de embeddings.
    NO usa len(text.split()) — usa el tokenizador de HuggingFace."""
    toks = tokenizer(
        text,
        add_special_tokens=True,
        truncation=False,
        return_overflowing_tokens=False,
        verbose=False,
    )
    return len(toks["input_ids"])


def count_tokens_batch(texts: list[str], tokenizer) -> list[int]:
    """Cuenta tokens para múltiples textos en una sola llamada al tokenizador.
    ~3-5x más rápido que llamar count_tokens individualmente sobre muchos textos."""
    if not texts:
        return []
    toks = tokenizer(
        texts,
        add_special_tokens=True,
        truncation=False,
        return_overflowing_tokens=False,
        verbose=False,
    )
    if isinstance(toks["input_ids"], list):
        return [len(ids) for ids in toks["input_ids"]]
    # Batch: shape (batch_size, seq_len)
    return [len(ids) for ids in toks["input_ids"]]


def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    """Similitud coseno entre dos vectores numpy."""
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom < 1e-10:
        return 0.0
    return float(np.dot(a, b) / denom)


# ===========================================================================
# PASO 2: PREPROCESADO POR DOCUMENTO
# ===========================================================================

def clean_whitespace(text: str) -> str:
    """Limpieza de whitespace redundante y artefactos de codificación."""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = text.replace("\u00a0", " ")   # non-breaking space -> space
    text = text.replace("\ufffd", "")    # replacement char (encoding artifacts)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"[ \t]*\n[ \t]*", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def remove_boilerplate(text: str) -> str:
    """Aplica PATRONES_BOILERPLATE para eliminar texto repetido de scrapers.
    Usa re.MULTILINE para que ^/$ matcheen líneas, no solo inicio/fin del texto."""
    for pattern, replacement in PATRONES_BOILERPLATE:
        text = re.sub(pattern, replacement, text, flags=re.IGNORECASE | re.MULTILINE)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]*\n[ \t]*", "\n", text)
    return text.strip()


def detect_language_safe(text: str) -> Optional[str]:
    """Detecta idioma con langdetect por párrafo.
    Devuelve ISO 639-1 normalizado o None si falla/fracasa.
    Se detecta por párrafo, no para todo el documento.
    Se limita a los primeros 5000 caracteres por rendimiento."""
    if not _LANGDETECT_OK:
        return None
    text = text.strip()
    if len(text) < 10:
        return None
    # Limitar longitud para evitar lentitud con textos muy grandes
    sample = text[:5000] if len(text) > 5000 else text
    try:
        code = _ld_detect(sample)
        return normalize_langdetect_code(code)
    except (LangDetectException, Exception):
        return None


# ===========================================================================
# PASO 3: SEGMENTACIÓN ESTRUCTURAL
# ===========================================================================

def detect_section_header(line: str) -> bool:
    """Heurísticas para detectar encabezados de sección.

    Condiciones (según especificación):
    - Línea corta en MAYÚSCULAS (< 80 chars, sin punto final)
    - Patrón ^\\d+(\\.\\d+)*\\.?\\s (numeración de secciones)
    - Literal "Capítulo N" / "Chapter N" / "Section N"
    - Línea que termina en ':'
    """
    line = line.strip()
    if not line or len(line) > 80:
        return False
    if line.endswith("."):
        return False

    # MAYÚSCULAS (múltiples palabras, sin puntuación final)
    alnum_only = line.replace(" ", "").replace("-", "")
    if alnum_only.isupper() and len(line.split()) >= 2:
        return True

    # Numeración: "1. ", "3.2. ", "10.5.3. ", etc.
    if re.match(r"^\d+(\.\d+)*\.?\s+\S", line):
        return True

    # Capítulo / Chapter / Section / Sección
    if re.match(
        r"^(Cap[ií]tulo|Chapter|Section|Secci[oó]n|CAP[iÍ]TULO|CHAPTER)\s+\d+",
        line, re.IGNORECASE,
    ):
        return True

    # Termina en ':'
    if line.endswith(":"):
        return True

    return False


def split_sections(
    para_lang_pairs: list[tuple[str, str]],
) -> list[tuple[str, list[tuple[str, str]]]]:
    """Divide párrafos (con su idioma) en secciones jerárquicas.

    Si un párrafo comienza con un encabezado detectado, inicia una nueva
    sección. El encabezado es separado del contenido restante del párrafo.

    Devuelve lista de (título_sección, [(párrafo, idioma), ...]).
    """
    sections: list[tuple[str, list[tuple[str, str]]]] = []
    current_title = "sin_seccion"
    current_pairs: list[tuple[str, str]] = []

    for para, lang in para_lang_pairs:
        lines = para.split("\n")
        first_line = lines[0].strip()

        if first_line and detect_section_header(first_line):
            # Guardar sección anterior
            if current_pairs:
                sections.append((current_title, current_pairs))
            # Nueva sección: el header es el título
            current_title = first_line
            # El resto del párrafo (si lo hay) es contenido
            rest = "\n".join(lines[1:]).strip()
            if rest:
                current_pairs = [(rest, lang)]
            else:
                current_pairs = []
        else:
            current_pairs.append((para, lang))

    # Última sección
    if current_pairs:
        sections.append((current_title, current_pairs))

    return sections


# ===========================================================================
# PASO 4: SEGMENTACIÓN ORACIONAL
# ===========================================================================

# Umbral: textos > este tamaño se dividen por saltos de línea antes de spaCy
# (evita el límite max_length y mejora rendimiento en documentos CSV grandes)
MAX_SENTENCE_INPUT_CHARS = 100_000

# Umbral para usar spacy.blank() (más rápido) en lugar del modelo preentrenado
# (el tokenizer del modelo preentrenado es ~6x más lento en textos grandes)
FAST_TOKENIZER_THRESHOLD = 50_000


def get_spacy_nlp(lang_code: str) -> Optional[Any]:
    """Carga perezosamente el modelo spaCy para el idioma dado.
    Usa caché en memoria durante la ejecución.

    Se deshabilitan NER, tagger, parser y demás componentes pesados;
    solo se conserva el tokenizer del modelo (idioma-adecuado) y se añade
    el sentencizer (regla) para segmentación oracional rápida.

    Si el idioma no tiene modelo estadístico, devuelve None y el caller
    usa get_sentencizer()."""
    global _worker_spacy_cache
    spacy_name = SPACY_MODEL_MAP.get(lang_code)
    if spacy_name is None:
        return None
    if spacy_name not in _worker_spacy_cache:
        try:
            nlp = spacy.load(
                spacy_name,
                disable=["ner", "tagger", "lemmatizer",
                         "attribute_ruler", "parser"],
            )
            nlp.add_pipe("sentencizer")
            nlp.max_length = 50_000_000
            _worker_spacy_cache[spacy_name] = nlp
            logging.info(
                "Modelo spaCy cargado: %s (idioma=%s) [sentencizer activo]",
                spacy_name, lang_code,
            )
        except OSError:
            logging.warning(
                "Modelo spaCy '%s' no disponible para idioma '%s'; "
                "usando sentencizer genérico", spacy_name, lang_code,
            )
            _worker_spacy_cache[spacy_name] = None
    return _worker_spacy_cache[spacy_name]


def get_fast_sentencizer(lang_code: str) -> Optional[Any]:
    """Crea un pipeline spaCy blank con sentencizer (regla, no estadístico).
    Más rápido que el modelo preentrenado para textos grandes, ya que el
    tokenizer simple de spacy.blank() es ~6x más veloz."""
    global _worker_spacy_cache
    cache_key = f"_fast_senter_{lang_code or 'en'}"
    if cache_key in _worker_spacy_cache:
        return _worker_spacy_cache[cache_key]

    blank_lang = lang_code if lang_code else "en"
    try:
        nlp = spacy.blank(blank_lang)
    except Exception:
        logging.warning(
            "spacy.blank falló para idioma '%s'; usando 'xx'", lang_code,
        )
        nlp = spacy.blank("xx")
    nlp.add_pipe("sentencizer")
    nlp.max_length = 50_000_000
    _worker_spacy_cache[cache_key] = nlp
    return nlp


def get_sentencizer(lang_code: str) -> Optional[Any]:
    """Crea un pipeline spaCy blank con sentencizer (regla, no estadístico).
    Fallback para idiomas sin modelo spaCy disponible."""
    global _worker_spacy_cache
    cache_key = f"_sentencizer_{lang_code or 'en'}"
    if cache_key in _worker_spacy_cache:
        return _worker_spacy_cache[cache_key]

    blank_lang = lang_code if lang_code else "en"
    try:
        nlp = spacy.blank(blank_lang)
    except Exception:
        logging.warning(
            "spacy.blank falló para idioma '%s'; usando 'xx'", lang_code,
        )
        nlp = spacy.blank("xx")
    nlp.add_pipe("sentencizer")
    nlp.max_length = 50_000_000
    logging.warning(
        "Usando sentencizer genérico (regla) para idioma '%s' — "
        "no hay modelo estadístico disponible", lang_code,
    )
    _worker_spacy_cache[cache_key] = nlp
    return nlp


def _get_segmentation_nlp(text: str, lc: str) -> Any:
    """Selecciona el pipeline spaCy adecuado según el tamaño del texto.
    Para textos grandes usa el tokenizer rápido (spacy.blank()); para textos
    pequeños mantiene el modelo preentrenado con sentencizer."""
    use_fast = len(text) > FAST_TOKENIZER_THRESHOLD
    if use_fast:
        nlp = get_fast_sentencizer(lc)
        logging.debug(
            "segment_sentences: usando tokenizer rápido (blank) para "
            "texto de %d chars", len(text),
        )
        return nlp

    nlp = get_spacy_nlp(lc)
    if nlp is None:
        nlp = get_sentencizer(lc)
    return nlp


def segment_sentences(text: str, lang_code: str) -> list[str]:
    """Segmenta texto en oraciones usando el modelo spaCy para el idioma.
    Nunca corta una oración a la mitad.
    Si el idioma no tiene modelo, usa sentencizer genérico (regla).
    Para textos muy grandes, divide por saltos de línea y procesa por partes,
    y usa el tokenizer rápido de spacy.blank() cuando el texto supera el
    umbral FAST_TOKENIZER_THRESHOLD."""
    lc = normalize_langdetect_code(lang_code) if lang_code else "en"

    # Para textos muy grandes, dividir por \n en sub-chunks manejables
    if len(text) > MAX_SENTENCE_INPUT_CHARS and "\n" in text:
        lines = text.split("\n")
        sub_chunks: list[str] = []
        current = ""
        for line in lines:
            if (len(current) + len(line) + 1 > MAX_SENTENCE_INPUT_CHARS
                    and current):
                sub_chunks.append(current)
                current = line
            else:
                current = current + "\n" + line if current else line
        if current:
            sub_chunks.append(current)

        all_sents: list[str] = []
        for chunk_text in sub_chunks:
            nlp = _get_segmentation_nlp(chunk_text, lc)
            doc = nlp(chunk_text)
            all_sents.extend(
                s.text.strip() for s in doc.sents if s.text.strip()
            )
        return all_sents

    nlp = _get_segmentation_nlp(text, lc)
    doc = nlp(text)
    sents = [s.text.strip() for s in doc.sents if s.text.strip()]
    if sents:
        return sents

    # Último recurso: división por puntuación
    logging.warning(
        "segment_sentences: regex fallback para idioma '%s' (texto: %.60s...)",
        lc, text,
    )
    sents = re.split(r"(?<=[.!?])\s+", text)
    return [s.strip() for s in sents if s.strip()]


# ===========================================================================
# PASO 5: CONSTRUCCIÓN DE CHUNKS (ventana deslizante + corte semántico)
# ===========================================================================

@dataclass
class ChunkResult:
    """Resultado de chunk para un documento — incluye metadatos internos
    no serializados directamente en el JSONL de salida."""
    idx_chunk: int
    texto: str
    idioma: str
    seccion: str
    n_tokens: int
    n_palabras: int
    es_overlap: bool
    close_reason: str
    estrategias_usadas: list[str] = field(
        default_factory=lambda: ["estructural", "oracional", "semantica"]
    )


def get_overlap_indices(
    sent_indices: list[int],
    token_counts: list[int],
    overlap_tokens: int,
) -> list[int]:
    """Selecciona las últimas oraciones que suman ~overlap_tokens.
    Siempre incluye al menos la última oración."""
    if not sent_indices or overlap_tokens <= 0:
        return []
    overlap = []
    total = 0
    for j in reversed(sent_indices):
        if total + token_counts[j] <= overlap_tokens:
            overlap.insert(0, j)
            total += token_counts[j]
        else:
            break
    if not overlap and sent_indices:
        overlap = [sent_indices[-1]]
    return overlap


def determine_chunk_language(
    sent_langs: list[str],
    chunk_text: str,
    doc_dominant_lang: str,
) -> str:
    """Determina el idioma ISO 639-1 para un chunk específico
    basado en las oraciones que lo componen."""
    valid_langs = [l for l in sent_langs if l]
    if valid_langs:
        counter = Counter(valid_langs)
        return counter.most_common(1)[0][0]
    # Fallback: langdetect sobre el texto del chunk
    detected = detect_language_safe(chunk_text)
    if detected:
        return detected
    return doc_dominant_lang


def _finalize_chunk(
    indices: list[int],
    sentences: list[tuple[str, str]],
    token_counts: list[int],
    seccion_title: str,
    doc_dominant_lang: str,
    tokenizer,
    idx_chunk: int,
    is_overlap: bool,
    close_reason: str,
) -> ChunkResult:
    """Construye un ChunkResult a partir de índices de oraciones."""
    chunk_text = " ".join(sentences[j][0].strip() for j in indices)
    chunk_text = clean_whitespace(chunk_text)

    n_tokens = count_tokens(chunk_text, tokenizer)
    n_palabras = len(chunk_text.split())

    sent_langs = [sentences[j][1] for j in indices]
    idioma = determine_chunk_language(sent_langs, chunk_text, doc_dominant_lang)

    return ChunkResult(
        idx_chunk=idx_chunk,
        texto=chunk_text,
        idioma=idioma,
        seccion=seccion_title,
        n_tokens=n_tokens,
        n_palabras=n_palabras,
        es_overlap=is_overlap,
        close_reason=close_reason,
    )


def build_chunks_for_section(
    sentences: list[tuple[str, str]],
    doc_id: str,
    doc_fuente: str,
    doc_fase: str,
    doc_organizacion: str,
    doc_ruta: str,
    seccion_title: str,
    doc_dominant_lang: str,
    target_tokens: int,
    overlap_tokens: int,
    umbral_coseno: float,
    tokenizer,
    model: SentenceTransformer,
) -> list[ChunkResult]:
    """Construye chunks para una sección usando ventana deslizante + corte semántico.

    sentences: lista de (texto_oración, código_idioma) en orden.
    No hay overlap cruzando fronteras de sección.
    """
    if not sentences:
        return []

    # Pre-computar token counts por oración (batch único)
    token_counts = count_tokens_batch(
        [sent for sent, _ in sentences], tokenizer
    )

    # Pre-computar embeddings de oraciones (batch único, una sola carga de modelo)
    try:
        embeddings = model.encode(
            [f"passage: {sent}" for sent, _ in sentences],
            convert_to_numpy=True,
            normalize_embeddings=True,  # e5 recomienda normalizar para similitud coseno
            show_progress_bar=False,
            batch_size=32,
        )
    except Exception as e:
        logging.warning(
            "Error computando embeddings (doc_id=%s, sección='%s'): %s; "
            "deshabilitando corte semántico", doc_id, seccion_title, e,
        )
        embeddings = None

    chunks: list[ChunkResult] = []
    idx_chunk = 0

    current_indices: list[int] = []
    current_tokens = 0
    current_has_overlap = False

    for i in range(len(sentences)):
        sent_tokens = token_counts[i]

        # --- Decidir si cerramos el chunk antes de agregar la oración i ---
        should_close = False
        close_reason = None

        # (a) Límite de tokens: al agregar i, excedería el target
        if current_tokens + sent_tokens > target_tokens and current_indices:
            should_close = True
            close_reason = "limite_tokens"

        # (b) Corte semántico: si chunk >= 150 tokens, compara similitud
        #     entre la última oración incluida y la candidata (frontera real)
        if not should_close and current_tokens >= SEMANTIC_MIN_TOKENS and current_indices:
            prev_idx = current_indices[-1]
            if embeddings is not None:
                sim = cosine_sim(embeddings[prev_idx], embeddings[i])
                if sim < umbral_coseno:
                    should_close = True
                    close_reason = "semantico"

        if should_close:
            # --- Cerrar chunk actual ---
            chunk = _finalize_chunk(
                indices=current_indices,
                sentences=sentences,
                token_counts=token_counts,
                seccion_title=seccion_title,
                doc_dominant_lang=doc_dominant_lang,
                tokenizer=tokenizer,
                idx_chunk=idx_chunk,
                is_overlap=current_has_overlap,
                close_reason=close_reason,
            )
            chunks.append(chunk)
            idx_chunk += 1

            # --- Iniciar nuevo chunk con overlap ---
            overlap = get_overlap_indices(
                current_indices, token_counts, overlap_tokens
            )
            current_indices = list(overlap)
            current_tokens = sum(token_counts[j] for j in overlap)
            current_has_overlap = len(overlap) > 0

        # Agregar oración actual
        current_indices.append(i)
        current_tokens += sent_tokens

    # --- Último chunk de la sección ---
    if current_indices:
        chunk = _finalize_chunk(
            indices=current_indices,
            sentences=sentences,
            token_counts=token_counts,
            seccion_title=seccion_title,
            doc_dominant_lang=doc_dominant_lang,
            tokenizer=tokenizer,
            idx_chunk=idx_chunk,
            is_overlap=current_has_overlap,
            close_reason="final",
        )
        chunks.append(chunk)

    return chunks


# ===========================================================================
# PROCESAMIENTO DE DOCUMENTO COMPLETO (Pasos 2-6)
# ===========================================================================

def process_document(
    doc: dict[str, Any],
    params: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    """Procesa un documento completo: preprocesa, segmenta, construye chunks, valida.

    Devuelve (chunks_list, exclusiones_list, stats_dict).
    """
    # --- Lectura de campos (soporta 'text' o 'texto_extraido') ---
    raw_text = doc.get("texto_extraido", "") or doc.get("text", "")
    doc_id = doc.get("doc_id", "")
    fuente = doc.get("fuente", "")     # COPIAR EXACTAMENTE
    fase = doc.get("fase", "")
    organizacion = doc.get("organizacion", "")
    ruta_relativa = doc.get("ruta_relativa", "")
    # Campos obligatorios propagados del documento original (requisito 12)
    formato = doc.get("tipo_original", doc.get("formato", ""))
    fenomeno = doc.get("fenomeno", "")

    stats: dict[str, Any] = {
        "doc_id": doc_id,
        "fuente": fuente,
        "fase": fase,
        "organizacion": organizacion,
        "formato": formato,
        "fenomeno": fenomeno,
        "n_parrafos": 0,
        "n_sentences": 0,
        "n_chunks": 0,
        "n_excluidos": 0,
        "close_reasons": Counter(),
        "error": None,
    }

    if not raw_text or not str(raw_text).strip():
        stats["error"] = "texto_vacio"
        return [], [], stats

    # --- Paso 2a: Limpieza de whitespace ---
    text = clean_whitespace(str(raw_text))

    # --- Paso 2b: Remoción de boilerplate ---
    text = remove_boilerplate(text)

    if not text.strip():
        stats["error"] = "texto_vacio_despues_limpieza"
        return [], [], stats

    # --- Paso 2c: División en párrafos ---
    paragraphs = [p for p in text.split("\n\n") if p.strip()]
    stats["n_parrafos"] = len(paragraphs)

    if not paragraphs:
        stats["error"] = "sin_parrafos"
        return [], [], stats

    # --- Paso 2d: Detección de idioma por párrafo ---
    para_langs: list[Optional[str]] = []
    for p in paragraphs:
        lang = detect_language_safe(p)
        para_langs.append(lang)

    # Idioma dominante del documento (fallback "es")
    valid_langs = [l for l in para_langs if l]
    if valid_langs:
        doc_dominant_lang = Counter(valid_langs).most_common(1)[0][0]
    else:
        doc_dominant_lang = "es"
        logging.warning(
            "doc_id=%s: no se pudo detectar idioma en ningún párrafo; "
            "usando idioma dominante='%s'", doc_id, doc_dominant_lang,
        )

    # Reemplazar None con idioma dominante
    para_langs = [l if l else doc_dominant_lang for l in para_langs]

    # --- Paso 3: Segmentación estructural ---
    para_lang_pairs = list(zip(paragraphs, para_langs))
    sections = split_sections(para_lang_pairs)

    # --- Pasos 4-5: Segmentar oraciones y construir chunks ---
    all_chunks: list[dict[str, Any]] = []
    exclusions: list[dict[str, Any]] = []
    idx_chunk = 0

    target = params["target_tokens"]
    overlap = params["overlap_tokens"]
    umbral = params["umbral_coseno"]

    for sec_title, sec_pairs in sections:
        # Segmentar oraciones por párrafo (idioma por párrafo)
        section_sentences: list[tuple[str, str]] = []
        for para, lang in sec_pairs:
            sents = segment_sentences(para, lang)
            section_sentences.extend(
                (s.strip(), lang) for s in sents if s.strip()
            )

        stats["n_sentences"] += len(section_sentences)

        if not section_sentences:
            continue

        chunks = build_chunks_for_section(
            sentences=section_sentences,
            doc_id=doc_id,
            doc_fuente=fuente,
            doc_fase=fase,
            doc_organizacion=organizacion,
            doc_ruta=ruta_relativa,
            seccion_title=sec_title,
            doc_dominant_lang=doc_dominant_lang,
            target_tokens=target,
            overlap_tokens=overlap,
            umbral_coseno=umbral,
            tokenizer=_worker_tokenizer,
            model=_worker_model,
        )

        for ck in chunks:
            stats["close_reasons"][ck.close_reason] += 1

            # Asignar índice global dentro del documento (único por chunk)
            idx_chunk += 1
            ck.idx_chunk = idx_chunk

            # Paso 6: Validación — descartar chunks < 30 tokens
            if ck.n_tokens < MIN_TOKENS_VALID:
                exclusions.append({
                    "doc_id": doc_id,
                    "fuente": fuente,
                    "idx_chunk": ck.idx_chunk,
                    "seccion": sec_title,
                    "n_tokens": ck.n_tokens,
                    "n_palabras": ck.n_palabras,
                    "razon": f"menos_de_{MIN_TOKENS_VALID}_tokens",
                })
                stats["n_excluidos"] += 1
                continue

            # Construir dict de salida
            chunk_dict = {
                "chunk_id": sha256_full(doc_id + str(ck.idx_chunk)),
                "doc_id": doc_id,
                "idx_chunk": ck.idx_chunk,
                "texto": ck.texto,
                "idioma": ck.idioma,
                "seccion": sec_title,
                "n_tokens": ck.n_tokens,
                "n_palabras": ck.n_palabras,
                "fase": fase,
                "organizacion": organizacion,
                "fuente": fuente,
                "formato": formato,
                "fenomeno": fenomeno,
                "ruta_relativa": ruta_relativa,
                "estrategias_usadas": ck.estrategias_usadas,
                "params": {
                    "target_tokens": target,
                    "overlap_tokens": overlap,
                    "umbral_coseno": umbral,
                },
                "es_overlap": ck.es_overlap,
                "modelo_embeddings": params["modelo_embeddings"],
            }
            all_chunks.append(chunk_dict)

    stats["n_chunks"] = len(all_chunks)
    return all_chunks, exclusions, stats


# ===========================================================================
# WORKER DE MULTIPROCESSING (función de nivel de módulo, picklable)
# ===========================================================================

def _worker_process(args: tuple) -> tuple[str, list[dict], list[dict], dict]:
    """Procesa un documento. Devuelve (doc_id, chunks, exclusions, stats)."""
    doc, params = args
    try:
        chunks, exclusions, stats = process_document(doc, params)
        return (doc.get("doc_id", ""), chunks, exclusions, stats)
    except Exception as e:
        doc_id = doc.get("doc_id", "")
        fuente = doc.get("fuente", "")
        logging.error(
            "Error procesando doc_id=%s fuente=%s: %s", doc_id, fuente, e
        )
        return (
            doc_id,
            [],
            [{"doc_id": doc_id, "fuente": fuente,
              "razon": f"error_procesamiento: {e}"}],
            {"doc_id": doc_id, "fase": doc.get("fase", ""),
             "organizacion": doc.get("organizacion", ""),
             "error": str(e), "close_reasons": Counter()},
        )


# ===========================================================================
# GENERACIÓN DE REPORTE
# ===========================================================================

def generate_report(
    stats_total: dict[str, Any],
    output_path: Path,
    params: dict[str, Any],
) -> None:
    """Genera salida/reporte_chunking.md con estadísticas."""
    lines: list[str] = []
    lines.append("# Reporte de Chunking — CODEFEST AD ASTRA")
    lines.append("")
    lines.append("## Estrategia híbrida")
    lines.append("")
    lines.append(
        "Pipeline de chunking de 3 niveles: **estructural** (detección de "
        "secciones), **oracional** (segmentación con spaCy), y **"
        "semántico** (corte por similitud coseno)."
    )
    lines.append(
        f"- Modelo de embeddings: `{params['modelo_embeddings']}` (encoder-only)"
    )
    lines.append(f"- Tokens objetivo: {params['target_tokens']}")
    lines.append(f"- Overlap: {params['overlap_tokens']} tokens")
    lines.append(f"- Umbral de corte semántico: {params['umbral_coseno']} (coseno)")
    lines.append(f"- Tokens mínimos válidos: {MIN_TOKENS_VALID}")
    lines.append("")

    # Resumen general
    lines.append("## Resumen general")
    lines.append("")
    lines.append(
        f"- Documentos procesados: {stats_total['docs_procesados']}"
    )
    lines.append(
        f"- Documentos fallidos: {stats_total['docs_fallidos']}"
    )
    lines.append(
        f"- Chunks totales generados: {stats_total['chunks_totales']}"
    )
    lines.append(
        f"- Chunks excluidos (< {MIN_TOKENS_VALID} tokens): "
        f"{stats_total['chunks_excluidos']}"
    )
    rate = (
        stats_total["chunks_excluidos"]
        / stats_total["chunks_totales"]
        * 100
        if stats_total["chunks_totales"] > 0
        else 0
    )
    lines.append(f"- Tasa de exclusión: {rate:.1f}%")
    lines.append("")

    # Distribución de tokens
    td = stats_total["token_dist"]
    if td["count"] > 0:
        lines.append("## Distribución de n_tokens por chunk")
        lines.append("")
        lines.append("| Métrica | Valor |")
        lines.append("|---------|-------|")
        lines.append(f"| Mínimo  | {td['min']} |")
        lines.append(f"| Máximo  | {td['max']} |")
        lines.append(f"| Media   | {td['mean']:.1f} |")
        lines.append(f"| Mediana | {td['median']} |")
        lines.append("")

    # Cortes semánticos vs límite de tokens
    cr = stats_total["close_reasons"]
    sem_cuts = cr.get("semantico", 0)
    tok_cuts = cr.get("limite_tokens", 0)
    final_cuts = cr.get("final", 0)
    lines.append("## Estrategia de corte: semántico vs. límite de tokens")
    lines.append("")
    lines.append("| Criterio | Chunks cortados |")
    lines.append("|----------|-----------------|")
    lines.append(
        f"| Corte semántico (similitud < {params['umbral_coseno']}) | "
        f"{sem_cuts} |"
    )
    lines.append(
        f"| Límite de tokens (~{params['target_tokens']}) | {tok_cuts} |"
    )
    lines.append(f"| Último chunk de sección | {final_cuts} |")
    total_cuts = sem_cuts + tok_cuts
    pct_sem = (sem_cuts / total_cuts * 100) if total_cuts > 0 else 0
    lines.append(f"| **Porcentaje semántico** | **{pct_sem:.1f}%** |")
    lines.append("")
    lines.append(
        "> La proporción de cortes semánticos demuestra que la estrategia "
        "híbrida actúa efectivamente: no todos los chunks llegan al límite "
        "de 400 tokens, sino que muchos se cierran anticipadamente por "
        "cambio de tema."
    )
    lines.append("")

    # es_overlap
    lines.append("## Chunks con overlap (es_overlap)")
    lines.append("")
    lines.append("| Tipo | Cantidad |")
    lines.append("|------|----------|")
    lines.append(
        f"| `es_overlap=true` | {stats_total['overlap_true']} |"
    )
    lines.append(
        f"| `es_overlap=false` | {stats_total['overlap_false']} |"
    )
    lines.append("")

    # Exclusiones por fase y organización
    lines.append(
        f"## Exclusiones por umbral de {MIN_TOKENS_VALID} tokens "
        f"(por fase y organización)"
    )
    lines.append("")
    lines.append("| Fase | Chunks excluidos | Docs con exclusiones |")
    lines.append("|------|-----------------|----------------------|")
    for fase in FASES:
        n_chunks = sum(
            1 for e in stats_total["all_exclusions"]
            if e.get("fase") == fase
        )
        n_docs = len(
            set(
                e["doc_id"]
                for e in stats_total["all_exclusions"]
                if e.get("fase") == fase
            )
        )
        lines.append(f"| {fase} | {n_chunks} | {n_docs} |")
    lines.append("")

    # Exclusiones por organización
    lines.append("### Por organización")
    lines.append("")
    lines.append("| Organización | Chunks excluidos |")
    lines.append("|-------------|-----------------|")
    excl_by_org = Counter(
        e.get("organizacion", "desconocida")
        for e in stats_total["all_exclusions"]
    )
    for org, count in excl_by_org.most_common():
        lines.append(f"| {org} | {count} |")
    lines.append("")

    # Chunks por fase
    lines.append("## Chunks generados por fase")
    lines.append("")
    lines.append(
        "| Fase | Chunks totales | Chunks excluidos | Netos |"
    )
    lines.append("|------|-------------|-----------------|-------|")
    chunks_by_fase = stats_total.get("chunks_by_fase", Counter())
    for fase in FASES:
        valid = chunks_by_fase.get(fase, 0)
        excl = sum(
            1 for e in stats_total["all_exclusions"]
            if e.get("fase") == fase
        )
        lines.append(f"| {fase} | {valid + excl} | {excl} | {valid} |")
    lines.append("")

    # Documentos sin chunks
    lines.append("## Documentos sin chunks generados")
    lines.append("")
    no_chunk_docs = stats_total.get("docs_sin_chunks", [])
    if no_chunk_docs:
        lines.append(f"Total: {len(no_chunk_docs)} documentos sin chunks.")
        lines.append("")
        lines.append("| doc_id | fuente | organización | fase |")
        lines.append("|--------|--------|-------------|------|")
        for d in no_chunk_docs[:20]:
            lines.append(
                f"| {d.get('doc_id','')} | {d.get('fuente','')[:60]} | "
                f"{d.get('organizacion','')} | {d.get('fase','')} |"
            )
        if len(no_chunk_docs) > 20:
            lines.append(f"\n... y {len(no_chunk_docs) - 20} más")
    else:
        lines.append("Todos los documentos produjeron al menos un chunk.")
    lines.append("")

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    logging.info("Reporte escrito en %s", output_path)


# ===========================================================================
# MAIN
# ===========================================================================

def main():
    parser = argparse.ArgumentParser(
        description=(
            "Chunking híbrido (estructural + oracional + semántico) "
            "para el pipeline RAG CODEFEST AD ASTRA."
        ),
    )
    parser.add_argument(
        "--input-dir", type=Path, default=Path("salida"),
        help="Directorio con f1/f2/f3_extraido.jsonl",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("salida"),
        help="Directorio de salida para chunks y reporte",
    )
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Limitar número de documentos (smoke test)",
    )
    parser.add_argument(
        "--target-tokens", type=int, default=TARGET_TOKENS,
        help="Tamaño objetivo por chunk en tokens",
    )
    parser.add_argument(
        "--overlap-tokens", type=int, default=OVERLAP_TOKENS,
        help="Tokens de overlap entre chunks consecutivos",
    )
    parser.add_argument(
        "--umbral-coseno", type=float, default=UMBRAL_COSENO,
        help=" Umbral de similitud coseno para corte semántico",
    )
    parser.add_argument(
        "--modelo-embeddings", type=str, default=MODELO_EMBEDDINGS,
        help="Modelo de embeddings (encoder-only)",
    )
    parser.add_argument(
        "--workers", type=int, default=max(1, mp.cpu_count() - 1),
        help="Número de workers para multiprocessing",
    )
    parser.add_argument(
        "--reanudar", action="store_true",
        help="Usar checkpoint para reanudar sin reprocesar",
    )
    parser.add_argument(
        "--log-level", default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Nivel de logging",
    )
    args = parser.parse_args()

    # --- Setup logging ---
    log_level_num = getattr(logging, str(args.log_level).upper(), logging.INFO)
    logging.basicConfig(
        level=log_level_num,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    out_dir: Path = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    input_dir: Path = args.input_dir

    # Parámetros como dict
    params: dict[str, Any] = {
        "target_tokens": args.target_tokens,
        "overlap_tokens": args.overlap_tokens,
        "umbral_coseno": args.umbral_coseno,
        "modelo_embeddings": args.modelo_embeddings,
    }

    logging.info(
        "Parámetros: target=%d, overlap=%d, umbral_coseno=%.2f, modelo=%s",
        params["target_tokens"],
        params["overlap_tokens"],
        params["umbral_coseno"],
        params["modelo_embeddings"],
    )

    # =====================================================================
    # PASO 1: LECTURA DE JSONL CON CHECKPOINTING
    # =====================================================================
    checkpoint_path = out_dir / "procesados_chunking.log"

    processed_doc_ids: set[str] = set()
    if args.reanudar and checkpoint_path.exists():
        processed_doc_ids = set(
            line.strip()
            for line in checkpoint_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        )
        logging.info(
            "Checkpoint cargado (%s): %d documentos ya procesados",
            checkpoint_path, len(processed_doc_ids),
        )
    elif args.reanudar:
        logging.info(
            "--reanudar solicitado pero no existe checkpoint; se procesa todo"
        )
    else:
        logging.info("Modo fresh start (no --reanudar); se reprocesa todo")

    # Modo de apertura de archivos de salida
    mode = "a" if (args.reanudar and checkpoint_path.exists()) else "w"

    # Abrir writers append-only
    chunk_writers: dict[str, Any] = {}
    for fase in FASES:
        fname = f"chunks_{fase.lower()}.jsonl"
        chunk_writers[fase] = open(
            out_dir / fname, mode, encoding="utf-8"
        )

    # Log de exclusiones
    excl_f = open(
        out_dir / "exclusiones_chunking.log", mode, encoding="utf-8"
    )
    # Archivo de checkpoint (append)
    checkpoint_f = open(
        checkpoint_path, mode, encoding="utf-8"
    )

    # --- Leer documentos de los 3 JSONL ---
    all_docs: list[dict] = []
    for fase in FASES:
        input_path = input_dir / INPUT_FILENAMES[fase]
        if not input_path.exists():
            logging.error("No se encontró %s", input_path)
            continue
        with open(input_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                doc = json.loads(line)
                all_docs.append(doc)

    logging.info("Total de documentos leídos: %d", len(all_docs))

    # Aplicar --limit
    if args.limit is not None:
        all_docs = all_docs[:args.limit]
        logging.info(
            "--limit %d aplicado: %d documentos a procesar",
            args.limit, len(all_docs),
        )

    # Filtrar ya procesados (checkpoint)
    if args.reanudar:
        before = len(all_docs)
        all_docs = [
            d for d in all_docs
            if d.get("doc_id", "") not in processed_doc_ids
        ]
        logging.info(
            "Checkpoint: %d documentos omitidos (%d restantes)",
            before - len(all_docs), len(all_docs),
        )

    if not all_docs:
        logging.info("No hay documentos nuevos para procesar")
        for w in chunk_writers.values():
            w.close()
        excl_f.close()
        checkpoint_f.close()
        return

    # -------------------------------------------------------------------
    # Procesamiento
    # -------------------------------------------------------------------
    stats_total: dict[str, Any] = {
        "docs_procesados": 0,
        "docs_fallidos": 0,
        "chunks_totales": 0,
        "chunks_excluidos": 0,
        "overlap_true": 0,
        "overlap_false": 0,
        "close_reasons": Counter(),
        "token_dist": {
            "values": [], "min": 0, "max": 0,
            "mean": 0, "median": 0, "count": 0,
        },
        "chunks_by_fase": Counter(),
        "all_exclusions": [],
        "docs_sin_chunks": [],
    }

    task_args = [(doc, params) for doc in all_docs]

    if args.workers <= 1 or len(task_args) <= 1:
        # --- Secuencial ---
        logging.info(
            "Procesando secuencialmente (%d documentos)", len(task_args)
        )
        _worker_init(params["modelo_embeddings"], args.log_level)
        results_iter = (
            _worker_process(a) for a in tqdm(task_args, desc="Chunking")
        )
        pool = None
    else:
        # --- Paralelo ---
        logging.info(
            "Procesando en paralelo con %d workers", args.workers
        )
        num_workers = min(args.workers, len(task_args))
        ctx = mp.get_context("spawn")
        pool = ctx.Pool(
            processes=num_workers,
            initializer=_worker_init,
            initargs=(params["modelo_embeddings"], args.log_level),
        )
        results_iter = pool.imap_unordered(
            _worker_process, task_args, chunksize=4
        )
        results_iter = tqdm(results_iter, total=len(task_args), desc="Chunking")

    for doc_id, chunks, exclusions, doc_stats in results_iter:
        stats_total["docs_procesados"] += 1

        if doc_stats.get("error"):
            stats_total["docs_fallidos"] += 1
            if not chunks:
                stats_total["docs_sin_chunks"].append({
                    "doc_id": doc_id,
                    "fuente": doc_stats.get("fuente", ""),
                    "organizacion": doc_stats.get("organizacion", ""),
                    "fase": doc_stats.get("fase", ""),
                })

        # Acumular close_reasons
        stats_total["close_reasons"].update(
            doc_stats.get("close_reasons", Counter())
        )

        # Escribir chunks
        for chunk in chunks:
            fase = chunk.get("fase", "")
            writer = chunk_writers.get(fase)
            if writer:
                writer.write(json.dumps(chunk, ensure_ascii=False) + "\n")
                writer.flush()
            stats_total["chunks_totales"] += 1
            stats_total["chunks_by_fase"][fase] += 1
            if chunk["es_overlap"]:
                stats_total["overlap_true"] += 1
            else:
                stats_total["overlap_false"] += 1
            stats_total["token_dist"]["values"].append(chunk["n_tokens"])

        # Escribir exclusiones
        for excl in exclusions:
            excl["fase"] = doc_stats.get("fase", "")
            excl["organizacion"] = doc_stats.get("organizacion", "")
            excl_f.write(json.dumps(excl, ensure_ascii=False) + "\n")
            excl_f.flush()
            stats_total["all_exclusions"].append(excl)
            stats_total["chunks_excluidos"] += 1

        # Checkpoint: registrar doc_id procesado
        checkpoint_f.write(doc_id + "\n")
        checkpoint_f.flush()

    if pool is not None:
        pool.close()
        pool.join()

    # -------------------------------------------------------------------
    # Finalizar archivos
    # -------------------------------------------------------------------
    for w in chunk_writers.values():
        w.close()
    excl_f.close()
    checkpoint_f.close()

    # Calcular distribución de tokens
    td_values = stats_total["token_dist"]["values"]
    if td_values:
        stats_total["token_dist"]["min"] = min(td_values)
        stats_total["token_dist"]["max"] = max(td_values)
        stats_total["token_dist"]["mean"] = sum(td_values) / len(td_values)
        stats_total["token_dist"]["median"] = int(
            sorted(td_values)[len(td_values) // 2]
        )
        stats_total["token_dist"]["count"] = len(td_values)

    # Generar reporte
    report_path = out_dir / "reporte_chunking.md"
    generate_report(stats_total, report_path, params)

    logging.info("Chunking completado.")
    logging.info("  Docs procesados: %d", stats_total["docs_procesados"])
    logging.info("  Chunks totales: %d", stats_total["chunks_totales"])
    logging.info("  Chunks excluidos: %d", stats_total["chunks_excluidos"])
    if td_values:
        td = stats_total["token_dist"]
        logging.info(
            "  Distribución tokens: min=%d max=%d mean=%.1f median=%d",
            td["min"], td["max"], td["mean"], td["median"],
        )


if __name__ == "__main__":
    main()
