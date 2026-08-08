"""Tests unitarios para los fixes de chunking.py.

Ejecutar:  python test_chunking_fixes.py
Requiere: numpy (para embeddings sintéticos)
"""
import numpy as np

from chunking import (
    build_chunks_for_section,
    remove_boilerplate,
    SEMANTIC_MIN_TOKENS,
    MIN_TOKENS_VALID,
)


# ---------------------------------------------------------------------------
# Mocks
# ---------------------------------------------------------------------------

class MockTokenizer:
    """Tokenizer mock que asigna token_count = len(texto.split()) + 2 (special tokens)."""
    def __call__(self, texts, **kwargs):
        if isinstance(texts, str):
            texts = [texts]
        input_ids = [[0] * (len(t.split()) + 2) for t in texts]
        return {"input_ids": input_ids}


class MockModel:
    """Modelo mock que devuelve embeddings predefinidos."""
    def __init__(self, embeddings):
        self._embeddings = embeddings
    def encode(self, sentences, **kwargs):
        idxs = list(range(len(sentences)))
        return self._embeddings[idxs] if hasattr(self._embeddings, "__getitem__") else self._embeddings


def _make_sentences(n, words_per_sent=80):
    """Crea n oraciones sintéticas con suficientes tokens para superar SEMANTIC_MIN_TOKENS."""
    sents = []
    for i in range(n):
        word = chr(65 + i % 26) * 10  # "AAAAAAAAAA", "BBBBBBBBBB", ...
        # Repetir para alcanzar words_per_sent tokens (~words_per_sent + 2)
        body = " ".join([word] * words_per_sent)
        sents.append((f"Oracion {i} {body}", "es"))
    return sents


# ---------------------------------------------------------------------------
# Bug 1: Corte semántico en la frontera correcta
# ---------------------------------------------------------------------------

def test_corte_semantico_frontera_correcta():
    """4 oraciones: A~B similares, B->C cae bajo umbral, C~D similares.

    El chunk debe cerrarse entre B y C, no entre A y B ni entre C y D.
    """
    sentences = _make_sentences(4)

    # Embeddings 2D controlados manualmente
    # A y B: similares. B->C: ortogonales (sim ~0). C y D: similares.
    fake_embeddings = np.array([
        [1.0, 0.0],   # 0: A
        [0.95, 0.05], # 1: B (similar a A)
        [0.0, 1.0],   # 2: C (ortogonal a B -> similitud ~0)
        [0.05, 0.95], # 3: D (similar a C)
    ])

    tokenizer = MockTokenizer()
    model = MockModel(fake_embeddings)

    # target alto para forzar corte semántico (no por tokens)
    chunks = build_chunks_for_section(
        sentences=sentences,
        doc_id="test_doc",
        doc_fuente="",
        doc_fase="F1",
        doc_organizacion="",
        doc_ruta="",
        seccion_title="test_section",
        doc_dominant_lang="es",
        target_tokens=10_000,       # nunca se cierra por límite de tokens
        overlap_tokens=0,           # sin overlap para claridad
        umbral_coseno=0.5,          # B->C sim ~0 < 0.5 -> corte
        tokenizer=tokenizer,
        model=model,
    )

    assert len(chunks) >= 2, f"Se esperaban >=2 chunks, got {len(chunks)}"

    # El primer chunk debe contener A y B (índices 0 y 1)
    assert chunks[0].close_reason == "semantico", \
        f"Primer chunk debería cerrarse por 'semantico', got '{chunks[0].close_reason}'"

    # chunk text debe contener oraciones A y B
    chunk0_text = chunks[0].texto
    assert "Oracion 0" in chunk0_text, "Chunk 0 debe contener oración A"
    assert "Oracion 1" in chunk0_text, "Chunk 0 debe contener oración B"
    assert "Oracion 2" not in chunk0_text, "Chunk 0 NO debe contener oración C"

    print("  [OK] Corte semántico: chunk 0 = [A, B], close_reason = 'semantico'")

    # El segundo chunk debe empezar en C (índice 2)
    chunk1_text = chunks[1].texto
    assert "Oracion 2" in chunk1_text, "Chunk 1 debe contener oración C"
    print("  [OK] Chunk 1 empieza en oración C (frontera correcta)")

    return True


def test_corte_semantico_no_dispara_entre_similares():
    """Si todas las oraciones son similares, no debe haber corte semántico.
    El chunk debe cerrarse solo por límite de tokens."""
    sentences = _make_sentences(3)

    # Todos similares -> similitud alta entre todos
    fake_embeddings = np.array([
        [1.0, 0.0],
        [0.99, 0.01],
        [0.98, 0.02],
    ])

    tokenizer = MockTokenizer()
    model = MockModel(fake_embeddings)

    chunks = build_chunks_for_section(
        sentences=sentences,
        doc_id="test_doc2",
        doc_fuente="",
        doc_fase="F1",
        doc_organizacion="",
        doc_ruta="",
        seccion_title="test_section",
        doc_dominant_lang="es",
        target_tokens=10_000,
        overlap_tokens=0,
        umbral_coseno=0.5,
        tokenizer=tokenizer,
        model=model,
    )

    # Con todas similares y target alto, debe haber solo 1 chunk (cierre "final")
    assert len(chunks) == 1, f"Se esperaba 1 chunk, got {len(chunks)}"
    assert chunks[0].close_reason == "final", \
        f"Debería cerrarse por 'final', got '{chunks[0].close_reason}'"
    print("  [OK] Sin corte semántico entre oraciones similares")


# ---------------------------------------------------------------------------
# Bug 2: remove_boilerplate con re.MULTILINE
# ---------------------------------------------------------------------------

def test_boilerplate_multiline_removal():
    """'Skip to content' en medio del texto multilínea debe eliminarse."""
    text = (
        "Introducción del documento.\n"
        "Skip to content\n"
        "Esta es una oración que sigue.\n"
        "Toggle navigation\n"
        "Más contenido aquí."
    )

    result = remove_boilerplate(text)

    assert "Skip to content" not in result, \
        f"'Skip to content' no fue eliminado: {result}"
    assert "Toggle navigation" not in result, \
        f"'Toggle navigation' no fue eliminado: {result}"
    assert "Introducción" in result, "El contenido válido se perdió"
    assert "Más contenido" in result, "El contenido válido se perdió"
    print("  [OK] Boilerplate en medio de texto multilínea eliminado correctamente")


def test_boilerplate_no_falso_positivo():
    """Patrones sin anclas ^/$ no deben cambiar con MULTILINE."""
    text = (
        "Some text here.\n"
        "Explore some of our related publications below.\n"
        "More text."
    )
    result = remove_boilerplate(text)
    assert "Explore some of our related publications below." not in result
    assert "Some text here." in result
    assert "More text." in result
    print("  [OK] No hay falsos positivos con patrones no-ancorados")


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def run_all():
    tests = [
        ("Bug 1: corte semántico en frontera correcta",
         test_corte_semantico_frontera_correcta),
        ("Bug 1: no corta entre oraciones similares",
         test_corte_semantico_no_dispara_entre_similares),
        ("Bug 2: boilerplate eliminado en multilínea",
         test_boilerplate_multiline_removal),
        ("Bug 2: sin falsos positivos",
         test_boilerplate_no_falso_positivo),
    ]
    passed = 0
    failed = 0
    for name, fn in tests:
        print(f"\n[RUN] {name}")
        try:
            fn()
            passed += 1
        except Exception as e:
            failed += 1
            print(f"  [FAIL] {e}")

    print(f"\n{'='*50}")
    print(f"Resultados: {passed} passed, {failed} failed")
    if failed:
        raise SystemExit(1)
    print("¡Todos los tests pasaron!")


if __name__ == "__main__":
    run_all()
