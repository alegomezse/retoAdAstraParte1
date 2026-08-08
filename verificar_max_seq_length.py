"""Verifica el límite real de max_seq_length del modelo de embeddings
configurado, contra el que asume chunking.py al contar tokens con
AutoTokenizer. No modifica chunking.py — solo diagnóstico."""
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer

MODELO = "paraphrase-multilingual-MiniLM-L12-v2"
TARGET_TOKENS_ACTUAL = 400  # valor fijo en chunking.py, no lo cambies aquí

model = SentenceTransformer(MODELO)
tokenizer = AutoTokenizer.from_pretrained(f"sentence-transformers/{MODELO}")

print(f"Modelo: {MODELO}")
print(f"max_seq_length (sentence-transformers, límite real de encode()): {model.max_seq_length}")
print(f"tokenizer.model_max_length (límite nominal del tokenizer crudo): {tokenizer.model_max_length}")
print(f"TARGET_TOKENS configurado en chunking.py: {TARGET_TOKENS_ACTUAL}")

if TARGET_TOKENS_ACTUAL > model.max_seq_length:
    exceso = TARGET_TOKENS_ACTUAL - model.max_seq_length
    print(
        f"\nALERTA: TARGET_TOKENS excede max_seq_length en {exceso} tokens. "
        f"Todo chunk que alcance el target real se truncará al vectorizar."
    )
else:
    print("\nOK: TARGET_TOKENS está dentro del límite real del encoder.")
