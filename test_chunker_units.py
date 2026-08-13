import sys, json
sys.path.insert(0, "src/2_chunking")
import chunker as c

# Caso 1: texto largo > 800 tokens, con abreviaturas, para forzar varios chunks + overlap
pal = "palabra"
# construir una oración larga con comas para probar división por comas
long_sentence = ", ".join(f"elemento de prueba numero {i} con datos adicionales" for i in range(200))
text = "Introducción inicial. " + long_sentence + ". Conclusión final del documento."
units = c.split_unidades(text)
print("num unidades:", len(units))
print("unidad0 tokens:", c.contar_tokens(units[0]), "unidad1(long) tokens:", c.contar_tokens(units[1]), "unidad-1 tokens:", c.contar_tokens(units[-1]))
chunks = c.chunk_text(text)
print("num chunks:", len(chunks))
for i,(t,forced) in enumerate(chunks):
    print(i, "tok=", c.contar_tokens(t), "forced=", forced, "fin=", repr(t[-30:]))

# Caso 2: abreviaturas no deben romper
ab = "El Dr. Smith y la Sra. Jones hab parlaron de p. ej. temas urgentes. Luego salieron."
u = c.split_unidades(ab)
print("ab rev units:", u)

# Caso 3: decimal y numeracion
num = "En 3.14 el valor era alto. Luego el ítem 1. Introducción cubre esto. Final."
print("num rev:", c.split_unidades(num))

# Caso 4: oración solitaria > 400 tokens sin comas (solo palabras)
one = (" ".join(["palabra"]*500)) + "."
ch = c.chunk_text(one)
print("single long word-split chunks:", len(ch), [c.contar_tokens(t) for t,_ in ch], "all<=400?", all(c.contar_tokens(t)<=400 for t,_ in ch))
