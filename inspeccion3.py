import pandas as pd
import json
import os

BASE = "data"

xl = pd.ExcelFile("data\\Indice_Datos_Codefest.xlsx")
inv = xl.parse("Inventario de Archivos")

# Construir mapa real: basename -> lista de rutas reales
archivos_reales = []
for raiz, _, files in os.walk(BASE):
    for f in files:
        rel = os.path.join(raiz, f).replace("\\", "/")
        archivos_reales.append(rel)
real_by_base = {}
for r in archivos_reales:
    real_by_base.setdefault(os.path.basename(r), []).append(r)

print("=== 97 sin match: agrupados por Carpeta del inventario ===")
no_match = []
for _, r in inv.iterrows():
    nombre = r['Nombre estandarizado']
    if isinstance(nombre, str) and os.path.basename(nombre) not in real_by_base:
        no_match.append(r)
df_no = pd.DataFrame(no_match)
print(df_no.groupby(['Carpeta','Tipo']).size().to_string())
print()
print("Ejemplos de no-match:")
for _, r in df_no.head(15).iterrows():
    print(f"  [{r['Tipo']}] {r['DOC_ID']} | {r['Nombre estandarizado']} | carpeta={r['Carpeta']}")

print()
print("=== Archivos reales NO cubiertos por el inventario (basename no en inventario) ===")
inv_bases = set(inv['Nombre estandarizado'].dropna().apply(os.path.basename))
no_cubiertos = [r for r in archivos_reales if os.path.basename(r) not in inv_bases]
print(f"Archivos reales sin entrada en inventario: {len(no_cubiertos)}")
from collections import Counter
ext = Counter(os.path.splitext(os.path.basename(r))[1] for r in no_cubiertos)
print("Por extension:", dict(ext))
print("Ejemplos:")
for r in no_cubiertos[:30]:
    print(f"  {r}")
