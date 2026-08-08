import json, os, glob
from collections import Counter

BASE = "data"

# Mapear carpeta observatorio -> codigo desde inventario
import pandas as pd
xl = pd.ExcelFile("data\\Indice_Datos_Codefest.xlsx")
inv = xl.parse("Inventario de Archivos")
folder2code = {}
for _, r in inv.iterrows():
    carp = str(r['Carpeta'])
    parts = carp.split("/")
    if len(parts) >= 2:
        folder2code[parts[1]] = r['Código Observatorio']
print("Mapeo carpeta->codigo:")
for k, v in folder2code.items():
    print(f"  {k} -> {v}")

# Escanear esquemas JSON
patrones_no_articulo = ("catalog", "catalogo", "registro")
stats = Counter()
ejemplos = []
all_json = glob.glob(os.path.join(BASE, "**", "*.json"), recursive=True)
print(f"\nTotal JSON en disco: {len(all_json)}")

# check catalogo/registro names
for p in all_json:
    bn = os.path.basename(p).lower()
    if any(pat in bn for pat in patrones_no_articulo):
        stats["metadata"] += 1
        continue
    try:
        with open(p, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        stats["error_parse"] += 1
        continue
    if isinstance(data, dict):
        keys = set(data.keys())
        if "body_text" in keys or "body_paragraphs" in keys:
            stats["con_body"] += 1
        elif "abstract" in keys or "excerpt" in keys:
            stats["con_abstract"] += 1
        elif "html" in keys or "content" in keys or "text" in keys:
            stats["con_html_content"] += 1
        elif "title" in keys or "titulo" in keys:
            stats["solo_titulo"] += 1
        else:
            stats["otro"] += 1
            if len(ejemplos) < 15:
                ejemplos.append(p)
    elif isinstance(data, list):
        stats["lista"] += 1
        if len(ejemplos) < 15:
            ejemplos.append(p)
    else:
        stats["otro"] += 1

print("Estadisticas esquemas JSON:")
for k, v in stats.most_common():
    print(f"  {k}: {v}")
print("\nEjemplos 'otro/lista':")
for e in ejemplos:
    print(f"  {e}")
