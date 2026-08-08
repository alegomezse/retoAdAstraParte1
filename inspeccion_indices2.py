import pandas as pd
import json
import os

BASE = "data"

# 1. Cargar Inventario de Archivos completo
xl = pd.ExcelFile("data\\Indice_Datos_Codefest.xlsx")
inv = xl.parse("Inventario de Archivos")
print("Inventario de Archivos:")
print(f"  Filas: {len(inv)}")
print(f"  Columnas: {list(inv.columns)}")
print(f"  Tipos unicos: {inv['Tipo'].unique().tolist()}")
col_fen = inv.columns[inv.columns.str.contains('Fen', case=False)].tolist()[0]
print(f"  Fenomenos: {inv[col_fen].unique().tolist()}")
print(f"  Observatorios: {inv['Observatorio'].unique().tolist()}")
print(f"  DOC_ID duplicados: {inv['DOC_ID'].duplicated().sum()}")
print(f"  DOC_ID nulos: {inv['DOC_ID'].isna().sum()}")
print()
print("Ejemplo doc_id -> nombre estandarizado -> carpeta:")
for _, r in inv.head(12).iterrows():
    print(f"  {r['DOC_ID']} | {r['Tipo']} | {r['Nombre estandarizado']}")

print()
# Chequear duplicidad de 'Nombre estandarizado'
print(f"  Nombres estandarizados duplicados: {inv['Nombre estandarizado'].duplicated().sum()}")

# 2. Comparar inventario contra archivos reales
print("\n=== Comparacion con archivos reales ===")
archivos_reales = set()
for raiz, _, files in os.walk(BASE):
    for f in files:
        rel = os.path.join(raiz, f).replace("\\", "/")
        archivos_reales.add(rel)
print(f"Archivos reales en data/: {len(archivos_reales)}")

# nombres estandarizados en inventario
inv_nombres = inv['Nombre estandarizado'].dropna().tolist()
inv_carpetas = inv['Carpeta'].dropna().tolist()
print(f"Inventario registra {len(inv_nombres)} nombres estandarizados")

# buscar coincidencia por nombre de archivo
real_basename = {os.path.basename(p) for p in archivos_reales}
match = 0
no_match = []
for nombre in inv_nombres:
    bn = os.path.basename(nombre)
    if bn in real_basename:
        match += 1
    else:
        no_match.append(nombre)
print(f"Nombres estandarizados con match exacto por basename: {match}")
print(f"Sin match ({len(no_match)}): {no_match[:20]}")

# DOC_ID por tipo
print("\nConteo DOC_ID por Tipo:")
print(inv.groupby('Tipo')['DOC_ID'].count().to_string())

# 3. Analizar FASE ORDENADA CODEFEST - referencias a documentos
print("\n\n=== FASE ORDENADA CODEFEST ===")
xl2 = pd.ExcelFile("data\\F3_Dinamicas_Territoriales\\FASE ORDENADA CODEFEST.xlsx")
for sheet in xl2.sheet_names:
    df = xl2.parse(sheet)
    print(f"\n--- Hoja {sheet}: {df.shape} ---")
    print(f"  Columnas: {list(df.columns)}")
    doc_col = "DOCUMENTO" if "DOCUMENTO" in df.columns else None
    if doc_col:
        docs = df[doc_col].dropna().tolist()
        # separar por \n
        referencias = []
        for d in docs:
            for part in str(d).split("\\n"):
                part = part.strip()
                if part:
                    referencias.append(part)
        print(f"  Referencias a documentos (unicas): {len(set(referencias))}")
        for r in sorted(set(referencias))[:40]:
            print(f"    {r}")

# 4. Inspeccionar catalogs csv de organizacion
print("\n\n=== Catalog-2 CSV ===")
catalog_csvs = [
    "data\\F1_IA_y_Capacidades_Estrategicas\\RutaN_GEIAL\\rutan_pdfs\\RUTAN_catalog-2.csv",
    "data\\F2_Seguridad_Entorno_Espacial\\ESA_Space_Debris\\sipri_full_registro.json",
    "data\\F3_Dinamicas_Territoriales\\SIPRI\\sipri_data\\SIPRI_publicaciones-2.csv",
    "data\\F3_Dinamicas_Territoriales\\CEOBS\\ceobs_data\\CEOBS_publicaciones-2.csv",
    "data\\F3_Dinamicas_Territoriales\\RESDAL\\resdal_atlas\\RESDAL_catalog-2.csv",
]
for c in catalog_csvs:
    if os.path.exists(c) and c.endswith(".csv"):
        try:
            dfc = pd.read_csv(c, nrows=5)
            print(f"\nCSV: {c}")
            print(f"  Columnas: {list(dfc.columns)}")
            print(dfc.head(3).to_string()[:800])
        except Exception as e:
            print(f"  Error: {e}")
