import pandas as pd
import openpyxl
import pdfplumber
import json
import os

BASE = "data"

def inspeccionar_xlsx(path):
    print("=" * 80)
    print(f"XLSX: {path}")
    print("=" * 80)
    try:
        xl = pd.ExcelFile(path)
        print(f"Hojas: {xl.sheet_names}")
        for sheet in xl.sheet_names:
            df = xl.parse(sheet, nrows=8)
            print(f"\n--- Hoja: {sheet} | shape(primeras 8 filas) = {df.shape} ---")
            print("Columnas:", list(df.columns))
            print(df.head(8).to_string())
    except Exception as e:
        print(f"ERROR leyendo xlsx: {e}")

def inspeccionar_pdf(path, max_paginas=3):
    print("=" * 80)
    print(f"PDF: {path}")
    print("=" * 80)
    try:
        with pdfplumber.open(path) as pdf:
            print(f"Total paginas: {len(pdf.pages)}")
            for i, page in enumerate(pdf.pages[:max_paginas]):
                txt = page.extract_text() or ""
                print(f"\n--- Pagina {i+1} ---")
                print(txt[:1500])
    except Exception as e:
        print(f"ERROR: {e}")

def inspeccionar_json(path):
    print("=" * 80)
    print(f"JSON: {path}")
    print("=" * 80)
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        print(f"Tipo raiz: {type(data).__name__}")
        if isinstance(data, list):
            print(f"Longitud lista: {len(data)}")
            if data:
                print("Elemento[0]:")
                for k, v in list(data[0].items())[:30]:
                    sv = str(v)
                    print(f"   {k}: {sv[:200]}")
        elif isinstance(data, dict):
            print("Claves:", list(data.keys())[:40])
            # mostrar primeras claves
            for k, v in list(data.items())[:15]:
                print(f"   {k}: {str(v)[:150]}")
    except Exception as e:
        print(f"ERROR: {e}")

def main():
    inspeccionar_xlsx("data\\Indice_Datos_Codefest.xlsx")
    inspeccionar_xlsx("data\\F3_Dinamicas_Territoriales\\FASE ORDENADA CODEFEST.xlsx")
    inspeccionar_pdf(r"data\Extracto_Preguntas_50_v2.pdf", max_paginas=3)

    # Catalogs / registros
    cat = [
        "data\\F3_Dinamicas_Territoriales\\SIPRI\\sipri_full_catalogo.json",
        "data\\F3_Dinamicas_Territoriales\\SIPRI\\sipri_full_registro.json",
        "data\\F3_Dinamicas_Territoriales\\MAPP_OEA\\mapp_catalogo.json",
        "data\\F3_Dinamicas_Territoriales\\MAPP_OEA\\mapp_registro.json",
        "data\\F3_Dinamicas_Territoriales\\RESDAL\\resdal_catalogo.json",
        "data\\F3_Dinamicas_Territoriales\\RESDAL\\resdal_registro.json",
    ]
    for c in cat:
        if os.path.exists(c):
            inspeccionar_json(c)
        else:
            print(f"(no existe) {c}")

if __name__ == "__main__":
    main()