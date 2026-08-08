import json

def ver(path, max_elem=2):
    print("=" * 90)
    print(path)
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    print("Tipo:", type(data).__name__)
    if isinstance(data, list):
        print("len:", len(data))
        for e in data[:max_elem]:
            print("  elem:", json.dumps(e, ensure_ascii=False)[:500])
    elif isinstance(data, dict):
        print("claves:", list(data.keys()))
        for k, v in list(data.items())[:max_elem]:
            sv = json.dumps(v, ensure_ascii=False)[:400]
            print(f"  {k}: {sv}")

ver(r"data\F1_IA_y_Capacidades_Estrategicas\RutaN_GEIAL\rutan_pdfs\RUTAN_catalog-2.json")
ver(r"data\F3_Dinamicas_Territoriales\SIPRI\sipri_data\SIPRI_catalog-2.json")
ver(r"data\F2_Seguridad_Entorno_Espacial\CSIS_Aerospace\csis_pdfs\CSIS_catalog-2.json")
