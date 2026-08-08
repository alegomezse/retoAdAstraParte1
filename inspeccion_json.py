import json
import os
import glob

def inspeccionar(path, max_items=4):
    print("=" * 90)
    print(f"JSON: {path}")
    print("=" * 90)
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        print(f"ERROR json: {e}")
        return
    def show(obj, depth=0):
        if isinstance(obj, dict):
            for k, v in obj.items():
                if isinstance(v, (dict, list)):
                    print(f"{'  '*depth}{k}: <{type(v).__name__} len={len(v)}>")
                    show(v, depth+1)
                else:
                    sv = str(v).replace("\n", " ")[:90]
                    print(f"{'  '*depth}{k}: {sv}")
        elif isinstance(obj, list) and obj:
            print(f"{'  '*depth}[0] =>")
            show(obj[0], depth+1)
    show(data)

archivos = [
    r"data\F1_IA_y_Capacidades_Estrategicas\Atlantic_Council\GeoTech_Cues\page_01\ATLCOUNCIL_01-europe-must-address-the-digital-sovereignty-triad.json",
    r"data\F2_Seguridad_Entorno_Espacial\SWF_Counterspace\articulos\SWF_1-2-2022-global-counterspace-capabilities-report.json",
    r"data\F2_Seguridad_Entorno_Espacial\INPE\noticias\INPE_2023-10-09-inpe-portas-abertas-2023.json",
    r"data\F3_Dinamicas_Territoriales\CEEEP\articulos\revista\CEEEP_issue10-55-las-consecuencias-que-se-derivan-de-la-consolidacion-del-eje-beijing-m.json",
]
for a in archivos:
    if os.path.exists(a):
        inspeccionar(a)
    else:
        print(f"(no existe) {a}")
