import json

def ver(path):
    print("=" * 90)
    print(path)
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    print("Claves:", list(data.keys()) if isinstance(data, dict) else type(data))
    if isinstance(data, dict):
        for k in data:
            v = data[k]
            sv = str(v)[:300].replace("\n", " ")
            print(f"  {k}: {sv}")
        # valores completos de campos de texto
        for k in ["body_text", "body_paragraphs", "abstract", "content", "text", "html", "resumen"]:
            if k in data:
                print(f"\n--- {k} (primeros 1200 chars) ---")
                v = data[k]
                if isinstance(v, list):
                    print(" | ".join(str(x)[:200] for x in v[:5]))
                else:
                    print(str(v)[:1200])

ver(r"data\F3_Dinamicas_Territoriales\CEEEP\articulos\revista\CEEEP_issue10-55-las-consecuencias-que-se-derivan-de-la-consolidacion-del-eje-beijing-m.json")
ver(r"data\F3_Dinamicas_Territoriales\CEOBS\articulos\CEOBS_2025-11-27-monitoring-sudans-artisanal-and-small-scale-gold-mining-from-space.json")
