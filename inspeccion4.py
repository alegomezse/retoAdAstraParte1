import pandas as pd
import os

xl = pd.ExcelFile("data\\Indice_Datos_Codefest.xlsx")
inv = xl.parse("Inventario de Archivos")

def listar(carpeta):
    p = os.path.join("data", carpeta.replace("/", os.sep))
    if os.path.isdir(p):
        return sorted(os.listdir(p))
    return None

for carpeta in [
    "F1_IA_y_Capacidades_Estrategicas/AI_Index_Stanford/recursos/AI_Index_Report/pdfs",
    "F1_IA_y_Capacidades_Estrategicas/AI_Index_Stanford/recursos/Annual_Reports/pdfs",
    "F1_IA_y_Capacidades_Estrategicas/CSET_Georgetown/pdfs/Reports",
    "F2_Seguridad_Entorno_Espacial/CSIS_Aerospace/pdfs_full/Otros",
    "F2_Seguridad_Entorno_Espacial/CSIS_Aerospace/pdfs_full/Space_Threat_Assessment",
    "F2_Seguridad_Entorno_Espacial/SWF_Counterspace/pdfs/Counterspace_Reports",
]:
    print(f"\n=== {carpeta} ===")
    contenido = listar(carpeta)
    if contenido is None:
        print("  (no existe)")
        continue
    rows = inv[inv['Carpeta'] == carpeta]
    inv_ct = len(rows)
    print(f"Inventario registra {inv_ct} archivos | En disco: {len(contenido)}")
    print("  Archivos en disco:")
    for f in contenido:
        print(f"    {f}")