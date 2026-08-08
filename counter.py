from collections import Counter
import os

# 1. Define la ruta de la carpeta principal
ruta_principal = r"J:\Documentos\Dev\retoAdAstraParte1"

extensiones = []
total_carpetas = 0

# 2. os.walk recorre de forma automática todas las subcarpetas
for raiz, carpetas, archivos in os.walk(ruta_principal):
    # Contamos las subcarpetas que va encontrando
    total_carpetas += len(carpetas)

    for archivo in archivos:
        # Separar el nombre de la extensión
        _, ext = os.path.splitext(archivo)

        if ext == "":
            ext = "Sin extensión"

        extensiones.append(ext.lower())

# 3. Contar los resultados
conteo = Counter(extensiones)

# 4. Mostrar el reporte en la terminal
print(f"==================================================")
print(f" ANÁLISIS RECURSIVO: {ruta_principal}")
print(f"==================================================")
print(f"Total de subcarpetas exploradas: {total_carpetas}")
print(f"Total de archivos totales encontrados: {len(extensiones)}")
print(f"--------------------------------------------------")
print("Detalle de archivos por tipo (en todo el proyecto):")

for tipo, cantidad in conteo.most_common():  # Ordenados de mayor a menor
    print(f" 📄 Archivos {tipo}: {cantidad}")
