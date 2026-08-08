import time
import fitz  # PyMuPDF
from PIL import Image
import io
import os


def pdf_a_imagenes(pdf_path, max_paginas=3):
    """Convierte las primeras N páginas del PDF a imagen PIL."""
    doc = fitz.open(pdf_path)
    total = min(len(doc), max_paginas)
    imagenes = []
    for page_num in range(total):
        page = doc.load_page(page_num)
        pix = page.get_pixmap(dpi=200)
        img = Image.open(io.BytesIO(pix.tobytes("png")))
        imagenes.append((page_num + 1, img))
    total_paginas = len(doc)
    doc.close()
    return imagenes, total_paginas


def probar_easyocr(imagenes):
    """Ejecuta EasyOCR sobre las imágenes y mide tiempo."""
    import easyocr

    print("Inicializando EasyOCR...")
    reader = easyocr.Reader(["es", "en"], gpu=False, verbose=False)

    texto_total = []
    tiempo_inicio = time.time()

    for page_num, img in imagenes:
        print(f"  EasyOCR - Procesando página {page_num}...")
        # Convertir a bytes para easyocr
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        buf.seek(0)
        resultados = reader.readtext(buf.getvalue(), detail=0)
        texto_pagina = " ".join(resultados)
        texto_total.append(texto_pagina)

    tiempo_total = time.time() - tiempo_inicio
    texto_completo = "\n".join(texto_total)
    return texto_completo, tiempo_total


def probar_paddleocr(imagenes):
    """Ejecuta PaddleOCR sobre las imágenes y mide tiempo."""
    from paddleocr import PaddleOCR

    print("Inicializando PaddleOCR...")
    ocr = PaddleOCR(use_angle_cls=True, lang="es", show_log=False)

    texto_total = []
    tiempo_inicio = time.time()

    for page_num, img in imagenes:
        print(f"  PaddleOCR - Procesando página {page_num}...")
        # PaddleOCR acepta numpy array o path, convertimos a bytes
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        buf.seek(0)
        import numpy as np

        img_array = np.array(img)
        resultados = ocr.ocr(img_array, cls=True)
        texto_pagina = ""
        if resultados and resultados[0]:
            texto_pagina = " ".join(
                [linea[1][0] for linea in resultados[0]]
            )
        texto_total.append(texto_pagina)

    tiempo_total = time.time() - tiempo_inicio
    texto_completo = "\n".join(texto_total)
    return texto_completo, tiempo_total


def analizar_texto(texto):
    """Retorna estadísticas básicas del texto."""
    caracteres = len(texto)
    palabras = len(texto.split())
    lineas = len(texto.split("\n"))
    return caracteres, palabras, lineas


def main():
    pdf_path = r"data\F3_Dinamicas_Territoriales\Alertas_Tempranas\pdfs\Informes\ALERTAS_informes003.pdf"

    if not os.path.exists(pdf_path):
        print(f"ERROR: No se encontró el PDF en {pdf_path}")
        return

    print("=" * 60)
    print("PRUEBA COMPARATIVA: PaddleOCR vs EasyOCR")
    print("=" * 60)
    print(f"PDF: {pdf_path}")

    # Convertir PDF a imágenes (solo primeras 3 páginas para la prueba)
    max_paginas = 3
    print(f"\nConvirtiendo PDF a imágenes (primeras {max_paginas} páginas)...")
    imagenes, total_paginas = pdf_a_imagenes(pdf_path, max_paginas)
    print(f"Páginas en el PDF: {total_paginas} | Procesando: {len(imagenes)}")

    # --- EasyOCR ---
    print("\n--- EASYOCR ---")
    texto_easyocr, tiempo_easyocr = probar_easyocr(imagenes)
    caracs_easy, pala_easy, lineas_easy = analizar_texto(texto_easyocr)

    # --- PaddleOCR ---
    print("\n--- PADDLEOCR ---")
    texto_paddle, tiempo_paddle = probar_paddleocr(imagenes)
    caracs_padd, pala_padd, lineas_padd = analizar_texto(texto_paddle)

    # --- Reporte ---
    print("\n" + "=" * 60)
    print("REPORTE COMPARATIVO")
    print("=" * 60)

    print(f"\n{'Métrica':<30} {'EasyOCR':<20} {'PaddleOCR':<20}")
    print("-" * 70)
    print(f"{'Tiempo (segundos)':<30} {tiempo_easyocr:<20.2f} {tiempo_paddle:<20.2f}")
    print(f"{'Caracteres extraídos':<30} {caracs_easy:<20} {caracs_padd:<20}")
    print(f"{'Palabras extraídas':<30} {pala_easy:<20} {pala_padd:<20}")
    print(f"{'Líneas extraídas':<30} {lineas_easy:<20} {lineas_padd:<20}")

    print("\n--- MUESTRA EasyOCR (primeros 500 caracteres) ---")
    print(texto_easyocr[:500] if texto_easyocr else "(sin texto)")

    print("\n--- MUESTRA PaddleOCR (primeros 500 caracteres) ---")
    print(texto_paddle[:500] if texto_paddle else "(sin texto)")

    # Guardar resultados en archivo
    with open("resultados_ocr.txt", "w", encoding="utf-8") as f:
        f.write("REPORTE COMPARATIVO: PaddleOCR vs EasyOCR\n")
        f.write("=" * 60 + "\n")
        f.write(f"PDF: {pdf_path}\n")
        f.write(f"Paginas totales: {total_paginas} | Procesadas: {len(imagenes)}\n\n")
        f.write(f"EasyOCR:\n")
        f.write(f"  Tiempo: {tiempo_easyocr:.2f}s\n")
        f.write(f"  Caracteres: {caracs_easy}\n")
        f.write(f"  Palabras: {pala_easy}\n")
        f.write(f"  Lineas: {lineas_easy}\n\n")
        f.write(f"PaddleOCR:\n")
        f.write(f"  Tiempo: {tiempo_paddle:.2f}s\n")
        f.write(f"  Caracteres: {caracs_padd}\n")
        f.write(f"  Palabras: {pala_padd}\n")
        f.write(f"  Lineas: {lineas_padd}\n\n")
        f.write("--- TEXTO COMPLETO EasyOCR ---\n")
        f.write(texto_easyocr + "\n\n")
        f.write("--- TEXTO COMPLETO PaddleOCR ---\n")
        f.write(texto_paddle + "\n")

    print(f"\nResultados guardados en: resultados_ocr.txt")
    print("=" * 60)


if __name__ == "__main__":
    main()
