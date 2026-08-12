# CODEFEST AD ASTRA 2026 - Módulo de Codificación Semántica e Indización Vectorial

Este repositorio contiene la implementación correspondiente a las Fases 3, 4 y 5 del reto CODEFEST AD ASTRA 2026. El objetivo de este módulo es generar los embeddings densos a partir de los datos procesados, construir el índice de búsqueda en FAISS y ejecutar la recuperación de información ante consultas en lenguaje natural (Español, Inglés y Portugués) sin el uso de modelos generativos (LLMs).

## Estructura del Módulo

* **encoder.py**: Lee los fragmentos de texto (chunks), genera los embeddings de 1024 dimensiones utilizando el modelo local BAAI/bge-m3, aplica normalización L2 y construye el índice FAISS.
* **entrega/generador.py**: Lee las consultas en lenguaje natural, realiza la búsqueda por similitud coseno en FAISS, selecciona el Top 3 de documentos mediante Max Pooling y entrega el Top 10 de fragmentos con recorte a 250 palabras por oraciones completas.
* **entrega/base_vectorial/encoder_bge_m3/**: Directorio que contiene el índice binario (index.faiss) y el almacén de metadatos (metadata.jsonl).
* **entrega/requirements.txt**: Dependencias requeridas para la ejecución.

## Integración con Fases 1 y 2 (Extracción y Chunking)

El flujo está preparado para recibir directamente los datos luego del  preprocesamiento y chunking:

1. Colocar el archivo de chunks en la raíz como `chunks.jsonl` (o en `data/chunks.jsonl`). Se conserva `datos.json` como archivo de prueba temporal.
2. Generar el índice vectorial:
   ```bash
   python encoder.py
   ```
3. Ejecutar la inferencia para las consultas de evaluación:
   ```bash
   python entrega/generador.py --consultas ruta/a/consultas.json
   ```

El archivo de salida final se generará en `entrega/resultados.jsonl`.
