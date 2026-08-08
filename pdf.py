import pdfplumber


texto = ""

with pdfplumber.open(r"data\Extracto_Preguntas_50_v2.pdf") as pdf:
    for page in pdf.pages:
        texto += page.extract_text() + "\n"
        print(texto)