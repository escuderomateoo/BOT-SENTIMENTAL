# Importamos la función 'pipeline' de la biblioteca transformers
from transformers import pipeline # type: ignore
#mport warnings
#warnings.filterwarnings("ignore", category=FutureWarning)
# 1. Creamos el pipeline de análisis de sentimiento
#    - Especificamos la tarea: "sentiment-analysis".
#    - Elegimos un modelo pre-entrenado. 'nlptown/bert-base-multilingual-uncased-sentiment'
#      es un modelo multilenguaje popular y eficiente que está disponible públicamente.
print("Cargando el modelo de análisis de sentimiento...")
analizador_sentimiento = pipeline(
    "sentiment-analysis",
    model="nlptown/bert-base-multilingual-uncased-sentiment" 
)
print("¡Modelo cargado con éxito! ✅")

# 2. Preparamos una lista de frases para analizar
frases_para_analizar = [
    "Estoy enojado",
    "Estoy triste",
    "Estoy feliz",
]

# 3. Usamos el pipeline para obtener el sentimiento de cada frase
print("\nAnalizando frases...")
resultados = analizador_sentimiento(frases_para_analizar)

# 4. Mostramos los resultados de una forma clara
for frase, resultado in zip(frases_para_analizar, resultados):
    sentimiento = resultado['label']
    confianza = resultado['score']
    
    # Añadimos un emoji para hacerlo más visual
    # El modelo 'nlptown/bert-base-multilingual-uncased-sentiment' devuelve etiquetas como '1 star', '2 stars', etc.
    # Acá mapeamos esas estrellas a sentimientos más generales para los emojis.
    emoji = "❓"
    if "star" in sentimiento:
        if sentimiento == '5 stars':
            emoji = "😊" # Muy positivo
        elif sentimiento == '4 stars':
            emoji = "🙂" # Positivo
        elif sentimiento == '3 stars':
            emoji = "😐" # Neutral
        elif sentimiento == '2 stars':
            emoji = "😟" # Negativo
        elif sentimiento == '1 star':
            emoji = "😠" # Muy negativo

    print(f"\nFrase: '{frase}'")
    print(f"  -> Sentimiento Detectado: {sentimiento.upper()} {emoji} (Confianza: {confianza:.2%})")