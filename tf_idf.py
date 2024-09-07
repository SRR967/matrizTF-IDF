import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Definimos los documentos
sent_1 = 'El que madruga, Dios lo ayuda'
sent_2 = 'A quien Dios no le da hijos, el diablo le da sobrinos'
sent_3 = 'Que Dios lo guarde y se le olvide donde'
sent_4 = 'Dios aprieta pero no ahorca'
sent_5 = 'Dios le da pan al que no tiene dientes'

# Corpus de documentos
corpus = [sent_1, sent_2, sent_3, sent_4, sent_5]

# Creamos el TfidfVectorizer
VectorizerTfidf = TfidfVectorizer()

# Calculamos la matriz TF-IDF
tfidf_matrix = VectorizerTfidf.fit_transform(corpus)

# Obtenemos los nombres de los términos (palabras)
terms = VectorizerTfidf.get_feature_names_out()

# Imprimir la matriz TF-IDF 
print("Matriz TF-IDF (Coordenadas -> Valor, Término):")
for doc_idx, doc in enumerate(tfidf_matrix):
    for term_idx, value in zip(doc.indices, doc.data):
        term = terms[term_idx]  # Obtener el término correspondiente al índice
        print(f"  ({doc_idx}, {term_idx}) -> {value:.4f}, {term}")

# Calculamos la similitud coseno entre el primer documento y los demás
similarities = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix)

# Mostramos las similitudes del primer documento con los demás
print("\nSimilitud coseno del documento 1 con los demás documentos:")
for i, similarity in enumerate(similarities[0]):
    if i != 0:
        print(f"Documento 1 con Documento {i+1}: {similarity:.4f}")
