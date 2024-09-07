from sklearn.metrics.pairwise import cosine_similarity

import numpy as np
vec1 = np.array ([[1,0,1,2,0]])
vec2 = np.array ([[0,1,1,1,2]])
vec3 = np.array ([[1,1,0,2,0]])
print ('Similaridad entre V1 y V2 \n', cosine_similarity (vec1, vec2))