import numpy as np
from vec_db import VecDB

# Create an instance of VecDB and random DB of size 10K
db = VecDB(db_size = 10**4)

# Retrieve similar images for a given query
query_vector = np.random.rand(1,70) # Query vector of dimension 70
similar_images = db.retrieve(query_vector, top_k=10)
print(similar_images)