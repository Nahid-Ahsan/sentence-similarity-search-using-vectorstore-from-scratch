import numpy as np 

class VectorStore:
    def __init__(self):
        """
         Initialize an empty dictionary to store vector data with vector_id as keys
         and corresponding vectors as values.
         
         Initialize an empty dictionary to store similarity index between vectors.
         It is a nested dictionary where each vector_id has a sub-dictionary with
         other vector_ids as keys and their similarities as values.
        """
        self.vector_data = {}
        self.vector_index = {}

    def addVector(self, vector_id, vector):
        """
         Add a new vector to the vector_data dictionary with vector_id as the key.
         Also, update the similarity index using the new vector.
        """
        self.vector_data[vector_id] = vector
        self._update_index(vector_id, vector)

    def getVector(self, vector_id):
        # Retrieve the vector associated with the given vector_id.
        return self.vector_data.get(vector_id)

    def _update_index(self, vector_id, vector):
        """
         Update the similarity index based on the cosine similarity between
         the new vector and existing vectors in the vector_data dictionary.
        """
        for existing_id, existing_vector in self.vector_data.items():
            # Calculate cosine similarity between the new vector and existing vectors.
            similarity = np.dot(vector, existing_vector) / (np.linalg.norm(vector) * np.linalg.norm(existing_vector))
            
            # If the existing vector's id is not in the vector_index, create a new entry.
            if existing_id not in self.vector_index:
                self.vector_index[existing_id] = {}

            # Update the similarity index for both vectors.
            self.vector_index[existing_id][vector_id] = similarity

    def findVector(self, query_vector, num_results=5):
        # Find the most similar vectors to the given query_vector based on cosine similarity.
        results = []

        # Iterate through each vector in vector_data.
        for vector_id, vector in self.vector_data.items():
            # Calculate cosine similarity between the query_vector and each vector.
            similarity = np.dot(query_vector, vector) / (np.linalg.norm(query_vector) * np.linalg.norm(vector))
            
            # Append vector_id and similarity to the results list.
            results.append((vector_id, similarity))

        # Sort the results based on similarity in descending order.
        results.sort(key=lambda x: x[1], reverse=True)

        # Return the top 'num_results' most similar vectors.
        return results[:num_results]
