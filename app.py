from vector import VectorStore  # Import the VectorStore class from the 'vector' module.
import numpy as np

# Create an instance of the VectorStore class.
vector_store = VectorStore()

# Define a list of sentences in Bengali.
sentences = [
    "আমি ভাত খাই",
    "ভাত আমার পছন্দের খাবার"
    "ভাত, পোলাও, খিচুড়ি একই ধরনের খাবার",
    "ভাত বাংলাদেশের প্রধান খাদ্য"
]

# Create a vocabulary set to store unique words across all sentences.
vocabulary = set()

# Populate the vocabulary set by tokenizing and lowercasing each sentence.
for sentence in sentences:
    tokens = sentence.lower().split()
    vocabulary.update(tokens)

# Create a dictionary mapping words to their indices in the vocabulary.
word_to_index = {word: i for i, word in enumerate(vocabulary)}

# Create vectors for each sentence based on word frequency in the vocabulary.
sentence_vectors = {}
for sentence in sentences:
    tokens = sentence.lower().split()
    vector = np.zeros(len(vocabulary))
    for token in tokens:
        vector[word_to_index[token]] += 1
    sentence_vectors[sentence] = vector

# Add each sentence vector to the VectorStore.
for sentence, vector in sentence_vectors.items():
    vector_store.addVector(sentence, vector)

# Create a query vector for a given query sentence.
query_sentence = "আমি পোলাও বাংলাদেশের পছন্দের খাবার"
query_vector = np.zeros(len(vocabulary))
query_tokens = query_sentence.lower().split()

# Update the query vector based on word frequency in the vocabulary.
for token in query_tokens:
    if token in word_to_index:
        query_vector[word_to_index[token]] += 1

# Find the most similar sentences to the query vector in the VectorStore.
similar_sentences = vector_store.findVector(query_vector, num_results=2)

# Print the query sentence and the most similar sentences.
print("Query Sentence: ", query_sentence)
print("Similar Sentences: ")
for sentence, similarity in similar_sentences:
    print(f"{sentence}: Similarity + {similarity: .4f}")
