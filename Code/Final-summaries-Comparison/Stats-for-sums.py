# -*- coding: utf-8 -*-
"""
Created on Wed May 24 15:08:38 2023

@author: NLPmodels
"""
from urllib import request
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import spacy
import matplotlib.pyplot as plt


# Load the spaCy model
nlp = spacy.load('en_core_web_sm')

# Define the reference summary and the generated summaries from different models
with open("forComparison-summarization.txt", "r", encoding="utf-8") as file:
    reference_summary  = file.read().strip()
 
with open("Pegasus-summarization.txt", "r", encoding="utf-8") as file:
     summary_pegasus = file.read().strip()
 
with open("Bart-summarization.txt", "r", encoding="utf-8") as file:
    summary_bart = file.read().strip()

with open("T5-summarization.txt", "r", encoding="utf-8") as file:
    summary_t5 = file.read().strip()

# Tokenize the sentences using spaCy
reference_summarytokens = " ".join([token.text for token in nlp(reference_summary)])
summary_pegasustokens = " ".join([token.text for token in nlp(summary_pegasus)])
summary_barttokens = " ".join([token.text for token in nlp(summary_bart)])
summary_t5tokens = " ".join([token.text for token in nlp(summary_t5)])

# Compute the vector representations of the sentences
reference_vector = nlp(reference_summarytokens).vector
summary_pegasus_vector = nlp(summary_pegasustokens).vector
summary_bart_vector = nlp(summary_barttokens).vector
summary_t5_vector = nlp(summary_t5tokens).vector

# Compute cosine similarity between the reference summary and the generated summaries
similarity_model_pegasus = cosine_similarity(reference_vector.reshape(1, -1), summary_pegasus_vector.reshape(1, -1))[0][0]
similarity_model_bart = cosine_similarity(reference_vector.reshape(1, -1), summary_bart_vector.reshape(1, -1))[0][0]
similarity_model_t5 = cosine_similarity(reference_vector.reshape(1,-1), summary_t5_vector.reshape(1, -1))[0][0]



# Define the model names and similarity scores
models = ["Pegasus", "BART", "T5"]
similarities = [similarity_model_pegasus, similarity_model_bart, similarity_model_t5]

# Plot the bar graph

plt.bar(models, similarities)
plt.xlabel("Models")
plt.ylabel("Similarity Score")
plt.title("Comparison of Similarity Scores between Models")

# Display the graph
plt.show()