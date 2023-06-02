# -*- coding: utf-8 -*-
"""
Created on Fri Jun  2 18:32:29 2023

@author: NLPmodels
"""

from rouge import Rouge


# Define the reference summary and the generated summaries from different models
with open("forComparison-summarization.txt", "r", encoding="utf-8") as file:
    reference_summary  = file.read().strip()
 
with open("Pegasus-summarization.txt", "r", encoding="utf-8") as file:
     summary_pegasus = file.read().strip()
 
with open("Bart-summarization.txt", "r", encoding="utf-8") as file:
    summary_bart = file.read().strip()

with open("T5-summarization.txt", "r", encoding="utf-8") as file:
    summary_t5 = file.read().strip()


# Initialize the ROUGE scorer
rouge = Rouge()

# Calculate ROUGE scores for each model
scores_pegasus = rouge.get_scores(summary_pegasus, reference_summary)
scores_bart = rouge.get_scores(summary_bart, reference_summary)
scores_t5 = rouge.get_scores(summary_t5, reference_summary)

# Print ROUGE scores
print("ROUGE Scores for Pegasus:")
print(scores_pegasus)

print("ROUGE Scores for BART:")
print(scores_bart)

print("ROUGE Scores for T5:")
print(scores_t5)
