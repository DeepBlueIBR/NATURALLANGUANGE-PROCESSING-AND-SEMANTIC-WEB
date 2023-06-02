# -*- coding: utf-8 -*-
"""
Created on Sat May 20 13:20:43 2023

@author: NLPmodels
"""
import urllib.request
import nltk
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# Led model
# This model requires "pip install -U -q transformers bitsandbytes accelerate"
# Read the book from a web TXT file.
url = "https://www.gutenberg.org/files/5200/5200-0.txt"  # Replace with the actual URL of the book
response = urllib.request.urlopen(url)
FileContent = response.read().decode('utf-8').strip()

# Remove the first 970 words
start_index = FileContent.index(" ", 970)
FileContentStart = FileContent[start_index:]

# Remove the last 19350 words
end_index = len(FileContentStart) - 19350
Book = FileContentStart[:end_index]

# Using Pegasus from the Google model.
checkpoint = "cnicu/t5-small-booksum"

tokenizer = AutoTokenizer.from_pretrained(checkpoint)
model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint)

# Extract the sentences from the document.
sentences = nltk.tokenize.sent_tokenize(Book)

# Creating chunks out of the book.
chunks = []
current_chunk = ""
for sentence in sentences:
    tokenized_sentence = tokenizer.tokenize(sentence)
    if len(current_chunk) + len(tokenized_sentence) <= tokenizer.model_max_length:
        current_chunk += sentence + " "
    else:
        chunks.append(current_chunk.strip())
        current_chunk = sentence + " "
if current_chunk:
    chunks.append(current_chunk.strip())

# Inputs to the model
inputs = [tokenizer(chunk, return_tensors="pt") for chunk in chunks]

# Getting the summarization
summaries = []
for input in inputs:
    output = model.generate(
        **input,
        max_length=50,  # Set the desired maximum length of the generated output
        num_beams=4,  # Use beam search for better summaries
        early_stopping=True  # Stop generation when the model predicts an end-of-sentence token
    )
    summary = tokenizer.decode(output[0], skip_special_tokens=True)
    summaries.append(summary)

# Write summaries to a text file
with open("summaries.txt", "w", encoding="utf-8") as file:
    file.writelines("\n".join(summaries))
