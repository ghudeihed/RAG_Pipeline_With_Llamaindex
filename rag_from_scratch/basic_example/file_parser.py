# file_parser.py

import os

import nltk
from nltk.tokenize import sent_tokenize

# Make sure NLTK data is downloaded
nltk.download('punkt')

def parse_file_with_nltk(filename, chunk_size=1000):
    try:
        # Open and read the file content
        with open(filename, encoding='utf-8-sig') as f:
            content = f.read()
        
        # Tokenize the content into sentences
        sentences = sent_tokenize(content)
        
        # Initialize variables for chunking
        paragraphs = []
        current_chunk = []
        current_length = 0

        # Group sentences into chunks of approximately `chunk_size` characters
        for sentence in sentences:
            current_length += len(sentence)
            current_chunk.append(sentence)
            
            if current_length >= chunk_size:
                # When the current chunk reaches the desired size, save it
                paragraphs.append(" ".join(current_chunk))
                current_chunk = []
                current_length = 0
        
        # Add the last chunk if any sentences remain
        if current_chunk:
            paragraphs.append(" ".join(current_chunk))
        
        return paragraphs

    except Exception as e:
        print(f"Error processing file {filename}: {e}")
        return []

# Example usage
if __name__ == "__main__":
    filename = "War_and_Peace.txt"
    paragraphs = parse_file_with_nltk(filename)
    for i, paragraph in enumerate(paragraphs):
        print(f"Paragraph {i+1}: {paragraph[:100]}...")  # Print the first 100 characters of each chunk
