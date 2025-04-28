import pandas as pd
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
from tqdm import tqdm
import pickle
import os

OUTPUT_DIRECTORY = 'embeddings'
INPUT_FILE = 'processed_data/processed_dataframe.csv'

class EmbeddingGenerator:
    def __init__(self, model_url='https://tfhub.dev/google/universal-sentence-encoder-large/4', batch_size=32):
        """Initialize with Universal Sentence Encoder from TF Hub"""
        self.model = hub.load(model_url)
        self.batch_size = batch_size
    
    def generate_embeddings(self, descriptions):
        """Generate embeddings for a list of text descriptions"""
        embeddings = []
        
        for i in tqdm(range(0, len(descriptions), self.batch_size)):
            batch = descriptions[i:i + self.batch_size]
            result = self.model(batch)
            batch_embeddings = result['outputs']
            embeddings.append(batch_embeddings)
        
        return np.vstack(embeddings)

def process_csv():
    """Generate embeddings from the preprocessed dataframe"""
    
    # Ensure output directory exists
    if not os.path.exists(OUTPUT_DIRECTORY):
        os.makedirs(OUTPUT_DIRECTORY)
    
    # Read CSV
    df = pd.read_csv(INPUT_FILE)
    
    # Clean descriptions
    df['Cleaned_Description'] = df['Cleaned_Description'].fillna('')
    descriptions = df['Cleaned_Description'].tolist()
    
    # Generate embeddings
    generator = EmbeddingGenerator()
    embeddings = generator.generate_embeddings(descriptions)
    
    # Save embeddings with matching name
    output_file = f"{OUTPUT_DIRECTORY}/embeddings.pkl"
    with open(output_file, 'wb') as f:
        pickle.dump(embeddings, f)
    
    print(f"Saved {len(embeddings)} embeddings to {output_file}")
    return output_file

# Usage example
if __name__ == "__main__":
    process_csv()
    
# # Load embeddings later for training
# with open('file_location.pkl', 'rb') as f:
#     data = pickle.load(f)
#     embeddings = data['embeddings']