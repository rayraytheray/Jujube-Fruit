import pandas as pd
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
from tqdm import tqdm
import pickle
import os

class EmbeddingGenerator:
    def __init__(self, model_url='https://tfhub.dev/google/universal-sentence-encoder/4', batch_size=32):
        """Initialize with Universal Sentence Encoder from TF Hub"""
        self.model = hub.load(model_url)
        self.batch_size = batch_size
    
    def generate_embeddings(self, descriptions):
        """Generate embeddings for a list of text descriptions"""
        embeddings = []
        
        for i in tqdm(range(0, len(descriptions), self.batch_size)):
            batch = descriptions[i:i + self.batch_size]
            batch_embeddings = self.model(batch).numpy()
            embeddings.append(batch_embeddings)
        
        return np.vstack(embeddings)

def process_monthly_csv(year, month, input_directory, output_directory):
    """Process a single month's CSV and save embeddings with that month's name"""
    
    # Ensure output directory exists
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
    
    csv_path = f"{input_directory}.csv"
    if not os.path.exists(csv_path):
        print(f"File not found: {csv_path}")
        return None
    
    # Read CSV
    df = pd.read_csv(csv_path)
    
    # Clean descriptions
    df['Sub-Vertical'] = df['Sub-Vertical'].fillna('')
    descriptions = df['Sub-Vertical'].tolist()
    
    # Generate embeddings
    generator = EmbeddingGenerator()
    embeddings = generator.generate_embeddings(descriptions)
    
    # Save embeddings with month's name
    output_file = f"{output_directory}/{month:02d}_{year}.pkl"
    with open(output_file, 'wb') as f:
        pickle.dump(embeddings, f)
    
    print(f"Saved {len(embeddings)} embeddings for {year}/{month:02d} to {output_file}")
    return output_file

def process_all_months(input_directory, output_directory):
    """Process all months and save embeddings by month name"""
    processed_files = {}
    
    for year in range(2015, 2022):
        for month in range(1, 13):
            output_file = process_monthly_csv(year, month, input_directory, output_directory)
            if output_file:
                processed_files[(year, month)] = output_file
    
    # Save a mapping file for easy reference
    mapping_file = f"{output_directory}/embedding_file_mapping.pkl"
    with open(mapping_file, 'wb') as f:
        pickle.dump(processed_files, f)
    
    print(f"Saved mapping of {len(processed_files)} processed files to {mapping_file}")
    return processed_files

# Usage example
if __name__ == "__main__":
    # For a single CSV file
    process_monthly_csv(2021, 9, 'Sep_2021', 'test_embeddings')
    
    # For all CSVs in directory structure
    # process_multiple_csvs('data', 'embeddings')
    
    
    
    
# # Load embeddings later for training
# with open('file_location.pkl', 'rb') as f:
#     data = pickle.load(f)
#     embeddings = data['embeddings']