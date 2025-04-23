import pandas as pd
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
from tqdm import tqdm
import pickle
import os

OUTPUT_DIRECTORY = 'test_embeddings'
INPUT_DIRECTORY = 'StartUp_FundingScrappingData'

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
    
def get_all_csv_files(directory):
    """Recursively get all CSV files from a directory and its subdirectories"""
    all_csv_files = []
    
    # Use os.walk to traverse directory tree
    for root, dirs, files in os.walk(directory):
        # Get only CSV files
        csv_files = [f for f in files if f.endswith('.csv')]
        
        # Add full path to each file
        csv_files_with_path = [os.path.join(root, file) for file in csv_files]
        all_csv_files.extend(csv_files_with_path)
    
    return all_csv_files

def process_monthly_csv(csv_file):
    """Process a single month's CSV and save embeddings with that month's name"""
    
    # Ensure output directory exists
    if not os.path.exists(OUTPUT_DIRECTORY):
        os.makedirs(OUTPUT_DIRECTORY)
    
    csv_path = f"{INPUT_DIRECTORY}/{csv_file}"
    if not os.path.exists(csv_path):
        print(f"File not found: {csv_path}")
        return None
    
    # Get base filename without extension
    base_name = os.path.splitext(csv_file)[0]
    
    # Read CSV
    df = pd.read_csv(csv_path)
    
    # Clean descriptions
    df['Sub-Vertical'] = df['Sub-Vertical'].fillna('')
    descriptions = df['Sub-Vertical'].tolist()
    
    # Generate embeddings
    generator = EmbeddingGenerator()
    embeddings = generator.generate_embeddings(descriptions)
    
    # Save embeddings with matching name
    output_file = f"{OUTPUT_DIRECTORY}/{base_name}_embeddings.pkl"
    with open(output_file, 'wb') as f:
        pickle.dump(embeddings, f)
    
    print(f"Saved {len(embeddings)} embeddings for {base_name} to {output_file}")
    return output_file

def process_all_months():
    """Process all months and save embeddings by month name"""
    
    # Get all CSV files in the input directory
    csv_files = get_all_csv_files(INPUT_DIRECTORY)
    
    for csv_file in csv_files:
        process_monthly_csv(csv_file)

# Usage example
if __name__ == "__main__":
    # For a single CSV file
    process_monthly_csv('Sep_2021.csv')
    
    # For all CSVs in directory structure
    # process_all_months('data', 'embeddings')
    
# # Load embeddings later for training
# with open('file_location.pkl', 'rb') as f:
#     data = pickle.load(f)
#     embeddings = data['embeddings']