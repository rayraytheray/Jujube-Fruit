import os
import pandas as pd
import numpy as np
import glob
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split

TextVectorization = tf.keras.layers.TextVectorization

DATA = "StartUp_FundingScrappingData"
MAX_VOCAB_SIZE = 10000
MAX_SEQUENCE_LENGTH = 200
EMBEDDING_DIM = 128
RANDOM_SEED = 42

def load_and_combine_data():
    """
    load and combine CSV files and turn into a dataframe
    """
    all_files = []
    
    for year in range(2015, 2022):
        year_dir = os.path.join(DATA, str(year))
        if os.path.exists(year_dir):
            csv_files = glob.glob(os.path.join(year_dir, "*.csv"))
            all_files.extend(csv_files)
    
    print(f"Found {len(all_files)} files")
    print(all_files)
    
    df_list = []
   
    for file in all_files:
        
        try:
            df = pd.read_csv(file)
            filename = os.path.basename(file)
            month, year = filename.replace('.csv', '').split('_')
            df['Year'] = year
            df['Month'] = month
            # Drop the first column (index not needed))
            df = df.drop(df.columns[0], axis=1)
            df_list.append(df)
       
        except Exception as e:
            print(f"err reading file {file}: {e}")
    
    if df_list:
        combined = pd.concat(df_list, ignore_index=True)
        
        print(f"Combined DataFrame shape: {combined.shape}")
        return combined
    
    else:
        raise ValueError("no valid files found")

def clean_amount_column(df):
    """
    Clean and convert the funding amount column
    """
    df = df.copy()
    
    def convert_amount(amount):
        
        if pd.isna(amount) or amount == 'N/A':
            return np.nan
        
        amount = str(amount).replace(',', '').replace('$', '')
        
        try:
            return float(amount)
        except ValueError:
            return np.nan
    
    df['Amount_Numeric'] = df['Amount(in USD)'].apply(convert_amount)
    df['Amount_Numeric'] = df['Amount_Numeric'].replace(0, 0.01)
    df['Amount_Log'] = np.log1p(df['Amount_Numeric'])
    return df

def process_text_data(df):

    df['Combined_Description'] = df['Sub-Vertical'].fillna('') + ' ' + df['Industry/Vertical'].fillna('')
    df['Cleaned_Description'] = ( df['Combined_Description']
        .fillna('')
        .str.lower()
        .str.replace(r'[^\w\s]', ' ', regex=True)
        .str.replace(r'\s+', ' ', regex=True)
        .str.strip()
    )

    vectorizer = TextVectorization(
        max_tokens=MAX_VOCAB_SIZE,
        output_mode='int',
        output_sequence_length=MAX_SEQUENCE_LENGTH
    )
    vectorizer.adapt(df['Cleaned_Description'].values)

    sequences = vectorizer(df['Cleaned_Description'].values)
    padded_sequences = sequences.numpy()

    return df, padded_sequences, vectorizer

def encode_categorical_features(df):
    """
    Encode categorical features
    """
    categorical_cols = ['City', 'Industry/Vertical', 'Investment Stage']
    encoders = {}
    df = df.copy()
   
    for col in categorical_cols: 
        if col in df.columns:
            df[col] = df[col].fillna('Unknown')
            le = LabelEncoder()
            df[col + '_Encoded'] = le.fit_transform(df[col])
            encoders[col] = le
    return df, encoders

def prepare_data_for_model(df, padded_sequences):
    """
    Prepare final dataset for model training
    """
    numerical_features = ['Year']
    categorical_features = [col for col in df.columns if col.endswith('_Encoded')]
    feature_columns = numerical_features + categorical_features
    features = df[feature_columns].values
    target = df['Amount_Log'].values
    valid_indices = ~np.isnan(target)
    features = features[valid_indices]
    padded_sequences = padded_sequences[valid_indices]
    target = target[valid_indices]

    X_text_train, X_text_test, X_feat_train, X_feat_test, y_train, y_test = train_test_split(
        padded_sequences, features, target, test_size=0.2, random_state=RANDOM_SEED
    )
   
    scaler = StandardScaler()
    X_feat_train = scaler.fit_transform(X_feat_train)
    X_feat_test = scaler.transform(X_feat_test)
    return (X_text_train, X_feat_train, y_train), (X_text_test, X_feat_test, y_test), scaler

def save_processed_data(train_data, test_data, tokenizer, scaler, encoders):
    """
    Save processed data and associated objects for later use
    """
    os.makedirs('processed_data', exist_ok=True)
    np.save('processed_data/X_text_train.npy', train_data[0])
    np.save('processed_data/X_feat_train.npy', train_data[1])
    np.save('processed_data/y_train.npy', train_data[2])
    np.save('processed_data/X_text_test.npy', test_data[0])
    np.save('processed_data/X_feat_test.npy', test_data[1])
    np.save('processed_data/y_test.npy', test_data[2])
    
    metadata = {
        'max_vocab_size': MAX_VOCAB_SIZE,
        'max_sequence_length': MAX_SEQUENCE_LENGTH,
        'embedding_dim': EMBEDDING_DIM
    }

    pd.to_pickle(metadata, 'processed_data/metadata.pkl')
    pd.to_pickle(tokenizer, 'processed_data/tokenizer.pkl')
    pd.to_pickle(scaler, 'processed_data/scaler.pkl')
    pd.to_pickle(encoders, 'processed_data/encoders.pkl')
    
    print("Processed data and metadata saved successfully")

def main():
    print("Starting data preprocessing...")
    
    df = load_and_combine_data()
  
    df = clean_amount_column(df)
    
    #drop unused columns
    df = df.drop('Date(dd/mm/yyyy)', axis=1)
    df = df.drop('Date', axis=1)
   
    df, padded_sequences, tokenizer = process_text_data(df)
  
    df, encoders = encode_categorical_features(df)
  
    train_data, test_data, scaler = prepare_data_for_model(df, padded_sequences)
   
    save_processed_data(train_data, test_data, tokenizer, scaler, encoders)
 
    df.to_csv('processed_data/processed_dataframe.csv', index=False)
  
    print("Data preprocessing completed successfully")

if __name__ == "__main__":
    main()
