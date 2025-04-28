import os
import pandas as pd
import numpy as np
import glob
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
import joblib

TextVectorization = tf.keras.layers.TextVectorization

DATA = "StartUp_FundingScrappingData"
MAX_VOCAB_SIZE = 10000
MAX_SEQUENCE_LENGTH = 200
EMBEDDING_DIM = 128
RANDOM_SEED = 42

def load_and_combine_data():

    all_files = []
    
    for year in range(2015, 2022):
        year_dir = os.path.join(DATA, str(year))
        if os.path.exists(year_dir):
            csv_files = glob.glob(os.path.join(year_dir, "*.csv"))
            all_files.extend(csv_files)
    
    print(f"Found {len(all_files)} files")
    
    df_list = []
   
    for file in all_files:
        
        try:
            df = pd.read_csv(file) 
            filename = os.path.basename(file)
            month, year = filename.replace('.csv', '').split('_')
            df['Year'] = year
            df['Month'] = month
            
 
            df = df.drop(df.columns[0], axis=1)
            
            for col_name in df.columns:
                if 'Industry' in col_name:
                    df.rename(columns={col_name: 'Industry'}, inplace=True)
                if 'City' in col_name:
                    df.rename(columns={col_name: 'City'}, inplace=True)
                if 'Investor' in col_name:
                    df.rename(columns={col_name: 'Investor'}, inplace=True)
                    
                #remove all spaces from all column names
                cleaned_name = col_name.replace(" ", "")
                df.rename(columns={col_name: cleaned_name}, inplace=True)
                
       
            df = process_investment_stage(df)
                    
   
            df_list.append(df)
       
        except Exception as e:
            print(f"err reading file {file}: {e}")
    
    if df_list:
        combined = pd.concat(df_list, ignore_index=True)
        
        print(f"Combined DataFrame shape: {combined.shape}")
        return combined
    
    else:
        raise ValueError("no valid files found")


def process_investment_stage(df):
   
    df = df.copy()
    
    if 'Remarks' in df.columns:
        df['Remarks'] = df['Remarks'].astype(str)
  
        def extract_series(text):     
            if 'series' in text.lower():
                
                series_index = text.lower().find('series')
                
        
                end_index = min(series_index + len('Series') + 2, len(text))
                
             
                investment_type = text[series_index:end_index]
                
                investment_type = investment_type.strip()
                
                return investment_type
            else:
                return ''
        
        df['Remarks'] = df['Remarks'].apply(extract_series)
        

        df.rename(columns={'Remarks': 'InvestmentStage'}, inplace=True)
    
    return df


def clean_amount_column(df):
    df = df.copy()
    scaler = MinMaxScaler()
    
    def convert_amount(amount):
        
        if pd.isna(amount) or amount == 'N/A':
            return np.nan
        
        amount = str(amount).replace(',', '').replace('$', '')
        
        try:
            return float(amount)
        except ValueError:
            return np.nan
    
    df['Amount_Numeric'] = df['Amount(inUSD)'].apply(convert_amount)
    df['Amount_Log'] = np.log1p(df['Amount_Numeric'])
    df['Amount_Log'] = scaler.fit_transform(df[['Amount_Log']]) #could return scaler as well for inverse_transform (if needed)
    joblib.dump(scaler, 'scaler.pkl')
    return df


def process_text_data(df):

    df['Combined_Description'] = df['Industry'].fillna('') + ' ' + df['Sub-Vertical'].fillna('') 
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

    categorical_cols = ['City', 'InvestmentType', 'InvestmentStage', 'Year', 'Month']
    encoders = {}
    df = df.copy()
   
    for col in categorical_cols:
        if col in df.columns:
            df[col] = df[col].fillna('Unknown')
            le = LabelEncoder()
            df[col + '_Encoded'] = le.fit_transform(df[col].astype(str))
            encoders[col] = le

    if 'Investor' in df.columns:
        df['Investor'] = df['Investor'].fillna('')
        investor_lists = df['Investor'].apply(lambda x: [i.strip() for i in x.split(',') if i.strip()])
        
        all_investors = sorted({inv for inv_list in investor_lists for inv in inv_list})
        investor_le = LabelEncoder()
        investor_le.fit(all_investors)
        encoders['Investor'] = investor_le

        df['Investor_Encoded'] = investor_lists.apply(
            lambda invs: investor_le.transform(invs).tolist() if invs else []
        )

    return df, encoders

def save_processed_data( tokenizer,  encoders): #train_data, test_data,scaler,
    os.makedirs('processed_data', exist_ok=True)
    
    metadata = {
        'max_vocab_size': MAX_VOCAB_SIZE,
        'max_sequence_length': MAX_SEQUENCE_LENGTH,
        'embedding_dim': EMBEDDING_DIM
    }

    pd.to_pickle(metadata, 'processed_data/metadata.pkl')
    pd.to_pickle(tokenizer, 'processed_data/tokenizer.pkl')
    pd.to_pickle(encoders, 'processed_data/encoders.pkl')
    
    print("Processed data and metadata saved successfully")


def main():
    print("Starting data preprocessing...")
    
    df = load_and_combine_data()
  
    df = clean_amount_column(df)
    
    df = df.drop('Date(dd/mm/yyyy)', axis=1)
    df = df.drop('Date', axis=1)
   
    df, padded_sequences, tokenizer = process_text_data(df)
  
    df, encoders = encode_categorical_features(df)

    save_processed_data( tokenizer, encoders) 
 
    final_cols = ['StartupName', 'City_Encoded', 'Investor_Encoded', 'InvestmentType_Encoded', 'Amount_Log', 'Year_Encoded', 'Month_Encoded', 'Cleaned_Description']
    df_final = df[final_cols]
    df_final.to_csv('processed_data/processed_dataframe.csv', index=False)
  
    print("Data preprocessing completed successfully")
    
if __name__ == "__main__":
    main()
