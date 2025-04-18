import os
import pandas as pd
import numpy as np
import re
import glob
import tensorflow as tf

DATA = "StartUp_FundingScrappingData"

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
    df_list = []
    
    for file in all_files:
        try:
            df = pd.read_csv(file)
            
            # creating year and month data
            filename = os.path.basename(file)
            month, year = filename.replace('.csv', '').split('_')
            df['Year'] = year
            df['Month'] = month
            df_list.append(df)
        
        except Exception as e:
            print(f"err reading file {file}: {e}")
  
    if df_list:
        # combines the datasets vertically 
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

def clean_text(text):
    if pd.isna(text):
        return ''
        
    text = str(text).lower()
    text = re.sub(r'[^\w\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def main():
    print("Starting data preprocessing...")
  
    df = load_and_combine_data()
  
    df = clean_amount_column(df)

if __name__ == "__main__":
    main()