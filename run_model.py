import os
import numpy as np
import pandas as pd
import tensorflow as tf
from model import FundingModel
from sklearn.model_selection import train_test_split

data_dir = 'processed_data'
model_dir = 'models'
embedding_path = 'embeddings/embeddings.pkl'
csv_path = os.path.join(data_dir, 'processed_dataframe.csv')

def list_to_multihot(encoded_list, size):
    hot = np.zeros(size)
    for idx in encoded_list:
        hot[idx] = 1
    return hot

def main():

    df = pd.read_csv(csv_path)
    embeddings = pd.read_pickle(embedding_path)
    df['Descriptions_Encoded'] = embeddings.tolist()

    print("Embeddings merged into DataFrame")


    selected_columns = [
        'City_Encoded', 'Investor_Encoded', 'InvestmentType_Encoded',
        'Year_Encoded', 'Month_Encoded', 'Descriptions_Encoded', 'Amount_Log'
    ]
    df_filtered = df[selected_columns].dropna()


    investor_lists = df_filtered['Investor_Encoded'].apply(eval)
    all_investor_ids = [i for sublist in investor_lists for i in sublist]
    num_investors = max(all_investor_ids) + 1

    investor_multihot = investor_lists.apply(lambda x: list_to_multihot(x, num_investors))
    investor_array = np.stack(investor_multihot.to_numpy())


    X_embedding = np.stack(df_filtered['Descriptions_Encoded'].to_numpy())
    X_additional = df_filtered[['City_Encoded', 'InvestmentType_Encoded', 'Year_Encoded', 'Month_Encoded']].to_numpy()
    X_additional = np.concatenate([X_additional, investor_array], axis=1)
    y = df_filtered['Amount_Log'].to_numpy()

    X_embed_train, X_embed_test, X_add_train, X_add_test, y_train, y_test = train_test_split(
        X_embedding, X_additional, y, test_size=0.2, random_state=42
    )

    embedding_dim = X_embed_train.shape[1]
    additional_dim = X_add_train.shape[1]

    print(f"Training model with embedding_dim={embedding_dim}, additional_dim={additional_dim}")

    model = FundingModel(embedding_dim, additional_dim)
    model.compile(optimizer=tf.keras.optimizers.Adam(), loss='mse', metrics=['mae'])


    model.fit(
        x=[X_embed_train, X_add_train],
        y=y_train,
        batch_size=32,
        epochs=10,
        validation_split=0.1
    )


    loss, mae = model.evaluate([X_embed_test, X_add_test], y_test)
    print(f"Test Loss: {loss:.4f}, Test MAE: {mae:.4f}")


    preds = model.predict([X_embed_test[:5], X_add_test[:5]])
    print("Predictions (log scale):", preds.flatten())
    print("Ground Truth:", y_test[:5])


    os.makedirs(model_dir, exist_ok=True)
    model.save(os.path.join(model_dir, 'funding_model.keras'))
    print(f" Model saved to {model_dir}/funding_model.keras")

if __name__ == "__main__":
    main()