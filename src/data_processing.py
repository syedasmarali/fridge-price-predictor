import pandas as pd
import os

def load_data():
    data_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'fridge_price_predictor_dataset_real_brands.csv')
    df = pd.read_csv(data_path)
    return df

def preprocess_data(df):
    # Loading the data
    df = load_data()

    # Dropping the empty and none values
    df = df.dropna()

    # Dropping negative values from capacity
    df = df[(df['Capacity'] >= 0)]

    # Encoding energy rating
    df = pd.get_dummies(df, columns=['Energy Rating'], dtype=int)

    # Encoding brand
    df = pd.get_dummies(df, columns=['Brand'], dtype=int)

    # Encoding door type
    df = pd.get_dummies(df, columns=['Type'], dtype=int)

    # Encoding features
    df = pd.get_dummies(df, columns=['Features'], dtype=int)

    # Encoding color
    df = pd.get_dummies(df, columns=['Color'], dtype=int)

    # Encoding material
    df = pd.get_dummies(df, columns=['Material'], dtype=int)

    # Encoding manufacturing country
    df = pd.get_dummies(df, columns=['Country of Manufacture'], dtype=int)

    # Dropping model and random features
    df = df.drop(columns=['Model', 'Random Numeric Feature 1', 'Random Text Feature 2', 'Irrelevant Category Feature 3'])

    # Print the dataset
    #pd.set_option('display.max_columns', None)
    #print(df.head())

    # Return df
    return df