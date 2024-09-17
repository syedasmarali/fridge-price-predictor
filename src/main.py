from data_processing import load_data, preprocess_data
from regression import perform_regression

def main():
    # Load the dataset
    df = load_data()

    # Preprocess the data
    df_cleaned = preprocess_data(df)

    # Perform regression analysis
    model = perform_regression(df_cleaned, target_column='Price')

if __name__ == '__main__':
    main()