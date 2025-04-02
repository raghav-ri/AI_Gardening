import pandas as pd

def preprocess_data(file_path):
    """
    Reads a dataset, fills missing values in numerical and categorical columns, and returns a cleaned DataFrame.
    """
    df = pd.read_csv(file_path)

    # Fill missing values in numerical columns with the median
    df.fillna(df.select_dtypes(include=['number']).median(), inplace=True)

    # Fill missing values in categorical columns with the most frequent value (mode)
    for col in df.select_dtypes(include=['object']).columns:
        df[col].fillna(df[col].mode()[0], inplace=True)

    return df
