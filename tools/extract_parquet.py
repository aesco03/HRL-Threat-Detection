import pandas as pd

file_path = 'data/processed/labeled/cleaned_pdfmalware2022.parquet'

try:
    # 2. Read the Parquet file into a pandas DataFrame
    df = pd.read_parquet(file_path)

    # 3. Get the list of all column names (features and the label)
    column_names = df.columns.tolist()

    # 4. Print the result
    print(f"Successfully read {len(df)} rows and {len(column_names)} columns.")
    print("\nFeature and Label Column Titles:")
    for column in column_names:
        print(f"- {column}")

except FileNotFoundError:
    print(f"Error: The file '{file_path}' was not found. Please check the file path.")
except Exception as e:
    print(f"An error occurred while reading the Parquet file: {e}")