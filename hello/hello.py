import dask.dataframe as dd
import pandas as pd

# Load the dataset
df = dd.read_csv('/mnt/data/Syn.csv')

# Remove leading and trailing spaces from all column names
df.columns = df.columns.str.strip()

# Print column names to ensure they're correct
print(df.columns.compute())

# Example: Convert 'Source Port' and 'Destination Port' to string
df['Source Port'] = df['Source Port'].astype(str)
df['Destination Port'] = df['Destination Port'].astype(str)

# Define a safe conversion function for numeric columns
def safe_convert(series):
    return pd.to_numeric(series, errors='coerce')  # Safely convert values to numeric, coercing errors

# Apply the safe conversion function to 'Source Port' and 'Destination Port'
df['Source Port'] = safe_convert(df['Source Port'])
df['Destination Port'] = safe_convert(df['Destination Port'])

# Example of other transformations that you can perform on the dataset
# You can replace missing values (NaNs) in 'Source Port' and 'Destination Port' with a default value (like 0)
df['Source Port'].fillna(0, inplace=True)
df['Destination Port'].fillna(0, inplace=True)

# Let's perform a sample computation to verify everything
# You can change this to any analysis or transformation you want to do
print(df[['Source Port', 'Destination Port']].head().compute())

# Example of filtering rows with a specific condition
# For example, filter rows where 'Source Port' is greater than 10000
filtered_df = df[df['Source Port'] > 10000]

# Show the first few rows of the filtered dataframe
print(filtered_df.head().compute())

# Save the cleaned DataFrame back to a CSV if needed
# df.to_csv('/mnt/data/cleaned_data.csv', index=False)

