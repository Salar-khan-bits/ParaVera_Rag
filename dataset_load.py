# Install if needed
# pip install datasets pandas

from datasets import load_dataset
import pandas as pd

# Load the dataset from Hugging Face
dataset = load_dataset("sentence-transformers/natural-questions")

# See available splits
print(dataset)

# Example: convert train split to pandas DataFrame
df_train = dataset["train"].to_pandas()

# Preview
print(df_train.head())
print(df_train.columns)