#Technique used to convert categorical data to numerical format. Useful when preparing data for Machine learning and Deep learning.


import os
import pandas as pd
from sklearn.preprocessing import OneHotEncoder

# Define the root directory where your data is stored
# file name: Thaat and Raga Forest (TRF) Dataset Output
# Directory structure: Thaat -> SubThaat -> RagaLabel
root_dir = r"C:\zzz\Solution_chall\Thaat and Raga Forest (TRF) Dataset Output"
# Check if directory exists and is not empty
if not os.path.exists(root_dir):
    raise FileNotFoundError(f"Directory '{root_dir}' does not exist.")
if not os.listdir(root_dir):
    raise ValueError(f"Directory '{root_dir}' is empty.")

# Extract categories
thaat_names = []
sub_thaat_names = []
raga_labels = []

# Traverse through the directory
for thaat in os.listdir(root_dir):
    thaat_path = os.path.join(root_dir, thaat)
    if os.path.isdir(thaat_path):
        for sub_thaat in os.listdir(thaat_path):
            sub_thaat_path = os.path.join(thaat_path, sub_thaat)
            if os.path.isdir(sub_thaat_path):
                for raga_label in os.listdir(sub_thaat_path):
                    raga_path = os.path.join(sub_thaat_path, raga_label)
                    if os.path.isdir(raga_path):
                        thaat_names.append(thaat)
                        sub_thaat_names.append(sub_thaat)
                        raga_labels.append(raga_label)

# Create DataFrame
df = pd.DataFrame({
    'Thaat': thaat_names,
    'SubThaat': sub_thaat_names,
    'RagaLabel': raga_labels
})

# Apply One-Hot Encoding
encoder = OneHotEncoder()
encoded_data = encoder.fit_transform(df[['Thaat', 'SubThaat', 'RagaLabel']]).toarray()

# Convert to DataFrame
encoded_df = pd.DataFrame(encoded_data, columns=encoder.get_feature_names_out(['Thaat', 'SubThaat', 'RagaLabel']))

# Save the encoded data for training
encoded_df.to_csv("encoded_music_data.csv", index=False)

#this code will one hot encode the data till the song folder level.

