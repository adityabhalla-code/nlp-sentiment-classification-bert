import urllib.request
import os

# Define the URL and the filename
dataset_url = 'https://cdn.iisc.talentsprint.com/AIandMLOps/MiniProjects/Datasets/Reviews.csv'
filename = dataset_url.split('/')[-1]  # Extracts 'Reviews.csv' from the URL

# Define the directory where you want to store the dataset
dataset_dir = 'dataset'

# Create the directory if it doesn't exist
if not os.path.exists(dataset_dir):
    os.makedirs(dataset_dir)

# Define the path to save the file
file_path = os.path.join(dataset_dir, filename)

# Download the dataset
print("Downloading....")
urllib.request.urlretrieve(dataset_url, file_path)

# Verify the file is saved by listing all CSV files in the dataset directory
for file in os.listdir(dataset_dir):
    if file.endswith(".csv"):
        print(f"Downloaded: {file}")
