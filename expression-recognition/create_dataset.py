import kagglehub
import shutil
import os

# Define the custom path for the dataset
custom_path = 'expression-dataset'

# Create the custom path directory if it doesn't exist
if not os.path.exists(custom_path):
    os.makedirs(custom_path)

# Download the dataset directly to the custom path
kagglehub.dataset_download("noamsegal/affectnet-training-data", path=custom_path)

print("Dataset has been downloaded to:", custom_path)
