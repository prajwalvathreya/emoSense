import kagglehub
import shutil
import os

# Define the custom path for the dataset
custom_path = 'expression-dataset'

# Create the custom path directory if it doesn't exist
if not os.path.exists(custom_path):
    os.makedirs(custom_path)

# Download the latest version of the dataset
dataset_path = kagglehub.dataset_download("noamsegal/affectnet-training-data")

# Move the dataset to your custom path
if os.path.exists(custom_path):
    shutil.rmtree(custom_path) # Ensure the directory is empty
    shutil.move(dataset_path, custom_path)

# Delete the original dataset files
if os.path.exists(dataset_path):
    shutil.rmtree(dataset_path)

# Print the new path where the dataset is located
print("Dataset has been moved to:", custom_path)
