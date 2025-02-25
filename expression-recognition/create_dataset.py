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
shutil.move(dataset_path, custom_path)

# Print the new path where the dataset is located
print("Dataset has been moved to:", custom_path)
