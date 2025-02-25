import kagglehub
import shutil
import os

# Path to the dataset folder

# check if expression-dataset folder exists
if not os.path.exists('emoSense/expression-dataset'):
    os.makedirs('emoSense/expression-dataset')

custom_path = 'emoSense/expression-dataset'

# Download the latest version of the dataset
dataset_path = kagglehub.dataset_download("noamsegal/affectnet-training-data")

# Move the dataset to your custom path
if not os.path.exists(custom_path):
    os.makedirs(custom_path)

# Assuming the dataset is a folder, you can move the entire folder to your custom path
shutil.move(dataset_path, custom_path)

# Print the new path where the dataset is located
print("Dataset has been moved to:", custom_path)
