import kagglehub
import shutil
import os

# Specify your own path for the dataset
custom_path = 'E:/Projects/EmoSense/emoSense/data'

# Download the latest version of the dataset
dataset_path = kagglehub.dataset_download("noamsegal/affectnet-training-data")

# Move the dataset to your custom path
if not os.path.exists(custom_path):
    os.makedirs(custom_path)

# Assuming the dataset is a folder, you can move the entire folder to your custom path
shutil.move(dataset_path, custom_path)

# Print the new path where the dataset is located
print("Dataset has been moved to:", custom_path)
