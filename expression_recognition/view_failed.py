import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt

def display_images_from_csv(csv_path):
    # Read the CSV file into a pandas DataFrame
    df = pd.read_csv(csv_path)
    
    # Loop through each row in the DataFrame
    for index, row in df.iterrows():
        image_path = row['image_path']
        
        # Open the image
        image = Image.open(image_path)
        
        # Display the image
        plt.imshow(image)
        plt.axis('off')  # Turn off axis
        plt.show()

# Call the function with the path to your CSV
display_images_from_csv('expression-landmarks/failed_detections.csv')
