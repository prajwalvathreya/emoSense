# emoSense

# Project Setup

This guide will walk you through the process of setting up and running the required scripts in the correct order for this project.

## Step 1: Install Dependencies

First, you'll need to install the required dependencies listed in `requirements.txt`. This file contains all the Python packages that the project depends on.

1. Open a terminal (or command prompt).
2. Navigate to the project directory where `requirements.txt` is located.
3. Run the following command to install the dependencies:

    ```bash
    pip install -r requirements.txt
    ```

## Step 2: Download Mediapipe model
```bash
curl -o face_landmarker_v2_with_blendshapes.task -L https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task
```

## Step 3: Create Dataset

Once the dependencies are installed, the next step is to run the `create_dataset.py` script. This script will generate the necessary dataset required for the rest of the pipeline.

1. Run the following command to execute `create_dataset.py` by changing the path:

    ```bash
    python create_dataset.py
    ```

This will create the dataset and save it to the appropriate location. **DO NOT** make changes to the already set path.

## Step 4: Create Landmarks

Next, you'll need to run the `create_landmarks.py` script, which will generate the landmarks based on the dataset created in the previous step.

1. Run the following command to execute `create_landmarks.py`:

    ```bash
    python create_landmarks.py
    ```

This will generate the landmarks and prepare them for visualization or further processing.

## Step 5: View Failed

Finally, run the `view_failed.py` script. This script will analyze and display any failures or issues that occurred during the previous steps.

1. Execute the following command to run `view_failed.py`:

    ```bash
    python view_failed.py
    ```

This will display any failed items or errors for you to review.