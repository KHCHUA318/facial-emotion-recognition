# FER Model: Facial Expression Recognition Model

This project implements a face detector that predicts the emotion of faces in images. The detector performs this task using three different models, a base model with CNN architecture, and three models that is built on top of a pretrained transfer learning model. It utilizes deep learning techniques and pre-trained models to perform the detection and inference tasks. 

## Table of Contents
- [Features](#features)
- [Dependencies](#dependencies)
- [Installation](#installation)
- [Usage](#usage)
- [Expected Files and Directories](#expected-files-and-directories)

## Features

- Detects faces in images.
- Predicts the emotion, which include 'Happy', 'Sad', 'Angry', 'Fear', 'Neutral' and 'Surprise'.

## Dependencies

```
Python 3.x
numpy==1.24.3
opencv-python==4.9.0.80
pillow==9.5.0
matplotlib==3.7.1
seaborn==0.12.2
scipy==1.10.1
pandas==1.5.3
scikit-learn==1.2.2
tensorflow==2.12.0
```

## Installation

1. Download the Python 3 installer package from the official website: python.org, if not installed previously

2. Run the following in your computer's terminal to install the required libraries

```bash
pip install -r requirements.txt
```

## Usage
### Testing The Todels:

1. Upload `emotion_test.ipynb` notebook to Google Colab

2. Change runtime type to `T4 GPU` 

3. Connect to session

4. Upload Test Set 

5. Click on `Run All`

### Run Live Face Detection:

1. Open `main.py`

2. Paste relevant model path into:

```bash
emotion_model = load_model('/path/to/model.h5')
```

3. Run Code

4. Press Q to exit live camera

## Expected Files and Directories

1. **Source Code:**
   - `emotion_base_model.ipynb`: Jupyter notebook for base model training, includes model evaluation and visualisations.
   - `emotion_MobileNetV2_48.ipynb`: Jupyter notebook for MobileNetV2 model training using 48x48 images, includes model evaluation and visualisations.
   - `emotion_MobileNetV2_224.ipynb`: Jupyter notebook for MobileNetV2 model training using 224x224 images
   - `emotion_ResNet50.ipynb`: Jupyter notebook for VGG16 model training using 48x48 images.
   - `emotion_test.ipynb`: Jupyter notebook to test models.
   - `main.py`: Python script for live facial expression detection.

2. **Models:**
   - `models/`: Directory containing the trained machine learning models.
     - `emotion_model_base.h5`: Trained base FER model.
     - `emotion_model_mobilenet_v2_48.h5`: Trained MobileNetV2 FER model.
     - `emotion_model_mobilenet_v2_224.h5`: Trained MobileNetV2 FER model.
     - `emotion_model_resnet50.h5`: Trained VGG16 FER model.

3. **Data:**
   - `datasets/`: Directory containing datasets used for training and testing the models.
     - `emotion/`: Directory containing FER2013 dataset.
       - `train/`: Directory containing images for training the models.
       - `val/`: Directory containing images for validating the models.
     - `test/`: Directory containing custom self taken images for testing the models.

4. **Documentation:**
   - `requirements.txt`: File listing the Python packages required for running the project.
   - `report.pdf`: PDF file of the FaceMaster project report
   - `README.md`: The README file providing an overview of the project and its usage instructions.
