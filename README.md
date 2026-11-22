# Chest X-Ray Pneumonia Detection Using Deep Learning

## Project Overview

This project develops a deep learning solution for detecting lung opacity in chest X-ray images using Convolutional Neural Networks (CNNs) and transfer learning.

The project uses the RSNA Pneumonia Detection Challenge dataset. The model predicts whether a chest X-ray shows lung opacity, which may be associated with pneumonia but is not a definitive diagnosis.

**Disclaimer:** This application is intended for educational and demonstration purposes only. It is not a medical diagnostic tool.

## Project Objectives

- Perform exploratory data analysis on chest X-ray images.
- Preprocess DICOM images for deep learning.
- Build and evaluate a baseline CNN.
- Implement transfer learning using MobileNetV2 and ResNet50.
- Compare original and modified transfer-learning architectures.
- Select the best-performing model using validation ROC-AUC.
- Develop an interactive prediction interface using Gradio.
- Package the application for Docker deployment.

## Dataset

**Source:** RSNA Pneumonia Detection Challenge

The available dataset contains 5,336 unique chest X-ray images.

For transfer-learning experiments, a stratified subset of 1,500 images was used.

| Split | Images |
|---|---:|
| Training | 1,050 |
| Validation | 225 |
| Testing | 225 |

Images were resized to 128 × 128 pixels and normalized.

## Models Developed

1. Baseline CNN
2. MobileNetV2 — Original
3. MobileNetV2 — Modified
4. ResNet50 — Original
5. ResNet50 — Modified

ResNet50 Original was selected based on the highest validation ROC-AUC among the four transfer-learning models.

## Final Selected Model Performance

The selected ResNet50 Original model achieved the following results on the held-out test subset:

| Metric | Score |
|---|---:|
| Accuracy | 76.44% |
| Precision | 47.54% |
| Recall | 58.00% |
| F1-score | 52.25% |
| ROC-AUC | 79.91% |

These results are experimental and do not establish clinical reliability.

## Application Features

The Gradio application allows users to:

- Upload a chest X-ray image.
- Generate a lung-opacity prediction.
- View the model's prediction scores.

The application was tested successfully in Google Colab.

## Technologies Used

- Python
- TensorFlow / Keras
- MobileNetV2 and ResNet50
- NumPy
- Pillow
- Gradio
- Docker
- Google Colab
- GitHub

## Docker Deployment

The repository is intended to contain the following deployment files:

```text
pneumonia-detection-deployment/
├── app.py
├── best_model.keras
├── requirements.txt
├── Dockerfile
├── .dockerignore
└── README.md
```

Once all deployment files are uploaded, the Docker image can be built with:

```bash
docker build -t pneumonia-detection .
```

Run the application using:

```bash
docker run -p 7860:7860 pneumonia-detection
```

Then open `http://localhost:7860` to access the application.

**Deployment status:** The Gradio interface has been tested in Google Colab. Docker image building and container testing are pending.

## Limitations

- The transfer-learning experiments used a subset of
