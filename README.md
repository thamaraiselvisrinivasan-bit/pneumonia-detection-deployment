# Chest X-Ray Pneumonia Detection Using Deep Learning

## Project Overview

This project develops and evaluates deep learning models for identifying lung opacity in chest X-ray images using a baseline Convolutional Neural Network (CNN) and transfer learning.

The project uses images and labels from the RSNA Pneumonia Detection Challenge. Lung opacity may be associated with pneumonia, but its presence alone does not establish a pneumonia diagnosis.

**Disclaimer:** This project is intended strictly for educational, research, and demonstration purposes. It is not a clinically validated medical diagnostic tool.

## Project Objectives

- Explore and analyze chest X-ray images and class distributions.
- Preprocess DICOM images for deep learning.
- Build and evaluate a baseline CNN.
- Implement MobileNetV2 and ResNet50 transfer-learning architectures.
- Compare original and modified classifier configurations.
- Select the best model using validation ROC-AUC.
- Evaluate the selected model on a held-out test dataset.
- Develop an interactive prediction application using Gradio.
- Prepare application files for Docker deployment.

## Dataset

**Source:** RSNA Pneumonia Detection Challenge

The project uses 5,336 unique labeled chest X-ray images.

### Dataset Split

The dataset was divided using stratified sampling.

| Dataset | Images | Percentage |
|---|---:|---:|
| Training | 3,735 | 70.00% |
| Validation | 800 | 14.99% |
| Testing | 801 | 15.01% |
| **Total** | **5,336** | **100%** |

### Preprocessing

- Read original grayscale DICOM images.
- Resize images to 128 × 128 pixels.
- Normalize pixel intensities to the range 0–1.
- Prepare image arrays for CNN and transfer-learning models.
- Save preprocessed data in compressed NPZ batches to support memory-efficient processing.

The prediction target is binary:

- **0:** No labeled lung opacity
- **1:** Lung opacity present

## Models Developed

Five model configurations were evaluated:

1. Baseline CNN
2. MobileNetV2 — Original
3. MobileNetV2 — Modified
4. ResNet50 — Original
5. ResNet50 — Modified

The transfer-learning models used pretrained ImageNet feature extractors with classification heads trained for binary prediction.

## Transfer-Learning Model Comparison

The following results were obtained on the validation dataset.

| Model | Accuracy | ROC-AUC |
|---|---:|---:|
| **MobileNetV2 Original** | **77.25%** | **0.8421** |
| ResNet50 Modified | 77.25% | 0.8186 |
| ResNet50 Original | 75.00% | 0.8056 |
| MobileNetV2 Modified | 74.75% | 0.7837 |

**MobileNetV2 Original** was selected because it achieved the highest validation ROC-AUC among the evaluated transfer-learning configurations.

## Final Selected Model Performance

The selected MobileNetV2 Original model was evaluated on the held-out test dataset of 801 images.

| Metric | Score |
|---|---:|
| Accuracy | 78.53% |
| Precision | 51.84% |
| Recall | 70.17% |
| F1-score | 59.62% |
| ROC-AUC | 0.8411 |

### Test Confusion Matrix

| | Predicted Negative | Predicted Positive |
|---|---:|---:|
| Actual Negative | 502 | 118 |
| Actual Positive | 54 | 127 |

The model demonstrates useful experimental discrimination between the two labeled categories, but false-positive and false-negative predictions remain important limitations.

These results do not establish clinical reliability.

## Application Features

An interactive Gradio application was developed to:

- Upload a chest X-ray image.
- Generate a binary lung-opacity prediction.
- Display model prediction scores.

The Gradio interface was launched and tested successfully in Google Colab using the selected model.

## Technologies Used

- Python
- TensorFlow / Keras
- MobileNetV2
- ResNet50
- NumPy
- Pillow
- Gradio
- Google Colab
- Docker
- GitHub

## Repository Structure

```text
pneumonia-detection-deployment/
├── notebooks/
│   └── Pneumonia_Detection_GitHub_Safe.ipynb
├── reports/
│   └── Thamarai_Selvi_S_Pneumonia_Detection_FinalReport(1).pdf
├── app.py
├── best_model.keras
├── requirements.txt
├── Dockerfile
├── .dockerignore
└── README.md
```

The `notebooks` folder contains the project notebook, while `reports` contains the final project documentation.

The deployment files at the repository root support the Gradio application.

## Docker Deployment

The repository includes a Dockerfile and supporting application files.

To build the Docker image:

```bash
docker build -t pneumonia-detection .
```

To run the application:

```bash
docker run -p 7860:7860 pneumonia-detection
```

If the container starts successfully, open:

`http://localhost:7860`

**Deployment status:** The Gradio application has been tested in Google Colab. Docker image building and container execution have not yet been verified.

## Limitations

- The dataset contains lung-opacity labels, which do not independently confirm pneumonia.
- The dataset has class imbalance.
- The model can generate false-positive and false-negative predictions.
- The images were resized to 128 × 128 pixels, potentially reducing fine anatomical detail.
- Generalization to images from other hospitals, equipment, and patient populations has not been established.
- Clinical validation has not been performed.
- Docker container execution remains unverified.

## Future Enhancements

- Evaluate performance using additional independent datasets.
- Investigate fine-tuning and additional model architectures.
- Improve sensitivity, precision, and calibration.
- Explore explainable AI techniques such as Grad-CAM.
- Test Docker deployment and application reproducibility.
- Conduct further validation before considering any clinical research use.

## Conclusion

This project demonstrates an end-to-end deep learning workflow for chest X-ray lung-opacity classification, including exploratory data analysis, preprocessing, CNN development, transfer-learning comparison, final model evaluation, and an interactive Gradio application.

MobileNetV2 Original achieved the highest validation ROC-AUC among the evaluated transfer-learning models and produced a test ROC-AUC of 0.8411.

The application provides an educational demonstration of AI-based medical image classification, with further validation required before any real-world medical use.
