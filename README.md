# cancerdl
# Cancer Prediction Using Deep Learning

## Overview
This project uses a deep learning model to predict the likelihood of cancer recurrence based on patient data. The model is trained on a breast cancer dataset, leveraging TensorFlow/Keras for implementation. The workflow includes data preprocessing, model training, evaluation, and prediction.

## Features
- **Data Preprocessing**: Handles categorical data, encodes target values, and normalizes features.
- **Deep Learning Model**: Implements a neural network with multiple dense layers and dropout for regularization.
- **Early Stopping**: Prevents overfitting by monitoring validation loss during training.
- **Model Evaluation**: Provides metrics such as accuracy, classification report, and confusion matrix.
- **Model Saving**: Saves the trained model for reuse.

## Dataset
The dataset includes features such as:
- `age`: Age group of the patient.
- `tumor-size`: Size of the tumor.
- `deg-malig`: Degree of malignancy.
- `breast`: Breast affected (left or right).
- `breast-quad`: Breast quadrant affected.
- `irradiat`: History of radiation therapy.

Target column:
- `class`: Indicates whether the patient experienced cancer recurrence (`recurrence-events`) or not (`false-recurrence-events`).

## Requirements
Install the following Python libraries before running the project:
```bash
pip install numpy pandas scikit-learn tensorflow matplotlib
```

## Project Structure
- `cancer_prediction_dl.py`: Main script containing the code for preprocessing, model training, and evaluation.
- `breast-cancer.csv`: Dataset file used for training and testing.
- `cancer_prediction_model.h5`: Saved model after training.

## Usage
1. **Load Dataset**:
   Replace the dataset path in the script if necessary.
   ```python
   data = pd.read_csv('/path/to/breast-cancer.csv')
   ```

2. **Run the Script**:
   Execute the script to preprocess data, train the model, and evaluate performance.
   ```bash
   python cancer_prediction_dl.py
   ```

3. **Evaluate Model**:
   Review the test loss, accuracy, classification report, and confusion matrix output to analyze the model's performance.

4. **Save and Reuse Model**:
   The trained model is saved as `cancer_prediction_model.h5` and can be loaded for future predictions.

## Evaluation Metrics
- **Accuracy**: Measures the proportion of correctly classified instances.
- **Classification Report**: Displays precision, recall, and F1-score.
- **Confusion Matrix**: Provides a detailed breakdown of true positives, true negatives, false positives, and false negatives.

## Customization
- **Hyperparameter Tuning**: Modify the neural network architecture, learning rate, or dropout rates to optimize performance.
- **Cross-Validation**: Implement k-fold cross-validation for robust evaluation.
- **Feature Engineering**: Add or remove features based on domain knowledge for improved predictions.

## Future Work
- Incorporate additional features or datasets for a more comprehensive analysis.
- Use advanced architectures like CNNs or transformers for feature extraction and prediction.
- Implement explainable AI (XAI) techniques to interpret predictions.



