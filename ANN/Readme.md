# ANN (CHURN PREDICTION)

This project uses an Artificial Neural Network (ANN) to predict customer churn. The notebook demonstrates the implementation of an ANN for churn prediction using a dataset of customer information.

## Table of Contents
- [Introduction](#introduction)
- [Dataset](#dataset)
- [Model Architecture](#model-architecture)
- [Training](#training)
- [Evaluation](#evaluation)
- [Results](#results)
- [Installation](#installation)
- [Usage](#usage)
- [Contributing](#contributing)
- [License](#license)

## Introduction
Customer churn prediction is crucial for businesses to retain customers. This project builds an ANN model to predict whether a customer will churn based on various features.

## Dataset
The dataset includes customer information such as tenure, services availed, contract type, payment method, and more. The target variable is whether the customer churned.

## Model Architecture
The model is a fully connected ANN with multiple hidden layers. It uses ReLU activation functions for hidden layers and a sigmoid activation function for the output layer.

## Training
The model is trained using the Adam optimizer and binary cross-entropy loss function. Training involves splitting the data into training and validation sets and applying early stopping to prevent overfitting.

## Evaluation
The model's performance is evaluated using accuracy, precision, recall, and F1-score metrics. The results show the model's effectiveness in predicting customer churn.

## Results
- **Training Accuracy:** High accuracy on the training dataset.
- **Validation Accuracy:** Good generalization with high accuracy on the validation dataset.
- Detailed evaluation metrics and confusion matrix are provided in the notebook.

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/SheemaMasood381/Deep-Learning.git
   cd Deep-Learning/ANN

2. Create a virtual environment (optional but recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`

3. Install the required dependencies:
   ```bash
   pip install -r requirements.txt

## Usage

Open the Jupyter notebook `01_churn-mpdeling-ANN sheema masood.ipynb` and run the cells to train and evaluate the model.

## Contributing

Contributions are welcome! If you have any improvements or new features to suggest, please open a pull request.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.
