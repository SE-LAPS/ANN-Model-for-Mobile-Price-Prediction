# Mobile Price Classification using Artificial Neural Network

This project implements an Artificial Neural Network (ANN) to classify mobile phone prices based on various features. The model is built using TensorFlow and trained on the Mobile Price Classification dataset.

## Table of Contents

- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Model Architecture](#model-architecture)
- [Results](#results)
- [Viewing Model Weights](#viewing-model-weights)
- [License](#license)

## Prerequisites

- Python 3.7+
- pip

## Installation

1. Clone this repository:
https://github.com/SE-LAPS/ANN-Model-for-Mobile-Price-Prediction

2. Install the required packages:
pip install -r requirements.txt

## Usage

1. Ensure you have the `Mobile_Price_Classification.csv` file in the project directory.

2. Run the main script:
python mobile_price_classification.py

3. The script will train the model and save the weights to `ann_model_weights.weights.h5`.

## Project Structure

mobile-price-classification/
│
├── mobile_price_classification.py  # Main script for training the model
├── view_h5.py                      # Script to view the contents of the .h5 file
├── Mobile_Price_Classification.csv # Dataset (not included in repo)
├── requirements.txt                # Required Python packages
└── README.md                       # This file

## Model Architecture

The ANN model consists of:
- Input layer: 20 features
- First hidden layer: 8 neurons with ReLU activation
- Second hidden layer: 4 neurons with ReLU activation
- Output layer: 1 neuron with Sigmoid activation

## Results

After training, the model achieves:
- Loss: [Your model's loss]
- Accuracy: [Your model's accuracy]

## Viewing Model Weights

To view the contents of the saved model weights:

1. Ensure you have `h5py` installed:
pip install h5py

2. Run the `view_h5.py` script:
python view_h5.py

This will display the structure and contents of the `ann_model_weights.weights.h5` file.

## License

This project is licensed under the MIT License - see the [LICENSE](https://bit.ly/Lahiru_Senavirathna) file for details.
