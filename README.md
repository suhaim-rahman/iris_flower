# Iris Species Prediction

This repository contains a simple machine learning model to predict the species of Iris flowers based on their features. The model uses the K-Nearest Neighbors (KNN) algorithm.

## Dataset

The dataset used for this project is `IRIS.csv`, which contains the following columns:

- `sepal_length`: Sepal length of the flower
- `sepal_width`: Sepal width of the flower
- `petal_length`: Petal length of the flower
- `petal_width`: Petal width of the flower
- `species`: Species of the flower

## Requirements

- Python 3.6+
- pandas
- scikit-learn

You can install the required libraries using the following command:

## The script performs the follwing steps:
-1. Loads the dataset and displays basic details.
-2. Separates the features and target variable.
-3. Splits the data into training and testing sets.
-4. Scales the features using StandardScaler.
-5. Trains a K-Nearest Neighbors (KNN) model with 3 neighbors.
-6. Makes predictions on the test set.
=7. Calculates and displays the accuracy of the model.
-8. Provides an example prediction for a new flower.

## Code Overview
-The iris_prediction.py script contains the following main sections:

1. Data Loading: Load the dataset and display basic details.
2. Data Preparation: Separate features and target, and split the data into training and testing sets.
3. Feature Scaling: Scale the features using StandardScaler.
4. Model Training: Train a K-Nearest Neighbors (KNN) model.
5. Model Evaluation: Evaluate the model using accuracy score.
6. Example Prediction: Make a prediction for a new flower based on its features.

#Results
After running the script, you will see the accuracy of the model and the predicted species for a new flower with given features.

#Contributing
If you would like to contribute to this project, please open an issue or submit a pull request.

#License
This project is licensed under the MIT License. See the LICENSE file for details.

#Acknowledgements
Special thanks to the creators of the dataset and the developers of the libraries used in this project.
