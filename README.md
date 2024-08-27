# Seria-A Match Outcome Prediction
## Project Overview
This project focuses on predicting the outcomes of Serie A football matches using various machine learning models, including Random Forest, Gradient Boosting, and Neural Networks. The dataset used contains historical match data, and the target variable is the Full-Time Result (FTR) of each match.
## Dataset
The dataset (seria_a.csv) contains the following key features:
* HomeTeam: Name of the home team.
* AwayTeam: Name of the away team.
* Date: Date of the match.
* FTR: Full-Time Result, which is the target variable to predict (Home Win, Draw, Away Win).
* Numerical features: Various match statistics.
* Categorical features: Various other match-related attributes.
## Preprocessing Steps
### 1. Missing Values:
* Numeric columns: Filled missing values with the median.
* Non-numeric columns: Filled missing values with the mode.
* Rows with missing target values (FTR) were dropped.
### 2. Feature Engineering:
* Extracted the day of the week from the match date.
* Dropped irrelevant columns based on certain keywords (e.g., B365, Max, Avg).
* Applied one-hot encoding to categorical features (HomeTeam, AwayTeam, HTR, matchDay).
* Label-encoded the target variable (FTR).
### 3. Data Normalisation:
* Standardized the numerical features using StandardScaler.
### 4. Dimensionality Reduction:
* Applied Principal Component Analysis (PCA) to reduce the feature dimensions to 60 components.
## Models Used
### 1. Random Forest Classifier
* Parameters: n_estimators=200
* Performance: The model was trained on 80% of the data, and accuracy was evaluated on the remaining 20%.
### 2. Gradient Boost CLassifier
* Parameters: n_estimators=200
* Performance: Similar to the Random Forest model, it was trained and tested on the same split.
### 3. Neural Networks
* Architecture:
* Multiple dense layers with ReLU activation.
* Dropout layers for regularization.
* Softmax output layer for multi-class classification.
* Optimizers: Adam, AdamW
* Loss Function: Categorical Crossentropy
* Callbacks: Early stopping, learning rate reduction on plateau.
* Training: The model was trained with different batch sizes and for up to 150 epochs, with 20% validation split.
## Input and Output
* Input: The input to the models is a set of preprocessed features extracted from the dataset. This includes numerical features that have been standardized and categorical features that have been one-hot encoded.
* Output: The models output the predicted class label for the Full-Time Result (FTR), which could be either Home Win, Draw, or Away Win.
## Parameters of Defined Functions
* Random Forest Classifier:
* n_estimators: Number of trees in the forest. Default is 200.
* Gradient Boosting Classifier:
* n_estimators: Number of boosting stages to be run. Default is 200.
* Neural Network Models:
* units: Number of neurons in each dense layer.
* activation: Activation function used in the dense layers. Default is 'relu'.
* dropout: Fraction of the input units to drop for the next layer.
* kernel_regularizer: Regularizer function applied to the kernel weights matrix.
## Results and Visualization
* The models' performance is evaluated using metrics such as accuracy, confusion matrix, and classification report.
* Training and validation accuracy and loss are plotted for the neural network models.
* The confusion matrix is visualized using a heatmap for both the Random Forest and Neural Network models.
  
  
