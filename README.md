# README for FindIt Project

## Project Overview
This project focuses on predicting COPPA (Children's Online Privacy Protection Act) risk for mobile applications based on various features such as developer information, user ratings, download statistics, and privacy-related attributes. The goal is to classify whether an app poses a risk to children's privacy (True/False).

## Dataset
The dataset consists of three main files:
- `train.csv`: Contains 7,000 records with 16 features
- `test.csv`: Contains 3,000 records with 17 features (includes an ID column)
- `target.csv`: Contains the COPPA risk labels (True/False) for the training data

## Data Preprocessing
Key preprocessing steps included:
1. Handling missing values:
   - Numerical columns: Imputed with mean values
   - Categorical columns: Imputed with most frequent values
2. Feature encoding:
   - Categorical features were one-hot encoded
3. Train-test split:
   - 80% training data, 20% validation data

## Model Architecture
The model uses a pipeline approach with:
1. Preprocessing:
   - ColumnTransformer for numerical and categorical features
   - SimpleImputer for missing values
   - OneHotEncoder for categorical features
2. Classifier:
   - RandomForestClassifier with 100 estimators

## Results
The model achieved an accuracy of **90.64%** on the validation set.

## Submission File
The predictions for the test set have been saved in `submission.csv` with the following format:
- First column: App ID
- Second column: Predicted COPPA risk (True/False)

## How to Run
1. Install required packages:
   ```bash
   pip install pandas scikit-learn xgboost
   ```
2. Run the notebook `findIt.ipynb` to train the model and generate predictions
3. The submission file will be created automatically

## Prediction Results
The model predicted that **all apps in the test set are NOT at risk for COPPA violations** (all predictions are False). This suggests that either:
1. The test set consists of apps that genuinely don't pose COPPA risks, or
2. The model may be biased toward predicting "False" due to class imbalance in the training data
