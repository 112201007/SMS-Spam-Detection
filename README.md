# SMS Spam Detection

## Overview

The SMS Spam Detection project is a machine learning-based system that predicts whether an SMS message is spam or not spam. The model uses Python and is deployed on the web using Streamlit, allowing users to input messages and receive predictions.

## Technology Used

- Python
- Scikit-learn
- Pandas
- NumPy
- Streamlit

## Features

- Data Collection: Collecting and preparing the dataset for training the model.
- Data Cleaning and Preprocessing: Handling missing data, cleaning text, and transforming it for model input.
- Exploratory Data Analysis (EDA): Analyzing and visualizing the dataset to gain insights.
- Model Building and Evaluation: Training and evaluating multiple models for spam detection.
- Web Deployment: Deploying the model using Streamlit to allow users to input SMS and receive predictions.

## Data Collection

The dataset used for this project is the SMS Spam Collection dataset, available from Kaggle. It contains over 5,500 SMS messages, each labeled as either spam or not spam.

https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset

## Data Cleaning and Preprocessing

The dataset was cleaned by:
- Handling missing values.
- Removing duplicate entries.
- Encoding labels.
- Preprocessing the text by:
  - Converting it to lowercase.
  - Tokenizing and stemming the text.
  - Removing special characters, stop words, and punctuation.

## Exploratory Data Analysis (EDA)

EDA was performed to understand the dataset better:
- Analyzed the count of characters, words, and sentences for each message.
- Created visualizations like bar charts, pie charts, word clouds, and heatmaps.
- Identified patterns in spam and non-spam messages.

## Model Building and Evaluation

Several machine learning classifiers were tested, including:
- Naive Bayes
- Random Forest
- KNN
- Decision Tree
- Logistic Regression
- Extra Trees Classifier
- Support Vector Classifier (SVC)

The model with the highest precision was selected, achieving 100% precision.

## Web Deployment

The model was deployed using Streamlit. The web app allows users to input an SMS message, and the model predicts whether it is spam or not spam.

You can try out the deployed model here:  

## Usage

To use the SMS Spam Detection model on your local machine, follow these steps:

1. **Clone the repository**:

   git clone [repository link]
   
   
2. **Install the required Python packages**:
   
   pip install -r requirements.txt
   

3. **Run the Streamlit app**:

   streamlit run app.py
   

4. **Visit** `localhost:8501` in your web browser to use the app.

## Contributions

If you find any issues or have suggestions for improvement,open an issue or a pull request.

