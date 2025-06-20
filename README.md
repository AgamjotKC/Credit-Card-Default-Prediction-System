# Credit-Card-Default-Prediction-System

A Streamlit web app that predicts whether a credit card user is likely to default based on their financial profile.

## Model

- Machine Learning model built with scikit-learn
- Features include payment history, bill amounts, and demographic data

## Files

- `app.py`: Streamlit app
- `model.save`: Trained ML model (joblib)
- `scaler.save`: Fitted StandardScaler

# Dataset
The model is trained on a dataset containing customers' default payments in Taiwan and compares the predictive accuracy of probability of default among six data mining methods. 
Link to dataset: https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients
## Run the App

To run locally:
```bash
streamlit run app.py
