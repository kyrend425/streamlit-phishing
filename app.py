import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix

# Title
st.title('Phishing Website Detection')

# Upload file
uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
if uploaded_file is not None:
    # Load data
    df = pd.read_csv(uploaded_file)
    st.write(df.head())

    # Preprocessing: Handle missing data, encode categorical columns, etc.
    df.fillna(df.mean(), inplace=True)  # Example of handling missing data

    # Example of feature scaling for machine learning models
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(df.select_dtypes(include=[np.number]))

    # Train-test split
    X = scaled_data
    y = df['label']  # Assuming 'label' is the target column
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Model: Logistic Regression (or you can use another simple model)
    model = LogisticRegression()
    model.fit(X_train, y_train)

    # Predict and evaluate
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    st.write(f"Accuracy: {accuracy * 100:.2f}%")

    # Show confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    st.write("Confusion Matrix:")
    st.write(cm)

    # Optional: Plot confusion matrix
    fig, ax = plt.subplots()
    ax.matshow(cm, cmap='Blues')
    for (i, j), value in np.ndenumerate(cm):
        ax.text(j, i, value, ha='center', va='center')
    st.pyplot(fig)
