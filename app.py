import streamlit as st
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

# Load model
MODEL_PATH = "model.h5"
model = load_model(MODEL_PATH)

# Define pages
def home_page():
    st.title("Welcome to the Machine Learning App")
    st.write(
        """
        Aplikasi ini memanfaatkan model machine learning untuk berbagai kebutuhan:
        """
    )
    st.write(
        """
        ### Fitur
        1. **EDA**: Eksplorasi data secara interaktif.
        2. **Prediksi Dataset**: Unggah file CSV untuk prediksi.
        3. **Prediksi Input Link**: Masukkan satu link untuk diprediksi.
        """
    )

def eda_page():
    st.title("Exploratory Data Analysis (EDA)")
    st.write("Unggah dataset Anda untuk analisis.")

    uploaded_file = st.file_uploader("Unggah file CSV", type=["csv"])
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        st.write("## Tinjauan Data")
        st.write(df.head())
        st.write("## Statistik Data")
        st.write(df.describe())
        st.write("## Informasi Dataset")
        st.write(df.info())

        # Visualisasi
        st.write("## Visualisasi Data")
        numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
        if numeric_cols:
            selected_col = st.selectbox("Pilih kolom untuk histogram", numeric_cols)
            fig, ax = plt.subplots()
            sns.histplot(df[selected_col], bins=20, kde=True, ax=ax)
            st.pyplot(fig)
        else:
            st.write("Tidak ada kolom numerik untuk divisualisasikan.")

def prediction_dataset_page():
    st.title("Machine Learning Prediction - Dataset")
    st.write("Unggah dataset untuk prediksi.")

    uploaded_file = st.file_uploader("Unggah file CSV", type=["csv"])
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        st.write("Dataset Anda:")
        st.write(df.head())

        # Preprocessing data
        st.write("### Preprocessing Data")
        scaler = StandardScaler()
        try:
            features = df.values
            features_scaled = scaler.fit_transform(features)
            st.write("Data telah dinormalisasi.")
        except Exception as e:
            st.write("Error dalam preprocessing data:", e)
            return

        # Prediction
        st.write("### Hasil Prediksi")
        predictions = model.predict(features_scaled)
        st.write("Prediksi:")
        st.write(predictions)

        # Add predictions to the dataset
        df["Prediction"] = predictions
        st.write("Dataset dengan Prediksi:")
        st.write(df)

        # Option to download the dataset
        csv = df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="Unduh Hasil Prediksi",
            data=csv,
            file_name="predicted_dataset.csv",
            mime="text/csv",
        )

def prediction_link_page():
    st.title("Machine Learning Prediction - Input Link")
    st.write("Masukkan link untuk prediksi.")

    # Input link
    input_link = st.text_input("Masukkan link di sini:")
    if input_link:
        st.write(f"Link yang dimasukkan: {input_link}")

        # Preprocessing link into a feature vector (dummy example here)
        # Replace this section with your actual feature extraction logic
        st.write("### Preprocessing Link")
        try:
            # Dummy feature extraction: convert link to vector
            # Example: length of link and number of slashes
            features = np.array([
                len(input_link),
                input_link.count("/")
            ]).reshape(1, -1)

            scaler = StandardScaler()
            features_scaled = scaler.fit_transform(features)

            st.write("Fitur yang dihasilkan:")
            st.write(features_scaled)
        except Exception as e:
            st.write("Error dalam preprocessing data:", e)
            return

        # Prediction
        st.write("### Hasil Prediksi")
        predictions = model.predict(features_scaled)
        st.write("Prediksi:")
        st.write(predictions)

# Page routing
PAGES = {
    "Home": home_page,
    "EDA": eda_page,
    "Prediction - Dataset": prediction_dataset_page,
    "Prediction - Link": prediction_link_page,
}

def main():
    st.sidebar.title("Navigasi")
    selection = st.sidebar.radio("Pilih halaman", list(PAGES.keys()))
    page = PAGES[selection]
    page()

if __name__ == "__main__":
    main()
