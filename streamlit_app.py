
import streamlit as st
import pandas as pd
import pickle
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler

# --- 1. Muat Model dan Scaler yang Telah Disimpan ---
def load_model_and_scalers():
    try:
        with open('best_model_gradient_boosting.pkl', 'rb') as model_file:
            loaded_model = pickle.load(model_file)
        with open('scaler.pkl', 'rb') as scaler_file:
            loaded_scaler = pickle.load(scaler_file)
        with open('scaler_y.pkl', 'rb') as scaler_file:
            scaler_y = pickle.load(scaler_file)
        return loaded_model, loaded_scaler, scaler_y
    except FileNotFoundError:
        st.error("Pastikan semua file model (.pkl) ada di direktori yang sama.")
        st.stop()
    except Exception as e:
        st.error(f"Terjadi kesalahan saat memuat file: {e}")
        st.stop()

loaded_model, loaded_scaler, scaler_y = load_model_and_scalers()

# --- 2. Fungsi Preprocessing Data Input Baru ---
def preprocess_input(kategori, nama_produk, jumlah_beli, metode_bayar, kota_pengiriman):
    # Create a DataFrame from new input
    input_df = pd.DataFrame([{
        'Kategori': kategori,
        'Nama_Produk': nama_produk,
        'Jumlah_Beli': jumlah_beli,
        'Metode_Bayar': metode_bayar,
        'Kota_Pengiriman': kota_pengiriman
    }])

    # Define unique values from the original training data (Hardcoded for deployment)
    # In a real app, these would be loaded from a config or fitted on a sample data subset if not saved.
    unique_nama_produk = ['Kemeja Flanel', 'Hand Sanitizer', 'Headset Gaming', 'Jaket Bomber', 'Masker Medis', 'Laptop', 'Mouse Wireless', 'Powerbank', 'Celana Chino', 'Sepatu Sneakers', 'Termometer', 'Vitamin C']
    unique_metode_bayar = ['COD', 'E-Wallet', 'Kartu Kredit', 'Transfer Bank']
    unique_kota_pengiriman = ['Bandung', 'Denpasar', 'Jakarta', 'Makassar', 'Medan', 'Semarang', 'Surabaya', 'Yogyakarta']
    unique_kategori = ['Elektronik', 'Fashion', 'Kesehatan']

    # Label Encoding
    le_nama_produk = LabelEncoder()
    le_metode_bayar = LabelEncoder()
    le_kota_pengiriman = LabelEncoder()

    # Fit LabelEncoders with all unique values they saw during training
    le_nama_produk.fit(unique_nama_produk)
    le_metode_bayar.fit(unique_metode_bayar)
    le_kota_pengiriman.fit(unique_kota_pengiriman)

    input_df['Nama_Produk'] = le_nama_produk.transform(input_df['Nama_Produk'])
    input_df['Metode_Bayar'] = le_metode_bayar.transform(input_df['Metode_Bayar'])
    input_df['Kota_Pengiriman'] = le_kota_pengiriman.transform(input_df['Kota_Pengiriman'])

    # One-Hot Encoding for 'Kategori'
    input_df['Kategori_Elektronik'] = (input_df['Kategori'] == 'Elektronik').astype(int)
    input_df['Kategori_Fashion'] = (input_df['Kategori'] == 'Fashion').astype(int)
    input_df['Kategori_Kesehatan'] = (input_df['Kategori'] == 'Kesehatan').astype(int)
    input_df = input_df.drop(columns=['Kategori'])

    # Ensure column order matches training data
    # This list must be exactly the same as `feature_cols` used during training
    expected_columns_order = [
        'Jumlah_Beli',
        'Nama_Produk',
        'Metode_Bayar',
        'Kota_Pengiriman',
        'Kategori_Elektronik',
        'Kategori_Fashion',
        'Kategori_Kesehatan'
    ]
    input_df = input_df[expected_columns_order]

    # Scale features
    input_df_scaled = loaded_scaler.transform(input_df)

    return pd.DataFrame(input_df_scaled, columns=expected_columns_order)

# --- 3. Streamlit App Layout ---
st.title('Prediksi Total Pembayaran E-commerce')
st.write('Aplikasi ini memprediksi total pembayaran berdasarkan detail transaksi.')

# Input fields
with st.sidebar:
    st.header('Input Data Transaksi')
    # Use unique values from the original df to populate selectbox options
    # Hardcoded for deployment, in a real app these lists would be passed or loaded.
    unique_kategori = ['Elektronik', 'Fashion', 'Kesehatan']
    unique_nama_produk = ['Kemeja Flanel', 'Hand Sanitizer', 'Headset Gaming', 'Jaket Bomber', 'Masker Medis', 'Laptop', 'Mouse Wireless', 'Powerbank', 'Celana Chino', 'Sepatu Sneakers', 'Termometer', 'Vitamin C']
    unique_metode_bayar = ['COD', 'E-Wallet', 'Kartu Kredit', 'Transfer Bank']
    unique_kota_pengiriman = ['Bandung', 'Denpasar', 'Jakarta', 'Makassar', 'Medan', 'Semarang', 'Surabaya', 'Yogyakarta']

    kategori_input = st.selectbox('Kategori Produk', unique_kategori)
    nama_produk_input = st.selectbox('Nama Produk', unique_nama_produk)
    jumlah_beli_input = st.number_input('Jumlah Beli', min_value=1, max_value=10, value=1)
    metode_bayar_input = st.selectbox('Metode Pembayaran', unique_metode_bayar)
    kota_pengiriman_input = st.selectbox('Kota Pengiriman', unique_kota_pengiriman)

    predict_button = st.button('Prediksi Total Bayar')

# --- 4. Logic Prediksi ---
if predict_button:
    # Preprocess the input data
    processed_input = preprocess_input(
        kategori_input,
        nama_produk_input,
        jumlah_beli_input,
        metode_bayar_input,
        kota_pengiriman_input
    )

    # Make prediction
    predicted_total_bayar_scaled = loaded_model.predict(processed_input)

    # Inverse transform the prediction to original scale
    predicted_total_bayar = scaler_y.inverse_transform(predicted_total_bayar_scaled.reshape(-1, 1))[0][0]

    st.subheader('Hasil Prediksi')
    st.success(f"Total Pembayaran yang diprediksi: Rp {predicted_total_bayar:,.2f}")

