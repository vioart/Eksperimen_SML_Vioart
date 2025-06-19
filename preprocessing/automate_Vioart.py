"""
automate_Vioart.py

File ini berisi fungsi-fungsi untuk preprocessing otomatis dataset Medical Learning Resources
dari Kaggle (https://www.kaggle.com/datasets/gopikrishnan2005/medical-learning-resources).
Preprocessing menghasilkan data yang siap untuk pelatihan model machine learning, seperti
vektorisasi teks untuk rekomendasi sumber daya pembelajaran medis.
"""

import pandas as pd
import re

def load_and_check_dataset(file_path):
    """
    Memuat dataset dari file CSV dan memeriksa informasi dasar.
    
    Args:
        file_path (str): Path ke file CSV.
    
    Returns:
        pd.DataFrame: Dataset yang dimuat.
    """
    df = pd.read_csv(file_path)
    print("Informasi Dataset:")
    print(df.info())
    print("\nJumlah Baris dan Kolom:", df.shape)
    print("\nJumlah Nilai Null:\n", df.isnull().sum())
    print("\nJumlah Data Duplikat:", df.duplicated().sum())
    return df

def drop_constant_features(df):
    """
    Menghapus kolom yang konstan atau hampir konstan (>95% nilai sama).
    
    Args:
        df (pd.DataFrame): Dataset input.
    
    Returns:
        pd.DataFrame: Dataset tanpa kolom konstan.
    """
    constant_cols = []
    for col in df:
        top_pct = df[col].value_counts(normalize=True).max()
        if top_pct > 0.95:
            constant_cols.append(col)
            print(f"Menghapus kolom konstan: {col} ({top_pct:.2%})")
    
    # Menghapus kolom yang diidentifikasi (misalnya, 'archived')
    df = df.drop(columns=constant_cols, errors='ignore')
    
    # Menghapus kolom metadata tambahan (seperti di eksperimen)
    metadata_cols = ['record_modified', 'resource_revised']
    df = df.drop(columns=[col for col in metadata_cols if col in df.columns], errors='ignore')
    
    print("Kolom yang dihapus:", constant_cols + metadata_cols)
    return df

def fill_missing_values(df):
    """
    Mengisi nilai kosong (NaN/NaT) dengan string kosong ('').
    
    Args:
        df (pd.DataFrame): Dataset input.
    
    Returns:
        pd.DataFrame: Dataset dengan nilai kosong diisi.
    """
    df = df.fillna('')
    print("Nilai kosong telah diisi dengan string kosong.")
    return df

def preprocess_individual_text(text_input):
    """
    Membersihkan dan menstandarisasi teks input.
    
    Args:
        text_input (any): Input teks (akan dikonversi ke string).
    
    Returns:
        str: Teks yang telah dibersihkan.
    """
    text = str(text_input)
    text = re.sub(r'\W', ' ', text)  # Hapus karakter non-alfanumerik
    text = re.sub(r'\s+[a-zA-Z]\s+', ' ', text)  # Hapus karakter tunggal diapit spasi
    text = re.sub(r'^[a-zA-Z]\s+', ' ', text)  # Hapus karakter tunggal di awal
    text = re.sub(r'\s+', ' ', text, flags=re.I)  # Ganti spasi ganda dengan tunggal
    text = re.sub(r'^b\s+', '', text)  # Hapus prefix 'b'
    text = text.lower().strip()  # Huruf kecil dan hapus spasi awal/akhir
    return text

def preprocess_text_columns(df, text_columns):
    """
    Menerapkan preprocessing teks ke kolom-kolom tertentu.
    
    Args:
        df (pd.DataFrame): Dataset input.
        text_columns (list): Daftar kolom teks yang akan diproses.
    
    Returns:
        pd.DataFrame: Dataset dengan kolom teks yang telah diproses.
    """
    for col in text_columns:
        if col in df.columns:
            df[col] = df[col].apply(preprocess_individual_text)
            print(f"Kolom {col} telah diproses.")
    return df

def create_content_soup(df):
    """
    Menggabungkan kolom teks menjadi kolom 'content_soup'.
    
    Args:
        df (pd.DataFrame): Dataset input.
    
    Returns:
        pd.DataFrame: Dataset dengan kolom baru 'content_soup'.
    """
    def combine_text(row):
        subject_areas = " ".join(row["subject_areas"].replace(';', ' ').split())
        intended_audiences = " ".join(row["intended_audiences"].replace(';', ' ').split())
        return (row["resource_name"] + ' ' +
                row["description"] + ' ' +
                subject_areas + ' ' +
                row["type"] + ' ' +
                intended_audiences + ' ' +
                row["authoring_organization"])
    
    df["content_soup"] = df.apply(combine_text, axis=1)
    print("Kolom 'content_soup' telah dibuat.")
    return df

def preprocess_data(file_path):
    """
    Fungsi utama untuk preprocessing otomatis dataset.
    
    Args:
        file_path (str): Path ke file CSV.
    
    Returns:
        pd.DataFrame: Dataset yang telah diproses, siap untuk pelatihan.
    """
    # Langkah 1: Muat dataset
    df = load_and_check_dataset(file_path)
    
    # Langkah 2: Hapus fitur konstan
    df = drop_constant_features(df)
    
    # Langkah 3: Isi nilai kosong
    df = fill_missing_values(df)
    
    # Langkah 4: Preprocessing teks
    text_columns = ['resource_name', 'description', 'intended_audiences',
                    'subject_areas', 'type', 'authoring_organization']
    df = preprocess_text_columns(df, text_columns)
    
    # Langkah 5: Buat content_soup
    df = create_content_soup(df)
    
    print("\nPreprocessing selesai. Dataset siap untuk pelatihan.")
    return df

if __name__ == "__main__":
    # Contoh penggunaan
    file_path = "Learning_Resources_Database_Cleaned.csv"
    processed_df = preprocess_data(file_path)
    print("\nDataFrame yang telah diproses:")
    print(processed_df.head())
    print("\nContoh content_soup (baris kedua):")
    print(processed_df['content_soup'].iloc[1])