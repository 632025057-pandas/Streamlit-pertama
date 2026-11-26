"""
TAS AI - Hana Meisaria (NIM 632025057)
Aplikasi Streamlit: Regresi Linear, Klasifikasi, Clustering
+ Integrasi MongoDB (dataset tambahan, retraining, komentar publik, & model ML)
+ Integrasi Groq AI (Asisten AI, analisis metrik, & analisis sentimen komentar via groq_helper.py)
+ Integrasi Fuzzy Logic (Fuzzy KNN & Fuzzy C-Means) pada modul Klasifikasi
+ Galeri Hasil Training (referensi antar pengunjung)
Sumber data: scikit-learn (Diabetes, California Housing, Iris, Wine, Breast Cancer)

VERSI 3.5 - Tema Lilac Modern Terang + Galeri Hasil Training
"""
import copy
import hashlib
import json
import os
import pickle
import re
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
import streamlit as st
from pymongo import MongoClient
from sklearn.cluster import KMeans
from sklearn.datasets import (
    fetch_california_housing,
    load_breast_cancer,
    load_diabetes,
    load_iris,
    load_wine,
)
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    davies_bouldin_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    silhouette_score,
)
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

from groq_helper import analisis_sentimen_komentar, page_ai_assistant, page_kuis_ml, tanya_groq
from fuzzy_helper import (
    FuzzyKNNClassifier,
    FuzzyCMeansClassifier,
    plot_membership,
)

# ----------------------------------------------------------------------------
# Konfigurasi Halaman
# ----------------------------------------------------------------------------
st.set_page_config(
    page_title="TAS AI - Hana Meisaria",
    page_icon="🪻",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ----------------------------------------------------------------------------
# Session State untuk Navigasi & Status Model Terakhir
# ----------------------------------------------------------------------------
if "nav_radio" not in st.session_state:
    st.session_state["nav_radio"] = "🏠 Dashboard"

if "last_run" not in st.session_state:
    st.session_state["last_run"] = None


def change_page(page_name):
    st.session_state["nav_radio"] = page_name


# ----------------------------------------------------------------------------
# Custom CSS & Template Plotly - Tema Lilac Terang Modern
# ----------------------------------------------------------------------------
LILAC_COLORS = [
    "#8b5cf6", "#a78bfa", "#c084fc", "#7c3aed", "#d8b4fe",
    "#6366f1", "#a855f7", "#818cf8", "#9333ea", "#c4b5fd",
]
LILAC_SCALE = [[0.0, "#f8f5ff"], [0.5, "#a78bfa"], [1.0, "#5b21b6"]]
_lilac_tpl = copy.deepcopy(pio.templates["plotly_white"])
_lilac_tpl.layout.colorway = LILAC_COLORS
_lilac_tpl.layout.font.color = "#2e1065"
_lilac_tpl.layout.xaxis.gridcolor = "rgba(139, 92, 246, 0.12)"
_lilac_tpl.layout.yaxis.gridcolor = "rgba(139, 92, 246, 0.12)"
pio.templates["lilac_light"] = _lilac_tpl

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');

:root {
    --bg-main: #fcfaff;
    --bg-side: #f3effa;
    --surface: #ffffff;
    --surface-soft: #f9f6ff;
    --border: #e6ddf7;
    --txt: #2e1065;
    --muted: #6d5c8c;
    --accent: #8b5cf6;
    --accent-dark: #5b21b6;
    --accent-light: #a78bfa;
}

html, body, [class*="css"] {
    font-family: 'Plus Jakarta Sans', sans-serif;
}

.stApp, [data-testid="stAppViewContainer"] {
    background: var(--bg-main) !important;
    color: var(--txt);
}
[data-testid="stHeader"] {
    background: rgba(252, 250, 255, 0.88);
}

.gradient-text {
    background: linear-gradient(90deg, #6d28d9, #a78bfa);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    font-weight: 800;
}
.main-title-container {
    font-size: 2.15rem;
    font-weight: 800;
    margin-bottom: 0.3rem;
    color: #3b0764;
}

/* === HERO BANNER BERANDA === */
.hero-banner {
    position: relative;
    background: linear-gradient(115deg, #7c52c9 0%, #a779eb 52%, #8b5cf6 100%);
    border-radius: 22px;
    padding: 32px 36px;
    color: #ffffff;
    box-shadow: 0 10px 26px rgba(124, 82, 201, 0.24);
    overflow: hidden;
    margin-bottom: 24px;
}
.hero-title {
    font-size: 2rem;
    font-weight: 800;
    margin: 0;
    letter-spacing: 0.4px;
    color: #ffffff;
}
.hero-subtitle {
    font-size: 0.98rem;
    color: #f3e8ff;
    margin-top: 6px;
    margin-bottom: 20px;
}
.pill-row {
    display: flex;
    flex-wrap: wrap;
    gap: 10px;
}
.pill-badge {
    background: #ffffff;
    color: #4c1d95;
    padding: 6px 14px;
    border-radius: 999px;
    font-size: 0.82rem;
    font-weight: 600;
    box-shadow: 0 2px 6px rgba(46, 16, 101, 0.1);
}
.deco-1 { position: absolute; right: 22%; top: 20%; font-size: 1.35rem; opacity: 0.78; }
.deco-2 { position: absolute; right: 6%; top: 14%; font-size: 2.2rem; opacity: 0.88; }
.deco-3 { position: absolute; right: 14%; bottom: 16%; font-size: 1.6rem; opacity: 0.82; }

.section-heading {
    font-size: 1.15rem;
    font-weight: 800;
    color: #3b0764;
    margin: 6px 0 14px 0;
}

/* === GRID FITUR APLIKASI (4 KOLOM RAPI = 2 BARIS) === */
.feature-grid {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 14px;
    margin-bottom: 18px;
}
@media (max-width: 1100px) {
    .feature-grid { grid-template-columns: repeat(2, 1fr); }
}
.feature-card {
    background: var(--surface-soft);
    border: 1px solid var(--border);
    border-radius: 16px;
    padding: 16px;
    transition: transform 0.16s ease, box-shadow 0.16s ease, border-color 0.16s ease;
}
.feature-card:hover {
    transform: translateY(-2px);
    box-shadow: 0 6px 16px rgba(139, 92, 246, 0.14);
    border-color: var(--accent);
}
.feature-icon { font-size: 1.5rem; margin-bottom: 8px; }
.feature-title { font-weight: 700; font-size: 0.92rem; color: var(--txt); margin-bottom: 4px; }
.feature-desc { font-size: 0.78rem; color: var(--muted); line-height: 1.45; }

/* === KARTU RINGKASAN BAWAH === */
.bottom-card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-top: 4px solid var(--accent);
    border-radius: 18px;
    padding: 20px;
    min-height: 185px;
    box-shadow: 0 4px 14px rgba(109, 40, 217, 0.06);
}
.bottom-card-title { font-size: 1.02rem; font-weight: 800; color: #4c1d95; margin-bottom: 14px; }
.metric-row { display: grid; grid-template-columns: 1fr 1fr; gap: 12px; }
.metric-box {
    background: var(--surface-soft);
    border: 1px solid var(--border);
    border-left: 4px solid var(--accent);
    border-radius: 12px;
    padding: 14px;
}
.metric-label { font-size: 0.84rem; color: var(--muted); font-weight: 600; }
.metric-value { font-size: 1.9rem; font-weight: 800; color: var(--txt); margin-top: 4px; }
.status-info-box {
    background: #f0f4ff;
    color: #3730a3;
    border-radius: 12px;
    padding: 15px;
    font-size: 0.9rem;
    line-height: 1.5;
}
.status-ok-box {
    background: #f5f0ff;
    color: #4c1d95;
    border: 1px solid #ddd6fe;
    border-radius: 12px;
    padding: 15px;
    font-size: 0.9rem;
    line-height: 1.5;
}
.score-center { text-align: center; padding: 4px 0; }
.score-label { font-weight: 700; font-size: 0.88rem; color: #4c1d95; margin-bottom: 10px; }
.score-ring {
    width: 92px;
    height: 92px;
    border-radius: 50%;
    border: 8px solid #ede9fe;
    border-top-color: var(--accent);
    margin: 0 auto;
    display: flex;
    align-items: center;
    justify-content: center;
    color: var(--accent-dark);
    font-size: 1rem;
    font-weight: 800;
    background: var(--bg-main);
}

/* === KOMPONEN STREAMLIT === */
[data-testid="stMetric"] {
    background-color: var(--surface-soft);
    border: 1px solid var(--border);
    border-left: 4px solid var(--accent);
    padding: 12px 16px;
    border-radius: 12px;
    box-shadow: 0 2px 8px rgba(109, 40, 217, 0.04);
}
[data-testid="stMetricLabel"] { color: var(--muted) !important; font-weight: 600; }
[data-testid="stMetricValue"] { font-size: 1.75rem; font-weight: 800; color: var(--txt) !important; }

.stTabs [data-baseweb="tab-list"] { gap: 8px; }
.stTabs [data-baseweb="tab"] {
    border-radius: 999px;
    padding: 8px 18px;
    background-color: var(--bg-side);
    color: #4c1d95;
    font-weight: 600;
}
.stTabs [aria-selected="true"] {
    background: linear-gradient(90deg, #7c3aed, #a78bfa) !important;
    color: #ffffff !important;
}
.stTabs [data-baseweb="tab-highlight"], .stTabs [data-baseweb="tab-border"] { display: none; }

[data-testid="stExpander"] {
    border: 1px solid var(--border);
    border-radius: 14px;
    background: var(--surface);
}
[data-testid="stExpander"] summary { color: #4c1d95; font-weight: 700; }
[data-testid="stDataFrame"] { border: 1px solid var(--border); border-radius: 12px; overflow: hidden; }
[data-testid="stAlert"] { border-radius: 12px; }

.stButton > button[kind="primary"], .stButton > button[data-testid="stBaseButton-primary"] {
    background: linear-gradient(90deg, #7c3aed, #a78bfa);
    border: none;
    color: #ffffff;
    border-radius: 10px;
    font-weight: 600;
    box-shadow: 0 4px 10px rgba(124, 58, 237, 0.24);
}
.stButton > button[kind="primary"]:hover, .stButton > button[data-testid="stBaseButton-primary"]:hover {
    background: linear-gradient(90deg, #6d28d9, #9333ea);
}
.stButton > button[kind="secondary"], .stButton > button[data-testid="stBaseButton-secondary"] {
    background-color: var(--surface-soft);
    border: 1px solid #ddd6fe;
    color: #4c1d95;
    border-radius: 10px;
    font-weight: 600;
}
.stButton > button[kind="secondary"]:hover, .stButton > button[data-testid="stBaseButton-secondary"]:hover {
    background-color: var(--bg-side);
    border-color: var(--accent);
    color: var(--accent-dark);
}
hr { border-color: var(--border) !important; }

.footer {
    text-align: center;
    color: #7c6f96;
    font-size: 0.85rem;
    padding: 20px 0 10px 0;
    border-top: 1px solid var(--border);
    margin-top: 36px;
}

/* Galeri Highlight Card */
.gal-best {
    background: linear-gradient(135deg, #f5f0ff, #ede4ff);
    border: 1px solid #d4c2fa;
    border-left: 5px solid var(--accent);
    border-radius: 14px;
    padding: 16px 18px;
    margin: 6px 0 14px 0;
}
.gal-best-title { font-weight: 800; color: #4c1d95; font-size: 1.05rem; }
.gal-best-sub { color: var(--muted); font-size: 0.88rem; margin-top: 4px; }

/* === SIDEBAR TEMA LILAC === */
[data-testid="stSidebar"] {
    background-color: var(--bg-side) !important;
    border-right: 1px solid #e4dbf5;
}
[data-testid="stSidebar"] [data-testid="stVerticalBlock"] { gap: 0.35rem; }
.sidebar-title {
    font-size: 1.6rem;
    font-weight: 800;
    color: var(--accent-dark);
    margin-bottom: 10px;
    letter-spacing: -0.5px;
}
.profile-card {
    background: #ffffff;
    border-radius: 16px;
    padding: 13px 15px;
    display: flex;
    align-items: center;
    gap: 12px;
    box-shadow: 0 2px 10px rgba(109, 40, 217, 0.07);
    border: 1px solid var(--border);
    margin-bottom: 14px;
}
.profile-avatar {
    width: 44px;
    height: 44px;
    border-radius: 50%;
    background: linear-gradient(135deg, #7c3aed, #c084fc);
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 1.3rem;
    color: #ffffff;
    flex-shrink: 0;
}
.profile-name { font-weight: 700; font-size: 0.94rem; color: var(--txt); line-height: 1.25; }
.profile-id { font-size: 0.78rem; color: var(--muted); }

.sidebar-section {
    font-size: 0.7rem;
    font-weight: 800;
    letter-spacing: 0.09em;
    text-transform: uppercase;
    color: #8470ad;
    margin: 10px 0 4px 4px;
}
[data-testid="stSidebar"] button[data-testid="stBaseButton-secondary"],
[data-testid="stSidebar"] button[data-testid="baseButton-secondary"],
[data-testid="stSidebar"] button[kind="secondary"] {
    background-color: transparent;
    border: 1px solid transparent;
    color: #3b0764;
    justify-content: flex-start;
    padding: 0.44rem 0.8rem;
    border-radius: 12px;
    font-weight: 500;
    transition: all 0.18s ease;
}
[data-testid="stSidebar"] button[data-testid="stBaseButton-secondary"]:hover,
[data-testid="stSidebar"] button[data-testid="baseButton-secondary"]:hover,
[data-testid="stSidebar"] button[kind="secondary"]:hover {
    background-color: #e9e0fb;
    border-color: #d8c8f8;
    color: var(--accent-dark);
    transform: translateX(2px);
}
[data-testid="stSidebar"] button[data-testid="stBaseButton-primary"],
[data-testid="stSidebar"] button[data-testid="baseButton-primary"],
[data-testid="stSidebar"] button[kind="primary"] {
    background: linear-gradient(90deg, #7c3aed, #a78bfa);
    border: none;
    color: #ffffff;
    justify-content: flex-start;
    padding: 0.44rem 0.8rem;
    border-radius: 12px;
    font-weight: 700;
    box-shadow: 0 4px 12px rgba(124, 58, 237, 0.28);
}

.mini-card {
    background: #ffffff;
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 10px 12px;
    margin-bottom: 8px;
}
.mini-card-title { font-weight: 700; font-size: 0.84rem; color: #4c1d95; }
.mini-card-desc { font-size: 0.74rem; color: var(--muted); margin-top: 2px; line-height: 1.35; }

.status-panel {
    background-color: #ffffff;
    border: 1px solid var(--border);
    border-radius: 14px;
    padding: 12px 14px;
    margin-top: 10px;
}
.status-row {
    display: flex;
    justify-content: space-between;
    align-items: center;
    font-size: 0.82rem;
    color: #3b0764;
    font-weight: 600;
    padding: 4px 0;
}
.badge-ok { background: #ecfdf5; color: #059669; border: 1px solid #a7f3d0; padding: 2px 8px; border-radius: 999px; font-size: 0.72rem; font-weight: 700; }
.badge-warn { background: #fffbeb; color: #d97706; border: 1px solid #fde68a; padding: 2px 8px; border-radius: 999px; font-size: 0.72rem; font-weight: 700; }
.badge-err { background: #fef2f2; color: #dc2626; border: 1px solid #fecaca; padding: 2px 8px; border-radius: 999px; font-size: 0.72rem; font-weight: 700; }
.mini-stats {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 6px;
    margin-top: 10px;
    padding-top: 10px;
    border-top: 1px solid #ede9fe;
    text-align: center;
}
.mini-stat-box { background: var(--surface-soft); border: 1px solid var(--border); border-radius: 9px; padding: 6px 4px; }
.mini-stat-val { font-size: 0.95rem; font-weight: 800; color: var(--accent); line-height: 1.2; }
.mini-stat-lbl { font-size: 0.68rem; color: var(--muted); }
</style>
""", unsafe_allow_html=True)

# ----------------------------------------------------------------------------
# Konfigurasi Dataset & Materi Edukatif
# ----------------------------------------------------------------------------
LOADERS = {
    "Diabetes": load_diabetes,
    "California Housing": fetch_california_housing,
    "Iris": load_iris,
    "Wine": load_wine,
    "Breast Cancer": load_breast_cancer,
}
TASK_DATASETS = {
    "regresi": ["Diabetes", "California Housing"],
    "klasifikasi": ["Iris", "Wine", "Breast Cancer"],
    "clustering": ["Iris", "Wine", "Breast Cancer"],
}
TASK_LABEL = {
    "regresi": "Regresi Linear",
    "klasifikasi": "Klasifikasi",
    "clustering": "Clustering",
}
DATASET_DESC = {
    "Diabetes": "Prediksi tingkat progres penyakit diabetes dalam 1 tahun berdasarkan 10 metrik kesehatan pasien (nilai sudah dinormalisasi).",
    "California Housing": "Prediksi rata-rata harga rumah di California (dalam $100.000) berdasarkan 6 fitur ekonomi & demografis wilayah.",
    "Iris": "Klasifikasi 3 spesies bunga Iris (Setosa, Versicolor, Virginica) berdasarkan panjang & lebar sepal serta petal (cm).",
    "Wine": "Klasifikasi 3 jenis kultivar wine Italia berdasarkan 13 kandungan kimia hasil uji laboratorium.",
    "Breast Cancer": "Klasifikasi tumor payudara jinak (benign) atau ganas (malignant) berdasarkan 8 fitur morfologi sel medis.",
}

METRIC_HELP = {
    "R² Score": (
        "Koefisien determinasi (maksimal 1.0). Mengukur seberapa besar variasi target yang "
        "berhasil dijelaskan oleh model. Makin mendekati 1.0 makin baik; nilai negatif berarti "
        "model lebih buruk daripada sekadar menebak rata-rata."
    ),
    "MAE": (
        "Mean Absolute Error: rata-rata selisih absolut antara nilai aktual dan nilai prediksi. "
        "Satuannya sama dengan target. Semakin kecil (mendekati 0), semakin akurat."
    ),
    "RMSE": (
        "Root Mean Squared Error: akar dari rata-rata kuadrat kesalahan. Mirip MAE, tetapi "
        "memberi penalti lebih berat untuk prediksi yang meleset jauh. Semakin kecil semakin baik."
    ),
    "Accuracy": (
        "Persentase total prediksi yang benar dari seluruh data uji (rentang 0–1). "
        "Semakin mendekati 1.0 (100%), semakin baik."
    ),
    "Precision": (
        "Ketepatan prediksi per kelas (Macro Average): dari semua data yang diprediksi masuk kelas X, "
        "berapa proporsi yang benar-benar kelas X. Penting untuk menekan kesalahan positif palsu (FP)."
    ),
    "Recall": (
        "Cakupan deteksi per kelas (Macro Average): dari semua data asli kelas X, berapa proporsi "
        "yang berhasil dikenali model. Sangat krusial pada kasus medis seperti Breast Cancer."
    ),
    "F1 Score": (
        "Rata-rata harmonis antara Precision dan Recall (rentang 0–1). Metrik paling seimbang "
        "saat jumlah data antar kelas tidak merata."
    ),
    "Silhouette": (
        "Silhouette Coefficient (rentang -1 hingga 1). Mengukur seberapa padat anggota di dalam "
        "cluster dan seberapa jauh jaraknya dari cluster lain. >0.5 baik, mendekati 0 tumpang tindih."
    ),
    "Davies-Bouldin": (
        "Davies-Bouldin Index (minimal 0). Mengukur rasio kemiripan antar cluster. "
        "Semakin kecil (mendekati 0), semakin tegas pemisahan antar cluster."
    ),
    "Inertia": (
        "Total jarak kuadrat setiap titik data ke pusat cluster-nya (centroid). "
        "Selalu turun saat k bertambah; gunakan titik siku (Elbow) untuk memilih k ideal."
    ),
}

FEATURE_GLOSSARY = {
    "Diabetes": {
        "age": "Usia pasien (skala terstandarisasi)",
        "sex": "Jenis kelamin (skala terstandarisasi)",
        "bmi": "Body Mass Index / indeks massa tubuh",
        "bp": "Rata-rata tekanan darah (blood pressure)",
        "s1": "Total serum kolesterol (tc)",
        "s2": "Low-density lipoproteins (ldl)",
        "s3": "High-density lipoproteins (hdl)",
        "s4": "Rasio total kolesterol / HDL (tch)",
        "s5": "Log kadar trigliserida serum (ltg)",
        "s6": "Kadar gula darah (glu)",
    },
    "California Housing": {
        "MedInc": "Pendapatan median rumah tangga (dalam $10.000)",
        "HouseAge": "Usia median bangunan rumah di blok tersebut (tahun)",
        "AveRooms": "Rata-rata jumlah ruangan per rumah tangga",
        "AveBedrms": "Rata-rata jumlah kamar tidur per rumah tangga",
        "Population": "Total jumlah penduduk di blok wilayah",
        "AveOccup": "Rata-rata jumlah penghuni per rumah tangga",
    },
    "Iris": {
        "sepal length (cm)": "Panjang kelopak luar (sepal) dalam cm",
        "sepal width (cm)": "Lebar kelopak luar (sepal) dalam cm",
        "petal length (cm)": "Panjang mahkota bunga (petal) dalam cm",
        "petal width (cm)": "Lebar mahkota bunga (petal) dalam cm",
    },
    "Wine": {
        "alcohol": "Kadar alkohol pada wine",
        "malic_acid": "Kadar asam malat",
        "ash": "Kandungan abu mineral",
        "alcalinity_of_ash": "Tingkat alkalinitas abu",
        "magnesium": "Kandungan magnesium",
        "total_phenols": "Total senyawa fenol",
        "flavanoids": "Kandungan flavonoid (memengaruhi rasa & warna)",
        "nonflavanoid_phenols": "Fenol non-flavonoid",
        "proanthocyanins": "Senyawa antioksidan proantosianin",
        "color_intensity": "Intensitas warna wine",
        "hue": "Rona warna (hue)",
        "od280/od315_of_diluted_wines": "Rasio absorbansi protein wine encer",
        "proline": "Kadar asam amino prolin",
    },
    "Breast Cancer": {
        "mean radius": "Rata-rata jarak dari pusat ke tepi inti sel",
        "mean texture": "Standar deviasi nilai keabuan tekstur sel",
        "mean perimeter": "Rata-rata keliling inti sel",
        "mean area": "Rata-rata luas permukaan inti sel",
        "mean smoothness": "Variasi lokal panjang jari-jari (kehalusan tepi)",
        "mean compactness": "Kekompakan bentuk sel: (keliling² / luas) - 1",
        "mean concavity": "Tingkat kecekungan kontur inti sel",
        "mean concave points": "Jumlah bagian cekung pada kontur sel",
    },
}

# ----------------------------------------------------------------------------
# MongoDB Connection
# ----------------------------------------------------------------------------
def _secret(name, default=None):
    try:
        return st.secrets[name]
    except Exception:
        return os.getenv(name, default)


@st.cache_resource
def init_db():
    fallback_uri = "mongodb://632025057:632025057_Hana@ac-z4y3rlw-shard-00-00.n0apbrv.mongodb.net:27017,ac-z4y3rlw-shard-00-01.n0apbrv.mongodb.net:27017,ac-z4y3rlw-shard-00-02.n0apbrv.mongodb.net:27017/?ssl=true&replicaSet=atlas-ikculp-shard-0&authSource=admin&appName=Cluster0"
    uri = _secret("MONGODB_URI", fallback_uri)
    if not uri:
        return None, "MONGODB_URI belum diatur."
    try:
        client = MongoClient(uri, serverSelectionTimeoutMS=5000)
        client.admin.command("ping")
        return client[_secret("MONGODB_DB", "tugas_ai")], None
    except Exception as e:
        return None, str(e)


db, db_error = init_db()


def now_wib(dt):
    return (dt + timedelta(hours=7)).strftime("%d %b %Y %H:%M WIB")


# ----------------------------------------------------------------------------
# Helper: Save Model ke DB
# ----------------------------------------------------------------------------
def save_model_to_db(task, dataset_name, model_obj, model_name, metrics):
    if db is None:
        return False
    try:
        doc = {
            "key": f"{task}:{dataset_name}",
            "task": task,
            "dataset": dataset_name,
            "model_name": model_name,
            "model_bytes": pickle.dumps(model_obj),
            "metrics": metrics,
            "created_at": datetime.now(timezone.utc),
        }
        db.models.insert_one(doc)
        return True
    except Exception:
        return False


# ----------------------------------------------------------------------------
# Galeri Hasil Training (hasil disimpan ke MongoDB, dibaca semua pengunjung)
# ----------------------------------------------------------------------------
PRIMARY_METRIC = {"regresi": "R² Score", "klasifikasi": "F1 Score", "clustering": "Silhouette"}
MODULE_PAGE = {"regresi": "📈 Regresi Linear", "klasifikasi": "🧪 Klasifikasi", "clustering": "🔵 Clustering"}
FUZZY_MODELS = ("Fuzzy KNN", "Fuzzy C-Means")
MAX_SAVE_PER_SESSION = 20


def make_params(task, model_name="", test_size=0.2, k=3, fuzzy_k=5, fuzzy_m=2.0):
    """Parameter yang relevan saja, dinormalisasi agar konfigurasi identik menghasilkan tanda tangan identik."""
    if task == "clustering":
        return {"k": int(k)}
    p = {"test_size": round(float(test_size), 3)}
    if task == "klasifikasi" and model_name in FUZZY_MODELS:
        p["fuzzy_m"] = round(float(fuzzy_m), 2)
        if model_name == "Fuzzy KNN":
            p["fuzzy_k"] = int(fuzzy_k)
    return p


def params_text(task, model_name, p):
    if task == "clustering":
        return f"k = {p.get('k', '-')}"
    txt = f"test {int(round(float(p.get('test_size', 0.2)) * 100))}%"
    if model_name in FUZZY_MODELS:
        txt += f", m = {p.get('fuzzy_m', '-')}"
        if model_name == "Fuzzy KNN":
            txt += f", K = {p.get('fuzzy_k', '-')}"
    return txt


def _bersihkan(teks, maks):
    teks = re.sub(r"[\[\]<>`|]", "", str(teks or "")).strip()
    return teks[:maks]


def save_training_result(task, dataset, model_name, params, metrics, n_train, n_test, n_extra,
                         nama="", catatan="", sumber="modul"):
    """Simpan konfigurasi + skor (bukan objek model). Konfigurasi identik tidak diduplikasi, hanya menambah penghitung 'dicoba'."""
    if db is None:
        return False, "MongoDB belum terhubung."
    raw = json.dumps([task, dataset, model_name, params, int(n_train), int(n_extra)], sort_keys=True)
    sig = hashlib.sha1(raw.encode()).hexdigest()
    try:
        r = db.hasil_training.update_one(
            {"sig": sig},
            {
                "$setOnInsert": {
                    "sig": sig, "task": task, "dataset": dataset, "model_name": model_name,
                    "params": params, "metrics": {k: float(v) for k, v in metrics.items()},
                    "n_train": int(n_train), "n_test": int(n_test), "n_extra": int(n_extra),
                    "nama": _bersihkan(nama, 30) or "Anonim", "catatan": _bersihkan(catatan, 200),
                    "sumber": sumber, "created_at": datetime.now(timezone.utc),
                },
                "$inc": {"dicoba": 1},
            },
            upsert=True,
        )
        fetch_training_results.clear()
        return True, ("baru" if r.upserted_id else "duplikat")
    except Exception as e:
        return False, str(e)


@st.cache_data(ttl=20, show_spinner=False)
def fetch_training_results(task, dataset=None):
    if db is None:
        return []
    q = {"task": task}
    if dataset:
        q["dataset"] = dataset
    try:
        return list(db.hasil_training.find(q, {"_id": 0}).sort("created_at", -1).limit(500))
    except Exception:
        return []


def _urut_terbaik(rows, task):
    pm = PRIMARY_METRIC[task]
    return sorted(rows, key=lambda d: d.get("metrics", {}).get(pm, float("-inf")), reverse=True)


def results_df(rows, task):
    out = []
    for d in rows:
        out.append({
            "Penyimpan": d.get("nama", "Anonim"),
            "Dataset": d.get("dataset", "-"),
            "Algoritma": d.get("model_name", "-"),
            "Parameter": params_text(task, d.get("model_name", ""), d.get("params", {})),
            "Data Latih": d.get("n_train", 0),
            "Data Tambahan": d.get("n_extra", 0),
            **{k: round(v, 4) for k, v in d.get("metrics", {}).items()},
            "Dicoba": d.get("dicoba", 1),
            "Catatan": d.get("catatan", ""),
            "Waktu": now_wib(d["created_at"]) if d.get("created_at") else "-",
        })
    return pd.DataFrame(out)


def apply_config(doc):
    """Callback tombol: isi widget modul dengan konfigurasi tersimpan lalu pindah halaman."""
    task, p = doc["task"], doc.get("params", {})
    pref = {"regresi": "reg", "klasifikasi": "clf", "clustering": "clu"}[task]
    st.session_state[f"{pref}_ds"] = doc["dataset"]
    st.session_state[f"{pref}_extra"] = doc.get("n_extra", 0) > 0
    if task == "regresi":
        st.session_state["reg_ts"] = float(p.get("test_size", 0.2))
    elif task == "klasifikasi":
        st.session_state["clf_model"] = doc["model_name"]
        st.session_state["clf_ts"] = float(p.get("test_size", 0.2))
        if "fuzzy_m" in p:
            st.session_state["clf_fuzzy_m"] = float(p["fuzzy_m"])
        if "fuzzy_k" in p:
            st.session_state["clf_fuzzy_k"] = int(p["fuzzy_k"])
    else:
        st.session_state["clu_k"] = int(p.get("k", 3))
    change_page(MODULE_PAGE[task])


def render_reference_hint(task, name, metrics):
    """Kotak referensi di dalam modul: bandingkan skor saat ini dengan hasil terbaik pengunjung lain."""
    if db is None:
        return
    pm = PRIMARY_METRIC[task]
    rows = _urut_terbaik(fetch_training_results(task, name), task)
    if not rows:
        st.caption("📚 Belum ada hasil tersimpan untuk dataset ini. Simpan hasil training kamu lewat panel **💾 Simpan Hasil Training** di bawah agar jadi referensi pengunjung lain.")
        return
    best = rows[0]
    best_val = best["metrics"].get(pm)
    cur = metrics.get(pm)
    if best_val is None or cur is None:
        return
    selisih = cur - best_val
    status = "🎉 **menyamai/melampaui** referensi terbaik!" if selisih >= -1e-9 else f"selisih **{selisih:+.4f}** dari terbaik."
    st.info(
        f"📚 **Referensi pengunjung lain ({len(rows)} hasil tersimpan untuk {name}):** skor {pm} tertinggi = **{best_val:.4f}** "
        f"({best['model_name']}, {params_text(task, best['model_name'], best.get('params', {}))}) oleh *{best.get('nama', 'Anonim')}*. "
        f"Skor kamu sekarang **{cur:.4f}** → {status}"
    )
    st.button("🏆 Lihat Galeri Lengkap", key=f"{task}_to_gallery", on_click=change_page, args=("🏆 Galeri Hasil Training",))


def render_save_panel(task, name, model_name, params, res, n_extra):
    with st.expander("💾 Simpan Hasil Training ini sebagai referensi pengunjung lain"):
        if db is None:
            st.info("Fitur simpan membutuhkan koneksi MongoDB Atlas.")
            return
        st.caption(
            "Yang disimpan: **dataset, algoritma, parameter, dan skor metrik** (bukan file model), "
            "lalu tampil di halaman **🏆 Galeri Hasil Training**. Konfigurasi yang sama persis tidak dobel, hanya menambah penghitung *Dicoba*."
        )
        c1, c2 = st.columns([1, 2])
        nama = c1.text_input("Nama / inisial (opsional)", max_chars=30, key=f"{task}_save_nama")
        catatan = c2.text_input(
            "Catatan (opsional)", max_chars=200, key=f"{task}_save_note",
            placeholder="mis. 'k=4 memberi silhouette tertinggi'",
        )
        if st.button("💾 Simpan ke Galeri", key=f"{task}_save_btn", type="primary"):
            if st.session_state.get("save_count", 0) >= MAX_SAVE_PER_SESSION:
                st.warning(f"Batas {MAX_SAVE_PER_SESSION} penyimpanan per sesi tercapai.")
            else:
                ok, info = save_training_result(
                    task, name, model_name, params, res["metrics"],
                    res.get("n_train", 0), res.get("n_test", 0), n_extra, nama, catatan,
                )
                if ok:
                    st.session_state["save_count"] = st.session_state.get("save_count", 0) + 1
                    if info == "baru":
                        st.success("✅ Tersimpan! Hasil kamu sekarang bisa dilihat pengunjung lain di Galeri.")
                    else:
                        st.info("ℹ️ Konfigurasi yang sama sudah ada di Galeri; penghitung *Dicoba* ditambah.")
                else:
                    st.error(f"Gagal menyimpan: {info}")


# ----------------------------------------------------------------------------
# Data Loaders
# ----------------------------------------------------------------------------
@st.cache_data
def load_base(name):
    d = LOADERS[name](as_frame=True)
    feats = list(d.feature_names)
    tn = getattr(d, "target_names", None)
    tn = [str(x) for x in tn] if tn is not None else None
    if name == "California Housing":
        feats = feats[:6]
    elif name == "Breast Cancer":
        feats = feats[:8]
    df = d.data[feats].copy()
    df["target"] = d.target.values
    return df, feats, tn


def get_extra(task, name, feats):
    cols = feats + ["target"]
    if db is None:
        return pd.DataFrame(columns=cols)
    try:
        docs = list(db.dataset_tambahan.find({"key": f"{task}:{name}"}))
    except Exception:
        return pd.DataFrame(columns=cols)
    rows = [list(d["values"]) + [d.get("target")] for d in docs if len(d.get("values", [])) == len(feats)]
    return pd.DataFrame(rows, columns=cols).astype(float)


def get_training_df(task, name, use_extra=True):
    base, feats, tn = load_base(name)
    extra = get_extra(task, name, feats) if use_extra else pd.DataFrame(columns=feats + ["target"])
    if task != "clustering":
        extra = extra.dropna(subset=["target"])
    df = pd.concat([base, extra], ignore_index=True) if len(extra) else base.copy()
    if task == "klasifikasi":
        df["target"] = df["target"].astype(int)
    return df, len(extra), feats, tn


def insert_rows(task, name, rows, sumber):
    docs = [
        {
            "key": f"{task}:{name}",
            "task": task,
            "dataset": name,
            "values": [float(v) for v in vals],
            "target": None if t is None else float(t),
            "sumber": sumber,
            "created_at": datetime.now(timezone.utc),
        }
        for vals, t in rows
    ]
    if docs:
        db.dataset_tambahan.insert_many(docs)
    return len(docs)


def feature_inputs(base, feats, prefix, ncols=3, dataset_name=None):
    vals = []
    cols = st.columns(ncols)
    glossary = FEATURE_GLOSSARY.get(dataset_name, {}) if dataset_name else {}
    for i, f in enumerate(feats):
        lo, hi = float(base[f].min()), float(base[f].max())
        pad = (hi - lo) * 0.5
        with cols[i % ncols]:
            v = st.number_input(
                f,
                value=float(base[f].median()),
                min_value=lo - pad,
                max_value=hi + pad,
                step=(hi - lo) / 100 or 0.01,
                format="%.4f",
                help=glossary.get(f, None),
                key=f"{prefix}_{f}",
            )
        vals.append(v)
    return vals


# ----------------------------------------------------------------------------
# Model Training
# ----------------------------------------------------------------------------
def make_classifier(model_name, seed, fuzzy_k=5, fuzzy_m=2.0):
    if model_name == "Fuzzy KNN":
        return make_pipeline(
            StandardScaler(),
            FuzzyKNNClassifier(n_neighbors=fuzzy_k, m=fuzzy_m),
        )
    if model_name == "Fuzzy C-Means":
        return make_pipeline(
            StandardScaler(),
            FuzzyCMeansClassifier(n_clusters=None, m=fuzzy_m, random_state=seed),
        )
    return {
        "Logistic Regression": make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)),
        "Random Forest": RandomForestClassifier(n_estimators=200, random_state=seed),
        "KNN": make_pipeline(StandardScaler(), KNeighborsClassifier()),
        "Decision Tree": DecisionTreeClassifier(random_state=seed),
    }[model_name]


@st.cache_resource(show_spinner=False)
def run_model(task, df, model_name="Random Forest", k=3, test_size=0.2,
              seed=42, fuzzy_k=5, fuzzy_m=2.0):
    if task == "regresi":
        X, y = df.drop(columns="target"), df["target"]
        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=test_size, random_state=seed)
        m = LinearRegression().fit(Xtr, ytr)
        pred = m.predict(Xte)
        return {
            "model": m,
            "metrics": {
                "R² Score": r2_score(yte, pred),
                "MAE": mean_absolute_error(yte, pred),
                "RMSE": mean_squared_error(yte, pred) ** 0.5,
            },
            "y_test": yte.values,
            "pred": pred,
            "n_train": len(Xtr),
            "n_test": len(Xte),
            "X_test": Xte,
            "y_train": ytr,
        }
    if task == "klasifikasi":
        X, y = df.drop(columns="target"), df["target"]
        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=test_size, random_state=seed, stratify=y)
        m = make_classifier(model_name, seed, fuzzy_k, fuzzy_m).fit(Xtr, ytr)
        pred = m.predict(Xte)
        classes = sorted(y.unique())
        return {
            "model": m,
            "metrics": {
                "Accuracy": accuracy_score(yte, pred),
                "Precision": precision_score(yte, pred, average="macro", zero_division=0),
                "Recall": recall_score(yte, pred, average="macro", zero_division=0),
                "F1 Score": f1_score(yte, pred, average="macro", zero_division=0),
            },
            "cm": confusion_matrix(yte, pred, labels=classes),
            "classes": classes,
            "n_train": len(Xtr),
            "n_test": len(Xte),
        }
    X = df.drop(columns="target", errors="ignore")
    sc = StandardScaler().fit(X)
    Xs = sc.transform(X)
    km = KMeans(n_clusters=k, n_init=10, random_state=seed).fit(Xs)
    labels = km.labels_
    return {
        "model": km,
        "scaler": sc,
        "labels": labels,
        "pca": PCA(n_components=2, random_state=seed).fit_transform(Xs),
        "metrics": {
            "Silhouette": silhouette_score(Xs, labels),
            "Davies-Bouldin": davies_bouldin_score(Xs, labels),
            "Inertia": km.inertia_,
        },
        "n_train": len(X),
    }


@st.cache_data(show_spinner=False)
def elbow_data(df, seed=42):
    X = StandardScaler().fit_transform(df.drop(columns="target", errors="ignore"))
    rows = []
    for k in range(2, 11):
        km = KMeans(n_clusters=k, n_init=10, random_state=seed).fit(X)
        rows.append({"k": k, "Inertia": km.inertia_, "Silhouette": silhouette_score(X, km.labels_)})
    return pd.DataFrame(rows).set_index("k")


def show_metrics(metrics):
    cols = st.columns(len(metrics))
    for c, (k, v) in zip(cols, metrics.items()):
        c.metric(k, f"{v:,.4f}", help=METRIC_HELP.get(k, ""))
    st.caption("ℹ️ Arahkan kursor ke ikon **?** di pojok kanan atas setiap kartu metrik untuk melihat definisi & cara membacanya.")


def show_quick_interpretation(task, metrics):
    if task == "regresi":
        r2 = metrics["R² Score"]
        if r2 >= 0.75:
            status = "Sangat Baik — model mampu menjelaskan sebagian besar variasi target."
        elif r2 >= 0.50:
            status = "Cukup Baik — hubungan linier cukup terlihat, namun masih ada variasi non-linier yang belum tertangkap."
        else:
            status = "Moderat/Rendah — pola data cenderung non-linier atau fitur belum cukup menjelaskan target secara linier."
        st.caption(f"🎓 **Interpretasi Cepat:** $R^2 = {r2:.2f}$ berarti sekitar **{max(0, r2)*100:.1f}%** variasi nilai target dijelaskan oleh fitur input. ({status})")
    elif task == "klasifikasi":
        acc = metrics["Accuracy"]
        f1 = metrics["F1 Score"]
        if acc >= 0.90:
            status = "Sangat Tinggi — model sangat konsisten memisahkan antar kelas."
        elif acc >= 0.75:
            status = "Baik — sebagian besar data uji berhasil diklasifikasikan dengan tepat."
        else:
            status = "Perlu Evaluasi — coba bandingkan dengan algoritma lain atau cek ketidakseimbangan kelas."
        st.caption(f"🎓 **Interpretasi Cepat:** Akurasi **{acc*100:.1f}%** dengan F1 Score **{f1:.4f}**. ({status})")
    else:
        sil = metrics["Silhouette"]
        if sil >= 0.50:
            status = "Struktur Cluster Kuat — tiap kelompok padat dan terpisah cukup jelas."
        elif sil >= 0.25:
            status = "Struktur Cluster Cukup — pemisahan wajar, coba bandingkan nilai k lain di grafik Elbow & Silhouette."
        else:
            status = "Tumpang Tindih — titik antar cluster saling berdekatan; coba ubah jumlah cluster (k)."
        st.caption(f"🎓 **Interpretasi Cepat:** Silhouette Score **{sil:.4f}**. ({status})")


def show_feature_glossary(name):
    glossary = FEATURE_GLOSSARY.get(name)
    if glossary:
        with st.expander(f"📘 Kamus Arti Fitur Dataset {name}"):
            df_g = pd.DataFrame([{"Nama Fitur": k, "Penjelasan": v} for k, v in glossary.items()])
            st.dataframe(df_g, use_container_width=True, hide_index=True)


def render_ai_insight(ringkasan, key):
    if st.button("🤖 Jelaskan Hasil Metrik dengan Groq", key=key):
        with st.spinner("Groq sedang menganalisis metrik..."):
            jawaban, err = tanya_groq(ringkasan)
        if err:
            st.error(err)
        else:
            st.success(jawaban)


# ============================================================================
# PENJELASAN MATEMATIKA INTERAKTIF
# ============================================================================
def _bagi(a, b):
    return a / b if b else 0.0


def _cek_cocok(manual, dari_model, nama):
    if abs(manual - dari_model) < 1e-6:
        st.success(f"✅ Hasil hitung manual **{nama} = {manual:.4f}** sama persis dengan kartu metrik di atas.")
    else:
        st.info(f"ℹ️ Hitung manual {nama} = {manual:.4f}, kartu metrik = {dari_model:.4f} (selisih kecil karena pembulatan/metode).")


def math_regresi(res, feats):
    y = np.asarray(res["y_test"], dtype=float)
    yh = np.asarray(res["pred"], dtype=float)
    e = y - yh
    n = len(y)
    model = res["model"]

    with st.expander("🧮 Penjelasan Matematika: Hitung Manual dari Hasil Model Anda"):
        st.caption("Semua angka di bawah diambil langsung dari model yang sedang aktif dan akan berubah saat dataset / test size diubah.")

        st.markdown(r"#### Langkah 1 — Dari mana prediksi $\hat{y}$ berasal?")
        idx = int(st.number_input("Pilih data uji ke-", min_value=1, max_value=n, value=1, step=1, key="math_reg_idx"))
        xi = np.asarray(res["X_test"].iloc[idx - 1], dtype=float)
        coef = np.asarray(model.coef_, dtype=float).ravel()
        b0 = float(model.intercept_)
        kontribusi = coef * xi
        st.dataframe(
            pd.DataFrame({"Fitur": feats, "Koefisien (β)": coef, "Nilai Fitur (x)": xi, "Kontribusi (β × x)": kontribusi}),
            hide_index=True, use_container_width=True,
        )
        pred_manual = b0 + kontribusi.sum()
        st.latex(rf"\hat{{y}} = \beta_0 + \sum_i \beta_i x_i = {b0:.4f} + ({kontribusi.sum():.4f}) = {pred_manual:.4f}")
        st.caption(
            f"Nilai aktual data ini **{y[idx - 1]:.4f}**, sehingga error = {y[idx - 1]:.4f} − {pred_manual:.4f} = **{y[idx - 1] - pred_manual:.4f}**."
        )

        st.markdown("#### Langkah 2 — Menghitung MAE, MSE & RMSE")
        n_show = min(5, n)
        contoh = pd.DataFrame({
            "Aktual (y)": y[:n_show],
            "Prediksi (ŷ)": yh[:n_show],
            "Error (y − ŷ)": e[:n_show],
            "|Error|": np.abs(e[:n_show]),
            "Error²": e[:n_show] ** 2,
        })
        contoh.index = [f"Data {i + 1}" for i in range(n_show)]
        st.dataframe(contoh, use_container_width=True)
        st.caption(f"Tabel hanya menampilkan {n_show} dari {n} data uji sebagai contoh. Rumus di bawah memakai seluruh {n} data uji.")
        sum_abs, sum_sq = np.abs(e).sum(), (e ** 2).sum()
        mae, mse = sum_abs / n, sum_sq / n
        rmse = mse ** 0.5
        st.latex(rf"\text{{MAE}} = \frac{{\sum |y_i - \hat{{y}}_i|}}{{n}} = \frac{{{sum_abs:.2f}}}{{{n}}} = {mae:.4f}")
        st.latex(rf"\text{{MSE}} = \frac{{\sum (y_i - \hat{{y}}_i)^2}}{{n}} = \frac{{{sum_sq:.2f}}}{{{n}}} = {mse:.4f}")
        st.latex(rf"\text{{RMSE}} = \sqrt{{\text{{MSE}}}} = \sqrt{{{mse:.4f}}} = {rmse:.4f}")
        st.caption(
            f"💡 RMSE ({rmse:.4f}) selalu ≥ MAE ({mae:.4f}). Semakin besar selisihnya, semakin banyak "
            "kesalahan besar yang 'dihukum' lebih berat karena dikuadratkan."
        )

        st.markdown(r"#### Langkah 3 — Menghitung $R^2$")
        ybar = y.mean()
        ss_res = (e ** 2).sum()
        ss_tot = ((y - ybar) ** 2).sum()
        r2 = 1 - ss_res / ss_tot
        st.markdown(
            rf"- $SS_{{res}}$ = total kuadrat error **model** = {ss_res:.2f}" "\n"
            rf"- $SS_{{tot}}$ = total kuadrat error jika **selalu menebak rata-rata** $\bar{{y}} = {ybar:.4f}$ = {ss_tot:.2f}"
        )
        st.latex(rf"R^2 = 1 - \frac{{SS_{{res}}}}{{SS_{{tot}}}} = 1 - \frac{{{ss_res:.2f}}}{{{ss_tot:.2f}}} = {r2:.4f}")
        if r2 >= 0:
            st.caption(f"💡 Model mengurangi total kesalahan kuadrat sebesar **{r2 * 100:.1f}%** dibandingkan hanya menebak nilai rata-rata.")
        else:
            st.caption("💡 R² negatif: model ini **lebih buruk** daripada sekadar menebak nilai rata-rata.")
        _cek_cocok(r2, res["metrics"]["R² Score"], "R²")


def math_klasifikasi(res, tn):
    cm = np.asarray(res["cm"], dtype=int)
    names = [tn[c] if tn else str(c) for c in res["classes"]]
    total = int(cm.sum())

    with st.expander("🧮 Penjelasan Matematika: Hitung Manual dari Confusion Matrix Anda"):
        st.caption("Semua angka di bawah diambil dari Confusion Matrix model yang sedang aktif.")

        st.markdown("#### Langkah 1 — Accuracy dari diagonal Confusion Matrix")
        st.dataframe(
            pd.DataFrame(cm, index=[f"Aktual: {x}" for x in names], columns=[f"Prediksi: {x}" for x in names]),
            use_container_width=True,
        )
        benar = int(np.trace(cm))
        st.latex(rf"\text{{Accuracy}} = \frac{{\text{{jumlah diagonal}}}}{{\text{{total data uji}}}} = \frac{{{benar}}}{{{total}}} = {benar / total:.4f}")

        st.markdown("#### Langkah 2 — Precision, Recall & F1 untuk satu kelas")
        st.caption("Untuk menghitung metrik satu kelas, kelas itu dianggap 'positif' dan kelas lainnya dianggap 'negatif'.")
        pilih = st.selectbox("Kelas yang dianggap positif:", range(len(names)), format_func=lambda i: names[i], key="math_clf_cls")
        TP = int(cm[pilih, pilih])
        FP = int(cm[:, pilih].sum() - TP)
        FN = int(cm[pilih, :].sum() - TP)
        TN = total - TP - FP - FN
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("TP", TP, help=f"Sebenarnya '{names[pilih]}' dan ditebak '{names[pilih]}'.")
        c2.metric("FP", FP, help=f"Sebenarnya BUKAN '{names[pilih]}' tetapi ditebak '{names[pilih]}'.")
        c3.metric("FN", FN, help=f"Sebenarnya '{names[pilih]}' tetapi ditebak kelas lain (terlewat).")
        c4.metric("TN", TN, help="Bukan kelas ini dan memang tidak ditebak sebagai kelas ini.")
        p = _bagi(TP, TP + FP)
        r = _bagi(TP, TP + FN)
        f1 = _bagi(2 * p * r, p + r)
        st.latex(rf"\text{{Precision}} = \frac{{TP}}{{TP + FP}} = \frac{{{TP}}}{{{TP} + {FP}}} = {p:.4f}")
        st.latex(rf"\text{{Recall}} = \frac{{TP}}{{TP + FN}} = \frac{{{TP}}}{{{TP} + {FN}}} = {r:.4f}")
        st.latex(rf"F_1 = 2 \times \frac{{{p:.4f} \times {r:.4f}}}{{{p:.4f} + {r:.4f}}} = {f1:.4f}")
        st.caption(
            f"💡 Dari semua data yang **ditebak** '{names[pilih]}', **{p * 100:.1f}%** memang benar (Precision). "
            f"Dari semua data yang **sebenarnya** '{names[pilih]}', **{r * 100:.1f}%** berhasil ditemukan (Recall)."
            + (f" Ada **{FN}** data yang terlewat (FN)." if FN else "")
        )

        st.markdown("#### Langkah 3 — Dari per kelas menjadi satu angka (Macro Average)")
        baris, P, R, F = [], [], [], []
        for i, nm in enumerate(names):
            tp = int(cm[i, i])
            fp = int(cm[:, i].sum() - tp)
            fn = int(cm[i, :].sum() - tp)
            pi, ri = _bagi(tp, tp + fp), _bagi(tp, tp + fn)
            fi = _bagi(2 * pi * ri, pi + ri)
            P.append(pi); R.append(ri); F.append(fi)
            baris.append({"Kelas": nm, "TP": str(tp), "FP": str(fp), "FN": str(fn), "Precision": pi, "Recall": ri, "F1": fi})
        baris.append({"Kelas": "Rata-rata (macro)", "TP": "-", "FP": "-", "FN": "-",
                      "Precision": np.mean(P), "Recall": np.mean(R), "F1": np.mean(F)})
        st.dataframe(pd.DataFrame(baris), hide_index=True, use_container_width=True)
        k = len(names)
        suku_p = " + ".join(f"{v:.4f}" for v in P)
        st.latex(rf"\text{{Precision}}_{{macro}} = \frac{{{suku_p}}}{{{k}}} = {np.mean(P):.4f}")
        st.caption(
            "💡 Macro average memberi bobot **sama** pada setiap kelas, jadi kelas kecil sama pentingnya dengan kelas besar. "
            "F1 macro adalah rata-rata F1 tiap kelas (bukan F1 dari Precision & Recall macro)."
        )
        _cek_cocok(float(np.mean(P)), res["metrics"]["Precision"], "Precision")
        _cek_cocok(float(np.mean(R)), res["metrics"]["Recall"], "Recall")
        _cek_cocok(float(np.mean(F)), res["metrics"]["F1 Score"], "F1 Score")


def math_clustering(res, df):
    from sklearn.metrics import silhouette_samples

    X = df.drop(columns="target", errors="ignore")
    feats = list(X.columns)
    sc, km = res["scaler"], res["model"]
    Xs = sc.transform(X)
    labels = np.asarray(res["labels"])
    cent = km.cluster_centers_
    k = len(cent)
    n = len(Xs)

    with st.expander("🧮 Penjelasan Matematika: Hitung Manual dari Hasil K-Means Anda"):
        st.caption("Semua angka di bawah diambil dari model K-Means yang sedang aktif.")
        idx = int(st.number_input("Pilih titik data ke-", min_value=1, max_value=n, value=1, step=1, key="math_clu_idx")) - 1

        st.markdown("#### Langkah 1 — Standarisasi fitur (StandardScaler)")
        z = Xs[idx]
        st.dataframe(
            pd.DataFrame({
                "Fitur": feats,
                "Nilai asli (x)": X.iloc[idx].values.astype(float),
                "Rata-rata (μ)": sc.mean_,
                "Std (σ)": sc.scale_,
                "Hasil z = (x − μ) / σ": z,
            }),
            hide_index=True, use_container_width=True,
        )
        x0, m0, s0 = float(X.iloc[idx, 0]), float(sc.mean_[0]), float(sc.scale_[0])
        st.latex(rf"z_{{1}} = \frac{{x - \mu}}{{\sigma}} = \frac{{{x0:.4f} - {m0:.4f}}}{{{s0:.4f}}} = {z[0]:.4f}")
        st.caption("Contoh perhitungan untuk fitur pertama. Setelah ini semua fitur berskala sebanding sehingga adil dalam perhitungan jarak.")

        st.markdown("#### Langkah 2 — Jarak ke setiap centroid & penentuan cluster")
        d_cent = np.linalg.norm(cent - z, axis=1)
        st.dataframe(
            pd.DataFrame({"Cluster": range(k), "Jarak Euclidean ke centroid": d_cent}).sort_values("Jarak Euclidean ke centroid"),
            hide_index=True, use_container_width=True,
        )
        j = int(np.argmin(d_cent))
        st.latex(rf"d(x, \mu_j) = \sqrt{{\sum_i (z_i - \mu_{{j,i}})^2}} \;\Rightarrow\; \min = {d_cent[j]:.4f} \text{{ (cluster {j})}}")
        st.caption(f"💡 Titik terdekat dengan centroid cluster **{j}**; label dari K-Means: **{int(labels[idx])}**.")

        st.markdown("#### Langkah 3 — Inertia (WCSS)")
        sse = [float(((Xs[labels == c] - cent[c]) ** 2).sum()) for c in range(k)]
        st.dataframe(
            pd.DataFrame({"Cluster": range(k), "Jumlah anggota": [int((labels == c).sum()) for c in range(k)], "SSE cluster": sse}),
            hide_index=True, use_container_width=True,
        )
        st.latex(rf"J = \sum_{{j=1}}^{{k}} \sum_{{x_i \in C_j}} \|x_i - \mu_j\|^2 = {' + '.join(f'{v:.2f}' for v in sse)} = {sum(sse):.4f}")
        _cek_cocok(sum(sse), res["metrics"]["Inertia"], "Inertia")

        st.markdown(f"#### Langkah 4 — Silhouette untuk titik data ke-{idx + 1}")
        d = np.linalg.norm(Xs - z, axis=1)
        sama = labels == labels[idx]
        a = d[sama].sum() / (sama.sum() - 1) if sama.sum() > 1 else 0.0
        lain = {int(c): float(d[labels == c].mean()) for c in np.unique(labels) if c != labels[idx]}
        st.markdown(f"- **a(i)** = rata-rata jarak ke anggota lain di cluster yang sama (cluster {int(labels[idx])}) = **{a:.4f}**")
        st.markdown("- Rata-rata jarak ke cluster lain: " + ", ".join(f"cluster {c} = {v:.4f}" for c, v in lain.items()))
        if lain and sama.sum() > 1:
            b = min(lain.values())
            s_i = (b - a) / max(a, b)
            st.latex(rf"s(i) = \frac{{b(i) - a(i)}}{{\max(a(i), b(i))}} = \frac{{{b:.4f} - {a:.4f}}}{{{max(a, b):.4f}}} = {s_i:.4f}")
            if s_i >= 0.5:
                st.caption("💡 Nilai mendekati 1: titik ini sangat cocok berada di clusternya.")
            elif s_i >= 0:
                st.caption("💡 Nilai kecil positif: titik ini berada dekat batas antar cluster.")
            else:
                st.caption("💡 Nilai negatif: titik ini justru lebih dekat ke cluster lain (kemungkinan salah cluster).")
        s_all = silhouette_samples(Xs, labels)
        st.caption(
            f"Silhouette Score model = rata-rata s(i) dari seluruh **{n}** titik = **{s_all.mean():.4f}**. "
            f"Terdapat **{int((s_all < 0).sum())}** titik dengan s(i) negatif."
        )
        _cek_cocok(float(s_all.mean()), res["metrics"]["Silhouette"], "Silhouette")

        st.markdown("#### Langkah 5 — Davies-Bouldin Index")
        sp = [float(np.linalg.norm(Xs[labels == c] - cent[c], axis=1).mean()) for c in range(k)]
        Rij = []
        for c in range(k):
            Rij.append(max((sp[c] + sp[o]) / np.linalg.norm(cent[c] - cent[o]) for o in range(k) if o != c))
        st.dataframe(
            pd.DataFrame({"Cluster": range(k), "Sebaran S (rata-rata jarak ke centroid)": sp, "R terburuk (dengan cluster lain)": Rij}),
            hide_index=True, use_container_width=True,
        )
        st.latex(rf"DB = \frac{{1}}{{k}} \sum_{{i=1}}^{{k}} \max_{{j \neq i}} \frac{{S_i + S_j}}{{d(\mu_i, \mu_j)}} = {np.mean(Rij):.4f}")
        st.caption("💡 Semakin **kecil** DB, semakin padat tiap cluster dan semakin jauh antar centroid.")
        _cek_cocok(float(np.mean(Rij)), res["metrics"]["Davies-Bouldin"], "Davies-Bouldin")


def select_data(task, key_prefix):
    c1, c2 = st.columns([1, 2])
    name = c1.selectbox("Pilih Dataset (scikit-learn)", TASK_DATASETS[task], key=f"{key_prefix}_ds")
    use_extra = c2.checkbox(
        "Sertakan data tambahan dari MongoDB saat training",
        value=True, disabled=db is None, key=f"{key_prefix}_extra",
    )
    st.caption(f"ℹ️ **Deskripsi Dataset:** {DATASET_DESC.get(name, '')}")
    df, n_extra, feats, tn = get_training_df(task, name, use_extra and db is not None)
    st.caption(f"📊 Total Data Latih: **{len(df)}** baris (Data Tambahan Pengguna: **{n_extra}**)")
    return name, df, n_extra, feats, tn


# ============================================================================
# HALAMAN: DASHBOARD BERANDA (TEMA LILAC TERANG)
# ============================================================================
def page_home():
    last = st.session_state.get("last_run")

    if last:
        b_task = last["task_label"]
        b_ds = last["dataset"]
        b_algo = last["model_name"]
        n_samples = last["n_samples"]
        n_feats = last["n_feats"]
    else:
        df_default, feats_default, _ = load_base("Diabetes")
        b_task = "Regresi"
        b_ds = "Diabetes"
        b_algo = "Linear Regression"
        n_samples = len(df_default)
        n_feats = len(feats_default)

    # 1. Hero Banner Lilac
    st.markdown(f"""
    <div class="hero-banner">
        <div class="deco-1">💜</div>
        <div class="deco-2">🪻</div>
        <div class="deco-3">✨</div>
        <div class="hero-title">HALO, SELAMAT DATANG! 👋</div>
        <div class="hero-subtitle">Regresi, Klasifikasi, Fuzzy Logic, dan Clustering secara interaktif ✨ · TAS AI — Hana Meisaria (632025057)</div>
        <div class="pill-row">
            <span class="pill-badge">🎯 {b_task}</span>
            <span class="pill-badge">📊 {b_ds}</span>
            <span class="pill-badge">🧩 {b_algo}</span>
            <span class="pill-badge">🎲 Seed 42</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # 2. Grid Fitur Aplikasi (8 kartu rapi dalam 2 baris x 4 kolom)
    st.markdown('<div class="section-heading">✨ Fitur Aplikasi</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="feature-grid">
        <div class="feature-card">
            <div class="feature-icon">📈</div>
            <div class="feature-title">Regresi Linear</div>
            <div class="feature-desc">Prediksi angka kontinu pada Diabetes & California Housing</div>
        </div>
        <div class="feature-card">
            <div class="feature-icon">🧩</div>
            <div class="feature-title">Klasifikasi</div>
            <div class="feature-desc">Random Forest, Logistic, KNN, & Decision Tree</div>
        </div>
        <div class="feature-card">
            <div class="feature-icon">🌫️</div>
            <div class="feature-title">Fuzzy Logic</div>
            <div class="feature-desc">Fuzzy KNN & Fuzzy C-Means + derajat keanggotaan</div>
        </div>
        <div class="feature-card">
            <div class="feature-icon">🔵</div>
            <div class="feature-title">Clustering</div>
            <div class="feature-desc">K-Means + proyeksi PCA 2D, Elbow & Silhouette</div>
        </div>
        <div class="feature-card">
            <div class="feature-icon">🏆</div>
            <div class="feature-title">Galeri Training</div>
            <div class="feature-desc">Leaderboard & referensi parameter antar pengunjung</div>
        </div>
        <div class="feature-card">
            <div class="feature-icon">🗄️</div>
            <div class="feature-title">MongoDB & Retrain</div>
            <div class="feature-desc">Tambah/hapus data, upload CSV, & retrain model</div>
        </div>
        <div class="feature-card">
            <div class="feature-icon">💬</div>
            <div class="feature-title">Komentar + Groq AI</div>
            <div class="feature-desc">Komentar publik, ringkasan & analisis sentimen LLM</div>
        </div>
        <div class="feature-card">
            <div class="feature-icon">📝</div>
            <div class="feature-title">Tutor & Mini Kuis</div>
            <div class="feature-desc">Asisten belajar AI & kuis interaktif evaluasi ML</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Tombol Pintas Modul
    q1, q2, q3, q4, q5, q6 = st.columns(6)
    q1.button("📈 Regresi", key="btn_nav_reg", use_container_width=True, on_click=change_page, args=("📈 Regresi Linear",))
    q2.button("🧪 Klasifikasi", key="btn_nav_clf", use_container_width=True, on_click=change_page, args=("🧪 Klasifikasi",))
    q3.button("🔵 Clustering", key="btn_nav_clu", use_container_width=True, on_click=change_page, args=("🔵 Clustering",))
    q4.button("🏆 Galeri", key="btn_nav_galeri", use_container_width=True, on_click=change_page, args=("🏆 Galeri Hasil Training",))
    q5.button("🤖 Asisten AI", key="btn_nav_ai", use_container_width=True, on_click=change_page, args=("🤖 Asisten AI (Groq)",))
    q6.button("📝 Mini Kuis", key="btn_nav_kuis", use_container_width=True, on_click=change_page, args=("📝 Mini Kuis ML",))

    st.write("")

    # 3. Baris Bawah 3 Kartu: Ringkasan Dataset, Status Model, Skor Model
    col1, col2, col3 = st.columns([1.2, 1.25, 0.95], gap="medium")

    with col1:
        st.markdown(f"""
        <div class="bottom-card">
            <div class="bottom-card-title">📊 Ringkasan Dataset ({b_ds})</div>
            <div class="metric-row">
                <div class="metric-box">
                    <div class="metric-label">📏 Sampel</div>
                    <div class="metric-value">{n_samples}</div>
                </div>
                <div class="metric-box">
                    <div class="metric-label">🎛️ Fitur</div>
                    <div class="metric-value">{n_feats}</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        if last:
            status_html = (
                f'<div class="status-ok-box">'
                f'Model aktif: <b>{last["model_name"]}</b> ({last["task_label"]})<br>'
                f'Dataset: <b>{last["dataset"]}</b> · Metrik utama: <b>{last["score_label"]} = {last["score_val"]:.4f}</b>'
                f'</div>'
            )
        else:
            status_html = (
                '<div class="status-info-box">'
                'Belum ada model dilatih di sesi ini. Pilih modul <b>Regresi, Klasifikasi,</b> atau <b>Clustering</b> di sidebar.'
                '</div>'
            )
        st.markdown(f"""
        <div class="bottom-card">
            <div class="bottom-card-title">🧠 Status Model</div>
            {status_html}
        </div>
        """, unsafe_allow_html=True)

    with col3:
        score_lbl = last["score_label"] if last else "Skor Utama"
        score_txt = f'{last["score_val"]:.3f}' if last else "--"
        st.markdown(f"""
        <div class="bottom-card">
            <div class="bottom-card-title" style="text-align:center;">🎯 Skor Model</div>
            <div class="score-center">
                <div class="score-label">{score_lbl}</div>
                <div class="score-ring">{score_txt}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.write("---")

    # 4. Peta Konsep Machine Learning
    st.markdown('<div class="section-heading">📚 Peta Konsep Machine Learning</div>', unsafe_allow_html=True)
    with st.expander("📖 Pelajari Perbedaan Supervised vs Unsupervised Learning & Alur Kerja ML", expanded=False):
        konsep_df = pd.DataFrame([
            {
                "Metode": "Regresi Linear",
                "Paradigma": "Supervised Learning (Ada Label Target)",
                "Bentuk Output": "Angka kontinu (mis. harga rumah, progres medis)",
                "Metrik Evaluasi": "R² Score, MAE, RMSE",
                "Pertanyaan yang Dijawab": "Berapa nilai yang diprediksi?",
            },
            {
                "Metode": "Klasifikasi (Crisp)",
                "Paradigma": "Supervised Learning (Ada Label Target)",
                "Bentuk Output": "Kategori / Kelas diskrit (mis. jinak vs ganas)",
                "Metrik Evaluasi": "Accuracy, Precision, Recall, F1 Score",
                "Pertanyaan yang Dijawab": "Termasuk kategori manakah data ini?",
            },
            {
                "Metode": "Fuzzy Klasifikasi (Fuzzy KNN / FCM)",
                "Paradigma": "Soft Computing / Supervised (Fuzzy)",
                "Bentuk Output": "Derajat keanggotaan 0–1 per kelas",
                "Metrik Evaluasi": "Accuracy, Precision, Recall, F1 Score + Membership",
                "Pertanyaan yang Dijawab": "Seberapa besar kecocokan data ini dengan setiap kelas?",
            },
            {
                "Metode": "Clustering (K-Means)",
                "Paradigma": "Unsupervised Learning (Tanpa Label)",
                "Bentuk Output": "Kelompok alami (Cluster 0, 1, 2, ...)",
                "Metrik Evaluasi": "Silhouette, Davies-Bouldin, Inertia",
                "Pertanyaan yang Dijawab": "Pola kelompok alami apa yang terbentuk?",
            },
        ])
        st.dataframe(konsep_df, use_container_width=True, hide_index=True)

        p1, p2, p3 = st.columns(3)
        p1.info("**1️⃣ Eksplorasi & Split Data**\n\nData dibagi menjadi *Data Latih* (belajar pola) dan *Data Uji* (menguji generalisasi agar tidak *overfitting*).")
        p2.info("**2️⃣ Pemodelan & Evaluasi**\n\nModel dilatih lalu diukur kinerjanya lewat metrik (arahkan kursor ke `?` di tiap metrik) dan grafik interaktif.")
        p3.info("**3️⃣ Simulasi & Retraining**\n\nUji prediksi dengan nilai fitur baru, lalu tambahkan data ke MongoDB untuk melihat dampak *retraining*.")

    # 5. Status Server & Database
    with st.expander("⚡ Status Server & Ringkasan Database MongoDB", expanded=False):
        if db is None:
            st.error(f"MongoDB belum terhubung: {db_error}")
        else:
            st.success("🟢 MongoDB Atlas terhubung aktif.")
            rows = []
            for task, names in TASK_DATASETS.items():
                for n in names:
                    rows.append({
                        "Modul": TASK_LABEL[task],
                        "Dataset": n,
                        "Deskripsi": DATASET_DESC.get(n, "-"),
                        "Data Tambahan": db.dataset_tambahan.count_documents({"key": f"{task}:{n}"}),
                        "Model Tersimpan": db.models.count_documents({"key": f"{task}:{n}"}),
                        "Hasil di Galeri": db.hasil_training.count_documents({"task": task, "dataset": n}),
                    })
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
            st.caption(f"💬 Total Komentar Publik: **{db.komentar.count_documents({})}** | 📦 Total Model: **{db.models.count_documents({})}**")

    st.markdown('<div class="footer">© 2026 Hana Meisaria (632025057) · TAS AI · Universitas Kristen Satya Wacana</div>', unsafe_allow_html=True)


# ============================================================================
# HALAMAN: REGRESI
# ============================================================================
def page_regresi():
    st.markdown('<h1 class="main-title-container">📈 <span class="gradient-text">Regresi Linear</span></h1>', unsafe_allow_html=True)
    st.info("💡 **Regresi Linear** termasuk *Supervised Learning* yang memprediksi nilai target berupa **angka kontinu** dengan mencari garis/bidang terbaik (*best-fit line*) melalui metode *Ordinary Least Squares (OLS)*.")

    with st.expander("📖 Pelajari Teori, Rumus Matematika & Cara Membaca Grafik"):
        st.markdown("**1. Persamaan Regresi Linear Berganda:**")
        st.latex(r"\hat{y} = \beta_0 + \beta_1 x_1 + \beta_2 x_2 + \dots + \beta_n x_n")
        st.markdown(
            r"- $\hat{y}$: nilai target hasil prediksi" "\n"
            r"- $\beta_0$: *intercept* (nilai dasar ketika seluruh fitur bernilai 0)" "\n"
            r"- $\beta_i$: koefisien bobot fitur $x_i$ — tanda **positif (+)** berarti kenaikan fitur menaikkan prediksi; tanda **negatif (-)** berarti menurunkan prediksi"
        )
        st.markdown("**2. Rumus Metrik Evaluasi Utama:**")
        r1, r2, r3 = st.columns(3)
        with r1:
            st.latex(r"R^2 = 1 - \frac{\sum (y_i - \hat{y}_i)^2}{\sum (y_i - \bar{y})^2}")
            st.caption("Mendekati 1 = variasi data dijelaskan dengan sangat baik.")
        with r2:
            st.latex(r"\text{MAE} = \frac{1}{n}\sum_{i=1}^{n}|y_i - \hat{y}_i|")
            st.caption("Rata-rata besar kesalahan absolut.")
        with r3:
            st.latex(r"\text{RMSE} = \sqrt{\frac{1}{n}\sum_{i=1}^{n}(y_i - \hat{y}_i)^2}")
            st.caption("Sensitif terhadap kesalahan besar (*outlier*).")
        st.markdown(
            "**3. Cara Membaca Grafik *Aktual vs Prediksi*:**\n"
            "- Garis putus-putus adalah **garis ideal** (Nilai Aktual = Nilai Prediksi).\n"
            "- Semakin rapat titik menempel pada garis ideal, semakin kecil kesalahan prediksi model."
        )

    name, df, n_extra, feats, tn = select_data("regresi", "reg")
    show_feature_glossary(name)

    test_size = st.slider(
        "Proporsi Data Uji (Test Size)",
        0.1, 0.5, 0.2, 0.05,
        help="Persentase data yang disisihkan sebagai ujian model. Sisanya digunakan untuk melatih model agar kita bisa mengevaluasi kemampuan generalisasi pada data baru.",
        key="reg_ts",
    )

    with st.spinner("Melatih model regresi..."):
        res = run_model("regresi", df, test_size=test_size)

    st.session_state["last_run"] = {
        "task_label": "Regresi",
        "dataset": name,
        "model_name": "Linear Regression",
        "n_samples": len(df),
        "n_feats": len(feats),
        "score_label": "R² Score",
        "score_val": res["metrics"]["R² Score"],
    }

    st.caption(f"🧪 Pembagian Data: **{res['n_train']}** baris Data Latih · **{res['n_test']}** baris Data Uji")
    show_metrics(res["metrics"])
    show_quick_interpretation("regresi", res["metrics"])
    st.write("")
    render_ai_insight(
        f"Dataset: {name}. Model: Regresi Linear. Metrik evaluasi: {res['metrics']}. "
        f"Jelaskan secara singkat dan edukatif arti metrik tersebut serta apakah performa model ini sudah baik.",
        key="reg_ai_explain",
    )
    st.write("")

    render_reference_hint("regresi", name, res["metrics"])
    render_save_panel("regresi", name, "Linear Regression", make_params("regresi", test_size=test_size), res, n_extra)
    st.write("")

    math_regresi(res, feats)
    t1, t2, t3 = st.tabs(["📊 Dataset Overview", "📈 Grafik & Koefisien", "🔮 Simulator Prediksi"])

    with t1:
        st.dataframe(df.tail(20) if n_extra else df.head(20), use_container_width=True)
        st.write("**Statistik Deskriptif (Mean, Std, Min, Kuartil, Max):**")
        st.dataframe(df.describe().T, use_container_width=True)

    with t2:
        c1, c2 = st.columns(2)
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=res["y_test"], y=res["pred"], mode="markers",
            marker=dict(size=8, color="#8b5cf6", opacity=0.75, line=dict(width=0.5, color="#ffffff")),
            name="Data Uji",
            hovertemplate="Aktual: %{x:.2f}<br>Prediksi: %{y:.2f}<extra></extra>",
        ))
        lims = [min(res["y_test"].min(), res["pred"].min()), max(res["y_test"].max(), res["pred"].max())]
        fig.add_trace(go.Scatter(
            x=lims, y=lims, mode="lines",
            line=dict(color="#4c1d95", dash="dash", width=2),
            name="Garis Ideal",
        ))
        fig.update_layout(
            title=f"Aktual vs Prediksi — {name}",
            xaxis_title="Nilai Aktual", yaxis_title="Nilai Prediksi",
            template="lilac_light", height=400,
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(255,255,255,0.85)",
        )
        c1.plotly_chart(fig, use_container_width=True)

        coef = pd.DataFrame({
            "Fitur": feats,
            "Koefisien": res["model"].coef_,
            "Arah Pengaruh": ["⬆️ Positif" if c >= 0 else "⬇️ Negatif" for c in res["model"].coef_],
        })
        c2.write(r"**Koefisien Bobot Regresi ($\beta_i$):**")
        c2.dataframe(coef, hide_index=True, use_container_width=True)
        c2.caption(f"Intercept ($\\beta_0$): **{res['model'].intercept_:.4f}**")

    with t3:
        base = load_base(name)[0]
        st.markdown(f"##### Input Nilai Fitur ({name}):")
        st.caption("💡 Arahkan kursor ke ikon `?` di tiap kolom input untuk melihat arti fitur. Nilai awal diisi median dataset.")
        vals = feature_inputs(base, feats, "reg_pred", dataset_name=name)
        if st.button("🚀 Jalankan Prediksi", key="reg_btn", type="primary"):
            with st.spinner("Menghitung prediksi..."):
                y = res["model"].predict(pd.DataFrame([vals], columns=feats))[0]
            st.balloons()
            st.success(f"🎯 Hasil Prediksi Target: **{y:.4f}**")
            hasil_df = pd.DataFrame([vals + [y]], columns=feats + ["Prediksi"])
            st.download_button(
                "📥 Unduh Hasil Prediksi (CSV)",
                hasil_df.to_csv(index=False).encode("utf-8"),
                file_name=f"prediksi_regresi_{name.lower().replace(' ', '_')}.csv",
                mime="text/csv",
            )


# ============================================================================
# HALAMAN: KLASIFIKASI (dengan Fuzzy)
# ============================================================================
def page_klasifikasi():
    st.markdown('<h1 class="main-title-container">🧪 <span class="gradient-text">Klasifikasi Multi-Algoritma</span></h1>', unsafe_allow_html=True)
    st.info(
        "💡 **Klasifikasi** termasuk *Supervised Learning* yang memetakan fitur input ke dalam **kelas/kategori diskrit**. "
        "Modul ini mendukung **4 algoritma crisp** (Random Forest, Logistic Regression, KNN, Decision Tree) dan "
        "**2 algoritma fuzzy** (Fuzzy KNN, Fuzzy C-Means) yang menghasilkan **derajat keanggotaan** per kelas."
    )

    with st.expander("📖 Pelajari Teori, Rumus Evaluasi, Perbandingan Algoritma & Konsep Fuzzy"):
        st.markdown("**1. Rumus Metrik Evaluasi Klasifikasi (berdasarkan TP, TN, FP, FN):**")
        k1, k2 = st.columns(2)
        with k1:
            st.latex(r"\text{Accuracy} = \frac{TP + TN}{TP + TN + FP + FN}")
            st.latex(r"\text{Precision} = \frac{TP}{TP + FP}")
        with k2:
            st.latex(r"\text{Recall} = \frac{TP}{TP + FN}")
            st.latex(r"F_1 = 2 \times \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}}")
        st.markdown(
            "- **TP (True Positive):** Kelas positif ditebak benar.\n"
            "- **FP (False Positive):** Sebenarnya negatif tetapi ditebak positif (*false alarm*).\n"
            "- **FN (False Negative):** Sebenarnya positif tetapi terlewat ditebak negatif (kritis pada deteksi kanker).\n\n"
            "**2. Cara Membaca *Confusion Matrix*:**\n"
            "- Angka pada **diagonal utama** (kiri atas ke kanan bawah) adalah jumlah prediksi yang **benar**.\n"
            "- Angka di luar diagonal menunjukkan jumlah data yang **salah diklasifikasikan** ke kelas lain.\n\n"
            "**3. Karakteristik Algoritma *Crisp* yang Tersedia:**\n"
            "- **Random Forest:** Ensemble 200 pohon keputusan, stabil & tahan *overfitting*.\n"
            "- **Logistic Regression:** Model probabilistik linier dengan fungsi sigmoid (butuh *StandardScaler*).\n"
            "- **KNN:** Voting tetangga terdekat berdasarkan jarak Euclidean (*StandardScaler*).\n"
            "- **Decision Tree:** Pohon keputusan bercabang; mudah diinterpretasi namun rentan *overfitting*."
        )

        st.markdown("**4. Konsep Fuzzy Logic (Logika Samar) — Perbedaan dengan *Crisp*:**")
        st.markdown(
            "Klasifikasi *crisp* hanya memberi keputusan **ya/tidak** (0 atau 1). "
            "Klasifikasi **fuzzy** memberi **derajat keanggotaan** di setiap kelas (0.0 – 1.0), "
            "sehingga model bisa menyatakan *\"saya 70% yakin kelas A, 30% yakin kelas B\"*. "
            "Pendekatan ini berguna ketika batas antar kelas **tidak tegas** (misal stadium awal penyakit, "
            "kualitas wine yang berkelanjutan, dsb.)."
        )
        st.markdown("**Fuzzy KNN — bobot tetangga & derajat keanggotaan:**")
        st.latex(r"w_i = \frac{1}{d_i^{\,2/(m-1)}}, \qquad u_c(x) = \frac{\sum_{i:\,y_i=c} w_i}{\sum_{i} w_i}")
        st.markdown(
            "dengan $d_i$ = jarak ke tetangga ke-$i$ dan $m$ = **fuzzifier** (m > 1). "
            "Semakin besar $m$, semakin **kabur** batas antar kelas; semakin mendekati 1, semakin **tegas** (mendekati crisp)."
        )
        st.markdown("**Fuzzy C-Means (FCM) — update derajat keanggotaan & centroid:**")
        st.latex(r"u_{ik} = \left[ \sum_{j=1}^{c} \left( \frac{d_{ik}}{d_{jk}} \right)^{2/(m-1)} \right]^{-1}, \quad v_k = \frac{\sum_i u_{ik}^m x_i}{\sum_i u_{ik}^m}")
        st.markdown(
            "Setelah FCM konvergen, setiap **cluster fuzzy** dipetakan ke **kelas mayoritas** anggotanya. "
            "Prediksi akhir = jumlah membership semua cluster yang memetakan ke kelas yang sama."
        )

    name, df, n_extra, feats, tn = select_data("klasifikasi", "clf")
    show_feature_glossary(name)

    c1, c2 = st.columns(2)
    model_name = c1.selectbox(
        "Pilih Algoritma ML",
        ["Random Forest", "Logistic Regression", "KNN", "Decision Tree",
         "Fuzzy KNN", "Fuzzy C-Means"],
        key="clf_model",
    )
    test_size = c2.slider(
        "Proporsi Data Uji",
        0.1, 0.5, 0.2, 0.05,
        help="Menggunakan Stratified Split agar proporsi tiap kelas tetap seimbang di Data Latih dan Data Uji.",
        key="clf_ts",
    )

    algo_desc = {
        "Random Forest": "🌲 **Random Forest:** Ensemble 200 Decision Tree, stabil & akurat.",
        "Logistic Regression": "📐 **Logistic Regression:** Model probabilistik linier dengan StandardScaler.",
        "KNN": "📍 **KNN:** Klasifikasi berdasarkan jarak ke 5 tetangga terdekat (dengan StandardScaler).",
        "Decision Tree": "🌿 **Decision Tree:** Pohon keputusan bercabang berbasis pemisahan Gini.",
        "Fuzzy KNN": "🌫️ **Fuzzy KNN:** KNN dengan bobot fuzzy 1/d^(2/(m−1)); setiap kelas memperoleh **derajat keanggotaan** 0–1.",
        "Fuzzy C-Means": "🎨 **Fuzzy C-Means:** Clustering fuzzy; setiap cluster dipetakan ke kelas mayoritas lalu membership-nya diagregasi.",
    }
    st.caption(algo_desc[model_name])

    fuzzy_k, fuzzy_m = 5, 2.0
    if model_name in ("Fuzzy KNN", "Fuzzy C-Means"):
        f1, f2 = st.columns(2)
        fuzzy_m = f1.slider(
            "Fuzzifier (m)", 1.1, 5.0, 2.0, 0.1,
            help=(
                "Mengatur seberapa 'kabur' batas antar kelas. "
                "m → 1.1 = hampir crisp (tegas). m besar = membership lebih merata / ambigu. "
                "Nilai ideal umumnya 1.5 – 2.5."
            ),
            key="clf_fuzzy_m",
        )
        if model_name == "Fuzzy KNN":
            fuzzy_k = f2.slider(
                "Jumlah Tetangga (K)", 1, 15, 5, 1,
                help="Jumlah tetangga terdekat yang diberi bobot fuzzy w_i = 1/d^(2/(m−1)).",
                key="clf_fuzzy_k",
            )
        else:
            f2.caption("ℹ️ FCM otomatis memakai c = jumlah kelas dataset.")

    with st.spinner("Melatih model klasifikasi..."):
        res = run_model(
            "klasifikasi", df,
            model_name=model_name,
            test_size=test_size,
            fuzzy_k=fuzzy_k,
            fuzzy_m=fuzzy_m,
        )

    st.session_state["last_run"] = {
        "task_label": "Klasifikasi",
        "dataset": name,
        "model_name": model_name,
        "n_samples": len(df),
        "n_feats": len(feats),
        "score_label": "F1 Score",
        "score_val": res["metrics"]["F1 Score"],
    }

    st.caption(f"🧪 Pembagian Data (Stratified): **{res['n_train']}** baris Data Latih · **{res['n_test']}** baris Data Uji")
    show_metrics(res["metrics"])
    show_quick_interpretation("klasifikasi", res["metrics"])
    st.write("")
    render_ai_insight(
        f"Dataset: {name}. Algoritma Klasifikasi: {model_name}. "
        f"Parameter fuzzy: fuzzy_k={fuzzy_k}, fuzzy_m={fuzzy_m}. Metrik evaluasi: {res['metrics']}. "
        f"Jelaskan secara singkat dan edukatif kualitas performa klasifikasi model ini serta saran perbaikan jika ada.",
        key="clf_ai_explain",
    )
    st.write("")

    render_reference_hint("klasifikasi", name, res["metrics"])
    render_save_panel(
        "klasifikasi", name, model_name,
        make_params("klasifikasi", model_name, test_size=test_size, fuzzy_k=fuzzy_k, fuzzy_m=fuzzy_m),
        res, n_extra,
    )
    st.write("")

    math_klasifikasi(res, tn)
    t1, t2, t3 = st.tabs(["📊 Distribusi Data", "🎯 Confusion Matrix", "🔮 Simulator Prediksi"])

    with t1:
        show = df.copy()
        show["kelas"] = show["target"].map(lambda i: tn[int(i)] if tn else str(int(i)))
        st.dataframe(show.tail(20) if n_extra else show.head(20), use_container_width=True)
        vc = show["kelas"].value_counts().reset_index()
        vc.columns = ["Kelas", "Jumlah"]
        fig = px.bar(vc, x="Kelas", y="Jumlah", color="Kelas", title=f"Distribusi Kelas — {name}")
        fig.update_layout(template="lilac_light", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(255,255,255,0.85)")
        st.plotly_chart(fig, use_container_width=True)

    with t2:
        labels_display = [tn[c] if tn else str(c) for c in res["classes"]]
        fig = px.imshow(
            res["cm"], text_auto=True, aspect="auto",
            labels=dict(x="Prediksi", y="Aktual", color="Jumlah"),
            x=labels_display, y=labels_display,
            color_continuous_scale=LILAC_SCALE,
        )
        fig.update_layout(
            title=f"Confusion Matrix — {model_name} ({name})",
            template="lilac_light",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(255,255,255,0.85)",
            height=450,
        )
        st.plotly_chart(fig, use_container_width=True)
        st.caption("💡 Semakin pekat warna di garis diagonal (kiri-atas ke kanan-bawah), semakin tinggi ketepatan klasifikasi model.")

    with t3:
        base = load_base(name)[0]
        st.markdown(f"##### Input Nilai Fitur ({name}):")
        st.caption("💡 Arahkan kursor ke ikon `?` di tiap kolom input untuk melihat arti fitur.")
        vals = feature_inputs(base, feats, "clf_pred", dataset_name=name)
        if st.button("🚀 Prediksi Kelas", key="clf_btn", type="primary"):
            Xn = pd.DataFrame([vals], columns=feats)
            with st.spinner("Memprediksi kelas..."):
                pred = int(res["model"].predict(Xn)[0])
                pred_label = tn[pred] if tn else str(pred)
            st.balloons()
            st.success(f"🎯 Hasil Prediksi Kelas: **{pred_label}**")
            if hasattr(res["model"], "predict_proba"):
                proba = res["model"].predict_proba(Xn)[0]
                idx_labels = [tn[c] if tn else str(c) for c in res["model"].classes_]
                proba_df = pd.DataFrame({"Kelas": idx_labels, "Probabilitas": proba})
                fig = px.bar(
                    proba_df, x="Probabilitas", y="Kelas", orientation="h",
                    color="Probabilitas", color_continuous_scale=LILAC_SCALE,
                    title="Tingkat Keyakinan (Probabilitas) per Kelas",
                )
                fig.update_layout(
                    template="lilac_light",
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(255,255,255,0.85)",
                    height=300,
                    showlegend=False,
                )
                st.plotly_chart(fig, use_container_width=True)

                if model_name in ("Fuzzy KNN", "Fuzzy C-Means"):
                    st.markdown("##### 🌫️ Derajat Keanggotaan Fuzzy (Membership)")
                    st.caption(
                        "Berbeda dengan probabilitas klasifikasi biasa, nilai di bawah adalah "
                        "**derajat keanggotaan fuzzy** — seberapa cocok data ini dengan tiap kelas "
                        "berdasarkan rumus fuzzy (bukan sigmoid/voting crisp)."
                    )
                    fig_mem = plot_membership(proba, idx_labels)
                    fig_mem.update_layout(
                        template="lilac_light",
                        paper_bgcolor="rgba(0,0,0,0)",
                        plot_bgcolor="rgba(255,255,255,0.85)",
                    )
                    st.plotly_chart(fig_mem, use_container_width=True)
                    yakin = float(np.max(proba))
                    if yakin >= 0.8:
                        st.success(f"🎯 Model **sangat yakin** ({yakin*100:.1f}%) memilih kelas ini.")
                    elif yakin >= 0.5:
                        st.info(
                            f"👍 Model **cukup yakin** ({yakin*100:.1f}%). "
                            "Coba ubah fuzzifier m untuk melihat pergeseran membership."
                        )
                    else:
                        st.warning(
                            f"⚠️ Model **kurang yakin** ({yakin*100:.1f}%) — kelas ini sangat ambigu. "
                            "Coba turunkan m (mendekati 1.1) agar keputusan lebih tegas."
                        )


# ============================================================================
# HALAMAN: CLUSTERING
# ============================================================================
def page_clustering():
    st.markdown('<h1 class="main-title-container">🔵 <span class="gradient-text">Clustering K-Means</span></h1>', unsafe_allow_html=True)
    st.info("💡 **Clustering K-Means** termasuk *Unsupervised Learning* yang mengelompokkan data **tanpa memakai label target**, sehingga data dengan kemiripan fitur tinggi berada dalam satu cluster.")

    with st.expander("📖 Pelajari Teori K-Means, Rumus Evaluasi, PCA & Cara Memilih k"):
        st.markdown("**1. Fungsi Objektif K-Means (Meminimalkan Inertia / WCSS):**")
        st.latex(r"J = \sum_{j=1}^{k} \sum_{x_i \in C_j} \| x_i - \mu_j \|^2")
        st.markdown(
            r"- $k$: jumlah cluster yang ditentukan pengguna" "\n"
            r"- $\mu_j$: titik pusat (*centroid*) dari cluster ke-$j$" "\n"
            r"- Sebelum K-Means dijalankan, seluruh fitur distandarisasi dengan **StandardScaler** ($z = \frac{x - \mu}{\sigma}$) agar fitur berskala besar tidak mendominasi perhitungan jarak."
        )
        st.markdown("**2. Rumus Silhouette Score untuk Evaluasi Kualitas Cluster:**")
        st.latex(r"s(i) = \frac{b(i) - a(i)}{\max(a(i), b(i))}")
        st.markdown(
            "- $a(i)$: rata-rata jarak titik $i$ ke titik lain di **cluster yang sama** (kohesi — makin kecil makin baik)\n"
            "- $b(i)$: rata-rata jarak titik $i$ ke titik di **cluster tetangga terdekat** (separasi — makin besar makin baik)\n\n"
            "**3. Mengapa Menggunakan PCA (Principal Component Analysis)?**\n"
            "- Dataset memiliki 4 hingga 13 dimensi fitur yang tidak bisa digambar langsung. **PCA** memampatkan seluruh fitur menjadi 2 sumbu utama (**PC1** dan **PC2**) agar pemisahan cluster dapat divisualisasikan secara 2D.\n\n"
            "**4. Cara Memilih Jumlah Cluster ($k$) Ideal:**\n"
            "- Lihat titik tekukan siku (*Elbow*) pada grafik **Inertia**, lalu konfirmasi pada nilai tertinggi di grafik **Silhouette Score**."
        )

    name, df, n_extra, feats, tn = select_data("clustering", "clu")
    show_feature_glossary(name)

    k = st.slider(
        "Jumlah Cluster (k)",
        2, 10, 3,
        help="Jumlah kelompok yang dibentuk K-Means. Bandingkan hasil beberapa nilai k menggunakan grafik Elbow & Silhouette di bawah.",
        key="clu_k",
    )

    with st.spinner("Menjalankan K-Means..."):
        res = run_model("clustering", df, k=k)

    st.session_state["last_run"] = {
        "task_label": "Clustering",
        "dataset": name,
        "model_name": f"K-Means (k={k})",
        "n_samples": len(df),
        "n_feats": len(feats),
        "score_label": "Silhouette",
        "score_val": res["metrics"]["Silhouette"],
    }

    show_metrics(res["metrics"])
    show_quick_interpretation("clustering", res["metrics"])
    st.write("")
    render_ai_insight(
        f"Dataset: {name}. Model: K-Means Clustering dengan k={k}. Metrik evaluasi: {res['metrics']}. "
        f"Jelaskan secara singkat dan edukatif kualitas pemisahan cluster berdasarkan Silhouette, Davies-Bouldin, dan Inertia.",
        key="clu_ai_explain",
    )
    st.write("")

    render_reference_hint("clustering", name, res["metrics"])
    render_save_panel("clustering", name, "K-Means", make_params("clustering", k=k), res, n_extra)
    st.write("")

    math_clustering(res, df)
    t1, t2, t3 = st.tabs(["🎨 Visualisasi Cluster (PCA)", "📊 Profiling Cluster", "🔮 Prediksi Cluster Baru"])

    with t1:
        c1, c2 = st.columns([1.2, 1])
        pca_df = pd.DataFrame(res["pca"], columns=["PC1", "PC2"])
        pca_df["Cluster"] = res["labels"].astype(str)
        fig = px.scatter(
            pca_df, x="PC1", y="PC2", color="Cluster",
            title=f"Proyeksi PCA 2D — K-Means (k={k}, {name})",
            color_discrete_sequence=LILAC_COLORS,
            opacity=0.85,
        )
        fig.update_traces(marker=dict(size=9, line=dict(width=0.7, color="#ffffff")))
        fig.update_layout(
            template="lilac_light",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(255,255,255,0.85)",
            height=450,
        )
        c1.plotly_chart(fig, use_container_width=True)

        c2.write("**Evaluasi Jumlah Cluster (Elbow & Silhouette):**")
        ed = elbow_data(df).reset_index()
        fig1 = px.line(ed, x="k", y="Inertia", markers=True, title="Elbow Method (Inertia — cari titik siku)")
        fig1.update_traces(line_color="#8b5cf6")
        fig1.update_layout(
            template="lilac_light",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(255,255,255,0.85)",
            height=205,
            margin=dict(t=40, b=10),
        )
        c2.plotly_chart(fig1, use_container_width=True)
        fig2 = px.line(ed, x="k", y="Silhouette", markers=True, title="Silhouette Score (cari titik puncak)")
        fig2.update_traces(line_color="#6d28d9")
        fig2.update_layout(
            template="lilac_light",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(255,255,255,0.85)",
            height=205,
            margin=dict(t=40, b=10),
        )
        c2.plotly_chart(fig2, use_container_width=True)

    with t2:
        X = df.drop(columns="target", errors="ignore").copy()
        X["cluster"] = res["labels"]
        c1, c2 = st.columns(2)
        vc = X["cluster"].value_counts().sort_index().reset_index()
        vc.columns = ["Cluster", "Jumlah"]
        fig = px.bar(
            vc, x="Cluster", y="Jumlah", color="Cluster",
            title="Jumlah Anggota per Cluster",
            color_continuous_scale=LILAC_SCALE,
        )
        fig.update_layout(
            template="lilac_light",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(255,255,255,0.85)",
            showlegend=False,
        )
        c1.plotly_chart(fig, use_container_width=True)
        c2.write("**Nilai Rata-rata Fitur per Cluster (Profiling):**")
        c2.dataframe(X.groupby("cluster").mean().style.format("{:.3f}"), use_container_width=True)
        st.caption("💡 Bandingkan nilai rata-rata antar baris untuk mengenali ciri khas tiap cluster (misal cluster dengan fitur tertinggi/terendah).")

    with t3:
        base = load_base(name)[0]
        st.markdown(f"##### Input Nilai Fitur ({name}):")
        st.caption("💡 Arahkan kursor ke ikon `?` di tiap kolom input untuk melihat arti fitur.")
        vals = feature_inputs(base, feats, "clu_pred", dataset_name=name)
        if st.button("🚀 Tentukan Cluster", key="clu_btn", type="primary"):
            Xn = res["scaler"].transform(pd.DataFrame([vals], columns=feats))
            cluster_id = int(res["model"].predict(Xn)[0])
            st.success(f"🎯 Data tersebut masuk ke **Cluster {cluster_id}**")


# ============================================================================
# HALAMAN: RETRAINING
# ============================================================================
def page_retrain():
    st.markdown('<h1 class="main-title-container">🗄️ <span class="gradient-text">Database & Retraining</span></h1>', unsafe_allow_html=True)
    with st.expander("📖 Mengapa Retraining Model Penting dalam Machine Learning?"):
        st.markdown(
            "Dalam praktik dunia nyata (*MLOps*), pola data terus berubah seiring waktu (**Data Drift**). "
            "Halaman ini menyimulasikan siklus pemeliharaan model:\n"
            "1. **Pengumpulan Data Baru:** Tambahkan observasi baru secara manual atau unggah CSV ke MongoDB.\n"
            "2. **Komparasi Sebelum vs Sesudah:** Model dilatih ulang lalu metriknya dibandingkan secara langsung terhadap dataset dasar.\n"
            "3. **Model Versioning:** Model terbaru diserialisasi (`pickle`) beserta riwayat metriknya ke database."
        )

    if db is None:
        st.error(f"Fitur ini membutuhkan koneksi MongoDB Atlas. Status: {db_error}")
        return

    c1, c2 = st.columns(2)
    task = c1.selectbox("Modul Pemodelan", list(TASK_LABEL), format_func=TASK_LABEL.get, key="rt_task")
    name = c2.selectbox("Dataset Tujuan", TASK_DATASETS[task], key="rt_ds")
    st.caption(f"ℹ️ **Dataset:** {DATASET_DESC.get(name, '')}")

    base, feats, tn = load_base(name)
    key = f"{task}:{name}"
    n_extra = db.dataset_tambahan.count_documents({"key": key})
    st.caption(f"📁 Data Asli Sklearn: **{len(base)}** baris | 📥 Data Tambahan MongoDB: **{n_extra}** baris")

    t1, t2, t3, t4 = st.tabs(["➕ Input Manual", "📄 Upload CSV", "🗂️ Kelola Data", "🔁 Retraining"])

    with t1:
        with st.form("form_manual", clear_on_submit=False):
            vals = feature_inputs(base, feats, "rt_man", dataset_name=name)
            target = None
            if task == "regresi":
                target = st.number_input("Target Value", value=float(base["target"].median()), key="rt_man_target")
            elif task == "klasifikasi":
                labels_opt = tn if tn else [str(i) for i in sorted(base["target"].unique())]
                label = st.selectbox("Label Kelas (Target)", labels_opt, key="rt_man_label")
                target = tn.index(label) if tn else float(label)
            if st.form_submit_button("💾 Simpan ke MongoDB"):
                insert_rows(task, name, [(vals, target)], "manual")
                st.toast("✅ Data berhasil disimpan ke MongoDB!", icon="🎉")
                st.rerun()

    with t2:
        cols_needed = feats + (["target"] if task != "clustering" else [])
        st.write("Struktur Kolom CSV:", ", ".join(cols_needed))
        st.download_button(
            "📥 Unduh Template CSV",
            base[cols_needed].head(3).to_csv(index=False),
            file_name=f"template_{name.lower().replace(' ', '_')}.csv",
            mime="text/csv",
        )
        up = st.file_uploader("Upload File CSV Data Baru", type="csv", key="rt_csv")
        if up is not None:
            try:
                new = pd.read_csv(up)
                missing = [c for c in cols_needed if c not in new.columns]
                if missing:
                    st.error(f"Kolom kurang: {missing}")
                else:
                    new = new[cols_needed].dropna()
                    if task == "klasifikasi" and new["target"].dtype == object and tn:
                        new["target"] = new["target"].map({n: i for i, n in enumerate(tn)})
                        new = new.dropna()
                    for c in cols_needed:
                        new[c] = pd.to_numeric(new[c], errors="raise")
                    st.dataframe(new.head(), use_container_width=True)
                    st.write(f"✓ Terbaca **{len(new)}** baris valid.")
                    if st.button("💾 Unggah Semua ke MongoDB", key="rt_csv_btn", type="primary"):
                        rows = [(r[feats].tolist(), r["target"] if task != "clustering" else None) for _, r in new.iterrows()]
                        n = insert_rows(task, name, rows, "csv")
                        st.toast(f"✅ {n} baris tersimpan!", icon="🎉")
                        st.rerun()
            except Exception as e:
                st.error(f"Gagal membaca CSV: {e}")

    with t3:
        extra = get_extra(task, name, feats)
        if extra.empty:
            st.info("Belum ada data tambahan.")
        else:
            st.dataframe(extra, use_container_width=True)
        if n_extra:
            if st.checkbox("Konfirmasi hapus seluruh data tambahan", key="rt_del_ok"):
                if st.button("🗑️ Hapus Data", key="rt_del", type="primary"):
                    db.dataset_tambahan.delete_many({"key": key})
                    st.toast("✅ Data dihapus!", icon="🗑️")
                    st.rerun()

    with t4:
        params = {}
        model_name_selected = "Linear Regression"
        if task == "klasifikasi":
            model_name_selected = st.selectbox(
                "Algoritma",
                ["Random Forest", "Logistic Regression", "KNN", "Decision Tree",
                 "Fuzzy KNN", "Fuzzy C-Means"],
                key="rt_algo",
            )
            params["model_name"] = model_name_selected
        elif task == "clustering":
            model_name_selected = "K-Means"
            params["k"] = st.slider("Jumlah Cluster (k)", 2, 10, 3, key="rt_k")

        st.markdown("Komparasi performa **Data Asli** vs **Data Asli + Tambahan**:")
        if st.button("🚀 Eksekusi Retraining & Simpan Model", type="primary", key="rt_go"):
            if n_extra == 0:
                st.warning("Belum ada data tambahan; hasil sebelum & sesudah akan sama.")
            with st.spinner("Melatih ulang model..."):
                df_new, n_e, _, _ = get_training_df(task, name, True)
                before = run_model(task, base, **params)
                after = run_model(task, df_new, **params)
            cmp = pd.DataFrame({
                "Sebelum": before["metrics"],
                "Sesudah": after["metrics"],
                "Selisih": {k: after["metrics"][k] - before["metrics"][k] for k in before["metrics"]},
            })
            st.dataframe(cmp.style.format("{:.4f}"), use_container_width=True)

            db.riwayat_training.insert_one({
                "key": key,
                "waktu": datetime.now(timezone.utc),
                "params": params,
                "n_asli": len(base),
                "n_tambahan": n_e,
                "metrik_sebelum": {k: float(v) for k, v in before["metrics"].items()},
                "metrik_sesudah": {k: float(v) for k, v in after["metrics"].items()},
            })
            saved = save_model_to_db(
                task, name, after["model"], model_name_selected,
                {k: float(v) for k, v in after["metrics"].items()},
            )
            ok_gal, _ = save_training_result(
                task, name, model_name_selected,
                make_params(task, model_name_selected, k=params.get("k", 3)),
                after["metrics"], after.get("n_train", 0), after.get("n_test", 0), n_e,
                nama="Retraining", catatan=f"Retraining dengan {n_e} data tambahan", sumber="retraining",
            )
            if saved:
                st.balloons()
                st.success(
                    "✅ Retraining selesai! Model terbaru tersimpan di MongoDB."
                    + (" Hasilnya juga muncul di 🏆 Galeri Hasil Training." if ok_gal else "")
                )
            else:
                st.warning("⚠️ Retraining selesai, tetapi model gagal disimpan ke MongoDB (kemungkinan kuota Atlas penuh atau ukuran model terlalu besar).")

        hist = list(db.riwayat_training.find({"key": key}).sort("waktu", -1).limit(10))
        if hist:
            st.write("**Riwayat Retraining Terakhir:**")
            st.dataframe(pd.DataFrame([{
                "Waktu": now_wib(h["waktu"]),
                "Data Tambahan": h["n_tambahan"],
                **{f"{k} (Baru)": round(v, 4) for k, v in h["metrik_sesudah"].items()},
            } for h in hist]), hide_index=True, use_container_width=True)


# ============================================================================
# HALAMAN: GALERI HASIL TRAINING
# ============================================================================
def page_galeri():
    st.markdown('<h1 class="main-title-container">🏆 <span class="gradient-text">Galeri Hasil Training</span></h1>', unsafe_allow_html=True)
    st.info(
        "💡 Halaman ini berisi hasil training yang **disimpan pengunjung lain**. Gunakan sebagai referensi: "
        "algoritma dan parameter apa yang memberi skor terbaik pada suatu dataset, lalu coba kalahkan skornya di modul."
    )
    if db is None:
        st.error(f"Galeri membutuhkan koneksi MongoDB Atlas. Status: {db_error}")
        return

    c1, c2 = st.columns(2)
    task = c1.selectbox("Modul Pemodelan", list(TASK_LABEL), format_func=TASK_LABEL.get, key="gal_task")
    ds = c2.selectbox("Dataset", ["Semua dataset"] + TASK_DATASETS[task], key="gal_ds")
    pm = PRIMARY_METRIC[task]

    rows = _urut_terbaik(fetch_training_results(task, None if ds == "Semua dataset" else ds), task)
    if not rows:
        st.warning("Belum ada hasil training tersimpan untuk pilihan ini. Buka modul, latih model, lalu klik **💾 Simpan ke Galeri**.")
        st.button(f"🚀 Ke Modul {TASK_LABEL[task]}", key="gal_go_empty", on_click=change_page, args=(MODULE_PAGE[task],))
        return

    df = results_df(rows, task)
    best = df.iloc[0]
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Total Hasil", len(df))
    m2.metric("Kontributor", df["Penyimpan"].nunique())
    m3.metric(f"{pm} Terbaik", f"{best[pm]:.4f}")
    m4.metric("Algoritma Terbaik", best["Algoritma"])
    st.markdown(
        f'<div class="gal-best"><div class="gal-best-title">🥇 Konfigurasi terbaik saat ini</div>'
        f'<div class="gal-best-sub">{best["Dataset"]} · {best["Algoritma"]} · {best["Parameter"]} · {pm} = {best[pm]:.4f} · oleh {best["Penyimpan"]}</div></div>',
        unsafe_allow_html=True,
    )

    t1, t2, t3, t4 = st.tabs(["🏅 Leaderboard", "📊 Perbandingan", "🕒 Terbaru", "🔁 Pakai Konfigurasi"])

    with t1:
        top = df.head(15).copy()
        medali = ["🥇", "🥈", "🥉"]
        top.insert(0, "Peringkat", [medali[i] if i < 3 else f"#{i + 1}" for i in range(len(top))])
        st.dataframe(top.drop(columns=["Waktu"]), use_container_width=True, hide_index=True)
        st.download_button(
            "⬇️ Unduh semua hasil (CSV)", df.to_csv(index=False).encode("utf-8"),
            file_name=f"galeri_{task}.csv", mime="text/csv", key="gal_dl",
        )

    with t2:
        xcol = "Algoritma" if task == "klasifikasi" else "Parameter"
        g = df.groupby([xcol, "Dataset"], as_index=False)[pm].max()
        fig = px.bar(g, x=xcol, y=pm, color="Dataset", barmode="group", text_auto=".3f",
                     title=f"Skor {pm} terbaik per {xcol.lower()}")
        fig.update_layout(
            template="lilac_light",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(255,255,255,0.85)",
            height=420,
        )
        st.plotly_chart(fig, use_container_width=True)
        if len(g) > 1:
            hi, lo = g.loc[g[pm].idxmax()], g.loc[g[pm].idxmin()]
            st.caption(
                f"💡 Skor tertinggi: **{hi[xcol]}** pada {hi['Dataset']} ({hi[pm]:.4f}); terendah: **{lo[xcol]}** pada {lo['Dataset']} ({lo[pm]:.4f}). "
                "Bandingkan antar algoritma/parameter pada **dataset yang sama** agar adil."
            )

    with t3:
        terbaru = results_df(sorted(rows, key=lambda d: d.get("created_at") or datetime.min, reverse=True)[:30], task)
        st.dataframe(terbaru, use_container_width=True, hide_index=True)

    with t4:
        st.caption("Pilih satu hasil, lalu terapkan: dataset, algoritma, dan parameter di modul akan terisi otomatis sehingga kamu bisa mereproduksi atau memperbaikinya.")
        pilihan = rows[:20]
        idx = st.selectbox(
            "Pilih konfigurasi", range(len(pilihan)), key="gal_pick",
            format_func=lambda i: (
                f"#{i + 1} · {pilihan[i]['dataset']} · {pilihan[i]['model_name']} · "
                f"{params_text(task, pilihan[i]['model_name'], pilihan[i].get('params', {}))} · "
                f"{pm} {pilihan[i]['metrics'].get(pm, 0):.4f}"
            ),
        )
        if pilihan[idx].get("catatan"):
            st.caption(f"📝 Catatan penyimpan: {pilihan[idx]['catatan']}")
        st.button("🔁 Terapkan di Modul", key="gal_apply", type="primary", on_click=apply_config, args=(pilihan[idx],))
        st.caption("ℹ️ Skor bisa sedikit berbeda jika data tambahan di MongoDB sudah berubah sejak hasil itu disimpan.")


# ============================================================================
# HALAMAN: KOMENTAR & ANALISIS SENTIMEN AI
# ============================================================================
def page_komentar():
    st.markdown(
        '<h1 class="main-title-container">💬 <span class="gradient-text">Komentar Publik dan Analisis AI</span></h1>',
        unsafe_allow_html=True,
    )
    st.caption(
        "Pengunjung dapat mengirim komentar yang disimpan di MongoDB. "
        "Groq LLM dapat merangkum komentar terbaru dan memberi label sentimen (Natural Language Processing)."
    )

    if db is None:
        st.error(f"Fitur ini membutuhkan MongoDB. Status: {db_error}")
        return

    with st.form("form_komentar", clear_on_submit=True):
        nama = st.text_input("Nama atau nama panggilan", max_chars=50)
        isi = st.text_area("Komentar", placeholder="Tulis komentar di sini...", max_chars=500)
        if st.form_submit_button("Kirim komentar"):
            if not nama.strip() or not isi.strip():
                st.warning("Mohon isi Nama dan Komentar.")
            else:
                db.komentar.insert_one({
                    "nama": nama.strip(),
                    "komentar": isi.strip(),
                    "created_at": datetime.now(timezone.utc),
                })
                st.toast("✅ Komentar terkirim!", icon="💬")
                st.rerun()

    docs = list(db.komentar.find().sort("created_at", -1).limit(30))

    st.divider()
    st.subheader("Analisis ringkasan dan sentimen")
    st.caption(
        "Analisis dilakukan saat tombol ditekan, terhadap maksimal 30 komentar terbaru. "
        "Hasil sentimen AI adalah estimasi, bukan penilaian manusia yang pasti."
    )

    if st.button(
        f"Analisis {len(docs)} komentar dengan Groq",
        type="primary",
        disabled=(len(docs) == 0),
        key="btn_analisis_groq",
    ):
        payload = [{"id": str(d["_id"]), "komentar": d["komentar"]} for d in docs]
        with st.spinner("Groq sedang menganalisis sentimen komentar..."):
            hasil, err = analisis_sentimen_komentar(payload)

        if err:
            st.error(err)
        else:
            id_map = {str(d["_id"]): d["_id"] for d in docs}
            for item in hasil.get("hasil", []):
                oid = id_map.get(str(item.get("id")))
                if oid is not None:
                    sent = str(item.get("sentimen", "Netral")).strip().capitalize()
                    if sent not in ("Positif", "Negatif", "Netral"):
                        sent = "Netral"
                    db.komentar.update_one(
                        {"_id": oid},
                        {
                            "$set": {
                                "sentimen": sent,
                                "alasan_sentimen": str(item.get("alasan", "")).strip(),
                                "analisis_oleh": "Groq",
                            }
                        },
                    )

            ringkasan = str(hasil.get("ringkasan", "")).strip()
            if ringkasan:
                db.ringkasan_komentar.replace_one(
                    {"_id": "terbaru"},
                    {
                        "_id": "terbaru",
                        "ringkasan": ringkasan,
                        "penyedia": "Groq",
                        "updated_at": datetime.now(timezone.utc),
                    },
                    upsert=True,
                )
            st.toast("✅ Analisis sentimen selesai!", icon="🤖")
            st.rerun()

    ringkasan_doc = db.ringkasan_komentar.find_one({"_id": "terbaru"})
    if ringkasan_doc and ringkasan_doc.get("ringkasan"):
        st.info(f"📝 **Ringkasan AI ({ringkasan_doc.get('penyedia', 'Groq')}):** {ringkasan_doc['ringkasan']}")

    st.divider()
    st.subheader("Komentar terbaru")

    if not docs:
        st.info("Belum ada komentar. Jadilah yang pertama!")
        return

    n_pos = sum(1 for d in docs if d.get("sentimen") == "Positif")
    n_neg = sum(1 for d in docs if d.get("sentimen") == "Negatif")
    n_net = sum(1 for d in docs if d.get("sentimen") == "Netral")

    c1, c2, c3 = st.columns(3)
    c1.metric("Positif", n_pos, help="Jumlah komentar dengan nada pujian, dukungan, atau kepuasan.")
    c2.metric("Negatif", n_neg, help="Jumlah komentar yang berisi keluhan, kritik tajam, atau ketidakpuasan.")
    c3.metric("Netral", n_net, help="Jumlah komentar bersifat informatif, pertanyaan, atau tanpa emosi kuat.")
    st.write("")

    for d in docs:
        with st.container(border=True):
            st.markdown(f"**{d['nama']}** · _{now_wib(d['created_at'])}_")
            st.write(d["komentar"])
            if d.get("sentimen"):
                alasan = d.get("alasan_sentimen", "")
                penyedia = d.get("analisis_oleh", "Groq")
                ket = f" · {alasan}" if alasan else ""
                st.caption(f"Sentimen: {d['sentimen']} · Analisis: {penyedia}{ket}")
            else:
                st.caption("Sentimen: Belum dianalisis")


# ============================================================================
# SIDEBAR & ROUTER
# ============================================================================
PAGES = {
    "🏠 Dashboard": page_home,
    "📈 Regresi Linear": page_regresi,
    "🧪 Klasifikasi": page_klasifikasi,
    "🔵 Clustering": page_clustering,
    "🏆 Galeri Hasil Training": page_galeri,
    "🗄️ Database & Retrain": page_retrain,
    "💬 Komentar": page_komentar,
    "🤖 Asisten AI (Groq)": page_ai_assistant,
    "📝 Mini Kuis ML": lambda: page_kuis_ml(db),
}

MENU_GROUPS = [
    ("MENU", ["🏠 Dashboard", "📈 Regresi Linear", "🧪 Klasifikasi", "🔵 Clustering", "🏆 Galeri Hasil Training", "🗄️ Database & Retrain", "💬 Komentar"]),
    ("EDUKASI & AI", ["🤖 Asisten AI (Groq)", "📝 Mini Kuis ML"]),
]

with st.sidebar:
    st.markdown('<div class="sidebar-title">Machine-Learning</div>', unsafe_allow_html=True)

    st.markdown(
        """
        <div class="profile-card">
            <div class="profile-avatar">🪻</div>
            <div>
                <div class="profile-name">Hana Meisaria</div>
                <div class="profile-id">632025057</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    choice = st.session_state.get("nav_radio", "🏠 Dashboard")
    if choice not in PAGES:
        choice = "🏠 Dashboard"

    for group_title, items in MENU_GROUPS:
        st.markdown(f'<div class="sidebar-section">{group_title}</div>', unsafe_allow_html=True)
        for label in items:
            is_active = (choice == label)
            st.button(
                label,
                key=f"sb_nav_{label}",
                use_container_width=True,
                type="primary" if is_active else "secondary",
                on_click=change_page,
                args=(label,),
            )

    st.caption("📊 Ringkasan dataset, status model, dan skor utama.")

    with st.expander("✨ Isi Aplikasi", expanded=True):
        st.markdown("""
        <div class="mini-card">
            <div class="mini-card-title">📈 Regresi</div>
            <div class="mini-card-desc">Linear Regression · Diabetes & California Housing</div>
        </div>
        <div class="mini-card">
            <div class="mini-card-title">🧩 Klasifikasi + Fuzzy</div>
            <div class="mini-card-desc">Random Forest, Logistic, KNN, Tree, Fuzzy KNN & FCM</div>
        </div>
        <div class="mini-card">
            <div class="mini-card-title">🔵 Clustering</div>
            <div class="mini-card-desc">K-Means + PCA 2D, Elbow & Silhouette</div>
        </div>
        """, unsafe_allow_html=True)

    mongo_badge = '<span class="badge-ok">● Aktif</span>' if db is not None else '<span class="badge-err">● Offline</span>'
    groq_badge = '<span class="badge-ok">● Aktif</span>' if _secret("GROQ_API_KEY") else '<span class="badge-warn">● No Key</span>'

    if db is not None:
        try:
            n_komen = db.komentar.count_documents({})
            n_model = db.models.count_documents({})
            n_kuis = db.riwayat_kuis.count_documents({})
        except Exception:
            n_komen, n_model, n_kuis = 0, 0, 0

        stats_html = (
            '<div class="mini-stats">'
            f'<div class="mini-stat-box"><div class="mini-stat-val">{n_komen}</div><div class="mini-stat-lbl">Komentar</div></div>'
            f'<div class="mini-stat-box"><div class="mini-stat-val">{n_model}</div><div class="mini-stat-lbl">Model</div></div>'
            f'<div class="mini-stat-box"><div class="mini-stat-val">{n_kuis}</div><div class="mini-stat-lbl">Kuis</div></div>'
            '</div>'
        )
    else:
        stats_html = ""

    st.markdown(
        (
            '<div class="status-panel">'
            f'<div class="status-row"><span>🍃 MongoDB Atlas</span>{mongo_badge}</div>'
            f'<div class="status-row"><span>⚡ Groq LLM</span>{groq_badge}</div>'
            f'{stats_html}'
            '</div>'
        ),
        unsafe_allow_html=True,
    )

    if db is None:
        st.write("")
        if st.button("🔄 Sambung Ulang MongoDB", use_container_width=True):
            init_db.clear()
            st.rerun()

PAGES[choice]()