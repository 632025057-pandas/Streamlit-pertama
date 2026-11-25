"""
TAS AI - Hana Meisaria (NIM 632025057)
Aplikasi Streamlit: Regresi Linear, Klasifikasi, Clustering
+ Integrasi MongoDB (dataset tambahan, retraining, komentar publik, & model ML)
+ Integrasi Groq AI (Asisten AI, analisis metrik, & analisis sentimen komentar via groq_helper.py)
Sumber data: scikit-learn (Diabetes, California Housing, Iris, Wine, Breast Cancer)

VERSI 3.2 - Sidebar Modern (Menu Pill & Status Panel) + Mode Edukatif,
            Penjelasan Matematika Interaktif, Mini Kuis ML, & Kamus Fitur Dataset
"""
import os
import pickle
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
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

# ----------------------------------------------------------------------------
# Konfigurasi Halaman
# ----------------------------------------------------------------------------
st.set_page_config(
    page_title="TAS AI - Hana Meisaria",
    page_icon="👩‍🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ----------------------------------------------------------------------------
# Session State untuk Navigasi
# ----------------------------------------------------------------------------
if "nav_radio" not in st.session_state:
    st.session_state["nav_radio"] = "🏠 Dashboard Beranda"


def change_page(page_name):
    st.session_state["nav_radio"] = page_name


# ----------------------------------------------------------------------------
# Custom CSS - Modern Dark Theme + Modern Sidebar
# ----------------------------------------------------------------------------
st.markdown("""
    <style>
    .stApp { background-color: #0e1117; }

    .gradient-text {
        background: linear-gradient(90deg, #4F46E5, #06B6D4);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 800;
    }
    .main-title-container {
        font-size: 2.5rem;
        font-weight: 800;
        margin-bottom: 0.2rem;
    }
    .custom-card {
        background-color: #1f2937;
        border-radius: 12px;
        padding: 20px;
        border: 1px solid #374151;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
        transition: transform 0.2s ease, box-shadow 0.2s ease;
        margin-bottom: 10px;
        height: 100%;
    }
    .custom-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.3);
        border-color: #4F46E5;
    }
    .card-title {
        color: #f3f4f6;
        font-size: 1.1rem;
        font-weight: 600;
        margin-bottom: 8px;
    }
    .card-desc { color: #9ca3af; font-size: 0.9rem; }

    [data-testid="stMetricValue"] {
        font-size: 1.8rem; font-weight: 700; color: #38bdf8;
    }
    [data-testid="stMetric"] {
        background-color: #1e293b;
        border: 1px solid #334155;
        padding: 12px 16px;
        border-radius: 10px;
    }
    .stTabs [data-baseweb="tab-list"] { gap: 8px; }
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px; padding: 8px 16px; background-color: #1e293b;
    }
    .stTabs [aria-selected="true"] {
        background-color: #4f46e5 !important; color: white !important;
    }
    .footer {
        text-align: center; color: #6b7280; font-size: 0.85rem;
        padding: 20px 0 10px 0; border-top: 1px solid #1f2937; margin-top: 40px;
    }

    /* === MODERN SIDEBAR === */
    [data-testid="stSidebar"] {
        background-color: #111827;
        border-right: 1px solid #1f2937;
    }
    [data-testid="stSidebar"] [data-testid="stVerticalBlock"] {
        gap: 0.38rem;
    }
    .sidebar-brand {
        background: linear-gradient(135deg, rgba(79,70,229,0.22), rgba(6,182,212,0.16));
        border: 1px solid rgba(99,102,241,0.35);
        border-radius: 14px;
        padding: 14px 16px;
        margin-bottom: 10px;
    }
    .sidebar-brand-title {
        font-size: 1.05rem;
        font-weight: 800;
        color: #f8fafc;
        margin: 0;
    }
    .sidebar-brand-sub {
        font-size: 0.78rem;
        color: #94a3b8;
        margin-top: 3px;
    }
    .sidebar-section {
        font-size: 0.7rem;
        font-weight: 700;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        color: #64748b;
        margin: 12px 0 4px 4px;
    }
    /* Tombol navigasi rata kiri (inactive) */
    [data-testid="stSidebar"] button[data-testid="stBaseButton-secondary"],
    [data-testid="stSidebar"] button[data-testid="baseButton-secondary"],
    [data-testid="stSidebar"] button[kind="secondary"] {
        background-color: transparent;
        border: 1px solid transparent;
        color: #cbd5e1;
        justify-content: flex-start;
        padding: 0.42rem 0.75rem;
        border-radius: 10px;
        font-weight: 500;
        transition: all 0.18s ease;
    }
    [data-testid="stSidebar"] button[data-testid="stBaseButton-secondary"]:hover,
    [data-testid="stSidebar"] button[data-testid="baseButton-secondary"]:hover,
    [data-testid="stSidebar"] button[kind="secondary"]:hover {
        background-color: #1e293b;
        border-color: #334155;
        color: #ffffff;
        transform: translateX(3px);
    }
    /* Tombol halaman aktif */
    [data-testid="stSidebar"] button[data-testid="stBaseButton-primary"],
    [data-testid="stSidebar"] button[data-testid="baseButton-primary"],
    [data-testid="stSidebar"] button[kind="primary"] {
        background: linear-gradient(90deg, #4f46e5, #0891b2);
        border: none;
        color: #ffffff;
        justify-content: flex-start;
        padding: 0.42rem 0.75rem;
        border-radius: 10px;
        font-weight: 600;
        box-shadow: 0 4px 12px rgba(79, 70, 229, 0.35);
    }
    /* Panel Status Sistem */
    .status-panel {
        background-color: #0f172a;
        border: 1px solid #1e293b;
        border-radius: 12px;
        padding: 12px 14px;
        margin-top: 14px;
    }
    .status-row {
        display: flex;
        justify-content: space-between;
        align-items: center;
        font-size: 0.82rem;
        color: #cbd5e1;
        padding: 4px 0;
    }
    .badge-ok {
        background: rgba(16, 185, 129, 0.15);
        color: #34d399;
        border: 1px solid rgba(16, 185, 129, 0.35);
        padding: 2px 8px;
        border-radius: 999px;
        font-size: 0.72rem;
        font-weight: 600;
    }
    .badge-warn {
        background: rgba(245, 158, 11, 0.15);
        color: #fbbf24;
        border: 1px solid rgba(245, 158, 11, 0.35);
        padding: 2px 8px;
        border-radius: 999px;
        font-size: 0.72rem;
        font-weight: 600;
    }
    .badge-err {
        background: rgba(239, 68, 68, 0.15);
        color: #f87171;
        border: 1px solid rgba(239, 68, 68, 0.35);
        padding: 2px 8px;
        border-radius: 999px;
        font-size: 0.72rem;
        font-weight: 600;
    }
    .mini-stats {
        display: grid;
        grid-template-columns: repeat(3, 1fr);
        gap: 6px;
        margin-top: 10px;
        padding-top: 10px;
        border-top: 1px solid #1e293b;
        text-align: center;
    }
    .mini-stat-box {
        background: #1e293b;
        border-radius: 8px;
        padding: 6px 4px;
    }
    .mini-stat-val {
        font-size: 0.95rem;
        font-weight: 700;
        color: #38bdf8;
        line-height: 1.2;
    }
    .mini-stat-lbl {
        font-size: 0.68rem;
        color: #94a3b8;
    }
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

# Penjelasan tooltip (?) pada setiap kartu metrik
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

# Kamus singkat fitur per dataset agar pengguna paham arti input
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
    df = d.frame[feats + ["target"]].copy()
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
def make_classifier(model_name, seed):
    return {
        "Logistic Regression": make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)),
        "Random Forest": RandomForestClassifier(n_estimators=200, random_state=seed),
        "KNN": make_pipeline(StandardScaler(), KNeighborsClassifier()),
        "Decision Tree": DecisionTreeClassifier(random_state=seed),
    }[model_name]


@st.cache_resource(show_spinner=False)
def run_model(task, df, model_name="Random Forest", k=3, test_size=0.2, seed=42):
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
        m = make_classifier(model_name, seed).fit(Xtr, ytr)
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
    """Interpretasi edukatif cepat secara otomatis berdasarkan nilai metrik utama."""
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
# PENJELASAN MATEMATIKA INTERAKTIF (HITUNG MANUAL DARI HASIL MODEL PENGGUNA)
# ============================================================================
def _bagi(a, b):
    """Pembagian aman: hasil 0 jika penyebut 0."""
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

        # ---- Langkah 1: dari mana prediksi berasal
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

        # ---- Langkah 2: MAE, MSE, RMSE
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

        # ---- Langkah 3: R²
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

        # ---- Langkah 1: Accuracy
        st.markdown("#### Langkah 1 — Accuracy dari diagonal Confusion Matrix")
        st.dataframe(
            pd.DataFrame(cm, index=[f"Aktual: {x}" for x in names], columns=[f"Prediksi: {x}" for x in names]),
            use_container_width=True,
        )
        benar = int(np.trace(cm))
        st.latex(rf"\text{{Accuracy}} = \frac{{\text{{jumlah diagonal}}}}{{\text{{total data uji}}}} = \frac{{{benar}}}{{{total}}} = {benar / total:.4f}")

        # ---- Langkah 2: Precision / Recall / F1 satu kelas
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

        # ---- Langkah 3: Macro average
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

        # ---- Langkah 1: standarisasi
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

        # ---- Langkah 2: jarak ke centroid
        st.markdown("#### Langkah 2 — Jarak ke setiap centroid & penentuan cluster")
        d_cent = np.linalg.norm(cent - z, axis=1)
        st.dataframe(
            pd.DataFrame({"Cluster": range(k), "Jarak Euclidean ke centroid": d_cent}).sort_values("Jarak Euclidean ke centroid"),
            hide_index=True, use_container_width=True,
        )
        j = int(np.argmin(d_cent))
        st.latex(rf"d(x, \mu_j) = \sqrt{{\sum_i (z_i - \mu_{{j,i}})^2}} \;\Rightarrow\; \min = {d_cent[j]:.4f} \text{{ (cluster {j})}}")
        st.caption(f"💡 Titik terdekat dengan centroid cluster **{j}**; label dari K-Means: **{int(labels[idx])}**.")

        # ---- Langkah 3: Inertia
        st.markdown("#### Langkah 3 — Inertia (WCSS)")
        sse = [float(((Xs[labels == c] - cent[c]) ** 2).sum()) for c in range(k)]
        st.dataframe(
            pd.DataFrame({"Cluster": range(k), "Jumlah anggota": [int((labels == c).sum()) for c in range(k)], "SSE cluster": sse}),
            hide_index=True, use_container_width=True,
        )
        st.latex(rf"J = \sum_{{j=1}}^{{k}} \sum_{{x_i \in C_j}} \|x_i - \mu_j\|^2 = {' + '.join(f'{v:.2f}' for v in sse)} = {sum(sse):.4f}")
        _cek_cocok(sum(sse), res["metrics"]["Inertia"], "Inertia")

        # ---- Langkah 4: Silhouette
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

        # ---- Langkah 5: Davies-Bouldin
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
# HALAMAN: DASHBOARD BERANDA
# ============================================================================
def page_home():
    st.markdown('<h1 class="main-title-container">🧠🤖 <span class="gradient-text">Dashboard Machine Learning</span></h1>', unsafe_allow_html=True)
    st.markdown(
        "Media pembelajaran & eksperimen interaktif **Regresi Linear**, **Klasifikasi**, dan **Clustering** "
        "terintegrasi *real-time* dengan **MongoDB Atlas** & **Groq AI**.<br>"
        "**TAS AI : Hana Meisaria 632025057**",
        unsafe_allow_html=True,
    )
    st.write("")

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("""
            <div class="custom-card">
                <div class="card-title">📈 Regresi Linear</div>
                <div class="card-desc"><b>Supervised Learning</b> untuk memprediksi angka kontinu pada dataset <b>Diabetes</b> & <b>California Housing</b>.</div>
            </div>
        """, unsafe_allow_html=True)
        st.button("🚀 Modul Regresi", key="btn_nav_reg", use_container_width=True, on_click=change_page, args=("📈 Regresi Linear",))
    with c2:
        st.markdown("""
            <div class="custom-card">
                <div class="card-title">🧪 Klasifikasi Multi-Algoritma</div>
                <div class="card-desc"><b>Supervised Learning</b> untuk memprediksi kelas diskrit pada dataset <b>Iris</b>, <b>Wine</b>, & <b>Breast Cancer</b>.</div>
            </div>
        """, unsafe_allow_html=True)
        st.button("🚀 Modul Klasifikasi", key="btn_nav_clf", use_container_width=True, on_click=change_page, args=("🧪 Klasifikasi",))
    with c3:
        st.markdown("""
            <div class="custom-card">
                <div class="card-title">🔵 Clustering K-Means</div>
                <div class="card-desc"><b>Unsupervised Learning</b> untuk mengelompokkan data tanpa label pada dataset <b>Iris</b>, <b>Wine</b>, & <b>Breast Cancer</b>.</div>
            </div>
        """, unsafe_allow_html=True)
        st.button("🚀 Modul Clustering", key="btn_nav_clu", use_container_width=True, on_click=change_page, args=("🔵 Clustering",))

    st.write("")
    a1, a2 = st.columns(2)
    with a1:
        st.markdown("""
            <div class="custom-card">
                <div class="card-title">🤖 Asisten AI (Groq)</div>
                <div class="card-desc">Tutor interaktif dengan tombol pertanyaan cepat untuk belajar konsep ML, arti rumus matematika, dan interpretasi metrik evaluasi.</div>
            </div>
        """, unsafe_allow_html=True)
        st.button("💬 Buka Asisten AI (Groq)", key="btn_nav_ai", use_container_width=True, on_click=change_page, args=("🤖 Asisten AI (Groq)",))
    with a2:
        st.markdown("""
            <div class="custom-card">
                <div class="card-title">📝 Mini Kuis Machine Learning</div>
                <div class="card-desc">Uji pemahaman Regresi, Klasifikasi, Clustering, dan evaluasi model dengan pembahasan otomatis serta generator soal baru dari Groq AI.</div>
            </div>
        """, unsafe_allow_html=True)
        st.button("🎯 Mulai Mini Kuis ML", key="btn_nav_kuis", use_container_width=True, on_click=change_page, args=("📝 Mini Kuis ML",))

    # --- PETA KONSEP EDUKATIF ---
    st.write("---")
    st.subheader("📚 Peta Konsep Machine Learning")
    with st.expander("📖 Pelajari Perbedaan Supervised vs Unsupervised Learning & Alur Kerja ML", expanded=True):
        konsep_df = pd.DataFrame([
            {
                "Metode": "Regresi Linear",
                "Paradigma": "Supervised Learning (Ada Label Target)",
                "Bentuk Output": "Angka kontinu (mis. harga rumah, progres medis)",
                "Metrik Evaluasi": "R² Score, MAE, RMSE",
                "Pertanyaan yang Dijawab": "Berapa nilai yang diprediksi?",
            },
            {
                "Metode": "Klasifikasi",
                "Paradigma": "Supervised Learning (Ada Label Target)",
                "Bentuk Output": "Kategori / Kelas diskrit (mis. jinak vs ganas)",
                "Metrik Evaluasi": "Accuracy, Precision, Recall, F1 Score",
                "Pertanyaan yang Dijawab": "Termasuk kategori manakah data ini?",
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

    st.write("---")
    st.subheader("⚡ Status Server & Database")
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
                })
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
        st.caption(f"💬 Total Komentar Publik: **{db.komentar.count_documents({})}** | 📦 Total Model: **{db.models.count_documents({})}**")

    st.markdown('<div class="footer">© 2026 Hana Meisaria · TAS AI · Universitas Kristen Satya Wacana</div>', unsafe_allow_html=True)


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
            "- Garis merah putus-putus adalah **garis ideal** (Nilai Aktual = Nilai Prediksi).\n"
            "- Semakin rapat titik biru menempel pada garis merah, semakin kecil kesalahan prediksi model."
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
            marker=dict(size=8, color="#38bdf8", opacity=0.7, line=dict(width=0)),
            name="Data Uji",
            hovertemplate="Aktual: %{x:.2f}<br>Prediksi: %{y:.2f}<extra></extra>",
        ))
        lims = [min(res["y_test"].min(), res["pred"].min()), max(res["y_test"].max(), res["pred"].max())]
        fig.add_trace(go.Scatter(
            x=lims, y=lims, mode="lines",
            line=dict(color="#f43f5e", dash="dash", width=2),
            name="Garis Ideal",
        ))
        fig.update_layout(
            title=f"Aktual vs Prediksi — {name}",
            xaxis_title="Nilai Aktual", yaxis_title="Nilai Prediksi",
            template="plotly_dark", height=400,
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
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
# HALAMAN: KLASIFIKASI
# ============================================================================
def page_klasifikasi():
    st.markdown('<h1 class="main-title-container">🧪 <span class="gradient-text">Klasifikasi Multi-Algoritma</span></h1>', unsafe_allow_html=True)
    st.info("💡 **Klasifikasi** termasuk *Supervised Learning* yang memetakan fitur input ke dalam **kelas/kategori diskrit** berdasarkan pola pada data latih berlabel.")

    with st.expander("📖 Pelajari Teori, Rumus Evaluasi & Perbandingan 4 Algoritma"):
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
            "**3. Karakteristik Algoritma yang Tersedia:**\n"
            "- **Random Forest:** Menggabungkan ratusan pohon keputusan (*ensemble*) sehingga stabil dan tahan *overfitting*.\n"
            "- **Logistic Regression:** Menghitung probabilitas kelas lewat fungsi sigmoid (membutuhkan *StandardScaler*).\n"
            "- **KNN (K-Nearest Neighbors):** Menentukan kelas berdasarkan voting tetangga terdekat berdasarkan jarak Euclidean.\n"
            "- **Decision Tree:** Membagi data dengan aturan *if-else* bercabang; mudah diinterpretasi namun rentan *overfitting*."
        )

    name, df, n_extra, feats, tn = select_data("klasifikasi", "clf")
    show_feature_glossary(name)

    c1, c2 = st.columns(2)
    model_name = c1.selectbox("Pilih Algoritma ML", ["Random Forest", "Logistic Regression", "KNN", "Decision Tree"], key="clf_model")
    algo_desc = {
        "Random Forest": "🌲 **Random Forest:** Ensemble 200 Decision Tree, stabil & akurat.",
        "Logistic Regression": "📐 **Logistic Regression:** Model probabilistik linier dengan StandardScaler.",
        "KNN": "📍 **KNN:** Klasifikasi berdasarkan jarak ke 5 tetangga terdekat (dengan StandardScaler).",
        "Decision Tree": "🌿 **Decision Tree:** Pohon keputusan bercabang berbasis pemisahan Gini.",
    }
    st.caption(algo_desc[model_name])
    test_size = c2.slider(
        "Proporsi Data Uji",
        0.1, 0.5, 0.2, 0.05,
        help="Menggunakan Stratified Split agar proporsi tiap kelas tetap seimbang di Data Latih dan Data Uji.",
        key="clf_ts",
    )

    with st.spinner("Melatih model klasifikasi..."):
        res = run_model("klasifikasi", df, model_name=model_name, test_size=test_size)
    st.caption(f"🧪 Pembagian Data (Stratified): **{res['n_train']}** baris Data Latih · **{res['n_test']}** baris Data Uji")
    show_metrics(res["metrics"])
    show_quick_interpretation("klasifikasi", res["metrics"])
    st.write("")
    render_ai_insight(
        f"Dataset: {name}. Algoritma Klasifikasi: {model_name}. Metrik evaluasi: {res['metrics']}. "
        f"Jelaskan secara singkat dan edukatif kualitas performa klasifikasi model ini serta saran perbaikan jika ada.",
        key="clf_ai_explain",
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
        fig.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig, use_container_width=True)

    with t2:
        labels_display = [tn[c] if tn else str(c) for c in res["classes"]]
        fig = px.imshow(
            res["cm"], text_auto=True, aspect="auto",
            labels=dict(x="Prediksi", y="Aktual", color="Jumlah"),
            x=labels_display, y=labels_display,
            color_continuous_scale="Plasma",
        )
        fig.update_layout(
            title=f"Confusion Matrix — {model_name} ({name})",
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            height=450,
        )
        st.plotly_chart(fig, use_container_width=True)
        st.caption("💡 Semakin terang angka di garis diagonal (kiri-atas ke kanan-bawah), semakin tinggi ketepatan klasifikasi model.")

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
                    color="Probabilitas", color_continuous_scale="Viridis",
                    title="Tingkat Keyakinan (Probabilitas) per Kelas",
                )
                fig.update_layout(
                    template="plotly_dark",
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    height=300,
                    showlegend=False,
                )
                st.plotly_chart(fig, use_container_width=True)


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
    show_metrics(res["metrics"])
    show_quick_interpretation("clustering", res["metrics"])
    st.write("")
    render_ai_insight(
        f"Dataset: {name}. Model: K-Means Clustering dengan k={k}. Metrik evaluasi: {res['metrics']}. "
        f"Jelaskan secara singkat dan edukatif kualitas pemisahan cluster berdasarkan Silhouette, Davies-Bouldin, dan Inertia.",
        key="clu_ai_explain",
    )
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
            color_discrete_sequence=px.colors.qualitative.Vivid,
            opacity=0.8,
        )
        fig.update_traces(marker=dict(size=9, line=dict(width=0.5, color="white")))
        fig.update_layout(
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            height=450,
        )
        c1.plotly_chart(fig, use_container_width=True)

        c2.write("**Evaluasi Jumlah Cluster (Elbow & Silhouette):**")
        ed = elbow_data(df).reset_index()
        fig1 = px.line(ed, x="k", y="Inertia", markers=True, title="Elbow Method (Inertia — cari titik siku)")
        fig1.update_layout(
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            height=200,
            margin=dict(t=40, b=10),
        )
        c2.plotly_chart(fig1, use_container_width=True)
        fig2 = px.line(ed, x="k", y="Silhouette", markers=True, title="Silhouette Score (cari titik puncak)")
        fig2.update_layout(
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            height=200,
            margin=dict(t=40, b=10),
        )
        c2.plotly_chart(fig2, use_container_width=True)

    with t2:
        X = df.drop(columns="target", errors="ignore").copy()
        X["cluster"] = res["labels"]
        c1, c2 = st.columns(2)
        vc = X["cluster"].value_counts().sort_index().reset_index()
        vc.columns = ["Cluster", "Jumlah"]
        fig = px.bar(vc, x="Cluster", y="Jumlah", color="Cluster", title="Jumlah Anggota per Cluster")
        fig.update_layout(
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
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
    st.markdown('<h1 class="main-title-container">🔁 <span class="gradient-text">Management Data & Retraining</span></h1>', unsafe_allow_html=True)
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
            model_name_selected = st.selectbox("Algoritma", ["Random Forest", "Logistic Regression", "KNN", "Decision Tree"], key="rt_algo")
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
            if saved:
                st.balloons()
                st.success("✅ Retraining selesai! Model terbaru tersimpan di MongoDB.")

        hist = list(db.riwayat_training.find({"key": key}).sort("waktu", -1).limit(10))
        if hist:
            st.write("**Riwayat Retraining Terakhir:**")
            st.dataframe(pd.DataFrame([{
                "Waktu": now_wib(h["waktu"]),
                "Data Tambahan": h["n_tambahan"],
                **{f"{k} (Baru)": round(v, 4) for k, v in h["metrik_sesudah"].items()},
            } for h in hist]), hide_index=True, use_container_width=True)


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
    "🏠 Dashboard Beranda": page_home,
    "📈 Regresi Linear": page_regresi,
    "🧪 Klasifikasi": page_klasifikasi,
    "🔵 Clustering": page_clustering,
    "🔁 Update Data & Retraining": page_retrain,
    "💬 Diskusi Komentar": page_komentar,
    "🤖 Asisten AI (Groq)": page_ai_assistant,
    "📝 Mini Kuis ML": lambda: page_kuis_ml(db),
}

MENU_GROUPS = [
    ("Utama", ["🏠 Dashboard Beranda"]),
    ("Modul Machine Learning", ["📈 Regresi Linear", "🧪 Klasifikasi", "🔵 Clustering"]),
    ("Data & Interaksi", ["🔁 Update Data & Retraining", "💬 Diskusi Komentar"]),
    ("Edukasi & AI", ["🤖 Asisten AI (Groq)", "📝 Mini Kuis ML"]),
]

with st.sidebar:
    st.markdown(
        """
        <div class="sidebar-brand">
            <div class="sidebar-brand-title">🧠 TAS AI Studio</div>
            <div class="sidebar-brand-sub">👩‍🎓 Hana Meisaria · 632025057</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    choice = st.session_state.get("nav_radio", "🏠 Dashboard Beranda")
    if choice not in PAGES:
        choice = "🏠 Dashboard Beranda"

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

    # Panel Status Sistem & Statistik Ringkas
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
