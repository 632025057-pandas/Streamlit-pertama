"""
TAS AI - Hana Meisaria
Aplikasi Streamlit: Regresi Linear, Klasifikasi, Clustering
+ Integrasi MongoDB (dataset tambahan, retraining, komentar publik, & model ML)
Sumber data: scikit-learn (Diabetes, California Housing, Iris, Wine, Breast Cancer)
"""
import os
import pickle
from datetime import datetime, timedelta, timezone

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
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
    ConfusionMatrixDisplay,
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

# Konfigurasi Halaman
st.set_page_config(
    page_title="TAS AI - Hana Meisaria", 
    page_icon="👩‍🎓", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# ----------------------------------------------------------------------------
# Custom CSS untuk UI/UX Modern
# ----------------------------------------------------------------------------
st.markdown("""
    <style>
    .stApp {
        background-color: #0e1117;
    }
    .main-title {
        background: linear-gradient(90deg, #4F46E5, #06B6D4);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 800;
        font-size: 2.5rem;
        margin-bottom: 0.2rem;
    }
    .custom-card {
        background-color: #1f2937;
        border-radius: 12px;
        padding: 20px;
        border: 1px solid #374151;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
        transition: transform 0.2s ease, box-shadow 0.2s ease;
        margin-bottom: 15px;
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
    .card-desc {
        color: #9ca3af;
        font-size: 0.9rem;
    }
    [data-testid="stMetricValue"] {
        font-size: 1.8rem;
        font-weight: 700;
        color: #38bdf8;
    }
    [data-testid="stMetric"] {
        background-color: #1e293b;
        border: 1px solid #334155;
        padding: 12px 16px;
        border-radius: 10px;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px;
        padding: 8px 16px;
        background-color: #1e293b;
    }
    .stTabs [aria-selected="true"] {
        background-color: #4f46e5 !important;
        color: white !important;
    }
    </style>
""", unsafe_allow_html=True)

# ----------------------------------------------------------------------------
# Konfigurasi Multi-Dataset (2-3 Dataset per Modul)
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
    except Exception as e:  # noqa: BLE001
        return None, str(e)


db, db_error = init_db()


def now_wib(dt):
    return (dt + timedelta(hours=7)).strftime("%d %b %Y %H:%M WIB")


# ----------------------------------------------------------------------------
# Data & Model Helpers
# ----------------------------------------------------------------------------
def save_model_to_db(task, dataset_name, model_obj, model_name, metrics):
    if db is None:
        return False
    try:
        model_bytes = pickle.dumps(model_obj)
        doc = {
            "key": f"{task}:{dataset_name}",
            "task": task,
            "dataset": dataset_name,
            "model_name": model_name,
            "model_bytes": model_bytes,
            "metrics": metrics,
            "created_at": datetime.now(timezone.utc),
        }
        db.models.insert_one(doc)
        return True
    except Exception:  # noqa: BLE001
        return False


@st.cache_data
def load_base(name):
    d = LOADERS[name](as_frame=True)
    feats = list(d.feature_names)
    tn = getattr(d, "target_names", None)
    tn = [str(x) for x in tn] if tn is not None else None
    
    # Batasi dataset California Housing atau Breast Cancer agar fitur tidak terlalu banyak di UI
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
    except Exception:  # noqa: BLE001
        return pd.DataFrame(columns=cols)
    rows = [
        list(d["values"]) + [d.get("target")]
        for d in docs
        if len(d.get("values", [])) == len(feats)
    ]
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


def feature_inputs(base, feats, prefix, ncols=3):
    vals = []
    cols = st.columns(ncols)
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
                "R2 Score": r2_score(yte, pred),
                "MAE": mean_absolute_error(yte, pred),
                "RMSE": mean_squared_error(yte, pred) ** 0.5,
            },
            "y_test": yte.values,
            "pred": pred,
            "n_train": len(Xtr),
            "n_test": len(Xte),
        }

    if task == "klasifikasi":
        X, y = df.drop(columns="target"), df["target"]
        Xtr, Xte, ytr, yte = train_test_split(
            X, y, test_size=test_size, random_state=seed, stratify=y
        )
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

    # Clustering
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
        c.metric(k, f"{v:,.4f}")


def select_data(task, key_prefix):
    c1, c2 = st.columns([1, 2])
    name = c1.selectbox("Pilih Dataset (scikit-learn)", TASK_DATASETS[task], key=f"{key_prefix}_ds")
    use_extra = c2.checkbox(
        "Sertakan data tambahan dari MongoDB saat training",
        value=True,
        disabled=db is None,
        key=f"{key_prefix}_extra",
    )
    df, n_extra, feats, tn = get_training_df(task, name, use_extra and db is not None)
    st.caption(f"📊 Total Data Latih: **{len(df)}** baris (Data Tambahan Pengguna: **{n_extra}**)")
    return name, df, n_extra, feats, tn


# ----------------------------------------------------------------------------
# Halaman Dashboard
# ----------------------------------------------------------------------------
def page_home():
    st.markdown('<h1 class="main-title">🧠🤖 Dashboard Machine Learning</h1>', unsafe_allow_html=True)
    st.markdown("Aplikasi interaktif pemodelan **Regresi Linear**, **Klasifikasi**, dan **Clustering** terintegrasi secara *real-time* dengan **MongoDB Atlas**. TAS AI : Hana Meisaria 632025057")
    st.write("")

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown('''
            <div class="custom-card">
                <div class="card-title">📈 Regresi Linear</div>
                <div class="card-desc">Prediksi nilai kontinu dengan pilihan dataset <b>Diabetes</b> & <b>California Housing</b>.</div>
            </div>
        ''', unsafe_allow_html=True)
    with c2:
        st.markdown('''
            <div class="custom-card">
                <div class="card-title">🧪 Klasifikasi Multi-Algoritma</div>
                <div class="card-desc">Prediksi kategori kelas dengan pilihan dataset <b>Iris</b>, <b>Wine</b>, & <b>Breast Cancer</b>.</div>
            </div>
        ''', unsafe_allow_html=True)
    with c3:
        st.markdown('''
            <div class="custom-card">
                <div class="card-title">🔵 Clustering K-Means</div>
                <div class="card-desc">Pengelompokan data tak berlabel dengan dataset <b>Iris</b>, <b>Wine</b>, & <b>Breast Cancer</b>.</div>
            </div>
        ''', unsafe_allow_html=True)

    st.subheader("⚡ Status Server & Database")
    if db is None:
        st.error(f"MongoDB belum terhubung: {db_error}")
    else:
        st.success("MongoDB Atlas terhubung aktif.")
        rows = []
        for task, names in TASK_DATASETS.items():
            for n in names:
                rows.append(
                    {
                        "Modul Pemodelan": TASK_LABEL[task],
                        "Dataset": n,
                        "Data Tambahan (MongoDB)": db.dataset_tambahan.count_documents({"key": f"{task}:{n}"}),
                        "Model Biner Tersimpan": db.models.count_documents({"key": f"{task}:{n}"}),
                    }
                )
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
        st.caption(f"💬 Total Komentar Publik Tersimpan: **{db.komentar.count_documents({})}**")


def page_regresi():
    st.markdown('<h1 class="main-title">📈 Regresi Linear</h1>', unsafe_allow_html=True)
    name, df, n_extra, feats, tn = select_data("regresi", "reg")
    test_size = st.slider("Proporsi Data Uji (Test Size)", 0.1, 0.5, 0.2, 0.05, key="reg_ts")
    
    res = run_model("regresi", df, test_size=test_size)
    show_metrics(res["metrics"])
    st.write("")

    t1, t2, t3 = st.tabs(["📊 Dataset Overview", "📈 Grafik & Koefisien Model", "🔮 Simulator Prediksi"])
    with t1:
        st.dataframe(df.tail(20) if n_extra else df.head(20), use_container_width=True)
        st.write("**Deskripsi Statistik Dataset:**")
        st.write(df.describe().T)
    with t2:
        c1, c2 = st.columns(2)
        fig, ax = plt.subplots(figsize=(6, 4.5))
        fig.patch.set_facecolor('#111827')
        ax.set_facecolor('#1f2937')
        ax.scatter(res["y_test"], res["pred"], alpha=0.7, color='#38bdf8', edgecolors='none')
        lim = [min(res["y_test"].min(), res["pred"].min()), max(res["y_test"].max(), res["pred"].max())]
        ax.plot(lim, lim, "--", color='#f43f5e', linewidth=2)
        ax.set_xlabel("Nilai Aktual", color='#9ca3af')
        ax.set_ylabel("Nilai Prediksi", color='#9ca3af')
        ax.set_title(f"Aktual vs Prediksi ({name})", color='#f3f4f6')
        ax.tick_params(colors='#9ca3af')
        for spine in ax.spines.values():
            spine.set_color('#374151')
        c1.pyplot(fig)

        coef = pd.DataFrame({"Fitur": feats, "Koefisien": res["model"].coef_})
        c2.write("**Koefisien Bobot Regresi:**")
        c2.dataframe(coef, hide_index=True, use_container_width=True)
        c2.caption(f"Intercept Nilai Konstan: **{res['model'].intercept_:.4f}**")

    with t3:
        base = load_base(name)[0]
        st.markdown(f"##### Input Nilai Fitur ({name}):")
        vals = feature_inputs(base, feats, "reg_pred")
        if st.button("🚀 Jalankan Prediksi", key="reg_btn", type="primary"):
            y = res["model"].predict(pd.DataFrame([vals], columns=feats))[0]
            st.balloons()
            st.success(f"Hasil Prediksi Target: **{y:.4f}**")


def page_klasifikasi():
    st.markdown('<h1 class="main-title">🧪 Klasifikasi Multi-Algoritma</h1>', unsafe_allow_html=True)
    name, df, n_extra, feats, tn = select_data("klasifikasi", "clf")
    c1, c2 = st.columns(2)
    model_name = c1.selectbox(
        "Pilih Algoritma ML", ["Random Forest", "Logistic Regression", "KNN", "Decision Tree"], key="clf_model"
    )
    test_size = c2.slider("Proporsi Data Uji", 0.1, 0.5, 0.2, 0.05, key="clf_ts")
    
    res = run_model("klasifikasi", df, model_name=model_name, test_size=test_size)
    show_metrics(res["metrics"])
    st.write("")

    t1, t2, t3 = st.tabs(["📊 Distribusi Data", "🎯 Confusion Matrix", "🔮 Simulator Prediksi"])
    with t1:
        show = df.copy()
        show["kelas"] = show["target"].map(lambda i: tn[int(i)] if tn else str(int(i)))
        st.dataframe(show.tail(20) if n_extra else show.head(20), use_container_width=True)
        st.bar_chart(show["kelas"].value_counts())
    with t2:
        fig, ax = plt.subplots(figsize=(5, 4))
        fig.patch.set_facecolor('#111827')
        ax.set_facecolor('#1f2937')
        labels_display = [tn[c] if tn else str(c) for c in res["classes"]]
        cm_display = ConfusionMatrixDisplay(res["cm"], display_labels=labels_display)
        cm_display.plot(ax=ax, colorbar=False, cmap='plasma')
        ax.set_title(f"Confusion Matrix ({model_name} - {name})", color='#f3f4f6')
        ax.tick_params(colors='#9ca3af')
        st.pyplot(fig)
    with t3:
        base = load_base(name)[0]
        st.markdown(f"##### Input Nilai Fitur ({name}):")
        vals = feature_inputs(base, feats, "clf_pred")
        if st.button("🚀 Prediksi Kelas", key="clf_btn", type="primary"):
            Xn = pd.DataFrame([vals], columns=feats)
            pred = int(res["model"].predict(Xn)[0])
            pred_label = tn[pred] if tn else str(pred)
            st.balloons()
            st.success(f"Hasil Prediksi Kelas: **{pred_label}**")
            if hasattr(res["model"], "predict_proba"):
                proba = res["model"].predict_proba(Xn)[0]
                st.caption("Distribusi Probabilitas Prediksi:")
                idx_labels = [tn[c] if tn else str(c) for c in res["model"].classes_]
                st.bar_chart(pd.Series(proba, index=idx_labels))


def page_clustering():
    st.markdown('<h1 class="main-title">🔵 Clustering K-Means</h1>', unsafe_allow_html=True)
    name, df, n_extra, feats, tn = select_data("clustering", "clu")
    k = st.slider("Jumlah Cluster (k)", 2, 10, 3, key="clu_k")
    
    res = run_model("clustering", df, k=k)
    show_metrics(res["metrics"])
    st.write("")

    t1, t2, t3 = st.tabs(["Visualisasi Cluster (PCA)", "📊 Profiling Cluster", "🔮 Prediksi Cluster Data Baru"])
    with t1:
        c1, c2 = st.columns(2)
        fig, ax = plt.subplots(figsize=(5, 4))
        fig.patch.set_facecolor('#111827')
        ax.set_facecolor('#1f2937')
        sc = ax.scatter(res["pca"][:, 0], res["pca"][:, 1], c=res["labels"], cmap="viridis", s=35, alpha=0.9)
        ax.set_xlabel("Principal Component 1 (PC1)", color='#9ca3af')
        ax.set_ylabel("Principal Component 2 (PC2)", color='#9ca3af')
        ax.set_title(f"Proyeksi K-Means (k={k} - {name})", color='#f3f4f6')
        ax.tick_params(colors='#9ca3af')
        c1.pyplot(fig)

        c2.write("**Evaluasi Jumlah Cluster (Elbow & Silhouette):**")
        ed = elbow_data(df)
        c2.line_chart(ed["Inertia"])
        c2.line_chart(ed["Silhouette"])
    with t2:
        X = df.drop(columns="target", errors="ignore").copy()
        X["cluster"] = res["labels"]
        st.write("**Jumlah Anggota per Cluster:**")
        st.bar_chart(X["cluster"].value_counts().sort_index())
        st.write("**Nilai Rata-rata Fitur per Cluster:**")
        st.dataframe(X.groupby("cluster").mean(), use_container_width=True)
    with t3:
        base = load_base(name)[0]
        st.markdown(f"##### Input Nilai Fitur ({name}):")
        vals = feature_inputs(base, feats, "clu_pred")
        if st.button("🚀 Tentukan Cluster", key="clu_btn", type="primary"):
            Xn = res["scaler"].transform(pd.DataFrame([vals], columns=feats))
            st.success(f"Data Tersebut Masuk ke **Cluster {int(res['model'].predict(Xn)[0])}**")


def page_retrain():
    st.markdown('<h1 class="main-title">🔁 Management Data & Retraining</h1>', unsafe_allow_html=True)
    if db is None:
        st.error(f"Fitur ini membutuhkan koneksi MongoDB Atlas. Status: {db_error}")
        return

    c1, c2 = st.columns(2)
    task = c1.selectbox("Modul Pemodelan", list(TASK_LABEL), format_func=TASK_LABEL.get, key="rt_task")
    name = c2.selectbox("Dataset Tujuan", TASK_DATASETS[task], key="rt_ds")
    base, feats, tn = load_base(name)
    key = f"{task}:{name}"
    n_extra = db.dataset_tambahan.count_documents({"key": key})
    st.caption(f"📁 Data Asli Sklearn: **{len(base)}** baris | 📥 Data Tambahan MongoDB: **{n_extra}** baris")

    t1, t2, t3, t4 = st.tabs(["➕ Input Manual Data", "📄 Upload Berkas CSV", "🗂️ Kelola Data Tambahan", "🔁 Retraining Model ML"])

    with t1:
        with st.form("form_manual", clear_on_submit=False):
            vals = feature_inputs(base, feats, "rt_man")
            target = None
            if task == "regresi":
                target = st.number_input(
                    "Target Value",
                    value=float(base["target"].median()),
                    key="rt_man_target",
                )
            elif task == "klasifikasi":
                labels_opt = tn if tn else [str(i) for i in sorted(base["target"].unique())]
                label = st.selectbox("Label Kelas (Target)", labels_opt, key="rt_man_label")
                target = tn.index(label) if tn else float(label)
            if st.form_submit_button("💾 Simpan Baris Data ke MongoDB"):
                insert_rows(task, name, [(vals, target)], "manual")
                st.success("Data baru berhasil disimpan ke MongoDB! Buka tab Retraining untuk melatih ulang model.")
                st.rerun()

    with t2:
        cols_needed = feats + (["target"] if task != "clustering" else [])
        st.write("Struktur Kolom CSV yang Diperlukan:", ", ".join(cols_needed))
        st.download_button(
            "📥 Unduh Template Format CSV",
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
                    st.error(f"Format Kolom Kurang Tepat: {missing}")
                else:
                    new = new[cols_needed].dropna()
                    if task == "klasifikasi" and new["target"].dtype == object and tn:
                        new["target"] = new["target"].map({n: i for i, n in enumerate(tn)})
                        new = new.dropna()
                    for c in cols_needed:
                        new[c] = pd.to_numeric(new[c], errors="raise")
                    st.dataframe(new.head(), use_container_width=True)
                    st.write(f"✓ Terbaca **{len(new)}** baris data valid.")
                    if st.button("💾 Unggah Semua Baris ke MongoDB", key="rt_csv_btn", type="primary"):
                        rows = [
                            (r[feats].tolist(), r["target"] if task != "clustering" else None)
                            for _, r in new.iterrows()
                        ]
                        n = insert_rows(task, name, rows, "csv")
                        st.success(f"{n} baris data berhasil disimpan ke MongoDB.")
                        st.rerun()
            except Exception as e:  # noqa: BLE001
                st.error(f"Gagal membaca file CSV: {e}")

    with t3:
        extra = get_extra(task, name, feats)
        if extra.empty:
            st.info("Belum ada data tambahan tersimpan untuk dataset ini.")
        else:
            st.dataframe(extra, use_container_width=True)
        if n_extra:
            if st.checkbox("Konfirmasi reset / hapus seluruh data tambahan dataset ini", key="rt_del_ok"):
                if st.button("🗑️ Hapus Data Tambahan", key="rt_del", type="primary"):
                    db.dataset_tambahan.delete_many({"key": key})
                    st.rerun()

    with t4:
        params = {}
        model_name_selected = "Linear Regression"
        if task == "klasifikasi":
            model_name_selected = st.selectbox(
                "Algoritma Evaluasi", ["Random Forest", "Logistic Regression", "KNN", "Decision Tree"], key="rt_algo"
            )
            params["model_name"] = model_name_selected
        elif task == "clustering":
            model_name_selected = "K-Means"
            params["k"] = st.slider("Jumlah Cluster (k)", 2, 10, 3, key="rt_k")

        st.markdown("Komparasi performa model yang dilatih dengan **Data Asli** vs **Data Asli + Tambahan**:")
        if st.button("🚀 Eksekusi Retraining & Simpan Model Biner", type="primary", key="rt_go"):
            if n_extra == 0:
                st.warning("Belum ada data tambahan tersimpan; hasil metrik sebelum & sesudah akan persis sama.")
            df_new, n_e, _, _ = get_training_df(task, name, True)
            before = run_model(task, base, **params)
            after = run_model(task, df_new, **params)
            cmp = pd.DataFrame(
                {"Sebelum (Data Asli)": before["metrics"], "Sesudah (Asli + Tambahan)": after["metrics"]}
            )
            st.dataframe(cmp.style.format("{:.4f}"), use_container_width=True)
            
            # Simpan Riwayat
            db.riwayat_training.insert_one(
                {
                    "key": key,
                    "waktu": datetime.now(timezone.utc),
                    "params": params,
                    "n_asli": len(base),
                    "n_tambahan": n_e,
                    "metrik_sebelum": {k: float(v) for k, v in before["metrics"].items()},
                    "metrik_sesudah": {k: float(v) for k, v in after["metrics"].items()},
                }
            )
            
            # Simpan Model Biner Objek ke MongoDB
            saved = save_model_to_db(
                task=task,
                dataset_name=name,
                model_obj=after["model"],
                model_name=model_name_selected,
                metrics={k: float(v) for k, v in after["metrics"].items()}
            )
            if saved:
                st.balloons()
                st.success("Retraining Selesai! Model ML biner terbaru berhasil diperbarui di MongoDB Atlas.")

        hist = list(db.riwayat_training.find({"key": key}).sort("waktu", -1).limit(10))
        if hist:
            st.write("**Riwayat Retraining Terakhir:**")
            st.dataframe(
                pd.DataFrame(
                    [
                        {
                            "Waktu": now_wib(h["waktu"]),
                            "Jumlah Data Tambahan": h["n_tambahan"],
                            **{f"{k} (Baru)": round(v, 4) for k, v in h["metrik_sesudah"].items()},
                        }
                        for h in hist
                    ]
                ),
                hide_index=True,
                use_container_width=True,
            )


def page_komentar():
    st.markdown('<h1 class="main-title">💬 Diskusi & Komentar Publik</h1>', unsafe_allow_html=True)
    if db is None:
        st.error(f"Fitur ini membutuhkan koneksi MongoDB. Status: {db_error}")
        return

    with st.form("form_komentar", clear_on_submit=True):
        nama = st.text_input("Nama Anda", max_chars=50)
        isi = st.text_area("Tuliskan Komentar / Masukan Anda", max_chars=500)
        if st.form_submit_button("🚀 Kirim Komentar"):
            if not nama.strip() or not isi.strip():
                st.warning("Mohon isi Nama dan Komentar terlebih dahulu.")
            else:
                db.komentar.insert_one(
                    {
                        "nama": nama.strip(),
                        "komentar": isi.strip(),
                        "created_at": datetime.now(timezone.utc),
                    }
                )
                st.success("Terima kasih! Komentar Anda berhasil tersimpan di database MongoDB.")

    st.subheader("💬 Tanggapan & Komentar Pengunjung")
    docs = list(db.komentar.find().sort("created_at", -1).limit(50))
    if not docs:
        st.info("Belum ada komentar tersimpan. Jadilah yang pertama!")
    for d in docs:
        with st.chat_message("user"):
            st.markdown(f"**{d['nama']}** · _{now_wib(d['created_at'])}_")
            st.text(d["komentar"])


# ----------------------------------------------------------------------------
# Sidebar & Router Navigasi
# ----------------------------------------------------------------------------
PAGES = {
    "🏠 Dashboard Beranda": page_home,
    "📈 Regresi Linear": page_regresi,
    "🧪 Klasifikasi": page_klasifikasi,
    "🔵 Clustering": page_clustering,
    "🔁 Update Data & Retraining": page_retrain,
    "💬 Diskusi Komentar": page_komentar,
}

st.sidebar.markdown("### 👩‍🎓 Menu Navigasi")
choice = st.sidebar.radio("Navigasi", list(PAGES), label_visibility="collapsed")
st.sidebar.divider()

if db is None:
    st.sidebar.error("🔴 MongoDB: Tidak Terhubung")
    if st.sidebar.button("Coba Sambung Ulang"):
        init_db.clear()
        st.rerun()
else:
    st.sidebar.success("🟢 MongoDB: Terhubung")

PAGES[choice]()