"""
TAS AI - Hana Meisaria
Aplikasi Streamlit: Regresi Linear, Klasifikasi, Clustering
+ Integrasi MongoDB (dataset tambahan, retraining, komentar publik)
Sumber data: scikit-learn (Diabetes, Iris, Wine)
"""
import os
from datetime import datetime, timedelta, timezone

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from pymongo import MongoClient
from sklearn.cluster import KMeans
from sklearn.datasets import load_diabetes, load_iris, load_wine
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

st.set_page_config(page_title="TAS AI - Hana Meisaria", page_icon="🤖", layout="wide")

# ----------------------------------------------------------------------------
# Konfigurasi dataset (semua dari scikit-learn)
# ----------------------------------------------------------------------------
LOADERS = {"Diabetes": load_diabetes, "Iris": load_iris, "Wine": load_wine}
TASK_DATASETS = {
    "regresi": ["Diabetes"],
    "klasifikasi": ["Iris", "Wine"],
    "clustering": ["Iris", "Wine"],
}
TASK_LABEL = {
    "regresi": "Regresi Linear",
    "klasifikasi": "Klasifikasi",
    "clustering": "Clustering",
}


# ----------------------------------------------------------------------------
# MongoDB
# ----------------------------------------------------------------------------
def _secret(name, default=None):
    try:
        return st.secrets[name]
    except Exception:
        return os.getenv(name, default)


@st.cache_resource
def init_db():
    # Memasukkan URL koneksi MongoDB secara langsung
    uri = "mongodb://632025057:632025057_Hana@ac-z4y3rlw-shard-00-00.n0apbrv.mongodb.net:27017,ac-z4y3rlw-shard-00-01.n0apbrv.mongodb.net:27017,ac-z4y3rlw-shard-00-02.n0apbrv.mongodb.net:27017/?ssl=true&replicaSet=atlas-ikculp-shard-0&authSource=admin&appName=Cluster0"
    if not uri:
        return None, "MONGODB_URI belum diatur ."
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
# Data helpers
# ----------------------------------------------------------------------------
@st.cache_data
def load_base(name):
    d = LOADERS[name](as_frame=True)
    feats = list(d.feature_names)
    tn = getattr(d, "target_names", None)
    tn = [str(x) for x in tn] if tn is not None else None
    return d.frame[feats + ["target"]].copy(), feats, tn


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
    """Gabungkan dataset sklearn + data tambahan dari MongoDB."""
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
# Model
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
    """Latih model & hitung metrik. Di-cache berdasarkan isi df,
    sehingga otomatis dilatih ulang saat ada data tambahan."""
    if task == "regresi":
        X, y = df.drop(columns="target"), df["target"]
        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=test_size, random_state=seed)
        m = LinearRegression().fit(Xtr, ytr)
        pred = m.predict(Xte)
        return {
            "model": m,
            "metrics": {
                "R2": r2_score(yte, pred),
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
                "Precision (macro)": precision_score(yte, pred, average="macro", zero_division=0),
                "Recall (macro)": recall_score(yte, pred, average="macro", zero_division=0),
                "F1 (macro)": f1_score(yte, pred, average="macro", zero_division=0),
            },
            "cm": confusion_matrix(yte, pred, labels=classes),
            "classes": classes,
            "n_train": len(Xtr),
            "n_test": len(Xte),
        }

    # clustering
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
    name = c1.selectbox("Dataset (scikit-learn)", TASK_DATASETS[task], key=f"{key_prefix}_ds")
    use_extra = c2.checkbox(
        "Sertakan data tambahan dari MongoDB saat training",
        value=True,
        disabled=db is None,
        key=f"{key_prefix}_extra",
    )
    df, n_extra, feats, tn = get_training_df(task, name, use_extra and db is not None)
    st.caption(f"Total data training: **{len(df)}** baris (data tambahan dari pengguna: **{n_extra}**)")
    return name, df, n_extra, feats, tn


# ----------------------------------------------------------------------------
# Halaman
# ----------------------------------------------------------------------------
def page_home():
    st.title("🤖 Tugas Akhir AI: Aplikasi Machine Learning")
    st.write(
        "Aplikasi ini memuat tiga jenis pemodelan (**Regresi Linear**, **Klasifikasi**, "
        "**Clustering**) dengan dataset dari **scikit-learn**. Pengguna dapat menambah data "
        "baru ke **MongoDB**, melakukan **retraining**, dan meninggalkan **komentar**."
    )
    c1, c2, c3 = st.columns(3)
    c1.info("**Regresi Linear**\n\nDiabetes: prediksi progres penyakit")
    c2.info("**Klasifikasi**\n\nIris / Wine: prediksi kelas")
    c3.info("**Clustering**\n\nK-Means pada Iris / Wine")

    st.subheader("Status")
    if db is None:
        st.warning(f"MongoDB belum terhubung: {db_error}")
    else:
        st.success("MongoDB terhubung.")
        rows = []
        for task, names in TASK_DATASETS.items():
            for n in names:
                rows.append(
                    {
                        "Modul": TASK_LABEL[task],
                        "Dataset": n,
                        "Data tambahan": db.dataset_tambahan.count_documents({"key": f"{task}:{n}"}),
                    }
                )
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
        st.caption(f"Jumlah komentar: {db.komentar.count_documents({})}")


def page_regresi():
    st.title("📈 Regresi Linear")
    name, df, n_extra, feats, tn = select_data("regresi", "reg")
    test_size = st.slider("Proporsi data uji", 0.1, 0.5, 0.2, 0.05, key="reg_ts")
    res = run_model("regresi", df, test_size=test_size)
    show_metrics(res["metrics"])

    t1, t2, t3 = st.tabs(["Dataset", "Hasil Model", "Prediksi"])
    with t1:
        st.dataframe(df.tail(20) if n_extra else df.head(20), width="stretch")
        st.write(df.describe().T)
    with t2:
        c1, c2 = st.columns(2)
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.scatter(res["y_test"], res["pred"], alpha=0.7)
        lim = [min(res["y_test"].min(), res["pred"].min()), max(res["y_test"].max(), res["pred"].max())]
        ax.plot(lim, lim, "r--")
        ax.set_xlabel("Aktual")
        ax.set_ylabel("Prediksi")
        ax.set_title("Aktual vs Prediksi (data uji)")
        c1.pyplot(fig)
        coef = pd.DataFrame({"Fitur": feats, "Koefisien": res["model"].coef_})
        c2.write("**Koefisien regresi**")
        c2.dataframe(coef, hide_index=True, width="stretch")
        c2.caption(f"Intercept: {res['model'].intercept_:.4f}")
    with t3:
        base = load_base(name)[0]
        st.write("Masukkan nilai fitur (sudah dalam skala dataset sklearn):")
        vals = feature_inputs(base, feats, "reg_pred")
        if st.button("Prediksi", key="reg_btn"):
            y = res["model"].predict(pd.DataFrame([vals], columns=feats))[0]
            st.success(f"Prediksi progres penyakit: **{y:.2f}**")


def page_klasifikasi():
    st.title("🧪 Klasifikasi")
    name, df, n_extra, feats, tn = select_data("klasifikasi", "clf")
    c1, c2 = st.columns(2)
    model_name = c1.selectbox(
        "Algoritma", ["Random Forest", "Logistic Regression", "KNN", "Decision Tree"], key="clf_model"
    )
    test_size = c2.slider("Proporsi data uji", 0.1, 0.5, 0.2, 0.05, key="clf_ts")
    res = run_model("klasifikasi", df, model_name=model_name, test_size=test_size)
    show_metrics(res["metrics"])

    t1, t2, t3 = st.tabs(["Dataset", "Hasil Model", "Prediksi"])
    with t1:
        show = df.copy()
        show["kelas"] = show["target"].map(lambda i: tn[int(i)])
        st.dataframe(show.tail(20) if n_extra else show.head(20), width="stretch")
        st.bar_chart(show["kelas"].value_counts())
    with t2:
        fig, ax = plt.subplots(figsize=(5, 4))
        ConfusionMatrixDisplay(res["cm"], display_labels=[tn[c] for c in res["classes"]]).plot(
            ax=ax, colorbar=False
        )
        ax.set_title("Confusion Matrix (data uji)")
        st.pyplot(fig)
    with t3:
        base = load_base(name)[0]
        vals = feature_inputs(base, feats, "clf_pred")
        if st.button("Prediksi", key="clf_btn"):
            Xn = pd.DataFrame([vals], columns=feats)
            pred = int(res["model"].predict(Xn)[0])
            st.success(f"Prediksi kelas: **{tn[pred]}**")
            if hasattr(res["model"], "predict_proba"):
                proba = res["model"].predict_proba(Xn)[0]
                st.bar_chart(pd.Series(proba, index=[tn[c] for c in res["model"].classes_]))


def page_clustering():
    st.title("🔵 Clustering (K-Means)")
    name, df, n_extra, feats, tn = select_data("clustering", "clu")
    k = st.slider("Jumlah cluster (k)", 2, 10, 3, key="clu_k")
    res = run_model("clustering", df, k=k)
    show_metrics(res["metrics"])

    t1, t2, t3 = st.tabs(["Visualisasi", "Profil Cluster", "Prediksi Cluster"])
    with t1:
        c1, c2 = st.columns(2)
        fig, ax = plt.subplots(figsize=(5, 4))
        sc = ax.scatter(res["pca"][:, 0], res["pca"][:, 1], c=res["labels"], cmap="viridis", s=25)
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        ax.set_title(f"K-Means k={k} (proyeksi PCA)")
        ax.legend(*sc.legend_elements(), title="Cluster")
        c1.pyplot(fig)
        c2.write("**Elbow & Silhouette** (k = 2..10)")
        ed = elbow_data(df)
        c2.line_chart(ed["Inertia"])
        c2.line_chart(ed["Silhouette"])
    with t2:
        X = df.drop(columns="target", errors="ignore").copy()
        X["cluster"] = res["labels"]
        st.write("Jumlah anggota per cluster")
        st.bar_chart(X["cluster"].value_counts().sort_index())
        st.write("Rata-rata fitur per cluster")
        st.dataframe(X.groupby("cluster").mean(), width="stretch")
    with t3:
        base = load_base(name)[0]
        vals = feature_inputs(base, feats, "clu_pred")
        if st.button("Tentukan cluster", key="clu_btn"):
            Xn = res["scaler"].transform(pd.DataFrame([vals], columns=feats))
            st.success(f"Data masuk ke **cluster {int(res['model'].predict(Xn)[0])}**")


def page_retrain():
    st.title("🔁 Update Dataset & Retraining")
    if db is None:
        st.error(f"Fitur ini membutuhkan MongoDB. Status: {db_error}")
        return

    c1, c2 = st.columns(2)
    task = c1.selectbox("Modul", list(TASK_LABEL), format_func=TASK_LABEL.get, key="rt_task")
    name = c2.selectbox("Dataset", TASK_DATASETS[task], key="rt_ds")
    base, feats, tn = load_base(name)
    key = f"{task}:{name}"
    n_extra = db.dataset_tambahan.count_documents({"key": key})
    st.caption(f"Data asli sklearn: **{len(base)}** baris | Data tambahan di MongoDB: **{n_extra}** baris")

    t1, t2, t3, t4 = st.tabs(["➕ Input Manual", "📄 Upload CSV", "🗂️ Data Tambahan", "🔁 Retraining"])

    with t1:
        with st.form("form_manual", clear_on_submit=False):
            vals = feature_inputs(base, feats, "rt_man")
            target = None
            if task == "regresi":
                target = st.number_input(
                    "Target (progres penyakit)",
                    value=float(base["target"].median()),
                    key="rt_man_target",
                )
            elif task == "klasifikasi":
                label = st.selectbox("Kelas (target)", tn, key="rt_man_label")
                target = tn.index(label)
            if st.form_submit_button("Simpan ke MongoDB"):
                insert_rows(task, name, [(vals, target)], "manual")
                st.success("Data tersimpan. Buka tab Retraining untuk melatih ulang model.")
                st.rerun()

    with t2:
        cols_needed = feats + (["target"] if task != "clustering" else [])
        st.write("Kolom CSV harus persis:", ", ".join(cols_needed))
        if task == "klasifikasi":
            st.caption(f"Kolom target boleh berisi indeks (0..{len(tn) - 1}) atau nama kelas: {tn}")
        st.download_button(
            "Unduh template CSV",
            base[cols_needed].head(3).to_csv(index=False),
            file_name=f"template_{name.lower()}.csv",
            mime="text/csv",
        )
        up = st.file_uploader("Upload CSV", type="csv", key="rt_csv")
        if up is not None:
            try:
                new = pd.read_csv(up)
                missing = [c for c in cols_needed if c not in new.columns]
                if missing:
                    st.error(f"Kolom kurang: {missing}")
                else:
                    new = new[cols_needed].dropna()
                    if task == "klasifikasi" and new["target"].dtype == object:
                        new["target"] = new["target"].map({n: i for i, n in enumerate(tn)})
                        new = new.dropna()
                    for c in cols_needed:
                        new[c] = pd.to_numeric(new[c], errors="raise")
                    st.dataframe(new.head(), width="stretch")
                    st.write(f"{len(new)} baris valid.")
                    if st.button("Simpan semua ke MongoDB", key="rt_csv_btn"):
                        rows = [
                            (r[feats].tolist(), r["target"] if task != "clustering" else None)
                            for _, r in new.iterrows()
                        ]
                        n = insert_rows(task, name, rows, "csv")
                        st.success(f"{n} baris tersimpan.")
                        st.rerun()
            except Exception as e:  # noqa: BLE001
                st.error(f"Gagal membaca CSV: {e}")

    with t3:
        extra = get_extra(task, name, feats)
        if extra.empty:
            st.info("Belum ada data tambahan.")
        else:
            st.dataframe(extra, width="stretch")
        if n_extra:
            if st.checkbox("Saya yakin ingin menghapus semua data tambahan dataset ini", key="rt_del_ok"):
                if st.button("Hapus data tambahan", key="rt_del"):
                    db.dataset_tambahan.delete_many({"key": key})
                    st.rerun()

    with t4:
        params = {}
        if task == "klasifikasi":
            params["model_name"] = st.selectbox(
                "Algoritma", ["Random Forest", "Logistic Regression", "KNN", "Decision Tree"], key="rt_algo"
            )
        if task == "clustering":
            params["k"] = st.slider("Jumlah cluster (k)", 2, 10, 3, key="rt_k")
        st.write(
            "Retraining membandingkan model yang dilatih dengan **data asli saja** "
            "dan model yang dilatih dengan **data asli + data tambahan**."
        )
        if st.button("🚀 Retrain model", type="primary", key="rt_go"):
            if n_extra == 0:
                st.warning("Belum ada data tambahan; hasil retraining akan sama dengan model awal.")
            df_new, n_e, _, _ = get_training_df(task, name, True)
            before = run_model(task, base, **params)
            after = run_model(task, df_new, **params)
            cmp = pd.DataFrame(
                {"Sebelum (data asli)": before["metrics"], "Sesudah (asli + tambahan)": after["metrics"]}
            )
            st.dataframe(cmp.style.format("{:.4f}"), width="stretch")
            st.caption(
                f"Data latih sebelum: {before['n_train']} | sesudah: {after['n_train']} "
                "(pembagian train/test berbeda karena jumlah data berubah)."
            )
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
            st.success("Retraining selesai & tercatat di riwayat. Halaman model kini memakai data terbaru.")

        hist = list(db.riwayat_training.find({"key": key}).sort("waktu", -1).limit(10))
        if hist:
            st.write("**Riwayat retraining (10 terakhir)**")
            st.dataframe(
                pd.DataFrame(
                    [
                        {
                            "Waktu": now_wib(h["waktu"]),
                            "Data tambahan": h["n_tambahan"],
                            **{f"{k} (baru)": round(v, 4) for k, v in h["metrik_sesudah"].items()},
                        }
                        for h in hist
                    ]
                ),
                hide_index=True,
                width="stretch",
            )


def page_komentar():
    st.title("💬 Komentar")
    if db is None:
        st.error(f"Fitur ini membutuhkan MongoDB. Status: {db_error}")
        return

    with st.form("form_komentar", clear_on_submit=True):
        nama = st.text_input("Nama", max_chars=50)
        isi = st.text_area("Komentar", max_chars=500)
        if st.form_submit_button("Kirim"):
            if not nama.strip() or not isi.strip():
                st.warning("Nama dan komentar wajib diisi.")
            else:
                db.komentar.insert_one(
                    {
                        "nama": nama.strip(),
                        "komentar": isi.strip(),
                        "created_at": datetime.now(timezone.utc),
                    }
                )
                st.success("Terima kasih, komentar Anda tersimpan!")

    st.subheader("Komentar Pengunjung")
    docs = list(db.komentar.find().sort("created_at", -1).limit(50))
    if not docs:
        st.info("Belum ada komentar. Jadilah yang pertama!")
    for d in docs:
        with st.chat_message("user"):
            st.markdown(f"**{d['nama']}** · _{now_wib(d['created_at'])}_")
            st.text(d["komentar"])


# ----------------------------------------------------------------------------
# Navigasi
# ----------------------------------------------------------------------------
PAGES = {
    "🏠 Beranda": page_home,
    "📈 Regresi Linear": page_regresi,
    "🧪 Klasifikasi": page_klasifikasi,
    "🔵 Clustering": page_clustering,
    "🔁 Update Dataset & Retraining": page_retrain,
    "💬 Komentar": page_komentar,
}

st.sidebar.title("Menu")
choice = st.sidebar.radio("Navigasi", list(PAGES), label_visibility="collapsed")
st.sidebar.divider()
if db is None:
    st.sidebar.error("MongoDB: tidak terhubung")
    if st.sidebar.button("Coba sambung ulang"):
        init_db.clear()
        st.rerun()
else:
    st.sidebar.success("MongoDB: terhubung")

PAGES[choice]()