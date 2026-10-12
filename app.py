import streamlit as st
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
from sklearn.linear_model import LinearRegression, Ridge, Lasso, LogisticRegression
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.datasets import load_diabetes, load_breast_cancer, load_iris
from sklearn.datasets import fetch_california_housing
import matplotlib.pyplot as plt
import seaborn as sns

# Header konfigurasi halaman
st.set_page_config(page_title="ML Studio: Regresi & Klasifikasi",
                   layout="wide",
                   initial_sidebar_state="expanded")

st.title("🎯 ML Studio: Regresi & Klasifikasi dengan Scikit-Learn")
st.markdown("Selamat datang! Pilih task, dataset, dan model di sidebar. Kemudian klik tombol Train untuk melihat hasil.")

# ----------------------------
# Helpers / Datastore
# ----------------------------
@st.cache_data
def load_dataset(name: str, task: str):
    """
    Mengembalikan X, y berdasarkan pilihan dataset dan tugas (regresi/klasifikasi).
    """
    if task == "Regresi":
        if name == "Diabetes (sklearn)":
            data = load_diabetes()
            return data.data, data.target
        elif name == "California Housing (fetch)":
            data = fetch_california_housing()
            return data.data, data.target
        else:
            raise ValueError("Dataset regresi tidak dikenali.")
    else:  # Klasifikasi
        if name == " Breast Cancer (sklearn)":
            data = load_breast_cancer()
            return data.data, data.target
        elif name == "Iris (sklearn)":
            data = load_iris()
            return data.data, data.target
        else:
            raise ValueError("Dataset klasifikasi tidak dikenali.")

# Daftar dataset
REGRESSION_DATASETS = ["Diabetes (sklearn)", "California Housing (fetch)"]
CLASSIFICATION_DATASETS = [" Breast Cancer (sklearn)", "Iris (sklearn)"]

# Model options
REGRESSION_MODELS = {
    "Linear Regression": LinearRegression(),
    "Ridge": Ridge(),
    "Lasso": Lasso(),
    "Decision Tree": DecisionTreeRegressor(random_state=42),
    "Random Forest": RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)
}

CLASSIFICATION_MODELS = {
    "Logistic Regression": Pipeline([("scaler", StandardScaler()), ("clf", LogisticRegression(max_iter=1000))]),
    "KNN": KNeighborsClassifier(),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1),
    "SVM": Pipeline([("scaler", StandardScaler()), ("clf", SVC(kernel="rbf", probability=True))])
}

# ----------------------------
# Sidebar UI
# ----------------------------
st.sidebar.title("🧭 Menu Utama")
task = st.sidebar.selectbox("Pilih Tugas", ["Regresi", "Klasifikasi"])

# Dataset pilihan
if task == "Regresi":
    dataset_choice = st.sidebar.selectbox("Dataset Regresi", REGRESSION_DATASETS)
    model_choice = st.sidebar.selectbox("Model Regresi", list(REGRESSION_MODELS.keys()))
    models_map = REGRESSION_MODELS
else:
    dataset_choice = st.sidebar.selectbox("Dataset Klasifikasi", CLASSIFICATION_DATASETS)
    model_choice = st.sidebar.selectbox("Model Klasifikasi", list(CLASSIFICATION_MODELS.keys()))
    models_map = CLASSIFICATION_MODELS

# Test size
test_size = st.sidebar.slider("Ukuran data test", min_value=0.1, max_value=0.5, value=0.2, step=0.05)

# Train button
train_button = st.sidebar.button("🚀 Train Model")

# Tampilkan status model di bawah tombol
if "trained" not in st.session_state:
    st.session_state.trained = False
if "metrics" not in st.session_state:
    st.session_state.metrics = None
if "y_true" not in st.session_state:
    st.session_state.y_true = None
if "y_pred" not in st.session_state:
    st.session_state.y_pred = None
if "model" not in st.session_state:
    st.session_state.model = None
if "X_train" not in st.session_state:
    st.session_state.X_train = None
if "X_test" not in st.session_state:
    st.session_state.X_test = None
if "y_train" not in st.session_state:
    st.session_state.y_train = None
if "y_test" not in st.session_state:
    st.session_state.y_test = None

# ----------------------------
# Main area: Tabs
# ----------------------------
tab1, tab2 = st.tabs(["📈 Actual vs Predicted / Confusion Matrix", "🧮 Evaluation Metrics"])
with tab1:
    st.subheader("Tab 1: Visualisasi")
    if st.session_state.trained:
        if task == "Regresi":
            # Plot Actual vs Predicted
            fig, ax = plt.subplots(figsize=(6.5, 4))
            ax.scatter(st.session_state.y_true, st.session_state.y_pred, alpha=0.6)
            ax.plot([st.session_state.y_true.min(), st.session_state.y_true.max()],
                    [st.session_state.y_true.min(), st.session_state.y_true.max()],
                    color="red", linestyle="--")
            ax.set_xlabel("Actual")
            ax.set_ylabel("Predicted")
            ax.set_title("Actual vs Predicted")
            st.pyplot(fig)
        else:
            # Confusion Matrix
            cm = confusion_matrix(st.session_state.y_true, st.session_state.y_pred)
            fig, ax = plt.subplots(figsize=(6, 4))
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
            ax.set_xlabel("Predicted")
            ax.set_ylabel("Actual")
            ax.set_title("Confusion Matrix")
            st.pyplot(fig)
    else:
        st.info("Klik Train untuk melihat grafik."

        )
with tab2:
    st.subheader("Tab 2: Metrics & Classification Report")
    if st.session_state.trained and st.session_state.metrics is not None:
        metrics = st.session_state.metrics
        if task == "Regresi":
            df = pd.DataFrame({
                "Metric": ["MAE", "MSE", "RMSE", "R²"],
                "Value": [metrics["MAE"], metrics["MSE"], metrics["RMSE"], metrics["R2"]]
            })
            st.table(df)
        else:
            # Classification metrics
            acc = metrics["Accuracy"]
            prec = metrics["Precision"]
            rec = metrics["Recall"]
            f1 = metrics["F1"]
            st.metric("Accuracy", f"{acc:.4f}")
            st.metric("Precision", f"{prec:.4f}")
            st.metric("Recall", f"{rec:.4f}")
            st.metric("F1 Score", f"{f1:.4f}")
            st.text("Classification Report:")
            st.code(metrics["Report"])
    else:
        st.info("Klik Train untuk menghitung metrics.")

# ----------------------------
# Training Logic
# ----------------------------
def train_model():
    # Load dataset
    X, y = load_dataset(dataset_choice, task)

    # If dataset has more than 1 feature, proceed. If label vector size mismatch, handle errors.
    if X is None or y is None:
        st.error("Gagal memuat dataset.")
        return

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42, stratify=y if task == "Klasifikasi" and len(np.unique(y)) > 2 else None)

    # Determine model
    model = models_map[model_choice]

    # For classification with SVM or others, ensure compatibility
    try:
        model.fit(X_train, y_train)
    except Exception as e:
        st.error(f"Gagal melatih model: {e}")
        return

    # Predict
    y_pred = model.predict(X_test)

    # Save to session_state
    st.session_state.trained = True
    st.session_state.model = model
    st.session_state.X_train, st.session_state.X_test = X_train, X_test
    st.session_state.y_train, st.session_state.y_test = y_train, y_test
    st.session_state.y_true = y_test
    st.session_state.y_pred = y_pred

    # Metrics
    if task == "Regresi":
        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        rmse = mean_squared_error(y_test, y_pred, squared=False)
        r2 = r2_score(y_test, y_pred)
        st.session_state.metrics = {"MAE": mae, "MSE": mse, "RMSE": rmse, "R2": r2}
    else:
        acc = accuracy_score(y_test, y_pred)
        # For binary/multiclass, compute per-class metrics if possible
        try:
            prec = precision_score(y_test, y_pred, average="weighted", zero_division=0)
            rec = recall_score(y_test, y_pred, average="weighted", zero_division=0)
            f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)
        except Exception:
            prec = rec = f1 = 0.0
        report = classification_report(y_test, y_pred, zero_division=0)
        st.session_state.metrics = {
            "Accuracy": acc,
            "Precision": prec,
            "Recall": rec,
            "F1": f1,
            "Report": report
        }

# Action: run train when button pressed
if train_button:
    train_model()

# ----------------------------
# UI niceties
# ----------------------------
st.sidebar.markdown("---")
st.sidebar.caption("Tips: ")
st.sidebar.write("- Gunakan dataset toy sklearn untuk eksplorasi cepat.")
st.sidebar.write("- Pilihan model tersusun rapi untuk regresi dan klasifikasi.")
st.sidebar.write("- Gunakan tombol Train untuk mulai fitting dan evaluasi.")

# Tampilkan ringkasan singkat di bawah tombol tombol train
if st.session_state.trained:
    st.sidebar.success("Model telah dilatih. Lihat tab untuk grafik dan metrics.")
