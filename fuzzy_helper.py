"""
fuzzy_helper.py — Implementasi Fuzzy Classifier untuk TAS AI
- FuzzyKNNClassifier     : KNN dengan bobot fuzzy (m = fuzzifier)
- FuzzyCMeansClassifier  : Fuzzy C-Means + pemetaan cluster → kelas
"""
import numpy as np
import plotly.express as px
import pandas as pd
from scipy.spatial.distance import cdist
from sklearn.base import BaseEstimator, ClassifierMixin


# ---------------------------------------------------------------------------
# 1. FUZZY K-NEAREST NEIGHBORS
# ---------------------------------------------------------------------------
class FuzzyKNNClassifier(BaseEstimator, ClassifierMixin):
    """
    Fuzzy KNN — setiap tetangga diberi bobot:
        w_i = 1 / d_i^(2/(m-1))
    Membership kelas c:
        u_c(x) = Σ w_i · I(y_i = c) / Σ w_i
    """
    def __init__(self, n_neighbors=5, m=2.0):
        self.n_neighbors = n_neighbors
        self.m = m

    def fit(self, X, y):
        self.classes_ = np.unique(y)
        self.X_train_ = np.asarray(X, dtype=float)
        self.y_train_ = np.asarray(y)
        return self

    def predict_proba(self, X):
        X = np.asarray(X, dtype=float)
        dist = cdist(X, self.X_train_)
        proba = np.zeros((len(X), len(self.classes_)))

        for i in range(len(X)):
            k = min(self.n_neighbors, len(self.X_train_))
            idx = np.argsort(dist[i])[:k]
            d = np.clip(dist[i][idx], 1e-10, None)
            w = 1.0 / (d ** (2.0 / (self.m - 1)))
            for j, c in enumerate(self.classes_):
                mask = self.y_train_[idx] == c
                proba[i, j] = w[mask].sum()
            total = proba[i].sum()
            if total > 0:
                proba[i] /= total
        return proba

    def predict(self, X):
        return self.classes_[np.argmax(self.predict_proba(X), axis=1)]


# ---------------------------------------------------------------------------
# 2. FUZZY C-MEANS CLASSIFIER
# ---------------------------------------------------------------------------
class FuzzyCMeansClassifier(BaseEstimator, ClassifierMixin):
    """
    FCM membentuk c cluster fuzzy, lalu setiap cluster dipetakan ke kelas
    mayoritas anggota training-nya. Prediksi = agregasi membership per kelas.
    """
    def __init__(self, n_clusters=None, m=2.0, max_iter=100, tol=1e-4,
                 random_state=42):
        self.n_clusters = n_clusters
        self.m = m
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state

    def _fcm(self, X):
        rng = np.random.default_rng(self.random_state)
        n, c = len(X), self.n_clusters
        U = rng.random((n, c))
        U /= U.sum(axis=1, keepdims=True)

        for _ in range(self.max_iter):
            U_old = U.copy()
            Um = U ** self.m
            centroids = (Um.T @ X) / np.clip(Um.sum(axis=0)[:, None], 1e-10, None)
            d = np.clip(cdist(X, centroids), 1e-10, None)
            inv = (1.0 / d) ** (2.0 / (self.m - 1))
            U = inv / inv.sum(axis=1, keepdims=True)
            if np.linalg.norm(U - U_old) < self.tol:
                break
        return U, centroids

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)
        self.classes_ = np.unique(y)
        if self.n_clusters is None:
            self.n_clusters = len(self.classes_)

        self.U_, self.centroids_ = self._fcm(X)
        hard_labels = np.argmax(self.U_, axis=1)
        self.cluster_to_class_ = {}
        for c in range(self.n_clusters):
            members = y[hard_labels == c]
            if len(members) == 0:
                self.cluster_to_class_[c] = self.classes_[0]
            else:
                vals, counts = np.unique(members, return_counts=True)
                self.cluster_to_class_[c] = vals[np.argmax(counts)]
        return self

    def predict_proba(self, X):
        X = np.asarray(X, dtype=float)
        d = np.clip(cdist(X, self.centroids_), 1e-10, None)
        inv = (1.0 / d) ** (2.0 / (self.m - 1))
        U = inv / inv.sum(axis=1, keepdims=True)

        proba = np.zeros((len(X), len(self.classes_)))
        for ci in range(self.n_clusters):
            j = np.where(self.classes_ == self.cluster_to_class_[ci])[0][0]
            proba[:, j] += U[:, ci]
        proba = proba / np.clip(proba.sum(axis=1, keepdims=True), 1e-10, None)
        return proba

    def predict(self, X):
        return self.classes_[np.argmax(self.predict_proba(X), axis=1)]


# ---------------------------------------------------------------------------
# 3. HELPER: Grafik Membership
# ---------------------------------------------------------------------------
def plot_membership(proba, classes, title="Derajat Keanggotaan (Membership) Fuzzy"):
    df = pd.DataFrame({"Kelas": list(classes), "Membership": list(proba)})
    fig = px.bar(
        df, x="Membership", y="Kelas", orientation="h",
        color="Membership", color_continuous_scale="Viridis",
        title=title, range_x=[0, 1], text="Membership",
    )
    fig.update_traces(texttemplate="%{text:.4f}", textposition="outside")
    fig.update_layout(
        template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)", height=300, showlegend=False,
        coloraxis_showscale=False,
    )
    return fig