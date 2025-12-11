# HUMAN VOICE CLASSIFICATION & CLUSTERING

import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

st.set_page_config(page_title="Human Voice App", layout="wide")

MODELS_DIR = Path("models")
METRICS_PATH = Path("model_metrics.csv")

# Load all models
@st.cache_resource
def load_models():
    models = {}
    if MODELS_DIR.exists():
        for file in MODELS_DIR.glob("*.pkl"):
            try:
                models[file.name] = joblib.load(file)
            except:
                pass
    return models

models = load_models()

def is_classifier(name):
    name = name.lower()
    return any(m in name for m in ["rf", "svm", "knn", "gb", "lr"])

def is_scaler(name):
    return "scaler" in name.lower()

# Read CSV or Excel File

def read_df(upload):
    try:
        if upload.name.endswith(".csv"):
            return pd.read_csv(upload)
        else:
            return pd.read_excel(upload)
    except:
        st.error("❌ Unable to read file")
        return None

st.title("🎙 Human Voice Classification & Clustering")
st.write("Upload dataset with **numerical voice features**.")

file = st.file_uploader("Upload CSV or Excel file", type=["csv","xlsx","xls"])

if file:
    df = read_df(file)
    if df is None:
        st.stop()

    st.success("File loaded successfully!")
    st.dataframe(df.head())

    # Remove unwanted columns automatically
    drop_cols = ['label', 'kmeans_cluster', 'dbscan_cluster']
    df = df.drop(columns=drop_cols, errors='ignore')

    # Detect numeric feature columns
    feature_cols = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
    st.header("📌 Step 2: Preprocessing")
    st.write("Detected numeric features:", feature_cols)

    # Scaling
    scaler = None
    for k, v in models.items():
        if is_scaler(k) and "clust" not in k:
            scaler = v

    if scaler:
        X = df[feature_cols]
        df_scaled = scaler.transform(X)
        st.info("✔ Scaler applied")
    else:
        df_scaled = df[feature_cols].values
        st.warning("⚠ No scaler found. Using raw values.")

    # KMeans Clustering
    st.header("🔵 Step 3: Clustering (KMeans)")
    if "kmeans_model.pkl" in models:
        kmeans = models["kmeans_model.pkl"]
        df["kmeans_cluster"] = kmeans.predict(df_scaled)
        st.success("✔ KMeans clustering applied")
    else:
        st.warning("⚠ KMeans model not found")

    st.dataframe(df.head())

    # Classification (Gender)
    st.header("🟢 Step 4: Classification (Gender Prediction)")
    classifier = None
    for name, mdl in models.items():
        if is_classifier(name):
            classifier = mdl
            break

    if classifier:
        df["pred_label"] = classifier.predict(df_scaled)
        df["Gender"] = df["pred_label"].map({0: "Female", 1: "Male"})
        st.success("✔ Classification applied")
    else:
        st.warning("⚠ No classifier model found")

    st.dataframe(df.head())

    # PCA Visualization
    st.header("📊 Step 5: PCA Visualization")
    try:
        pca = PCA(n_components=2)
        pca_vals = pca.fit_transform(df_scaled)

        fig, ax = plt.subplots()
        scatter = ax.scatter(
            pca_vals[:, 0],
            pca_vals[:, 1],
            c=df["kmeans_cluster"],
            cmap="tab10",
            s=40
        )

        ax.set_xlabel("PCA 1")
        ax.set_ylabel("PCA 2")
        plt.colorbar(scatter, label="Cluster")
        st.pyplot(fig)
    except Exception as e:
        st.error(f"❌ PCA Error: {e}")

    # Download Final CSV
    st.header("⬇ Step 6: Download Final Output")
    st.download_button(
        "Download Results CSV",
        df.to_csv(index=False).encode("utf-8"),
        "voice_results.csv",
        "text/csv"
    )

# Sidebar — Model Metrics
st.sidebar.title("📈 Model Performance")
if METRICS_PATH.exists():
    metrics_df = pd.read_csv(METRICS_PATH)
    st.sidebar.dataframe(metrics_df)
else:
    st.sidebar.write("No metrics available.")

# Extras: Manual Entry / Audio Upload
st.header("🟡Manual Feature Entry / Audio Upload")

extra_option = st.radio(
    "Choose an extra option:",
    ["None", "Manual Feature Entry", "Upload Audio File"]
)

# 1️⃣ Manual Feature Entry
if extra_option == "Manual Feature Entry":
    st.subheader("Enter 44 Numeric Features Manually")
    manual_features = [st.number_input(f"Feature {i}", value=0.0, format="%.4f") for i in range(1, 45)]
    
    if st.button("Predict from Manual Entry"):
        if classifier and scaler:
            manual_array = np.array(manual_features).reshape(1, -1)
            manual_scaled = scaler.transform(manual_array)
            
            pred_label = classifier.predict(manual_scaled)[0]
            gender = "Male" if pred_label == 1 else "Female"
            st.success(f"✅ Predicted Gender: {gender}")
            
            if "kmeans_model.pkl" in models:
                cluster = models["kmeans_model.pkl"].predict(manual_scaled)[0]
                st.info(f"🔵 Predicted Cluster: {cluster}")
            else:
                st.warning("⚠ KMeans model not found")
        else:
            st.error("⚠ Classifier or scaler not found!")

# 2️⃣ Audio File Upload
if extra_option == "Upload Audio File":
    st.subheader("Upload Audio File (WAV/MP3)")
    audio_file = st.file_uploader("Choose audio file", type=["wav","mp3"])
    
    if audio_file:
        import librosa
        try:
            y, sr = librosa.load(audio_file, sr=None)
            mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
            mfcc_mean = mfccs.mean(axis=1)
            feature_array = np.hstack([mfcc_mean]).reshape(1, -1)  # Add more features if needed
            
            feature_scaled = scaler.transform(feature_array) if scaler else feature_array
            
            if classifier:
                pred_label = classifier.predict(feature_scaled)[0]
                gender = "Male" if pred_label == 1 else "Female"
                st.success(f"✅ Predicted Gender: {gender}")
            else:
                st.warning("⚠ Classifier model not found")
            
            if "kmeans_model.pkl" in models:
                cluster = models["kmeans_model.pkl"].predict(feature_scaled)[0]
                st.info(f"🔵 Predicted Cluster: {cluster}")
            else:
                st.warning("⚠ KMeans model not found")
        except Exception as e:
            st.error(f"❌ Audio processing error: {e}")
