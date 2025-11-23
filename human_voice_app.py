import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
import io
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA

# Configuration
MODELS_DIR = Path("models")
METRICS_PATH = Path("model_metrics.csv")
CLUST_CSV = Path("voice_features_with_clusters.csv")
FEATURES_META = MODELS_DIR / "features.pkl"

# Helpers
@st.cache_resource
def load_models(model_dir=MODELS_DIR):
    models = {}
    if not model_dir.exists():
        return models
    for fp in sorted(model_dir.glob("*.pkl")):
        try:
            models[fp.name] = joblib.load(fp)
        except Exception:
            models[fp.name] = None
    return models

models = load_models()

def read_df_from_upload(u):
    data = u.read()
    u.seek(0)
    return pd.read_csv(io.BytesIO(data)) if u.name.endswith(".csv") else pd.read_excel(io.BytesIO(data))

def detect_feature_cols(df):
    if FEATURES_META.exists():
        try:
            meta = joblib.load(FEATURES_META)
            saved = meta.get("feature_columns", [])
            used = [c for c in saved if c in df.columns]
            if used: 
                return used
        except Exception:
            pass
    drop = {'label','cluster','filename','id','index'}
    return [c for c in df.select_dtypes(include=[np.number]).columns if c.lower() not in drop]

def save_metrics(model_name, acc, prec, rec, f1, n_samples):
    row = {'model_name': model_name, 'accuracy': float(acc), 'precision': float(prec),
           'recall': float(rec), 'f1': float(f1), 'n_samples': int(n_samples)}
    if METRICS_PATH.exists():
        dfm = pd.read_csv(METRICS_PATH)
        if 'model_name' not in dfm.columns:
            dfm = dfm.reset_index().rename(columns={'index':'model_name'})
        dfm = dfm[dfm['model_name'] != model_name]
        dfm = pd.concat([dfm, pd.DataFrame([row])], ignore_index=True)
    else:
        dfm = pd.DataFrame([row])
    dfm.to_csv(METRICS_PATH, index=False)

# UI 
st.set_page_config(page_title="Human Voice Classification", layout="wide")
st.title("📊 Human Voice Dataset Analysis & Model Performance Dashboard")

page = st.sidebar.radio("Menu", ["Home","Upload & Predict","Clustering Visualization","Model Performance","About"])

# Home 
if page == "Home":
    st.header("Welcome")
    st.write("This application provides an end-to-end platform to analyze voice datasets and evaluate Machine Learning model performance using pre-computed features stored in CSV/Excel files. Users can upload any dataset containing extracted voice features and instantly explore clustering, model performance, and dataset statistics.")

# Upload & Predict 
elif page == "Upload & Predict":
    st.header("Upload dataset and run batch prediction")
    uploaded = st.file_uploader("Upload CSV / Excel", type=['csv','xls','xlsx'])
    classifier_files = sorted([k for k in models.keys() if k and any(s in k.lower() for s in ['rf','gb','svm','knn','lr'])])
    clf_choice = st.selectbox("Choose classifier (built model file)", options=['--select--'] + classifier_files)
    assign_clusters = st.checkbox("Also assign clusters (kmeans/dbscan if available)", value=True)

    if uploaded:
        df = read_df_from_upload(uploaded)
        st.write(f"Loaded {uploaded.name} — {df.shape[0]} rows")
        st.dataframe(df.head(5))
        features = detect_feature_cols(df)
        st.write("Using feature columns:", features[:20])

        if len(features) == 0:
            st.error("No numeric feature columns found.")
        else:
            if st.button("Run predictions"):
                X = df[features].fillna(0).values

                # scaler for classification
                scaler = models.get('scaler.pkl', None)
                if scaler is None:
                    for name,obj in models.items():
                        if name and 'scaler' in name.lower() and 'clust' not in name.lower():
                            scaler = obj
                            break
                if scaler is not None:
                    try:
                        Xs = scaler.transform(X)
                    except Exception:
                        Xs = X
                else:
                    Xs = X

                # Classification prediction
                if clf_choice != '--select--' and clf_choice in models and models[clf_choice] is not None:
                    clf = models[clf_choice]
                    preds = clf.predict(Xs)
                    df['pred_label'] = preds
                    try:
                        df['pred_gender'] = df['pred_label'].apply(lambda x: 'Male' if int(x)==1 else 'Female')
                    except Exception:
                        df['pred_gender'] = df['pred_label']
                    if hasattr(clf, 'predict_proba'):
                        try:
                            df['pred_prob_max'] = clf.predict_proba(Xs).max(axis=1)
                        except Exception:
                            pass
                    st.success(f"Predictions done with {clf_choice}")
                else:
                    st.warning("No classifier selected or model missing. Prediction skipped.")

                # Cluster assignment using KMeans/DBSCAN
                if assign_clusters:
                    # scaler for clustering
                    scaler_clust = models.get('scaler_clust.pkl', None)
                    Xc_base = df[features].fillna(0).values
                    try:
                        Xc = scaler_clust.transform(Xc_base) if scaler_clust is not None else Xc_base
                    except Exception:
                        Xc = Xc_base

                    # KMeans
                    kmeans = models.get('kmeans_model.pkl', None)
                    if kmeans is not None:
                        try:
                            df['kmeans_cluster'] = kmeans.predict(Xc)
                            st.info("KMeans clustering applied.")
                        except Exception:
                            st.warning("KMeans predict failed for uploaded data.")

                    # DBSCAN
                    db = models.get('dbscan_model.pkl', None)
                    if db is not None and hasattr(db, 'labels_') and db.labels_ is not None and len(db.labels_) == len(df):
                        df['dbscan_cluster'] = db.labels_
                        st.info("DBSCAN labels_ used from saved object.")
                    else:
                        st.info("Saved DBSCAN labels_ not usable for this data (skipped).")

                # Compute metrics if label present
                if 'label' in df.columns and 'pred_label' in df.columns:
                    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
                    y_true = df['label'].astype(int)
                    y_pred = df['pred_label'].astype(int)
                    acc = accuracy_score(y_true, y_pred)
                    prec = precision_score(y_true, y_pred, zero_division=0)
                    rec = recall_score(y_true, y_pred, zero_division=0)
                    f1 = f1_score(y_true, y_pred, zero_division=0)
                    st.write({'accuracy':acc,'precision':prec,'recall':rec,'f1':f1})
                    if clf_choice != '--select--':
                        save_metrics(clf_choice, acc, prec, rec, f1, len(df))
                        st.success("Metrics saved to model_metrics.csv")

                # Show predictions
                st.subheader("Sample predictions")
                st.dataframe(df.head(20))
                buf = io.BytesIO()
                df.to_csv(buf, index=False)
                buf.seek(0)
                st.download_button("Download predictions CSV", data=buf, file_name="predictions.csv", mime="text/csv")

# Clustering Visualization 
elif page == "Clustering Visualization":
    st.header("Clustering Visualization (built-model prediction option)")
    if CLUST_CSV.exists():
        df = pd.read_csv(CLUST_CSV)
        st.info(f"Loaded {CLUST_CSV} — {df.shape}")
    else:
        uploaded = st.file_uploader("Upload CSV (features or features+clusters)", type=['csv','xls','xlsx'])
        if not uploaded: st.stop()
        df = read_df_from_upload(uploaded)

    st.dataframe(df.sample(min(200, len(df))).reset_index(drop=True))
    existing = [c for c in df.columns if any(s in c.lower() for s in ('kmeans','dbscan','cluster'))]
    if not existing:
        st.info("No cluster columns and no kmeans/dbscan models found.")
        st.stop()

    choice = st.selectbox("Choose cluster source:", existing)
    features = detect_feature_cols(df)
    if len(features) == 0: st.error("No numeric features found for PCA/clustering."); st.stop()

    cluster_col = choice if choice in df.columns else None
    Xnum = df[features].fillna(0).values
    X2 = None
    if 'pca_2d.pkl' in models and models['pca_2d.pkl'] is not None:
        try: X2 = models['pca_2d.pkl'].transform(Xnum)
        except Exception: X2 = None
    if X2 is None:
        X2 = PCA(n_components=2, random_state=42).fit_transform(Xnum)

    fig, ax = plt.subplots(figsize=(8,5))
    if cluster_col and cluster_col in df.columns:
        vals = df[cluster_col].astype(int).values
        unique = np.unique(vals)
        palette = sns.color_palette(n_colors=len(unique))
        ax.scatter(X2[:,0], X2[:,1], c=vals, cmap='tab10', s=12)
        handles = [plt.Line2D([0],[0], marker='o', color='w', markerfacecolor=palette[i], markersize=8, label=f'cluster {c}') for i,c in enumerate(unique)]
        ax.legend(handles=handles, bbox_to_anchor=(1.02,1), loc='upper left', title='Clusters')
    else:
        ax.scatter(X2[:,0], X2[:,1], s=12)
    ax.set_xlabel("PC1"); ax.set_ylabel("PC2"); ax.set_title("PCA 2D clustering view")
    st.pyplot(fig)

    if cluster_col in df.columns:
        st.write("Cluster counts:", df[cluster_col].value_counts().sort_index())
        if 'label' in df.columns:
            ct = pd.crosstab(df[cluster_col], df['label'])
            st.dataframe(ct)
            st.write("Per-cluster purity:", (ct.max(axis=1)/ct.sum(axis=1)).round(3))

# Model Performance 
elif page == "Model Performance":
    st.header("Model Performance")
    if METRICS_PATH.exists():
        mdf = pd.read_csv(METRICS_PATH)
        st.subheader("Metrics Table")
        st.dataframe(mdf)

        metric_cols = [c for c in ['accuracy','precision','recall','f1','roc_auc'] if c in [col.lower() for col in mdf.columns]]
        col_map = {c.lower():c for c in mdf.columns}
        best_col = 'accuracy' if 'accuracy' in metric_cols else (metric_cols[0] if metric_cols else None)

        if best_col:
            actual_name = col_map[best_col]
            best_idx = mdf[actual_name].idxmax()
            best_row = mdf.loc[best_idx]
            st.markdown(f"**Best model (by {best_col}):** `{best_row.get('model_name', best_row.get('model', 'unknown'))}` — {best_col} = {best_row[actual_name]:.4f}")

            top3 = mdf.sort_values(by=actual_name, ascending=False).head(3)[[c for c in ['model_name','model'] if c in mdf.columns] + [actual_name]]
            st.subheader(f"Top 3 models by {best_col}")
            st.dataframe(top3)

            st.subheader("Metric summary (mean / median)")
            stats = {}
            for mc in metric_cols:
                an = col_map[mc]
                vals = pd.to_numeric(mdf[an], errors='coerce').dropna()
                stats[mc] = {'mean': float(vals.mean()) if not vals.empty else None,
                             'median': float(vals.median()) if not vals.empty else None}
            st.json(stats)

            if 'accuracy' in metric_cols:
                fig, ax = plt.subplots(figsize=(6,3))
                names = mdf['model_name'] if 'model_name' in mdf.columns else mdf.index.astype(str)
                ax.bar(names, pd.to_numeric(mdf[col_map['accuracy']], errors='coerce').fillna(0))
                ax.set_ylabel("Accuracy")
                ax.set_xticklabels(names, rotation=45, ha='right')
                st.pyplot(fig)
        else:
            st.info("No standard metric columns (accuracy/precision/recall/f1/roc_auc) found in model_metrics.csv.")

        buf = io.BytesIO()
        mdf.to_csv(buf, index=False)
        buf.seek(0)
        st.download_button("Download metrics CSV", data=buf, file_name="model_metrics.csv", mime="text/csv")
    else:
        st.info("No model_metrics.csv found. Run Upload & Predict with a labeled CSV to auto-save metrics, or upload one here.")
        up = st.file_uploader("Upload metrics CSV", type=['csv'])
        if up:
            st.dataframe(pd.read_csv(up))

# About
else:
    st.header("About")
    st.write("""
    ## 📘 About This Application

    This application is designed to analyze **human voice feature datasets** using pre-trained 
    Machine Learning and Clustering models.

    Instead of processing raw audio files, the app focuses on **uploaded CSV/Excel datasets** 
    that already contain extracted voice features such as MFCC, pitch, energy, 
    frequency-based measurements, and other audio statistics.

    ### 🎯 The goal of this application is to provide:

    - Exploring voice datasets  
    - Understanding feature patterns  
    - Visualizing clustering results  
    - Comparing multiple ML model performances  
    - Supporting research, learning, and project demonstrations  

    With built-in support for **KMeans** and **DBSCAN** clustering, along with classification models 
    such as **Random Forest, SVM, Logistic Regression, Gradient Boosting, and KNN**, this tool 
    acts as a complete dashboard for analyzing voice-based datasets.
    """)
