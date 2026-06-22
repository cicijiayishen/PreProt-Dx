import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# ============================================
# 1. Configuration
# ============================================

# 16-protein panel (must match training)
FEATURE_COLS = [
    "P43652", "P41222", "P00450", "P02787", "O75594",
    "Q92692", "P00441", "P19823", "P98172",
    "P01009", "P02768", "P98160", "P02760", "P02748",
    "P01023", "P45974"
]

# Gene name mapping for display
GENE_NAMES = {
    "P43652": "AFM", "P41222": "PTGDS", "P00450": "CP",
    "P02787": "TF", "O75594": "PGLYRP1", "Q92692": "NECTIN2",
    "P00441": "SOD1", "P19823": "ITIH2", "P98172": "EFNB1",
    "P01009": "SERPINA1", "P02768": "ALB", "P98160": "HSPG2",
    "P02760": "AMBP", "P02748": "C9", "P01023": "A2M",
    "P45974": "USP5"
}

CLASS_MAP = {
    0: 'HC/HT (Healthy Control & Hypertension)',
    1: 'PE (Preeclampsia)'
}

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, 'final_model.joblib')
SCALER_PATH = os.path.join(BASE_DIR, 'final_scaler.joblib')
PROTEIN_PEPTIDE_MAP_PATH = os.path.join(BASE_DIR, 'Protein-Peptide.csv')

# Skyline CSV column mapping
RAW_HEADER_MAP = {
    'Peptide Sequence': 'Peptide.Sequence',
    'Median Peak Area': 'Median.Peak.Area',
    'Total Area MS1': 'Total.Area.MS1',
    'Replicate': 'Replicate'
}
PP_MAP_HEADER_MAP = {
    'Peptide Sequence': 'Peptide.Sequence',
    'Protein Accession': 'Protein_Group'
}
RAW_NUMERIC_COLS = ['Median.Peak.Area', 'Total.Area.MS1']


# ============================================
# 2. Load artifacts (cached)
# ============================================
@st.cache_resource
def load_all_artifacts():
    """Load model, scaler, and protein-peptide map."""
    artifacts = {}

    # Check required files
    missing_files = []
    for f, label in [(MODEL_PATH, 'Model'), (SCALER_PATH, 'Scaler'),
                     (PROTEIN_PEPTIDE_MAP_PATH, 'Protein-Peptide map')]:
        if not os.path.exists(f):
            missing_files.append(label)

    if missing_files:
        st.error(f"Missing files: {', '.join(missing_files)}")
        return None

    try:
        # Load model and scaler
        artifacts['model'] = joblib.load(MODEL_PATH)
        artifacts['scaler'] = joblib.load(SCALER_PATH)

        # Load Protein-Peptide map
        pp_map = pd.read_csv(PROTEIN_PEPTIDE_MAP_PATH)
        pp_map = pp_map.rename(columns=PP_MAP_HEADER_MAP)
        artifacts['protein_peptide_map'] = pp_map[['Peptide.Sequence', 'Protein_Group']].drop_duplicates()

        st.success("All artifacts loaded successfully.")
        return artifacts

    except Exception as e:
        st.error(f"Error loading artifacts: {e}")
        return None


# ============================================
# 3. Data preprocessing
# ============================================
def preprocess_skyline_data(df_raw, protein_peptide_map):
    """Preprocess Skyline PRM report to sample × protein matrix."""
    try:
        df = df_raw.rename(columns=RAW_HEADER_MAP)
        df = df[~df['Replicate'].str.contains("BATCH_MIX", na=False)]

        for col in RAW_NUMERIC_COLS:
            df[col] = pd.to_numeric(df[col], errors='coerce')

        wide_tic = df.pivot_table(index='Peptide.Sequence', columns='Replicate', values='Median.Peak.Area')
        wide_ms1 = df.pivot_table(index='Peptide.Sequence', columns='Replicate', values='Total.Area.MS1')

        # Replace NA with row minimum
        abundance = wide_ms1.apply(lambda x: x.fillna(x.min()), axis=0)

        # Normalize: each sample divided by its own TIC median
        sample_medians = wide_tic.median(axis=0, skipna=True)
        sample_medians = sample_medians.replace(0, 1e-10)
        overall_median = sample_medians.median()
        normalized = (abundance.div(sample_medians, axis=1)*overall_median)

        # Log2 transform
        normalized_log2 = np.log2(normalized + 1e-10)

        # Peptide-to-protein mapping
        df_melted = normalized_log2.reset_index().melt(
            id_vars='Peptide.Sequence', var_name='Replicate', value_name='Abundance'
        )
        df_mapped = pd.merge(df_melted, protein_peptide_map, on='Peptide.Sequence', how='left')

        # Warn about unmapped peptides
        if df_mapped['Protein_Group'].isnull().any():
            missing = df_mapped[df_mapped['Protein_Group'].isnull()]['Peptide.Sequence'].unique()
            st.warning(f"{len(missing)} peptides not mapped to proteins. Showing first 5: {missing[:5]}")
            df_mapped = df_mapped.dropna(subset=['Protein_Group'])

        if df_mapped.empty:
            st.error("No peptides mapped to proteins.")
            return None

        # Aggregate to protein level
        df_mapped['Linear'] = 2 ** df_mapped['Abundance']
        protein_means = df_mapped.groupby(['Protein_Group', 'Replicate'])['Linear'].mean()
        protein_log2 = np.log2(protein_means + 1e-10)

        # Pivot to sample × protein matrix
        df_final = protein_log2.unstack(level='Protein_Group')
        df_final.index.name = "Sample"

        # Replace negative values with minimum positive
        def replace_neg(col):
            pos = col[col > 0]
            if len(pos) > 0:
                col[col < 0] = pos.min()
            else:
                col[col < 0] = 0
            return col

        df_final = df_final.apply(replace_neg, axis=0)
        return df_final

    except Exception as e:
        st.error(f"Preprocessing failed: {e}")
        st.exception(e)
        return None


# ============================================
# 4. Streamlit UI
# ============================================
st.set_page_config(page_title="PreProt-Dx: Preeclampsia Diagnosis", layout="wide")

# Custom CSS
st.markdown("""
<style>
    .stApp { background-color: #f0f2f6; }
    h1 { color: #004d40; font-size: 2.5em; }
    h2 { color: #00695c; font-size: 2em; border-bottom: 2px solid #e0e0e0; padding-bottom: 0.5rem; }
    h3 { color: #00796b; font-size: 1.5em; }
</style>
""", unsafe_allow_html=True)

st.image("Logo.jpg", width=500)
st.markdown("**A machine learning tool for early detection and diagnosis of Preeclampsia (PE) using targeted urinary proteomics data.**")

# ============================================
# 5. Load artifacts
# ============================================
artifacts = load_all_artifacts()
if artifacts is None:
    st.error("Application initialization failed.")
    st.stop()

model = artifacts['model']
scaler = artifacts['scaler']
protein_peptide_map = artifacts['protein_peptide_map']

# ============================================
# 6. Model information
# ============================================
st.markdown("---")
st.subheader("Model Information")
st.write(f"**{type(model).__name__}** classifier trained on **{len(FEATURE_COLS)} protein markers**.")

panel_genes = [f"{GENE_NAMES.get(p, p)}" for p in FEATURE_COLS]
st.write(f"**Protein panel:** {', '.join(panel_genes)}")

st.info(
    f"**Normalization Method:** Self-normalization (each sample divided by its own median TIC).\n\n"
    f"**Input:** Skyline PRM report (.csv) with columns: "
    f"`{', '.join(RAW_HEADER_MAP.keys())}`.\n\n"
    f"**Important:** Samples should be processed under the same conditions as the training cohort "
    f"for optimal performance."
)

# ============================================
# 7. File upload
# ============================================
st.markdown("---")
st.subheader("Upload Skyline PRM Report")
uploaded_file = st.file_uploader(
    f"CSV file with columns: {', '.join(RAW_HEADER_MAP.keys())}",
    type=["csv"]
)

if uploaded_file is not None:
    try:
        df_raw = pd.read_csv(uploaded_file)

        missing_cols = set(RAW_HEADER_MAP.keys()) - set(df_raw.columns)
        if missing_cols:
            st.error(f"Missing columns: {', '.join(missing_cols)}")
        else:
            st.success("File uploaded. Running preprocessing...")

            df_processed = preprocess_skyline_data(df_raw, protein_peptide_map)

            if df_processed is not None:
                st.write("Preprocessing complete.")

                # Check for required features
                missing_features = set(FEATURE_COLS) - set(df_processed.columns)
                if missing_features:
                    st.error(f"Missing protein markers after preprocessing: {', '.join(missing_features)}")
                else:
                    # Scale and predict
                    X_new = df_processed[FEATURE_COLS].astype(float)
                    X_scaled = scaler.transform(X_new)

                    predictions = model.predict(X_scaled)
                    probabilities = model.predict_proba(X_scaled)[:, 1]

                    # Results table
                    st.markdown("---")
                    st.header("Diagnostic Results")

                    df_results = pd.DataFrame({
                        'Sample': df_processed.index,
                        'Predicted Diagnosis': [CLASS_MAP[p] for p in predictions],
                        'PE Probability': probabilities
                    })

                    def color_prob(val):
                        return 'background-color: #ffcccc' if val > 0.5 else 'background-color: #ccffcc'

                    st.dataframe(
                        df_results.style.format({'PE Probability': '{:.2%}'}),
                        use_container_width=True
                    )

                    # Summary metrics
                    n_pe = (predictions == 1).sum()
                    n_nonpe = (predictions == 0).sum()

                    col1, col2, col3 = st.columns(3)
                    col1.metric("Total Samples", len(predictions))
                    col2.metric("Predicted PE", n_pe)
                    col3.metric("Predicted HC/HT", n_nonpe)

                    # Individual details (for ≤10 samples)
                    if len(df_results) <= 10:
                        st.subheader("Sample Details")
                        for i, (sid, prob) in enumerate(zip(df_results['Sample'], probabilities)):
                            with st.expander(f"{sid}: PE Probability = {prob:.1%}"):
                                st.progress(float(prob))
                                if prob > 0.5:
                                    st.warning(f"High risk of PE")
                                else:
                                    st.success(f"Low risk of PE")

    except pd.errors.EmptyDataError:
        st.error("Uploaded file is empty.")
    except Exception as e:
        st.error(f"Error: {e}")
        st.exception(e)

st.markdown("---")
st.caption("PreProt-Dx | 16-protein urinary panel for preeclampsia diagnosis | Research Use Only")

with open("test_data.csv", "rb") as f:
    st.download_button(
        "Download Example Input",
        f,
        file_name="test_data.csv"
    )