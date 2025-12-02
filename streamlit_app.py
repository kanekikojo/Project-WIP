import streamlit as st
import pandas as pd
import json
import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

# Page config
st.set_page_config(
    page_title="Fragrance Alchemy - AI Perfume Predictor",
    page_icon="🌸",
    layout="wide"
)

# Custom CSS for luxury theme
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@400;700&family=Montserrat:wght@300;400;600&display=swap');
    
    .stApp {
        background: linear-gradient(135deg, #0a0a0a 0%, #1a1520 100%);
    }
    
    h1 {
        font-family: 'Playfair Display', serif;
        background: linear-gradient(135deg, #d4af37 0%, #f9e892 50%, #d4af37 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        font-size: 3.5rem;
        letter-spacing: 2px;
        margin-bottom: 0;
    }
    
    .subtitle {
        text-align: center;
        color: #a8a8a8;
        font-family: 'Montserrat', sans-serif;
        letter-spacing: 4px;
        font-size: 0.9rem;
        text-transform: uppercase;
        margin-bottom: 3rem;
    }
    
    .stSelectbox label, .stMultiSelect label {
        color: #d4af37 !important;
        font-weight: 600 !important;
        text-transform: uppercase;
        letter-spacing: 1px;
        font-size: 0.9rem;
    }
    
    .stButton>button {
        width: 100%;
        background: linear-gradient(135deg, #d4af37 0%, #f9e892 100%);
        color: #0a0a0a;
        font-weight: 700;
        font-size: 1.1rem;
        text-transform: uppercase;
        letter-spacing: 2px;
        border: none;
        padding: 0.75rem 2rem;
        border-radius: 10px;
        transition: all 0.3s ease;
    }
    
    .stButton>button:hover {
        transform: translateY(-3px);
        box-shadow: 0 10px 30px rgba(212, 175, 55, 0.5);
    }
    
    .result-card {
        background: rgba(212, 175, 55, 0.1);
        border: 2px solid #d4af37;
        border-radius: 15px;
        padding: 2rem;
        text-align: center;
        margin-top: 2rem;
    }

    
    .rating-value {
        font-size: 4rem;
        font-family: 'Playfair Display', serif;
        color: #d4af37;
        font-weight: 700;
    }
    
    .verdict {
        font-size: 1.5rem;
        color: #f0f0f0;
        margin-top: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# Load and train model
@st.cache_resource
def get_model():
    """Loads data and trains the model"""
    try:
        # Load data with relative path
        df = pd.read_csv('fra_cleaned.csv', sep=';', encoding='latin-1')
        
        # Clean Rating Value
        if df['Rating Value'].dtype == object:
            df['Rating Value'] = df['Rating Value'].str.replace(',', '.').astype(float)
        
        # Clean Rating Count
        df['Rating Count'] = pd.to_numeric(df['Rating Count'], errors='coerce').fillna(0)
        
        # Drop rows with missing data
        df = df.dropna(subset=['Rating Value', 'Rating Count', 'Gender'])
        
        # Filter for reliable data
        df = df[df['Rating Count'] >= 10].copy()
        
        # Combine features
        accord_cols = [f'mainaccord{i}' for i in range(1, 6)]
        
        def combine_features(row):
            parts = []
            for col in accord_cols:
                if pd.notna(row.get(col)):
                    parts.append(str(row[col]).replace(' ', '_'))
            for col in ['Top', 'Middle', 'Base']:
                if pd.notna(row.get(col)):
                    notes = [n.strip().replace(' ', '_') for n in str(row[col]).split(',')]
                    parts.extend(notes)
            return ' '.join(parts)

        df['Composition'] = df.apply(combine_features, axis=1)
        
        # Prepare data
        X = df[['Gender', 'Composition']]
        y = df['Rating Value']
        
        # Build pipeline
        preprocessor = ColumnTransformer(
            transformers=[
                ('gender', OneHotEncoder(handle_unknown='ignore', sparse_output=False), ['Gender']),
                ('composition', Pipeline([
                    ('tfidf', TfidfVectorizer(min_df=5, max_features=5000)),
                    ('svd', TruncatedSVD(n_components=100, random_state=42))
                ]), 'Composition'),
            ]
        )
        
        pipeline = Pipeline([
            ('preprocessor', preprocessor),
            ('regressor', HistGradientBoostingRegressor(max_iter=500, random_state=42, learning_rate=0.1, max_depth=10))
        ])
        
        # Train
        pipeline.fit(X, y)
        return pipeline
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

# Load dropdown data
@st.cache_data
def get_dropdown_data():
    with open('web_data.json', 'r', encoding='utf-8') as f:
        return json.load(f)

# Header
st.markdown("<h1>Fragrance Alchemy</h1>", unsafe_allow_html=True)
st.markdown("<p class='subtitle'>AI-Powered Perfume Rating Predictor</p>", unsafe_allow_html=True)

# Load resources
try:
    model = get_model()
    data = get_dropdown_data()
    
    # Form
    col1, col2 = st.columns([1, 1])
    
    with col1:
        gender = st.selectbox(
            "Target Gender",
            options=[''] + data['genders'],
            format_func=lambda x: "Select Gender..." if x == '' else x.capitalize()
        )
        
        accords = st.multiselect(
            "Main Accords",
            options=data['accords'],
            format_func=lambda x: x.capitalize()
        )
        
        top_notes = st.multiselect(
            "Top Notes",
            options=data['notes'][:100],
            format_func=lambda x: x.capitalize()
        )
    
    with col2:
        middle_notes = st.multiselect(
            "Middle Notes",
            options=data['notes'][:100],
            format_func=lambda x: x.capitalize()
        )
        
        base_notes = st.multiselect(
            "Base Notes",
            options=data['notes'][:100],
            format_func=lambda x: x.capitalize()
        )
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    if st.button("✨ Predict Rating"):
        if not gender:
            st.error("Please select a target gender")
        else:
            with st.spinner("Analyzing composition..."):
                # Prepare input
                parts = []
                parts.extend([a.replace(' ', '_') for a in accords])
                parts.extend([n.replace(' ', '_') for n in top_notes])
                parts.extend([n.replace(' ', '_') for n in middle_notes])
                parts.extend([n.replace(' ', '_') for n in base_notes])
                
                composition = ' '.join(parts) if parts else 'unknown'
                
                input_data = pd.DataFrame({
                    'Gender': [gender],
                    'Composition': [composition]
                })
                
                # Predict
                prediction = model.predict(input_data)[0]
                
                # Determine verdict
                if prediction >= 4.2:
                    verdict = "🌟 Signature Scent - Likely a HIT!"
                elif prediction >= 3.8:
                    verdict = "✨ Premium Fragrance"
                elif prediction >= 3.5:
                    verdict = "👍 Solid Composition"
                else:
                    verdict = "📉 Needs Refinement"
                
                # Display result
                st.markdown(f"""
                <div class='result-card'>
                    <div class='rating-value'>{prediction:.2f} / 5.0</div>
                    <div class='verdict'>{verdict}</div>
                </div>
                """, unsafe_allow_html=True)

except FileNotFoundError as e:
    st.error(f"Required file not found: {e}")
    st.info("Please ensure `fra_cleaned.csv` and `web_data.json` are in the correct location.")
