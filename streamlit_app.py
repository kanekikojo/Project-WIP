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
    """Loads data and trains the advanced model"""
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
        
        # Filter for reliable data (Increased to 50 votes as requested)
        df = df[df['Rating Count'] >= 50].copy()
        
        # ACCORD STRENGTH: Weight accords by position (5x for 1st, 1x for 5th)
        accord_cols = [f'mainaccord{i}' for i in range(1, 6)]
        weights = [5, 4, 3, 2, 1]
        
        weighted_accords = []
        for idx, row in df.iterrows():
            accord_parts = []
            for i, col in enumerate(accord_cols):
                if pd.notna(row.get(col)):
                    accord = str(row[col]).replace(' ', '_')
                    accord_parts.extend([accord] * weights[i])
            weighted_accords.append(' '.join(accord_parts))
        
        df['Accords_Weighted'] = weighted_accords
        
        # SEPARATE NOTE LAYERS
        def clean_notes(note_str):
            if pd.isna(note_str): return ''
            return ' '.join([n.strip().replace(' ', '_') for n in str(note_str).split(',')])
        
        df['Top_Clean'] = df['Top'].apply(clean_notes)
        df['Middle_Clean'] = df['Middle'].apply(clean_notes)
        df['Base_Clean'] = df['Base'].apply(clean_notes)
        
        # Prepare data
        X = df[['Gender', 'Accords_Weighted', 'Top_Clean', 'Middle_Clean', 'Base_Clean']]
        y = df['Rating Value']
        
        # Sample weights (trust popular perfumes more)
        sample_weights = np.sqrt(df['Rating Count'])
        
        # Build pipeline with advanced features
        preprocessor = ColumnTransformer(
            transformers=[
                ('gender', OneHotEncoder(handle_unknown='ignore', sparse_output=False), ['Gender']),
                ('accords', Pipeline([
                    ('tfidf', TfidfVectorizer(min_df=5, max_features=3000)),
                    ('svd', TruncatedSVD(n_components=50, random_state=42))
                ]), 'Accords_Weighted'),
                ('top', Pipeline([
                    ('tfidf', TfidfVectorizer(min_df=5, max_features=2000)),
                    ('svd', TruncatedSVD(n_components=40, random_state=42))
                ]), 'Top_Clean'),
                ('middle', Pipeline([
                    ('tfidf', TfidfVectorizer(min_df=5, max_features=2000)),
                    ('svd', TruncatedSVD(n_components=40, random_state=42))
                ]), 'Middle_Clean'),
                ('base', Pipeline([
                    ('tfidf', TfidfVectorizer(min_df=5, max_features=2000)),
                    ('svd', TruncatedSVD(n_components=40, random_state=42))
                ]), 'Base_Clean'),
            ]
        )
        
        # Use HistGradientBoosting (Reliable & Fast)
        pipeline = Pipeline([
            ('preprocessor', preprocessor),
            ('regressor', HistGradientBoostingRegressor(max_iter=500, random_state=42, learning_rate=0.05, max_depth=12))
        ])
        
        # Train
        pipeline.fit(X, y, regressor__sample_weight=sample_weights)
        return pipeline
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

# Load dropdown data
@st.cache_data(ttl=3600)  # Cache for 1 hour, then reload
def get_dropdown_data():
    """Loads dropdown options from JSON file"""
    with open('web_data.json', 'r', encoding='utf-8') as f:
        return json.load(f)

# Header
st.markdown("<h1>Fragrance Alchemy</h1>", unsafe_allow_html=True)
st.markdown("<p class='subtitle'>AI-Powered Perfume Rating Predictor</p>", unsafe_allow_html=True)

# Load resources
try:
    model = get_model()
    data = get_dropdown_data()
    
    # Premium Mode Toggle
    st.markdown("---")
    premium_mode = st.toggle("🌟 Premium Mode", value=False, 
                             help="Auto-fill with ML-validated combinations for high ratings (4.2+)")
    
    if premium_mode:
        st.info("**Premium Mode Active**: Using top-rated accords and notes identified by AI analysis of 18,000+ perfumes.")
    
    st.markdown("---")
    
    # Form
    col1, col2 = st.columns([1, 1])
    
    with col1:
        gender = st.selectbox(
            "Target Gender",
            options=[''] + data['genders'],
            format_func=lambda x: "Select Gender..." if x == '' else x.capitalize()
        )
        
        # Set default premium accords if premium mode
        default_accords = data.get('premium_accords', [])[:5] if premium_mode else []
        
        accords = st.multiselect(
            "Main Accords" + (" (Premium Suggestions Pre-filled)" if premium_mode else ""),
            options=data['accords'],
            default=default_accords,
            format_func=lambda x: x.capitalize(),
            help="Search or scroll through all 84 available accords"
        )
        
        # Set default premium top notes if premium mode
        default_top = data.get('premium_notes', [])[:3] if premium_mode else []
        
        top_notes = st.multiselect(
            "Top Notes" + (" (Premium Suggestions Pre-filled)" if premium_mode else ""),
            options=data['notes'],
            default=default_top,
            format_func=lambda x: x.capitalize(),
            help="Search through all 1,671 available notes. Type to filter."
        )
    
    with col2:
        # Set default premium middle notes if premium mode
        default_middle = data.get('premium_notes', [])[3:6] if premium_mode else []
        
        middle_notes = st.multiselect(
            "Middle Notes" + (" (Premium Suggestions Pre-filled)" if premium_mode else ""),
            options=data['notes'],
            default=default_middle,
            format_func=lambda x: x.capitalize(),
            help="Search through all 1,671 available notes. Type to filter."
        )
        
        # Set default premium base notes if premium mode
        default_base = data.get('premium_notes', [])[6:9] if premium_mode else []
        
        base_notes = st.multiselect(
            "Base Notes" + (" (Premium Suggestions Pre-filled)" if premium_mode else ""),
            options=data['notes'],
            default=default_base,
            format_func=lambda x: x.capitalize(),
            help="Search through all 1,671 available notes. Type to filter."
        )
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    if st.button("✨ Predict Rating"):
        if not gender:
            st.error("Please select a target gender")
        else:
            with st.spinner("Analyzing composition..."):
                # Prepare input with Advanced Features
                
                # 1. Weighted Accords
                weighted_accords_list = []
                # We don't know the exact order user selected, so we treat all selected as equal high importance
                # or just join them. For the UI, simple joining is best approximation.
                # To mimic training, we could repeat them, but simple presence is key.
                # Let's repeat each selected accord 3 times to give them "average high" weight
                for a in accords:
                    accord = a.replace(' ', '_')
                    weighted_accords_list.extend([accord] * 3)
                accords_str = ' '.join(weighted_accords_list)
                
                # 2. Clean Notes
                top_str = ' '.join([n.replace(' ', '_') for n in top_notes])
                middle_str = ' '.join([n.replace(' ', '_') for n in middle_notes])
                base_str = ' '.join([n.replace(' ', '_') for n in base_notes])
                
                input_data = pd.DataFrame({
                    'Gender': [gender],
                    'Accords_Weighted': [accords_str],
                    'Top_Clean': [top_str],
                    'Middle_Clean': [middle_str],
                    'Base_Clean': [base_str]
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
                
                # Feature Insights
                st.markdown("---")
                st.markdown("### 📊 AI Insights")
                
                col_a, col_b = st.columns(2)
                
                with col_a:
                    st.markdown("#### 📈 Rating Boosters")
                    st.info("**Oud & Resins**: Indian Oud, Agarwood, Natural Musk\n\n**Natural Florals**: Taif Rose, Rose de Mai")
                    
                with col_b:
                    st.markdown("#### 📉 Rating Killers")
                    st.warning("**Polarizing Textures**: Soapy, Earthy\n\n**Sharp Notes**: Birch Leaf, Blood Orange, Yuzu")
                    
                st.caption("*Based on analysis of 18,000+ perfumes. 'Main Accords' are 2x more important than specific notes.*")

except FileNotFoundError as e:
    st.error(f"Required file not found: {e}")
    st.info("Please ensure `fra_cleaned.csv` and `web_data.json` are in the correct location.")
