import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, classification_report
import joblib

# File path
DATA_PATH = r"c:\Users\pc\OneDrive\Documents\Fragrantica\fra_cleaned.csv"

def load_and_clean_data(filepath):
    """Loads and cleans the dataset."""
    print(f"Loading data from {filepath}...")
    try:
        df = pd.read_csv(filepath, sep=';', encoding='latin-1')
    except Exception as e:
        print(f"Error loading CSV: {e}")
        return None

    # Clean Rating Value
    if df['Rating Value'].dtype == object:
        df['Rating Value'] = df['Rating Value'].str.replace(',', '.').astype(float)
    
    # Clean Rating Count
    df['Rating Count'] = pd.to_numeric(df['Rating Count'], errors='coerce').fillna(0)
    
    # Drop rows with missing target or critical features
    df = df.dropna(subset=['Rating Value', 'Rating Count', 'Gender'])
    
    # Filter for perfumes with enough votes to be reliable
    df = df[df['Rating Count'] >= 10].copy()
    
    # Combine Accords and Notes into a single text feature
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
    
    return df

def train_model(df):
    """Trains the Gradient Boosting Regressor."""
    print(f"\nTraining on {len(df)} perfumes...")
    
    X = df[['Gender', 'Composition']]
    y = df['Rating Value']
    
    # Preprocessing
    # Use SVD to reduce dimensionality of TF-IDF for HistGradientBoosting
    preprocessor = ColumnTransformer(
        transformers=[
            ('gender', OneHotEncoder(handle_unknown='ignore', sparse_output=False), ['Gender']),
            ('composition', Pipeline([
                ('tfidf', TfidfVectorizer(min_df=5, max_features=5000)),
                ('svd', TruncatedSVD(n_components=100, random_state=42))
            ]), 'Composition'),
        ]
    )
    
    # Model: Histogram-based Gradient Boosting Regression Tree
    # Much faster for large datasets (n_samples >= 10,000)
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('regressor', HistGradientBoostingRegressor(max_iter=500, random_state=42, learning_rate=0.1, max_depth=10))
    ])
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print("Fitting model (Gradient Boosting)...")
    pipeline.fit(X_train, y_train)
    
    # Evaluate Regression
    print("Evaluating model...")
    y_pred = pipeline.predict(X_test)
    
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    
    print(f"\n--- Regression Performance ---")
    print(f"MAE: {mae:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"R²: {r2:.4f}")
    
    # Evaluate Classification (Precision/Recall for "Loved" threshold)
    threshold = 4.20
    y_test_class = (y_test >= threshold).astype(int)
    y_pred_class = (y_pred >= threshold).astype(int)
    
    print(f"\n--- Classification Performance (Threshold >= {threshold}) ---")
    print(classification_report(y_test_class, y_pred_class, target_names=['Not Loved', 'Loved']))
    
    return pipeline

def analyze_gender_features(model, df):
    """Analyzes top features for each gender."""
    print("\n--- Feature Analysis by Gender ---")
    
    for gender in ['women', 'men', 'unisex']:
        print(f"\nTop Drivers for {gender.upper()} (High Rated > 4.2):")
        subset = df[(df['Gender'] == gender) & (df['Rating Value'] >= 4.2)]
        
        if len(subset) < 10:
            print("  (Not enough data)")
            continue
            
        all_terms = []
        for text in subset['Composition']:
            all_terms.extend(text.split())
        
        from collections import Counter
        counts = Counter(all_terms).most_common(10)
        for term, count in counts:
            print(f"  - {term.replace('_', ' ')}: {count}")

def predict_interactive(model):
    """Interactive prediction loop."""
    print("\n--- Interactive Rating Predictor ---")
    print("Enter perfume details to predict the Rating (1.0 - 5.0).")
    
    while True:
        try:
            gender = input("\nGender (women/men/unisex): ").strip().lower()
            if not gender: break
            
            accords = input("Main Accords (comma separated): ").strip()
            top = input("Top Notes (comma separated): ").strip()
            middle = input("Middle Notes (comma separated): ").strip()
            base = input("Base Notes (comma separated): ").strip()
            
            parts = []
            if accords: parts.extend([x.strip().replace(' ', '_') for x in accords.split(',')])
            if top: parts.extend([x.strip().replace(' ', '_') for x in top.split(',')])
            if middle: parts.extend([x.strip().replace(' ', '_') for x in middle.split(',')])
            if base: parts.extend([x.strip().replace(' ', '_') for x in base.split(',')])
            
            composition = ' '.join(parts)
            
            input_data = pd.DataFrame({
                'Gender': [gender],
                'Composition': [composition]
            })
            
            prediction = model.predict(input_data)[0]
            print(f"\n>>> Predicted Rating: {prediction:.2f} / 5.0")
            
            if prediction >= 4.2:
                print("🌟 Likely a HIT!")
            elif prediction >= 3.8:
                print("👍 Solid fragrance.")
            else:
                print("📉 Below average.")
                
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    df = load_and_clean_data(DATA_PATH)
    if df is not None:
        model = train_model(df)
        analyze_gender_features(model, df)
        predict_interactive(model)
