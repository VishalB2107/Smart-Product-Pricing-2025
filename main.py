
# ML Challenge 2025 - Product Price Prediction
# Main Training Script

import pandas as pd
import numpy as np
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import re

def extract_text_features(text):
    '''Extract features from catalog content'''
    features = {}
    text = str(text).lower()
    
    # Basic features
    features['text_length'] = len(text)
    features['word_count'] = len(text.split())
    
    # Extract numbers
    numbers = re.findall(r'\d+\.?\d*', text)
    features['num_count'] = len(numbers)
    features['max_number'] = max([float(n) for n in numbers]) if numbers else 0
    features['min_number'] = min([float(n) for n in numbers]) if numbers else 0
    
    # Brand detection
    premium_brands = ['apple', 'samsung', 'sony', 'nike', 'adidas', 'lego']
    features['is_premium'] = any(brand in text for brand in premium_brands)
    
    # Quantity indicators
    features['has_pack'] = 'pack' in text or 'set' in text
    features['has_bundle'] = 'bundle' in text or 'combo' in text
    
    return features

def train_model():
    '''Train the price prediction model'''
    print("Loading training data...")
    train_df = pd.read_csv('dataset/train.csv')
    
    # Extract features
    print("Extracting features...")
    X_features = train_df['catalog_content'].apply(extract_text_features)
    X = pd.DataFrame(X_features.tolist())
    
    # TF-IDF features
    print("Creating TF-IDF features...")
    tfidf = TfidfVectorizer(max_features=100, stop_words='english')
    tfidf_features = tfidf.fit_transform(train_df['catalog_content'])
    tfidf_df = pd.DataFrame(tfidf_features.toarray(), 
                            columns=[f'tfidf_{i}' for i in range(100)])
    
    # Combine features
    X = pd.concat([X.reset_index(drop=True), tfidf_df], axis=1)
    y = train_df['price']
    
    # Split data
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Train model
    print("Training model...")
    model = RandomForestRegressor(
        n_estimators=100,
        max_depth=20,
        min_samples_split=5,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    
    # Evaluate
    val_pred = model.predict(X_val)
    val_mae = mean_absolute_error(y_val, val_pred)
    print(f"Validation MAE: ${val_mae:.2f}")
    
    # Save model
    joblib.dump(model, 'model.pkl')
    joblib.dump(tfidf, 'tfidf.pkl')
    print("Model saved!")
    
    return model, tfidf

def make_predictions():
    '''Generate predictions for test set'''
    print("Loading model...")
    model = joblib.load('model.pkl')
    tfidf = joblib.load('tfidf.pkl')
    
    print("Loading test data...")
    test_df = pd.read_csv('dataset/test.csv')
    
    # Extract features
    print("Extracting features...")
    X_features = test_df['catalog_content'].apply(extract_text_features)
    X_test = pd.DataFrame(X_features.tolist())
    
    # TF-IDF features
    tfidf_features = tfidf.transform(test_df['catalog_content'])
    tfidf_df = pd.DataFrame(tfidf_features.toarray(), 
                            columns=[f'tfidf_{i}' for i in range(100)])
    
    # Combine features
    X_test = pd.concat([X_test.reset_index(drop=True), tfidf_df], axis=1)
    
    # Predict
    print("Making predictions...")
    predictions = model.predict(X_test)
    predictions = np.maximum(predictions, 0.1)  # Ensure positive
    
    # Create output
    output_df = pd.DataFrame({
        'sample_id': test_df['sample_id'],
        'price': predictions
    })
    
    # Save
    output_df.to_csv('test_out.csv', index=False)
    print(f"Predictions saved! Total: {len(output_df)}")
    
    return output_df

if __name__ == "__main__":
    # Train model
    model, tfidf = train_model()
    
    # Make predictions
    predictions = make_predictions()
    
    print("Done!")
