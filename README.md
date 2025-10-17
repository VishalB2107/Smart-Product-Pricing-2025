# ML Challenge 2025 - Product Price Prediction
## Submission by Vishal Bhandarakavathe and Team(WildStone)

### Date: 2025-10-11

## Overview
This project was developed for the Amazon ML Challenge 2025 – Smart Product Pricing competition, where the goal was to predict optimal product prices based on product descriptions and specifications. The solution implements a Random Forest Regressor trained on textual data extracted from the catalog_content field. Using a combination of custom text-based features (like length, numeric indicators, and brand/quantity presence) and TF-IDF representations of product descriptions, the model learns complex relationships between product details and pricing. It achieved a validation Mean Absolute Error (MAE) that translated to an approximate SMAPE score of 42% on the public leaderboard. The codebase supports full training, evaluation, and generation of submission-ready predictions (test_out.csv).

## Approach
- **Method**: Random Forest Regression with Text Feature Engineering
- **Features**: 
  - Text-based features (length, word count, numbers)
  - Brand detection (premium vs. standard)
  - TF-IDF vectorization (100 features)
  - Quantity indicators (pack, bundle, set)

## Model Details
- **Algorithm**: Random Forest Regressor
- **Parameters**:
  - n_estimators: 100
  - max_depth: 20
  - min_samples_split: 5
- **Features**: 108 total (8 engineered + 100 TF-IDF)

## Files Included
- `main.py` - Main training and prediction script
- `README.md` - This file
- `requirements.txt` - Required Python packages

## How to Run

### Installation
```bash
pip install -r requirements.txt
```

### Training and Prediction
```bash
python main.py
```

This will:
1. Load training data from `dataset/train.csv`
2. Train the model
3. Generate predictions on `dataset/test.csv`
4. Save output to `test_out.csv`

## Requirements
- Python 3.8+
- scikit-learn
- pandas
- numpy
- joblib

## Performance
- Validation MAE: ~$16-20
- Training time: ~5-10 minutes on CPU
- Prediction time: ~1-2 minutes

## Model Interpretability
Key features in order of importance:
1. TF-IDF features (product keywords)
2. Maximum number in text (often indicates quantity/size)
3. Text length
4. Brand indicators

## Limitations & Future Improvements
- Currently uses only text features (no images)
- Could benefit from:
  - Image features using CNN
  - More sophisticated NLP (BERT, GPT)
  - Ensemble methods
  - Price range-specific models

## Contact
bvishal2107@gmail.com
