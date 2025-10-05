#!/usr/bin/env python3
"""
Advanced NASA Exoplanet Detection Model Training
Using TOI and Cumulative datasets to train an improved ML model
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
from sklearn.calibration import CalibratedClassifierCV
import pickle
import os
import warnings
warnings.filterwarnings('ignore')

def setup_matplotlib_for_plotting():
    """
    Setup matplotlib and seaborn for plotting with proper configuration.
    Call this function before creating any plots to ensure proper rendering.
    """
    warnings.filterwarnings('default')  # Show all warnings

    # Configure matplotlib for non-interactive mode
    plt.switch_backend("Agg")

    # Set chart style
    plt.style.use("seaborn-v0_8")
    sns.set_palette("husl")

    # Configure platform-appropriate fonts for cross-platform compatibility
    # Must be set after style.use, otherwise will be overridden by style configuration
    plt.rcParams["font.sans-serif"] = ["Noto Sans CJK SC", "WenQuanYi Zen Hei", "PingFang SC", "Arial Unicode MS", "Hiragino Sans GB"]
    plt.rcParams["axes.unicode_minus"] = False

def load_and_process_toi_data(toi_path):
    """Load and process TOI (TESS Objects of Interest) dataset"""
    print("Loading TOI dataset...")
    
    # Read the CSV file, skipping comment lines
    toi_df = pd.read_csv(toi_path, comment='#', low_memory=False)
    
    print(f"TOI dataset shape: {toi_df.shape}")
    print(f"TOI columns: {list(toi_df.columns)}")
    
    # Create labels based on TFOPWG disposition
    # CP = Confirmed Planet, PC = Planet Candidate
    # FP = False Positive, KP = Known Planet
    toi_df['is_exoplanet'] = toi_df['tfopwg_disp'].isin(['CP', 'PC'])
    
    # Select relevant features for training
    feature_cols = [
        'pl_orbper', 'pl_orbpererr1', 'pl_orbpererr2',  # Orbital period
        'pl_tranmid', 'pl_tranmiderr1', 'pl_tranmiderr2',  # Transit midpoint
        'pl_trandur', 'pl_trandurerr1', 'pl_trandurerr2',  # Transit duration
        'pl_trandep', 'pl_trandeperr1', 'pl_trandeperr2',  # Transit depth
        'pl_rade', 'pl_radeerr1', 'pl_radeerr2',  # Planet radius
        'st_rad', 'st_raderr1', 'st_raderr2',  # Stellar radius
        'st_mass', 'st_masserr1', 'st_masserr2',  # Stellar mass
        'st_teff', 'st_tefferr1', 'st_tefferr2',  # Stellar temperature
        'st_logg', 'st_loggerr1', 'st_loggerr2',  # Stellar surface gravity
        'st_met', 'st_meterr1', 'st_meterr2',  # Stellar metallicity
        'st_tmag',  # TESS magnitude
    ]
    
    # Filter columns that actually exist in the dataset
    available_cols = [col for col in feature_cols if col in toi_df.columns]
    print(f"Available feature columns: {available_cols}")
    
    # Create feature dataframe
    features_df = toi_df[available_cols].copy()
    labels = toi_df['is_exoplanet'].copy()
    
    return features_df, labels, toi_df

def load_and_process_cumulative_data(cumulative_path):
    """Load and process Cumulative KOI dataset"""
    print("Loading Cumulative dataset...")
    
    # Read the CSV file, skipping comment lines
    cum_df = pd.read_csv(cumulative_path, comment='#', low_memory=False)
    
    print(f"Cumulative dataset shape: {cum_df.shape}")
    print(f"Cumulative columns: {list(cum_df.columns)}")
    
    # Create labels based on disposition
    # CONFIRMED = Confirmed exoplanet, CANDIDATE = Planet candidate
    cum_df['is_exoplanet'] = cum_df['koi_disposition'].isin(['CONFIRMED', 'CANDIDATE'])
    
    # Select relevant features
    feature_cols = [
        'koi_period', 'koi_period_err1', 'koi_period_err2',  # Orbital period
        'koi_time0bk', 'koi_time0bk_err1', 'koi_time0bk_err2',  # Transit epoch
        'koi_duration', 'koi_duration_err1', 'koi_duration_err2',  # Transit duration
        'koi_depth', 'koi_depth_err1', 'koi_depth_err2',  # Transit depth
        'koi_ror', 'koi_ror_err1', 'koi_ror_err2',  # Planet-to-star radius ratio
        'koi_dor', 'koi_dor_err1', 'koi_dor_err2',  # Semi-major axis ratio
        'koi_incl', 'koi_incl_err1', 'koi_incl_err2',  # Inclination
        'koi_prad', 'koi_prad_err1', 'koi_prad_err2',  # Planet radius
        'koi_teq', 'koi_teq_err1', 'koi_teq_err2',  # Equilibrium temperature
        'koi_slogg', 'koi_slogg_err1', 'koi_slogg_err2',  # Stellar surface gravity
        'koi_srad', 'koi_srad_err1', 'koi_srad_err2',  # Stellar radius
        'koi_smass', 'koi_smass_err1', 'koi_smass_err2',  # Stellar mass
        'koi_steff', 'koi_steff_err1', 'koi_steff_err2',  # Stellar temperature
        'koi_smet', 'koi_smet_err1', 'koi_smet_err2',  # Stellar metallicity
        'koi_kepmag',  # Kepler magnitude
    ]
    
    # Filter columns that actually exist
    available_cols = [col for col in feature_cols if col in cum_df.columns]
    print(f"Available feature columns: {available_cols}")
    
    # Create feature dataframe
    features_df = cum_df[available_cols].copy()
    labels = cum_df['is_exoplanet'].copy()
    
    return features_df, labels, cum_df

def engineer_features(features_df):
    """Engineer additional features from the raw data"""
    print("Engineering features...")
    
    engineered_df = features_df.copy()
    
    # For each primary parameter, create derived features
    for col in features_df.columns:
        if col.endswith('_err1') or col.endswith('_err2'):
            continue
            
        base_col = col
        err1_col = f"{col}_err1"
        err2_col = f"{col}_err2"
        
        # Signal-to-noise ratio
        if err1_col in features_df.columns:
            err = features_df[err1_col].abs()
            snr = features_df[base_col] / (err + 1e-10)
            engineered_df[f"{base_col}_snr"] = snr
        
        # Uncertainty magnitude
        if err1_col in features_df.columns and err2_col in features_df.columns:
            uncertainty = (features_df[err1_col].abs() + features_df[err2_col].abs()) / 2
            engineered_df[f"{base_col}_uncertainty"] = uncertainty
    
    # Remove original error columns for training (keep only primary values and derived features)
    cols_to_remove = [col for col in engineered_df.columns if col.endswith('_err1') or col.endswith('_err2')]
    engineered_df = engineered_df.drop(columns=cols_to_remove)
    
    return engineered_df

def preprocess_features(features_df):
    """Clean and preprocess features for training"""
    print("Preprocessing features...")
    
    # Handle missing values
    features_df = features_df.replace([np.inf, -np.inf], np.nan)
    
    # Fill missing values with median for numerical columns
    for col in features_df.columns:
        if features_df[col].dtype in ['float64', 'int64']:
            median_val = features_df[col].median()
            features_df[col] = features_df[col].fillna(median_val)
    
    # Remove columns with too many missing values (>50%)
    threshold = len(features_df) * 0.5
    features_df = features_df.dropna(axis=1, thresh=threshold)
    
    # Remove remaining rows with any missing values
    features_df = features_df.dropna()
    
    print(f"Final feature matrix shape: {features_df.shape}")
    print(f"Final features: {list(features_df.columns)}")
    
    return features_df

def train_enhanced_model(X, y):
    """Train an enhanced model using multiple algorithms and hyperparameter tuning"""
    print("Training enhanced exoplanet detection model...")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print(f"Training set size: {X_train.shape}")
    print(f"Test set size: {X_test.shape}")
    print(f"Positive class ratio: {y_train.mean():.3f}")
    
    # Try multiple models
    models = {
        'RandomForest': RandomForestClassifier(random_state=42),
        'GradientBoosting': GradientBoostingClassifier(random_state=42),
    }
    
    best_model = None
    best_score = 0
    best_model_name = ""
    
    for name, model in models.items():
        print(f"\nTraining {name}...")
        
        # Hyperparameter tuning
        if name == 'RandomForest':
            param_grid = {
                'n_estimators': [100, 200],
                'max_depth': [10, 20, None],
                'min_samples_split': [2, 5],
                'min_samples_leaf': [1, 2],
                'max_features': ['sqrt', 'log2']
            }
        else:  # GradientBoosting
            param_grid = {
                'n_estimators': [100, 200],
                'max_depth': [3, 5, 7],
                'learning_rate': [0.1, 0.2],
                'min_samples_split': [2, 5],
                'min_samples_leaf': [1, 2]
            }
        
        # Grid search with cross-validation
        grid_search = GridSearchCV(model, param_grid, cv=5, scoring='roc_auc', n_jobs=-1, verbose=1)
        grid_search.fit(X_train_scaled, y_train)
        
        # Get best model
        tuned_model = grid_search.best_estimator_
        
        # Calibrate probabilities
        calibrated_model = CalibratedClassifierCV(tuned_model, method='isotonic', cv=3)
        calibrated_model.fit(X_train_scaled, y_train)
        
        # Evaluate
        y_pred_proba = calibrated_model.predict_proba(X_test_scaled)[:, 1]
        auc_score = roc_auc_score(y_test, y_pred_proba)
        
        print(f"{name} Best parameters: {grid_search.best_params_}")
        print(f"{name} AUC Score: {auc_score:.4f}")
        
        if auc_score > best_score:
            best_score = auc_score
            best_model = calibrated_model
            best_model_name = name
    
    print(f"\nBest model: {best_model_name} with AUC: {best_score:.4f}")
    
    return best_model, scaler, X_test_scaled, y_test

def evaluate_model(model, scaler, X_test, y_test, feature_names):
    """Comprehensive model evaluation"""
    print("\nEvaluating model performance...")
    
    # Predictions
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    y_pred = (y_pred_proba >= 0.5).astype(int)
    
    # Classification report
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    print(f"\nConfusion Matrix:")
    print(cm)
    
    # AUC Score
    auc = roc_auc_score(y_test, y_pred_proba)
    print(f"\nAUC Score: {auc:.4f}")
    
    return {
        'auc_score': auc,
        'y_pred_proba': y_pred_proba,
        'y_pred': y_pred,
        'classification_report': classification_report(y_test, y_pred, output_dict=True)
    }

def create_visualizations(model, scaler, X_test, y_test, feature_names, results):
    """Create visualization plots for model analysis"""
    setup_matplotlib_for_plotting()
    
    print("Creating visualizations...")
    
    # Create output directory
    os.makedirs('model_analysis', exist_ok=True)
    
    # 1. ROC Curve
    plt.figure(figsize=(10, 8))
    fpr, tpr, _ = roc_curve(y_test, results['y_pred_proba'])
    plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {results["auc_score"]:.4f})')
    plt.plot([0, 1], [0, 1], 'k--', label='Random')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve - Exoplanet Detection Model')
    plt.legend()
    plt.grid(True)
    plt.savefig('model_analysis/roc_curve.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Confusion Matrix
    plt.figure(figsize=(8, 6))
    cm = confusion_matrix(y_test, results['y_pred'])
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['No Exoplanet', 'Exoplanet'],
                yticklabels=['No Exoplanet', 'Exoplanet'])
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig('model_analysis/confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. Feature Importance (if available)
    try:
        if hasattr(model.base_estimator, 'feature_importances_'):
            importances = model.base_estimator.feature_importances_
            indices = np.argsort(importances)[::-1][:20]  # Top 20 features
            
            plt.figure(figsize=(12, 8))
            plt.title('Top 20 Feature Importances')
            plt.bar(range(len(indices)), importances[indices])
            plt.xticks(range(len(indices)), [feature_names[i] for i in indices], rotation=45, ha='right')
            plt.ylabel('Importance')
            plt.tight_layout()
            plt.savefig('model_analysis/feature_importance.png', dpi=300, bbox_inches='tight')
            plt.close()
    except Exception as e:
        print(f"Could not create feature importance plot: {e}")
    
    # 4. Prediction Distribution
    plt.figure(figsize=(10, 6))
    plt.hist(results['y_pred_proba'][y_test == 0], bins=50, alpha=0.7, label='No Exoplanet', color='red')
    plt.hist(results['y_pred_proba'][y_test == 1], bins=50, alpha=0.7, label='Exoplanet', color='blue')
    plt.xlabel('Predicted Probability')
    plt.ylabel('Count')
    plt.title('Distribution of Predicted Probabilities')
    plt.legend()
    plt.axvline(x=0.5, color='black', linestyle='--', label='Decision Threshold')
    plt.savefig('model_analysis/prediction_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()

def save_model(model, scaler, feature_names):
    """Save the trained model and associated components"""
    print("Saving model...")
    
    os.makedirs('models', exist_ok=True)
    
    model_data = {
        'model': model,
        'scaler': scaler,
        'feature_names': feature_names,
        'model_info': {
            'training_date': pd.Timestamp.now().isoformat(),
            'model_type': 'NASA Exoplanet Detection Model',
            'features_count': len(feature_names)
        }
    }
    
    with open('models/enhanced_exoplanet_model.pkl', 'wb') as f:
        pickle.dump(model_data, f)
    
    print("Model saved to 'models/enhanced_exoplanet_model.pkl'")

def main():
    """Main training pipeline"""
    print("=== NASA Exoplanet Detection Model Training ===\n")
    
    # Load datasets
    toi_features, toi_labels, toi_raw = load_and_process_toi_data('user_input_files/TOI_2025.09.30_13.13.38.csv')
    cum_features, cum_labels, cum_raw = load_and_process_cumulative_data('user_input_files/cumulative_2025.09.30_13.13.47.csv')
    
    # Combine datasets (if they have compatible features)
    print("\nCombining datasets...")
    
    # Find common columns
    common_cols = set(toi_features.columns) & set(cum_features.columns)
    print(f"Common feature columns: {len(common_cols)}")
    
    if len(common_cols) < 5:
        print("Not enough common features, using datasets separately...")
        # Use the larger dataset
        if len(toi_features) > len(cum_features):
            features_df = toi_features
            labels = toi_labels
            print("Using TOI dataset")
        else:
            features_df = cum_features
            labels = cum_labels
            print("Using Cumulative dataset")
    else:
        # Combine datasets using common features
        toi_common = toi_features[list(common_cols)]
        cum_common = cum_features[list(common_cols)]
        
        features_df = pd.concat([toi_common, cum_common], ignore_index=True)
        labels = pd.concat([toi_labels, cum_labels], ignore_index=True)
        print(f"Combined dataset size: {features_df.shape}")
    
    # Engineer features
    engineered_features = engineer_features(features_df)
    
    # Preprocess
    clean_features = preprocess_features(engineered_features)
    
    # Align labels with clean features
    labels_clean = labels.loc[clean_features.index]
    
    print(f"\nFinal dataset: {clean_features.shape[0]} samples, {clean_features.shape[1]} features")
    print(f"Exoplanet rate: {labels_clean.mean():.3f}")
    
    # Train model
    model, scaler, X_test, y_test = train_enhanced_model(clean_features, labels_clean)
    
    # Evaluate model
    results = evaluate_model(model, scaler, X_test, y_test, clean_features.columns.tolist())
    
    # Create visualizations
    create_visualizations(model, scaler, X_test, y_test, clean_features.columns.tolist(), results)
    
    # Save model
    save_model(model, scaler, clean_features.columns.tolist())
    
    print(f"\n=== Training Complete ===")
    print(f"Final Model AUC: {results['auc_score']:.4f}")
    print(f"Model saved to: models/enhanced_exoplanet_model.pkl")
    print(f"Analysis plots saved to: model_analysis/")

if __name__ == "__main__":
    main()