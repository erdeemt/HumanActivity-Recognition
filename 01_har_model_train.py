# -*- coding: utf-8 -*-
"""
SIGNAL PROCESSING AND MACHINE LEARNING 2024-2025 FALL

FINAL : HUMAN ACTIVITY RECOGNITION

Script 01 : Model Training with Advanced Feature Extraction (34 Features)


"""

# ==========================================
# 1. IMPORTATIONS
# ==========================================

import pandas as pd
import numpy as np
import glob
import time
import warnings
import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import xgboost as xgb
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv1D, MaxPooling1D, LSTM
from tensorflow.keras.utils import to_categorical

warnings.filterwarnings("ignore")

# ==========================================
# 2. SETTINGS AND CONSTANTS
# ==========================================
RANDOM_SEED = 47
WINDOW_SIZE = 40  
STEP_SIZE = 20    
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def extract_features_for_ml(segments):
    """Advanced Feature Extraction with 34 Features"""
    features_list = []
    for segment in segments:
        # Basic Stats (24 attributes)
        mean = np.mean(segment, axis=0)
        std = np.std(segment, axis=0)
        max_val = np.max(segment, axis=0)
        min_val = np.min(segment, axis=0)
        
        # Magnitude (4 features)
        gyro_mag = np.sqrt(np.sum(np.square(segment[:, 0:3]), axis=1))
        accel_mag = np.sqrt(np.sum(np.square(segment[:, 3:6]), axis=1))
        mag_stats = [np.mean(gyro_mag), np.std(gyro_mag), np.mean(accel_mag), np.std(accel_mag)]
        
        # Zero Crossing Rate (6 özellik)
        zcr = np.mean(np.diff(np.sign(segment), axis=0) != 0, axis=0)
        
        # Total: 34 Features
        features = np.concatenate([mean, std, max_val, min_val, mag_stats, zcr])
        features_list.append(features)
    return np.array(features_list)

def load_data(target_folder="human_activity"):
    search_path = os.path.join(BASE_DIR, target_folder, "*.csv")
    files = glob.glob(search_path)
    data_frames = []
    for file in files:
        df = pd.read_csv(file)
        df['Label'] = df['Label'].apply(lambda x: '_'.join(str(x).split('_')[:-1]) if str(x).split('_')[-1].isdigit() else str(x))
        data_frames.append(df)
    full_df = pd.concat(data_frames, ignore_index=True)
    le = LabelEncoder()
    full_df['Label_Encoded'] = le.fit_transform(full_df['Label'])
    return full_df, le

def create_segments(df, time_steps, step, features):
    segments, labels = [], []
    for i in range(0, len(df) - time_steps, step):
        window = df[features].iloc[i: i + time_steps].values
        if window.shape[0] == time_steps:
            segments.append(window)
            labels.append(df['Label_Encoded'].iloc[i: i + time_steps].mode()[0])
    return np.array(segments), np.array(labels)

if __name__ == "__main__":
    print("=== HAR TRAINING IS STARTING ===")
    
    try:
        df, label_encoder = load_data("human_activity")
    except ValueError as e:
        print(e)
        exit()

    feature_cols = ['gyro_x', 'gyro_y', 'gyro_z', 'accel_x', 'accel_y', 'accel_z']
    
    print("[ACTION] Data is checked in numerical format...")
    for col in feature_cols:
        
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Completely delete bad (NaN) rows
    eski_boyut = len(df)
    df.dropna(subset=feature_cols, inplace=True)
    
    if len(df) != eski_boyut:
        print(f"[WARNING] {eski_boyut - len(df)} Bad rows were cleaned from the data set.")

    # --- NOW NORMALIZATION CAN BE DONE SAFELY ---
    scaler = StandardScaler()
    df[feature_cols] = scaler.fit_transform(df[feature_cols])

    print("\n[ACTION] Segmentation in progress...")
    X_segments, y_labels = create_segments(df, WINDOW_SIZE, STEP_SIZE, feature_cols)
    X_train_dl, X_test_dl, y_train, y_test = train_test_split(X_segments, y_labels, test_size=0.2, stratify=y_labels, random_state=47)
    
    X_train_ml = extract_features_for_ml(X_train_dl)
    X_test_ml = extract_features_for_ml(X_test_dl)
    
    # XGBoost usually gives the best performance
    model = xgb.XGBClassifier(random_state=47)
    model.fit(X_train_ml, y_train)
    
    joblib.dump(model, "best_model_ml.joblib")
    joblib.dump(scaler, "scaler.joblib")
    joblib.dump(label_encoder, "label_encoder.joblib")
    print("Model and auxiliary files updated with 34 features!")