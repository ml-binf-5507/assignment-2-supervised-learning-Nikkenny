"""
Data loading and preprocessing functions for heart disease dataset.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def load_heart_disease_data(filepath):
    """
    Load the heart disease dataset from CSV.
    
    Parameters
    ----------
    filepath : str
        Path to the heart disease CSV file
        
    Returns
    -------
    pd.DataFrame
        Raw dataset with all features and targets
        
    Raises
    ------
    FileNotFoundError
        If the CSV file does not exist
    ValueError
        If the CSV is empty or malformed
        
    Examples
    --------
    >>> df = load_heart_disease_data('data/heart_disease_uci.csv')
    >>> df.shape
    (270, 15)
    """
    # Hint: Use pd.read_csv()
    # Hint: Check if file exists and raise helpful error if not
    # TODO: Implement data loading
    heart_disease = pd.read_csv(filepath)
    print(heart_disease.head())
    return heart_disease


def preprocess_data(df):
    """
    Handle missing values, encode categorical variables, and clean data.
    
    Parameters
    ----------
    df : pd.DataFrame
        Raw dataset
        
    Returns
    -------
    pd.DataFrame
        Cleaned and preprocessed dataset
    """
    # TODO: Implement preprocessing
    # - Handle missing values
    missing_data = heart_disease.isnull().sum()
    print(missing_data)
    int_fill = ['trestbps', 'chol', 'thalch', 'oldpeak', 'ca']
    cat_fill = ['fbs', 'restecg', 'exang', 'slope', 'thal']
    heart_disease[int_fill] = heart_disease[int_fill].fillna(heart_disease[int_fill].mean())
    for col in cat_fill:
        heart_disease = heart_disease[col].fillna(heart_disease[col].mode()[0], inplace = True)
    # - Encode categorical variables (e.g., sex, cp, fbs, etc.)
    target = 'num'
    
    encoded_columns = []

    encoded = pd.get_dummies(heart_disease, columns=[cat_fill], dtype = int)
    encoded_columns.extend(encoded.columns.tolist())

    heart_disease = pd.concat([heart_disease, encoded], axis=1)    
    
    # - Ensure all columns are numeric
    return heart_disease
    

def prepare_regression_data(df, target='chol'):
    """
    Prepare data for linear regression (predicting serum cholesterol).
    
    Parameters
    ----------
    df : pd.DataFrame
        Preprocessed dataset
    target : str
        Target column name (default: 'chol')
        
    Returns
    -------
    tuple
        (X, y) feature matrix and target vector
    """
    # TODO: Implement regression data preparation
    # - Remove rows with missing chol values
    heart_clean = heart_disease.dropna(subset=['chol'])
    print(heart_clean)
    # - Exclude chol from features
    heart_clean = heart_disease.dropna('chol', axis=1)

    # - Return X (features) and y (target)
    X = heart_clean('num')
    y = heart_clean.drop(['num'], axis=1)
    return X, y

def prepare_classification_data(df, target='num'):
    """
    Prepare data for classification (predicting heart disease presence).
    
    Parameters
    ----------
    df : pd.DataFrame
        Preprocessed dataset
    target : str
        Target column name (default: 'num')
        
    Returns
    -------
    tuple
        (X, y) feature matrix and target vector (binary)
    """
    # TODO: Implement classification data preparation
    # - Binarize target variable

    # - Exclude target from features
    heart_clean = heart_disease.dropna('num', axis=1)

    # - Exclude chol from features
    heart_clean = heart_disease.dropna('chol', axis=1)

    # - Return X (features) and y (target)
    X = heart_clean('num')
    y = heart_clean.drop(['num'], axis=1)
    return X, y


def split_and_scale(X, y, test_size=0.2, random_state=42):
    """
    Split data into train/test sets and scale features.
    
    Parameters
    ----------
    X : pd.DataFrame or np.ndarray
        Feature matrix
    y : pd.Series or np.ndarray
        Target vector
    test_size : float
        Proportion of data to use for testing
    random_state : int
        Random seed for reproducibility
        
    Returns
    -------
    tuple
        (X_train_scaled, X_test_scaled, y_train, y_test, scaler)
        where scaler is the fitted StandardScaler
    """
    # TODO: Implement train/test split and scaling
    # - Use train_test_split with provided parameters
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    # - Fit StandardScaler on training data only
    scaler = StandardScaler()
    # - Transform both train and test data
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.fit_transform(X_test)
    # - Return scaled data and scaler object
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler

heart_disease = load_heart_disease_data("Assignments/assignment-2-supervised-learning-Nikkenny/data/heart_disease_uci.csv")