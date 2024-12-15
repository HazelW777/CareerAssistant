# scripts/prepare_training_data.py
import os
import sys
from pathlib import Path
from pprint import pprint
import pandas as pd
import numpy as np
from collections import Counter

# Add the project root directory to the Python path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from src.data.preprocessor import DataPreprocessor
from src.config import RAW_DATA_DIR, PROCESSED_DATA_DIR

def analyze_data_distribution(df: pd.DataFrame):
    """Analyze data distribution"""
    print("\n=== Data Distribution Analysis ===")
    
    print("\nField Distribution:")
    print(df['field'].value_counts())
    
    print("\nTier Distribution:")
    print(df['tier'].value_counts())
    
    print("\nSubfield Distribution:")
    print(df['subfield'].value_counts())
    
    print("\nNumber of questions per field:")
    field_subfield = df.groupby(['field', 'subfield']).size().unstack(fill_value=0)
    print(field_subfield)

def validate_data_quality(df: pd.DataFrame):
    """Validate data quality"""
    print("\n=== Data Quality Check ===")
    
    # Check for missing values
    print("\nMissing Value Check:")
    print(df.isnull().sum())
    
    # Check text lengths
    print("\nQuestion Length Statistics:")
    df['question_length'] = df['question'].str.len()
    print(df['question_length'].describe())
    
    print("\nAnswer Length Statistics:")
    df['answer_length'] = df['answer'].str.len()
    print(df['answer_length'].describe())
    
    # Check for outliers
    print("\nPotential Outliers:")
    short_questions = df[df['question_length'] < 20]
    if not short_questions.empty:
        print("\nQuestions that are too short:")
        print(short_questions[['question', 'field']].head())
    
    long_questions = df[df['question_length'] > 500]
    if not long_questions.empty:
        print("\nQuestions that are too long:")
        print(long_questions[['question', 'field']].head())

def prepare_training_data():
    """Prepare training data"""
    print("=== Starting Training Data Preparation ===")
    
    # Initialize preprocessor
    preprocessor = DataPreprocessor()
    
    # Load raw data
    data_path = os.path.join(RAW_DATA_DIR, 'interview_questions.json')
    questions = preprocessor.load_raw_data(data_path)
    print(f"\nLoaded {len(questions)} questions")
    
    # Convert to DataFrame
    df = pd.DataFrame(questions)
    
    # Analyze data distribution
    analyze_data_distribution(df)
    
    # Validate data quality
    validate_data_quality(df)
    
    # Prepare training data
    train_df = preprocessor.prepare_for_training(questions)
    train_data, eval_data = preprocessor.split_train_eval(train_df)
    
    # Save processed data
    os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
    train_data.to_csv(os.path.join(PROCESSED_DATA_DIR, 'train.csv'), index=False)
    eval_data.to_csv(os.path.join(PROCESSED_DATA_DIR, 'eval.csv'), index=False)
    
    print(f"\nTraining Set Size: {len(train_data)} entries")
    print(f"Validation Set Size: {len(eval_data)} entries")
    
    # Save label mappings
    label_mappings = preprocessor.get_label_mappings()
    print("\nLabel Mappings:")
    pprint(label_mappings)

def main():
    try:
        prepare_training_data()
        print("\n=== Data Preparation Complete ===")
    except Exception as e:
        print(f"\nError: {str(e)}")
        raise

if __name__ == "__main__":
    main()
