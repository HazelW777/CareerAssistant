import os
import sys
from pprint import pprint

# Add the project root directory to the Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from src.data.preprocessor import DataPreprocessor
from src.data.vectordb import VectorDB
from src.config import RAW_DATA_DIR, PROCESSED_DATA_DIR

def test_preprocessing():
    print("=== Starting Data Preprocessing Test ===")
    
    # Initialize preprocessor
    preprocessor = DataPreprocessor()
    
    # Test data loading
    data_path = os.path.join(RAW_DATA_DIR, 'interview_questions.json')
    questions = preprocessor.load_raw_data(data_path)
    print("\n1. Raw Data Loading:")
    pprint(questions[0])  # Print the first data entry
    
    # Test vector database data preparation
    embed_df = preprocessor.prepare_for_embedding(questions)
    print("\n2. Data Prepared for Embedding:")
    print(embed_df.head(1))
    print("\nColumns:", embed_df.columns.tolist())
    
    # Test training data preparation
    train_df = preprocessor.prepare_for_training(questions)
    train_data, eval_data = preprocessor.split_train_eval(train_df)
    print("\n3. Training Data Preparation:")
    print("Training Set Size:", len(train_data))
    print("Evaluation Set Size:", len(eval_data))
    print("\nTraining Data Sample:")
    print(train_data.head(1))
    
    return embed_df

def test_vector_db(embed_df):
    print("\n=== Starting Vector Database Test ===")
    
    # Initialize vector database
    vector_db = VectorDB()
    
    # Test data upload
    print("\n1. Uploading Data to Vector Database...")
    vector_db.upsert_questions(embed_df.to_dict('records'))
    
    # Test similarity search
    test_query = "Python backend developer interview"
    print(f"\n2. Testing Similarity Search, Query: '{test_query}'")
    results = vector_db.search_similar_questions(test_query)
    
    print("\nSearch Results:")
    for i, result in enumerate(results, 1):
        print(f"\nResult {i}:")
        print(f"Similarity Score: {result['score']:.4f}")
        print("Question:", result['metadata']['question'])
        print("Answer:", result['metadata']['answer'])

def main():
    try:
        # Ensure the output directory exists
        os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
        
        # Test preprocessing
        embed_df = test_preprocessing()
        
        # Test vector database
        test_vector_db(embed_df)
        
        print("\n=== Testing Complete ===")
        
    except Exception as e:
        print(f"\nError: {str(e)}")
        raise

if __name__ == "__main__":
    main()
