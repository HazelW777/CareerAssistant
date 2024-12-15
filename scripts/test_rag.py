import os
import sys
from pathlib import Path
from pprint import pprint

# Add the project root directory to the Python path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from src.models.rag import RAGSystem
from src.data.preprocessor import DataPreprocessor
from src.data.vectordb import VectorDB
from src.config import RAW_DATA_DIR

def test_rag_system():
    print("=== Starting RAG System Test ===\n")
    
    # Initialize RAG system
    rag = RAGSystem()
    
    # Test scenario 1: Backend Developer Interview
    test_background1 = "I am a Python backend developer with 3 years of experience, currently preparing for an advanced developer interview"
    print("Test Scenario 1 - Backend Developer Interview:")
    print(f"User Background: {test_background1}")
    
    # First round of conversation
    response1, state1 = rag.chat(test_background1, [])
    print("\nInterviewer Question:", response1)
    
    # Simulate user answer
    test_answer1 = input("\nPlease provide your answer (or press Enter for a sample answer): ") or \
                   "In Python, the Global Interpreter Lock (GIL) ensures that only one thread executes Python bytecode at a time. This has a significant impact on CPU-bound multi-threaded programs, as it prevents leveraging multi-core processors. Solutions include using multiprocessing or asynchronous programming."
    print("\nUser Answer:", test_answer1)
    
    # Save the complete conversation history
    conversation_history = [{
        "state": "awaiting_answer",
        "current_question": response1,
        "reference_answer": state1["reference_answer"],
        "user_background": test_background1
    }]
    
    response2, state2 = rag.chat(test_answer1, conversation_history)
    print("\nInterviewer Evaluation:", response2)


def main():
    try:
        # Ensure data is processed and loaded into the vector database
        print("Ensuring vector database is ready...\n")
        preprocessor = DataPreprocessor()
        data_path = os.path.join(RAW_DATA_DIR, 'interview_questions.json')
        questions = preprocessor.load_raw_data(data_path)
        embed_df = preprocessor.prepare_for_embedding(questions)
        
        vector_db = VectorDB()
        vector_db.upsert_questions(embed_df.to_dict('records'))
        
        # Test the RAG system
        test_rag_system()
        
    except Exception as e:
        print(f"\nError: {str(e)}")
        raise

if __name__ == "__main__":
    main()
