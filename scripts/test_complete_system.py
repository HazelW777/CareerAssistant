import os
import sys
from pathlib import Path
from pprint import pprint

# Add the project root directory to Python path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from src.models.rag import RAGSystem

def test_system():
    print("=== Starting Complete System Test ===\n")
    
    # Initialize the RAG system
    rag = RAGSystem()
    
    # Test cases
    test_cases = [
        {
            "background": "I am a Python backend developer with 3 years of experience, preparing for a senior developer interview.",
            "sample_answer": "I use asynchronous IO and multiprocessing to handle high-concurrency scenarios, specifically asyncio for IO-bound tasks and multiprocessing for CPU-bound tasks."
        },
        {
            "background": "I am a frontend development intern primarily using React, preparing for a campus recruitment interview.",
            "sample_answer": "React's virtual DOM improves rendering performance by comparing the differences between the old and new DOM trees to reduce actual DOM operations."
        },
        {
            "background": "I am a systems architect with 5 years of experience, preparing for a technical expert interview.",
            "sample_answer": "When designing distributed systems, I consider the CAP theorem and balance consistency and availability based on business needs."
        }
    ]
    
    for idx, case in enumerate(test_cases, 1):
        print(f"\nTest Case {idx}: {case['background']}")
        print("-" * 50)
        
        # 1. Test Background Classification
        print("\n1. Test Background Classification:")
        classifications = rag.classify_background(case['background'])
        print("Classification Results:")
        pprint(classifications)
        
        # 2. Test Question Generation
        print("\n2. Test Question Generation:")
        conversation_history = []
        response1, state1 = rag.chat(case['background'], conversation_history)
        print("\nGenerated Question:", response1)
        
        # 3. Test Answer Evaluation
        print("\n3. Test Answer Evaluation:")
        print("Simulated Answer:", case['sample_answer'])
        conversation_history = [{
            "state": "awaiting_answer",
            "current_question": response1,
            "reference_answer": state1["reference_answer"],
            "user_background": case['background']
        }]
        response2, state2 = rag.chat(case['sample_answer'], conversation_history)
        print("\nEvaluation Result:", response2)
        
        # Wait for user confirmation to continue
        input("\nPress Enter to proceed to the next test case...\n")

def main():
    try:
        test_system()
        print("\n=== System Testing Completed ===")
    except Exception as e:
        print(f"\nError: {str(e)}")
        raise

if __name__ == "__main__":
    main()
