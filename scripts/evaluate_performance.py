import os
import sys
from pathlib import Path
import json
import time
import numpy as np
from tqdm import tqdm

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from src.models.rag import RAGSystem
from src.config import MODEL_ARTIFACTS_DIR


def main():
    print("=== Starting System Performance Evaluation ===")
    
    try:
        print("\n1. Evaluating Model Performance")
        evaluate_model_performance()
        
        print("\n2. Evaluating Response Quality")
        evaluate_response_quality()
        
        print("\n3. Evaluating System Robustness")
        evaluate_system_robustness()
        
        print("\n=== Performance Evaluation Completed ===")
    except Exception as e:
        print(f"\nError during evaluation: {str(e)}")
        import traceback
        print("\nDetailed traceback:")
        print(traceback.format_exc())

def evaluate_model_performance():
    """Evaluate the classification performance of the model"""
    print("\n=== Model Performance Evaluation ===")
    
    # Load evaluation results
    results_path = os.path.join(project_root, "model_evaluation_results.json")
    with open(results_path, 'r') as f:
        results = json.load(f)
    
    print("\nPerformance Metrics for the Best Model:")
    print(f"Field Classification Accuracy: {results['accuracies']['field']:.4f}")
    print(f"Tier Classification Accuracy: {results['accuracies']['tier']:.4f}")
    print(f"Subfield Classification Accuracy: {results['accuracies']['subfield']:.4f}")
    print(f"Overall Average Accuracy: {results['accuracies']['overall_average']:.4f}")

def evaluate_response_quality():
    """Evaluate the quality of system responses"""
    print("\n=== Response Quality Evaluation ===")
    
    rag = RAGSystem()
    test_cases = [
        {
            "background": "I am a Python backend developer with 3 years of experience, preparing for a senior developer interview.",
            "sample_answer": "I use asynchronous IO and multiprocessing to handle high-concurrency scenarios. Specifically, asyncio for IO-bound tasks and multiprocessing for CPU-bound tasks."
        },
        {
            "background": "I am a frontend development intern primarily using React, preparing for a campus recruitment interview.",
            "sample_answer": "React's virtual DOM improves rendering performance by comparing the differences between the old and new DOM trees to reduce actual DOM operations."
        }
    ]
    
    response_metrics = {
        'relevance_scores': [],
        'response_times': [],
        'valid_responses': 0
    }
    
    for case in tqdm(test_cases, desc="Evaluating Response Quality"):
        try:
            # Test question generation and answer evaluation
            conversation_history = []
            start_time = time.time()
            response1, state1 = rag.chat(case['background'], conversation_history)
            response_time = time.time() - start_time
            response_metrics['response_times'].append(response_time)
            
            if state1.get('state') == 'awaiting_answer':
                response_metrics['valid_responses'] += 1
            
            # Evaluate the answer
            conversation_history = [{
                "state": "awaiting_answer",
                "current_question": response1,
                "reference_answer": state1["reference_answer"],
                "user_background": case['background']
            }]
            response2, _ = rag.chat(case['sample_answer'], conversation_history)
            
            # Extract relevance score from evaluation feedback
            try:
                score = float(response2.split("Overall Score (1-5):")[1].split("\n")[0].strip())
                response_metrics['relevance_scores'].append(score)
            except:
                print(f"Warning: Could not extract score from response: {response2}")
        except Exception as e:
            print(f"Error processing case: {str(e)}")
    
    # Calculate average metrics
    if response_metrics['response_times']:
        avg_response_time = np.mean(response_metrics['response_times'])
        print(f"\nAverage Response Time: {avg_response_time:.2f} seconds")
    
    if response_metrics['relevance_scores']:
        avg_relevance = np.mean(response_metrics['relevance_scores'])
        print(f"Average Relevance Score: {avg_relevance:.2f}/5.0")
    
    response_rate = response_metrics['valid_responses'] / len(test_cases)
    print(f"Valid Response Rate: {response_rate:.2%}")

def evaluate_system_robustness():
    """Evaluate the robustness of the system"""
    print("\n=== System Robustness Evaluation ===")
    
    rag = RAGSystem()
    edge_cases = [
        "",  # Empty input
        "Hello",  # Irrelevant input
        "I want a job",  # Incomplete background
        "I am an engineer" * 50,  # Overly long input
    ]
    
    robustness_metrics = {
        'handled_cases': 0,
        'valid_responses': 0
    }
    
    for case in edge_cases:
        try:
            response, state = rag.chat(case, [])
            robustness_metrics['handled_cases'] += 1
            if state.get('state') == 'awaiting_answer':
                robustness_metrics['valid_responses'] += 1
        except Exception as e:
            print(f"Error with input '{case[:20]}...': {str(e)}")
    
    success_rate = robustness_metrics['handled_cases'] / len(edge_cases)
    valid_rate = robustness_metrics['valid_responses'] / len(edge_cases)
    
    print(f"\nError Handling Success Rate: {success_rate:.2%}")
    print(f"Valid Response Rate: {valid_rate:.2%}")
