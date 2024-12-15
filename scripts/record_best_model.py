import json
import datetime
import os
from pathlib import Path

project_root = Path(__file__).parent.parent

def record_best_model_results():
    results = {
        "best_model": "First Training - model_epoch_3",
        "accuracies": {
            "overall_average": 0.8041,
            "field": 0.9899,
            "tier": 1.0000,
            "subfield": 0.4223
        },
        "model_path": os.path.join("data/processed/model_artifacts", "model_epoch_3"),
        "evaluation_date": datetime.datetime.now().strftime("%Y-%m-%d %H:%M"),
        "comparison_results": {
            "First Training - model_epoch_1": {
                "overall_average": 0.5732,
                "field": 0.5068,
                "tier": 0.9932,
                "subfield": 0.2196
            },
            "First Training - model_epoch_2": {
                "overall_average": 0.7736,
                "field": 0.9797,
                "tier": 0.9966,
                "subfield": 0.3446
            },
            "First Training - model_epoch_3": {
                "overall_average": 0.8041,
                "field": 0.9899,
                "tier": 1.0000,
                "subfield": 0.4223
            },
            "Second Training - run_1213_2013": {
                "overall_average": 0.7725,
                "field": 0.9392,
                "tier": 0.9966,
                "subfield": 0.3818
            }
        }
    }
    
    # Save to model_evaluation_results.json in the project root directory
    output_path = project_root / "model_evaluation_results.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=4)
    
    print(f"Results have been saved to: {output_path}")
    
    # Update the model path used in the RAG system
    rag_path = project_root / "src" / "models" / "rag.py"
    print(f"\nPlease update the model path in {rag_path} to:")
    print(f"best_model_path = os.path.join(MODEL_ARTIFACTS_DIR, 'model_epoch_3')")

if __name__ == "__main__":
    record_best_model_results()
