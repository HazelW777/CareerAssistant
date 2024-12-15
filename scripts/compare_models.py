# scripts/compare_models.py
import os
import sys
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoConfig
from tqdm import tqdm
import pandas as pd

# Add the project root directory to the Python path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from src.data.dataset import InterviewDataset
from src.models.model import InterviewQuestionClassifier
from src.config import MODEL_NAME, PROCESSED_DATA_DIR, MODEL_ARTIFACTS_DIR

def find_model_paths():
    """Find all model paths"""
    model_paths = []
    artifacts_dir = Path(MODEL_ARTIFACTS_DIR)
    
    # Locate models from the first training (folders under model_artifacts with "model_epoch_*")
    for item in artifacts_dir.glob("model_epoch_*"):
        if item.is_dir():
            model_paths.append(("First Training - " + item.name, str(item)))
    
    # Locate models from the second training (best_model folders under run_* directories)
    for run_dir in artifacts_dir.glob("run_*"):
        if run_dir.is_dir():
            best_model_path = run_dir / "best_model"
            if best_model_path.exists():
                model_paths.append((f"Second Training - {run_dir.name}", str(best_model_path)))
    
    return model_paths

def evaluate_model(model, eval_loader, device):
    """Evaluate model performance"""
    model.eval()
    correct_predictions = {'field': 0, 'tier': 0, 'subfield': 0}
    total_predictions = 0
    
    with torch.no_grad():
        for batch in tqdm(eval_loader, desc="Evaluating"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            field_labels = batch['field_label'].to(device)
            tier_labels = batch['tier_label'].to(device)
            subfield_labels = batch['subfield_label'].to(device)
            
            outputs = model(input_ids, attention_mask)
            
            field_preds = torch.argmax(outputs['field_logits'], dim=1)
            tier_preds = torch.argmax(outputs['tier_logits'], dim=1)
            subfield_preds = torch.argmax(outputs['subfield_logits'], dim=1)
            
            correct_predictions['field'] += (field_preds == field_labels).sum().item()
            correct_predictions['tier'] += (tier_preds == tier_labels).sum().item()
            correct_predictions['subfield'] += (subfield_preds == subfield_labels).sum().item()
            total_predictions += input_ids.size(0)
    
    # Calculate accuracies
    accuracies = {
        task: correct / total_predictions 
        for task, correct in correct_predictions.items()
    }
    accuracies['average'] = sum(accuracies.values()) / len(accuracies)
    
    return accuracies

def main():
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load evaluation data
    eval_data = pd.read_csv(os.path.join(PROCESSED_DATA_DIR, 'eval.csv'))
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    eval_dataset = InterviewDataset(eval_data, tokenizer)
    eval_loader = DataLoader(eval_dataset, batch_size=4)
    
    # Get all model paths
    model_paths = find_model_paths()
    print("\nFound the following models:")
    for name, path in model_paths:
        print(f"{name}: {path}")
    
    # Evaluate each model
    results = {}
    for model_name, model_path in model_paths:
        print(f"\nEvaluating model: {model_name}")
        
        try:
            # Load model
            config = AutoConfig.from_pretrained(model_path)
            model = InterviewQuestionClassifier.from_pretrained(model_path, config=config)
            model.to(device)
            
            # Evaluate model
            accuracies = evaluate_model(model, eval_loader, device)
            results[model_name] = accuracies
            
            # Print detailed results
            print("\nAccuracies:")
            for task, acc in accuracies.items():
                print(f"{task}: {acc:.4f}")
                
        except Exception as e:
            print(f"Error evaluating model {model_name}: {str(e)}")
    
    # Compare results
    print("\n=== Model Comparison ===")
    print("\nAccuracy comparison across models:")
    for model_name, accuracies in results.items():
        print(f"\n{model_name}:")
        print(f"Overall Average Accuracy: {accuracies['average']:.4f}")
        print(f"Field Accuracy: {accuracies['field']:.4f}")
        print(f"Tier Accuracy: {accuracies['tier']:.4f}")
        print(f"Subfield Accuracy: {accuracies['subfield']:.4f}")
    
    # Identify the best model
    if results:
        best_model = max(results.items(), key=lambda x: x[1]['average'])
        print(f"\nBest Model: {best_model[0]}")
        print(f"Best Average Accuracy: {best_model[1]['average']:.4f}")
    else:
        print("\nNo models were successfully evaluated")

if __name__ == "__main__":
    main()
