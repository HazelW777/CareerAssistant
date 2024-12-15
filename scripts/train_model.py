import os
import sys
from pathlib import Path
import pandas as pd
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoConfig, AdamW, get_linear_schedule_with_warmup
from tqdm import tqdm
import datetime

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from src.data.dataset import InterviewDataset
from src.models.model import InterviewQuestionClassifier
from src.config import MODEL_NAME, PROCESSED_DATA_DIR, MODEL_ARTIFACTS_DIR

def train_model(model, train_loader, eval_loader, device, save_dir, num_epochs=1):
    """训练模型的主函数"""
    # 初始化优化器
    optimizer = AdamW(model.parameters(), lr=5e-5)
    
    # 创建学习率调度器
    total_steps = len(train_loader) * num_epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=0,
        num_training_steps=total_steps
    )
    
    # 损失函数
    criterion = torch.nn.CrossEntropyLoss()
    
    best_eval_accuracy = 0
    
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch + 1}/{num_epochs}")
        
        # 训练阶段
        model.train()
        train_loss = 0
        progress_bar = tqdm(train_loader, desc="Training")
        
        for batch in progress_bar:
            optimizer.zero_grad()
            
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            field_labels = batch['field_label'].to(device)
            tier_labels = batch['tier_label'].to(device)
            subfield_labels = batch['subfield_label'].to(device)
            
            outputs = model(input_ids, attention_mask)
            
            field_loss = criterion(outputs['field_logits'], field_labels)
            tier_loss = criterion(outputs['tier_logits'], tier_labels)
            subfield_loss = criterion(outputs['subfield_logits'], subfield_labels)
            
            loss = field_loss + tier_loss + subfield_loss
            train_loss += loss.item()
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            
            progress_bar.set_postfix({'loss': loss.item()})
        
        avg_train_loss = train_loss / len(train_loader)
        print(f"\nAverage training loss: {avg_train_loss:.4f}")
        
        # 评估阶段
        model.eval()
        eval_loss = 0
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
        
        # 打印评估结果
        print("\nEvaluation Results:")
        total_accuracy = 0
        for task in correct_predictions:
            accuracy = correct_predictions[task] / total_predictions
            print(f"{task} Accuracy: {accuracy:.4f}")
            total_accuracy += accuracy
        
        avg_accuracy = total_accuracy / len(correct_predictions)
        
        # 保存最佳模型
        if avg_accuracy > best_eval_accuracy:
            best_eval_accuracy = avg_accuracy
            best_model_path = os.path.join(save_dir, 'best_model')
            os.makedirs(best_model_path, exist_ok=True)
            model.save_pretrained(best_model_path)
            print(f"\nSaved best model to {best_model_path}")
        
        # 保存当前epoch的模型
        checkpoint_path = os.path.join(save_dir, f'model_epoch_{epoch + 1}')
        os.makedirs(checkpoint_path, exist_ok=True)
        model.save_pretrained(checkpoint_path)
        print(f"Saved model checkpoint to {checkpoint_path}")

def main():
    try:
        # 设置设备
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {device}")
        
        # 加载数据
        print("\nLoading data...")
        train_data = pd.read_csv(os.path.join(PROCESSED_DATA_DIR, 'train.csv'))
        eval_data = pd.read_csv(os.path.join(PROCESSED_DATA_DIR, 'eval.csv'))
        
        print(f"Training data size: {len(train_data)}")
        print(f"Evaluation data size: {len(eval_data)}")
        
        # 初始化tokenizer
        print("\nInitializing tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        
        # 创建数据集
        print("Creating datasets...")
        train_dataset = InterviewDataset(train_data, tokenizer)
        eval_dataset = InterviewDataset(eval_data, tokenizer)
        
        # 创建数据加载器
        train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
        eval_loader = DataLoader(eval_dataset, batch_size=4)
        
        # 获取标签数量
        num_fields = len(train_data['field'].unique())
        num_tiers = len(train_data['tier'].unique())
        num_subfields = len(train_data['subfield'].unique())
        
        print(f"\nNumber of unique fields: {num_fields}")
        print(f"Number of unique tiers: {num_tiers}")
        print(f"Number of unique subfields: {num_subfields}")
        
        # 初始化模型配置
        print("\nInitializing model...")
        config = AutoConfig.from_pretrained(MODEL_NAME)
        config.field_size = num_fields
        config.tier_size = num_tiers
        config.subfield_size = num_subfields
        
        # 初始化模型
        model = InterviewQuestionClassifier.from_pretrained(MODEL_NAME, config=config)
        model.to(device)
        
        # 设置保存路径
        timestamp = datetime.datetime.now().strftime("%m%d_%H%M")
        model_save_dir = os.path.join(MODEL_ARTIFACTS_DIR, f'run_{timestamp}')
        print(f"\n模型将保存到: {model_save_dir}")
        
        # 训练模型
        train_model(model, train_loader, eval_loader, device, save_dir=model_save_dir)
        
        print("\nTraining completed successfully!")
        
    except Exception as e:
        print(f"\nError during training: {str(e)}")
        raise

if __name__ == "__main__":
    main()