# import os
# from typing import Dict, List, Tuple
# import openai
# from ..data.vectordb import VectorDB
# from ..config import OPENAI_API_KEY, MODEL_NAME, MODEL_ARTIFACTS_DIR
# from openai import OpenAI
# import torch
# from transformers import AutoTokenizer, AutoConfig
# from ..models.model import InterviewQuestionClassifier

# class RAGSystem:
#     def __init__(self):
#         # Initialize the vector database and OpenAI client
#         self.vector_db = VectorDB()
#         self.client = OpenAI(api_key=OPENAI_API_KEY)
    
#         # Load fine-tuned model
#         best_model_path = os.path.join(MODEL_ARTIFACTS_DIR, "model_epoch_3")
    
#         # Load configuration
#         config = AutoConfig.from_pretrained(best_model_path)
        
#         # Ensure necessary parameters are included in the configuration
#         if not hasattr(config, 'field_size'):
#             config.field_size = 10  # Set based on your classification categories
#         if not hasattr(config, 'tier_size'):
#             config.tier_size = 3
#         if not hasattr(config, 'subfield_size'):
#             config.subfield_size = 10
    
#         # Load the classifier model
#         self.classifier = InterviewQuestionClassifier.from_pretrained(best_model_path, config=config)
#         self.classifier.eval()
#         self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    
#         self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#         self.classifier.to(self.device)

# def classify_background(self, user_background: str):
#     """Classify the user's background using the fine-tuned model"""
#     inputs = self.tokenizer(
#         user_background,
#         return_tensors="pt",
#         padding=True,
#         truncation=True,
#         max_length=512
#     ).to(self.device)
    
#     with torch.no_grad():
#         outputs = self.classifier(**inputs)
#         field_pred = torch.argmax(outputs['field_logits'], dim=1)
#         tier_pred = torch.argmax(outputs['tier_logits'], dim=1)
#         subfield_pred = torch.argmax(outputs['subfield_logits'], dim=1)
    
#     return {
#         'field': field_pred.item(),
#         'tier': tier_pred.item(),
#         'subfield': subfield_pred.item()
#     }

# def _create_context(self, similar_questions: List[Dict]) -> str:
#     """Create a context string based on similar questions"""
#     context = "Relevant interview questions and answers:\n\n"
#     for idx, q in enumerate(similar_questions, 1):
#         metadata = q['metadata']
#         context += f"{idx}. Field: {metadata['field']}\n"
#         context += f"   Difficulty: {metadata['tier']}\n"
#         context += f"   Question: {metadata['question']}\n"
#         context += f"   Reference Answer: {metadata['answer']}\n\n"
#     return context

# def get_next_question(self, user_background: str) -> Dict:
#     """Get the next question based on the user's background"""
#     # Classify the user's background
#     classifications = self.classify_background(user_background)
    
#     # Search for similar questions
#     similar_questions = self.vector_db.search_similar_questions(user_background, top_k=3)
    
#     # Create a prompt using the classification results and similar questions
#     prompt = self._get_question_prompt(user_background, similar_questions, classifications)
    
#     # Generate the question using OpenAI API
#     response = self.client.chat.completions.create(
#         model="gpt-3.5-turbo",
#         messages=[
#             {"role": "system", "content": "You are a professional technical interviewer."},
#             {"role": "user", "content": prompt}
#         ],
#         temperature=0.7
#     )
    
#     selected_question = response.choices[0].message.content
#     reference_answer = similar_questions[0]['metadata']['answer']
    
#     return {
#         "question": selected_question,
#         "reference_answer": reference_answer
#     }

# def _get_question_prompt(self, user_background: str, similar_questions: List[Dict], classifications: Dict) -> str:
#     """Enhance the question generation prompt"""
#     context = self._create_context(similar_questions)
#     prompt = f"""As an interviewer, please create or select an appropriate interview question based on the candidate's background and similar questions.

# Candidate Background: {user_background}

# System Analysis:
# - Technical Field: Level {classifications['field']}
# - Interview Difficulty: Level {classifications['tier']}
# - Specialization: Level {classifications['subfield']}

# Reference Questions:
# {context}

# Please generate or select the most suitable question based on:
# 1. The candidate's background and experience level.
# 2. Relevance to the candidate's technical field.
# 3. Alignment with the candidate's specialization.

# Only output the question itself, without any additional explanation."""
#     return prompt

# def evaluate_answer(self, question: str, user_answer: str, reference_answer: str) -> str:
#     """Evaluate the user's answer"""
#     response = self.client.chat.completions.create(
#         model="gpt-3.5-turbo",
#         messages=[
#             {"role": "system", "content": "You are a professional technical interviewer."},
#             {"role": "user", "content": self._get_evaluation_prompt(
#                 question, user_answer, reference_answer)}
#         ],
#         temperature=0.7
#     )
    
#     return response.choices[0].message.content

import os
from typing import Dict, List, Tuple
import openai
from ..data.vectordb import VectorDB
from ..config import OPENAI_API_KEY, MODEL_NAME, MODEL_ARTIFACTS_DIR
from openai import OpenAI
import torch
from transformers import AutoTokenizer, AutoConfig
from ..models.model import InterviewQuestionClassifier

class RAGSystem:
    def __init__(self):
        # Initialize the vector database and OpenAI client
        self.vector_db = VectorDB()
        self.client = OpenAI(api_key=OPENAI_API_KEY)
    
        # Load fine-tuned model
        best_model_path = os.path.join(MODEL_ARTIFACTS_DIR, "model_epoch_3")
    
        # Load configuration
        config = AutoConfig.from_pretrained(best_model_path)
        
        # Ensure necessary parameters are included in the configuration
        if not hasattr(config, 'field_size'):
            config.field_size = 10  # Set based on your classification categories
        if not hasattr(config, 'tier_size'):
            config.tier_size = 3
        if not hasattr(config, 'subfield_size'):
            config.subfield_size = 10
    
        # Load the classifier model
        self.classifier = InterviewQuestionClassifier.from_pretrained(best_model_path, config=config)
        self.classifier.eval()
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.classifier.to(self.device)

    def classify_background(self, user_background: str):
        """Classify the user's background using the fine-tuned model"""
        inputs = self.tokenizer(
            user_background,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512
        ).to(self.device)
        
        with torch.no_grad():
            outputs = self.classifier(**inputs)
            field_pred = torch.argmax(outputs['field_logits'], dim=1)
            tier_pred = torch.argmax(outputs['tier_logits'], dim=1)
            subfield_pred = torch.argmax(outputs['subfield_logits'], dim=1)
        
        return {
            'field': field_pred.item(),
            'tier': tier_pred.item(),
            'subfield': subfield_pred.item()
        }

    def _create_context(self, similar_questions: List[Dict]) -> str:
        """Create a context string based on similar questions"""
        context = "Relevant interview questions and answers:\n\n"
        for idx, q in enumerate(similar_questions, 1):
            metadata = q['metadata']
            context += f"{idx}. Field: {metadata['field']}\n"
            context += f"   Difficulty: {metadata['tier']}\n"
            context += f"   Question: {metadata['question']}\n"
            context += f"   Reference Answer: {metadata['answer']}\n\n"
        return context

    def get_next_question(self, user_background: str) -> Dict:
        """Get the next question based on the user's background"""
        # Classify the user's background
        classifications = self.classify_background(user_background)
        
        # Search for similar questions
        similar_questions = self.vector_db.search_similar_questions(user_background, top_k=3)
        
        # Create a prompt using the classification results and similar questions
        prompt = self._get_question_prompt(user_background, similar_questions, classifications)
        
        # Generate the question using OpenAI API
        response = self.client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a professional technical interviewer."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7
        )
        
        selected_question = response.choices[0].message.content
        reference_answer = similar_questions[0]['metadata']['answer']
        
        return {
            "question": selected_question,
            "reference_answer": reference_answer
        }

    def _get_question_prompt(self, user_background: str, similar_questions: List[Dict], classifications: Dict) -> str:
        """Enhance the question generation prompt"""
        context = self._create_context(similar_questions)
        prompt = f"""As an interviewer, please create or select an appropriate interview question based on the candidate's background and similar questions.

Candidate Background: {user_background}

System Analysis:
- Technical Field: Level {classifications['field']}
- Interview Difficulty: Level {classifications['tier']}
- Specialization: Level {classifications['subfield']}

Reference Questions:
{context}

Please generate or select the most suitable question based on:
1. The candidate's background and experience level.
2. Relevance to the candidate's technical field.
3. Alignment with the candidate's specialization.

Only output the question itself, without any additional explanation."""
        return prompt

    def evaluate_answer(self, question: str, user_answer: str, reference_answer: str) -> str:
        """Evaluate the user's answer"""
        response = self.client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a professional technical interviewer."},
                {"role": "user", "content": self._get_evaluation_prompt(
                    question, user_answer, reference_answer)}
            ],
            temperature=0.7
        )
        
        return response.choices[0].message.content

    def _get_evaluation_prompt(self, question: str, user_answer: str, reference_answer: str) -> str:
        """Generate the evaluation prompt"""
        prompt = f"""As an interviewer, please evaluate the candidate's answer.

Interview Question: {question}

Candidate's Answer: {user_answer}

Reference Answer: {reference_answer}

Please provide detailed feedback including:
1. Strengths of the answer
2. Areas for improvement
3. Additional knowledge points to consider
4. Overall score (1-5)

Please provide professional and constructive feedback."""
        return prompt

    def chat(self, user_input: str, conversation_history: List[Dict]) -> Tuple[str, Dict]:
        """Process user input and maintain conversation state"""
        if not conversation_history:
            # Get first question
            question_data = self.get_next_question(user_input)
            return question_data["question"], {
                "state": "awaiting_answer",
                "current_question": question_data["question"],
                "reference_answer": question_data["reference_answer"],
                "user_background": user_input
            }
        else:
            current_state = conversation_history[-1]
            if current_state["state"] == "awaiting_answer":
                evaluation = self.evaluate_answer(
                    current_state["current_question"],
                    user_input,
                    current_state["reference_answer"]
                )
                
                question_data = self.get_next_question(current_state["user_background"])
                return f"{evaluation}\n\nNext question: {question_data['question']}", {
                    "state": "awaiting_answer",
                    "current_question": question_data["question"],
                    "reference_answer": question_data["reference_answer"],
                    "user_background": current_state["user_background"]
                }
            
            return "I don't understand your input.", current_state
        