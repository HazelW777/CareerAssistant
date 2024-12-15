import streamlit as st
import sys
from pathlib import Path
import os

# Add the project root directory to the Python path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from src.models.rag import RAGSystem

def initialize_session_state():
    """Initialize session state variables"""
    if 'rag_system' not in st.session_state:
        st.session_state.rag_system = RAGSystem()
    if 'conversation_history' not in st.session_state:
        st.session_state.conversation_history = []
    if 'user_background' not in st.session_state:
        st.session_state.user_background = ""

def reset_conversation():
    """Reset the conversation"""
    st.session_state.conversation_history = []
    st.session_state.user_background = ""

def main():
    st.title("Interview Assistant")
    st.write("Welcome to the Interview Assistant! Please enter your background, and I will simulate an interview for you.")

    initialize_session_state()

    # Sidebar settings
    with st.sidebar:
        st.header("Settings")
        if st.button("Restart"):
            reset_conversation()

    # If no background has been entered, display the input box for background
    if not st.session_state.user_background:
        user_background = st.text_area("Please enter your background (e.g., I am a Python backend developer with 3 years of experience, preparing for a senior developer interview.)")
        if st.button("Start Interview"):
            if user_background:
                st.session_state.user_background = user_background
                response, state = st.session_state.rag_system.chat(user_background, [])
                st.session_state.conversation_history.append(state)
                st.experimental_rerun()

    # Display conversation history
    if st.session_state.conversation_history:
        st.write("---")
        st.write("Interviewer: Please introduce yourself.")
        st.write(f"Candidate: {st.session_state.user_background}")
        
        for state in st.session_state.conversation_history[:-1]:
            st.write("---")
            st.write(f"Interviewer: {state['current_question']}")
            # Find the corresponding answer from the conversation history
            idx = st.session_state.conversation_history.index(state)
            if idx + 1 < len(st.session_state.conversation_history):
                next_state = st.session_state.conversation_history[idx + 1]
                if 'last_answer' in next_state:
                    st.write(f"Candidate: {next_state['last_answer']}")
                if 'last_evaluation' in next_state:
                    st.write("Interviewer Feedback:")
                    st.write(next_state['last_evaluation'])

        # Display the current question
        current_state = st.session_state.conversation_history[-1]
        st.write("---")
        st.write(f"Interviewer: {current_state['current_question']}")
        
        # Input box for user to enter their answer
        user_answer = st.text_area("Please enter your answer", key="current_answer")
        if st.button("Submit Answer"):
            if user_answer:
                response, new_state = st.session_state.rag_system.chat(
                    user_answer, 
                    st.session_state.conversation_history
                )
                # Save the user's answer and the evaluation result
                new_state['last_answer'] = user_answer
                new_state['last_evaluation'] = response.split("\n\nNext question:")[0]
                st.session_state.conversation_history.append(new_state)
                st.experimental_rerun()

if __name__ == "__main__":
    main()
