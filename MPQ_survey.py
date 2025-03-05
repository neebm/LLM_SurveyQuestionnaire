# model_name = "distilbert-base-uncased"  # or "bert-base-uncased"
# tokenizer = AutoTokenizer.from_pretrained(model_name)
# model = AutoModel.from_pretrained(model_name)

pip install transformers sentence-transformers faiss-cpu streamlit


import streamlit as st
import torch
from transformers import AutoModel, AutoTokenizer
from sentence_transformers import SentenceTransformer, util
import faiss
import numpy as np

# Load DistilBERT-based embedding model
embedding_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# Sample MPQ questions
mpq_questions = [
    "Describe your pain in your own words.",
    "Is the pain throbbing, shooting, stabbing, or burning?",
    "How intense is your pain on a scale of 1-10?",
    "Does the pain get worse with movement or at rest?",
    "How does the pain affect your emotions?",
]

# Predefined answers to common patient questions
faq = {
    "What is this questionnaire for?": "The McGill Pain Questionnaire helps doctors assess the nature of your pain.",
    "What does 'throbbing' mean?": "Throbbing pain is like a steady or pulsing ache.",
    "Can I skip a question?": "Yes, but answering all questions helps provide a full picture of your pain.",
}

# Encode FAQ for similarity search
faq_questions = list(faq.keys())
faq_embeddings = embedding_model.encode(faq_questions, convert_to_tensor=True)
index = faiss.IndexFlatL2(faq_embeddings.shape[1])
index.add(faq_embeddings.cpu().numpy())

# Initialize session state
if "step" not in st.session_state:
    st.session_state["step"] = 0
if "responses" not in st.session_state:
    st.session_state["responses"] = []

# Streamlit UI
st.title("McGill Pain Questionnaire Facilitator")

# Display current question
if st.session_state["step"] < len(mpq_questions):
    question = mpq_questions[st.session_state["step"]]
    st.subheader(question)

    user_input = st.text_input("Your response:", "")

    if st.button("Submit"):
        if user_input.strip():
            # Check if user asked a question
            input_embedding = embedding_model.encode(user_input, convert_to_tensor=True)
            D, I = index.search(input_embedding.cpu().numpy().reshape(1, -1), 1)
            if D[0][0] < 0.5:  # Threshold for similarity
                st.write(f"🤖 {faq[faq_questions[I[0][0]]]}")  # Answer detected question
            else:
                st.session_state["responses"].append((question, user_input))
                st.session_state["step"] += 1

# Show summary when complete
if st.session_state["step"] >= len(mpq_questions):
    st.subheader("Summary of Your Responses")
    for q, r in st.session_state["responses"]:
        st.write(f"**{q}**\n{r}\n")
    st.success("Thank you for completing the questionnaire!")

