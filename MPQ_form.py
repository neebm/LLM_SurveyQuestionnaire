import streamlit as st 
import torch
from sentence_transformers import SentenceTransformer, util
import faiss
import numpy as np
import re
import pandas as pd
import os

#---------------------------------------------set up-----------------------------------
torch.classes.__path__ = []

embedding_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

CSV_FILE = "/home/bhux/mikayla/ProjectLLM-hlth/LLM_survey/patient_data.csv"
LOGIN_FILE = "/home/bhux/mikayla/ProjectLLM-hlth/LLM_survey/Patient_login_data.csv"

#-------------------------------------load logins ------------------------------------
def load_login_data():
    if os.path.exists(LOGIN_FILE):
        with open(LOGIN_FILE, "r") as file:
            lines = file.readlines()[1:]  # Skip header
        logins = {}
        for line in lines:
            parts = line.strip().split(", ")
            if len(parts) == 3:
                name, patient_id, password = parts
                logins[patient_id] = {"name": name, "password": password}
        return logins
    return {}

login_data = load_login_data()

#------------------------load patient data--------------------------------------------
if os.path.exists(CSV_FILE):
    data = pd.read_csv(CSV_FILE)
else:
    # Define structure if file doesn't exist
    data = pd.DataFrame(columns=["Patient_ID", "Response_1", "Response_2", "Response_3","Response_4","Response_5","Response_6","Response_7","Response_8","Response_9"])
    data.to_csv(CSV_FILE, index=False)
#--------------------------------------------------------------------------------------
if "page" not in st.session_state:
    st.session_state["page"] = "login"  # Start on login page

if "logged_in" not in st.session_state:
    st.session_state["logged_in"] = False

# --------------------------- LOGIN PAGE ----------------------------------------------
def verify_login(patient_id, password):
    if patient_id in login_data and login_data[patient_id]["password"] == password:
        return True
    return False

def get_next_patient_id():
    if login_data:
        existing_ids = sorted([int(pid) for pid in login_data.keys()])
        return str(existing_ids[-1] + 1).zfill(2)  # Ensures two-digit format
    return "01"

def register_user(name, password):
    patient_id = get_next_patient_id()
    with open(LOGIN_FILE, "a") as file:
        file.write(f"\n{name}, {patient_id}, {password}")
    return patient_id


def login_page():
    st.title("Patient Login")
    patient_id = st.text_input("Enter your Patient ID:")
    password = st.text_input("Enter your password:", type="password")
    if st.button("Login"):
        if verify_login(patient_id, password):
            st.session_state["patient_id"] = patient_id
            st.session_state["patient_name"] = login_data[patient_id]["name"]
            st.session_state["logged_in"] = True
            st.session_state["page"] = "questionnaire"
            st.rerun()
        else:
            st.error("Invalid Patient ID or Password")
    
    if st.button("Forgot Password?"):
        st.info("Please contact support to reset your password.")

    if st.button("I don't have a Patient ID"):
        st.session_state["page"] = "register"
        st.rerun()

# ---- Registration Page ----
def registration_page():
    st.title("Register New Account")
    name = st.text_input("Enter your full name:")
    password = st.text_input("Create a password:", type="password")
    if st.button("Register"):
        if name and password:
            patient_id = register_user(name, password)
            st.success(f"Registration successful! Your Patient ID is {patient_id}. Please return to the login page.")
        else:
            st.error("All fields are required.")
    if st.button("Back to Login"):
        st.session_state["page"] = "login"
        st.rerun()

#--------------------------Questionnaire page-------------
def questionnaire_page():
    st.title("McGill Pain Questionnaire")

    # Logout button
    sideb = st.sidebar
    if sideb.button("Logout"):
        st.session_state["logged_in"] = False
        st.session_state["page"] = "login"
        st.rerun()

# Sample MPQ questions
    mpq_questions = [
        "Describe your pain in your own words. Include as much detail as you can.",
        "How would you describe the temporality of the pain? Do any of the following apply: flickering, quivering, pulsing, throbbing, beating, pouding",
        "How would you desribe the spatiality of the pain? Does the pain move around or remain still?",
        "How would you describe the pressure of the pain? Do any of the following apply: boring, cutting, pinching, tugging...",
        "How does your pain change with time?",
        "Does the pain get worse with movement or at rest?",
        "How intense is your pain on a scale of 1-10?",
        "How does the pain affect your emotions?", 
        "Do any of the following increase or decrease your pain: liqor, stimulants (coffee), eating, heat, cold, damp, weather changes, massage, pressure, distraction, tension"
    ]

    # Predefined answers to common patient questions
    faq = {
        "What is this questionnaire for": "The McGill Pain Questionnaire helps doctors assess the nature of your pain.",
        "What does 'throbbing' mean": "Throbbing pain is like a steady or pulsing ache.",
        "What does 'flickering' pain mean" : "Flickering pain can refer to muscle spasms which are sudden involuntary muscle contractions that can be painful",
        "What does 'quivering' pain mean" : "Quivering pain can refer to rapid, small repetetive pulses similar to muscle twitches",
        "What does temporality of pain mean?" : "Temporality refers to the duration, onset and fluctuation of pain over time",
        "What does spatiality of pain mean?" : "Spatiality refers to the area and location of the pain and how it may change",
        "Can I skip a question": "Yes, but answering all questions helps provide a full picture of your pain.",
    }

    # Encode FAQ for similarity search
    faq_questions = list(faq.keys())
    faq_embeddings = embedding_model.encode(faq_questions, convert_to_tensor=True)
    index = faiss.IndexFlatL2(faq_embeddings.shape[1])
    index.add(faq_embeddings.cpu().numpy())

    # Common question words
    QUESTION_WORDS = {"who", "what", "where", "when", "why", "how", "can", "does", "is", "are", "do", "should", "would"}

    # Function to check if input is a question
    def is_question(user_input):
        user_input = user_input.lower().strip()

        # Rule 1: Ends with a question mark
        if user_input.endswith("?"):
            return True

        # Rule 2: Starts with a common question word
        first_word = user_input.split(" ")[0]
        if first_word in QUESTION_WORDS:
            return True

        # Rule 3: Check similarity with known question-like structures
        input_embedding = embedding_model.encode(user_input, convert_to_tensor=True)
        D, I = index.search(input_embedding.cpu().numpy().reshape(1, -1), 1)
        if D[0][0] < 0.5:  # If it matches an FAQ
            return True

        return False  # Otherwise, assume it's an answer

    # Initialize session state
    if "step" not in st.session_state:
        st.session_state["step"] = 0
    if "responses" not in st.session_state:
        st.session_state["responses"] = {}
    if "chat_history" not in st.session_state:
        st.session_state["chat_history"] = []

    # Check if questionnaire is completed
    if st.session_state["step"] < len(mpq_questions):
        question = mpq_questions[st.session_state["step"]]
        st.subheader(question)

        # Display chat history
        for msg in st.session_state["chat_history"]:
            st.chat_message(msg["role"]).write(msg["content"])

        # User input
        user_input = st.chat_input("Type your response or ask a question...")

        if user_input:
            # Store user input in chat history
            st.session_state["chat_history"].append({"role": "user", "content": user_input})

            # Detect if it's a question
            if is_question(user_input):
                # Try to find an answer in the FAQ
                input_embedding = embedding_model.encode(user_input, convert_to_tensor=True)
                D, I = index.search(input_embedding.cpu().numpy().reshape(1, -1), 1)

                if D[0][0] < 0.5:  # If it matches an FAQ
                    answer = faq[faq_questions[I[0][0]]]
                else:
                    answer = "I'm not sure about that, but I can still help with your pain assessment."

                # Respond to the question
                st.session_state["chat_history"].append({"role": "bot", "content": answer})
                st.chat_message("bot").write(answer)

            else:
                # Save response and move to next question
                st.session_state["responses"][question] = user_input
                st.session_state["step"] += 1
                st.session_state["chat_history"] = []  # Clear chat history for next question
                st.rerun()

    if "patient_id" in st.session_state:
        patient_id = st.session_state["patient_id"]

        if st.session_state["step"] >= len(mpq_questions):
            st.subheader("Summary of Your Responses")

            responses_list = []
            for q, r in st.session_state["responses"].items():
                st.write(f"**{q}**\n{r}\n")
                responses_list.append({"Patient_ID": patient_id, "Question": q, "Response": r})

            st.success("Thank you for completing the questionnaire!")

             # Convert responses into a DataFrame row
            new_row = {"Patient_ID": patient_id}
            new_row.update(st.session_state["responses"])

            # Append new row to DataFrame
            df = pd.read_csv(CSV_FILE)
            df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
            df.loc[len(df)] = new_row

            # Save to CSV
            df.to_csv(CSV_FILE, index=False)

    else:
        st.error("You must log in before completing the questionnaire.")

#----------------------------------navigation--------------------------
if st.session_state["page"] == "login":
    login_page()
elif st.session_state["page"] == "register":
    registration_page()
elif st.session_state["logged_in"]:
    questionnaire_page()
else:
    st.session_state["page"] = "login"
    st.rerun()
