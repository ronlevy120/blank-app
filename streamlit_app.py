import streamlit as st
import pypdf
import requests

st.set_page_config(page_title="שאל את הבוט על הפוליסה שלך", layout="wide")
st.title("שאל את הבוט על הפוליסה שלך")

@st.cache_data
def extract_text_from_pdf(uploaded_file):
    reader = pypdf.PdfReader(uploaded_file)
    text = ""
    for page in reader.pages:
        text += page.extract_text() + "\n"
    return text

@st.cache_data
def ask_question_llm(context, question):
    API_URL = "https://api-inference.huggingface.co/models/deepset/roberta-base-squad2"
    headers = {
        "Authorization": "Bearer hf_your_huggingface_token_here"  # ניתן להסיר את השורה הזו אם אין לך טוקן
    }
    payload = {
        "context": context,
        "question": question
    }
    response = requests.post(API_URL, headers=headers, json=payload)
    return response.json().get("answer", "לא הצלחתי למצוא תשובה.")

uploaded_file = st.file_uploader("העלה את מסמך הפוליסה שלך (PDF)", type=["pdf"])

if uploaded_file:
    with st.spinner("טוען את הפוליסה..."):
        policy_text = extract_text_from_pdf(uploaded_file)
    st.success("המסמך נטען בהצלחה!")

    question = st.text_input("מה ברצונך לשאול את הבוט?")
    if question:
        with st.spinner("חושב על התשובה..."):
            answer = ask_question_llm(policy_text, question)
        st.subheader("תשובת הבוט:")
        st.write(answer)