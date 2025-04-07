import streamlit as st
import PyPDF2
import requests

st.set_page_config(page_title="שאל את הבוט על הפוליסה שלך", layout="wide")
st.title("שאל את הבוט על הפוליסה שלך")

@st.cache_data
def extract_text_from_pdf(uploaded_file):
    reader = PyPDF2.PdfReader(uploaded_file)
    text = ""
    for page in reader.pages:
        text += page.extract_text() + "\n"
    return text

@st.cache_data
def ask_question_llm(context, question):
    API_URL = "https://api-inference.huggingface.co/models/google/flan-t5-base"
    headers = {
        # ניתן להוסיף טוקן של Hugging Face כאן אם יש צורך
        # "Authorization": "Bearer hf_xxx"
    }
    prompt = f"Answer the following question based on the policy text:\nQuestion: {question}\nPolicy: {context}"
    payload = {"inputs": prompt}

    response = requests.post(API_URL, headers=headers, json=payload)

    if response.status_code != 200:
        st.error(f"שגיאה בתשובה מה-LLM: {response.status_code} - {response.text}")
        return "לא הצלחתי להבין את התשובה מהמודל."

    try:
        return response.json()[0]['generated_text']
    except Exception as e:
        st.error(f"שגיאה בפענוח JSON: {e}")
        return "הייתה שגיאה בתהליך הפענוח."

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