import streamlit as st
import PyPDF2
import requests
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title="שאל את הבוט על הפוליסה שלך - RAG", layout="wide")
st.title("שאל את הבוט על הפוליסה שלך (RAG עם טוקן מוגן)")

@st.cache_data
def extract_text_from_pdf(uploaded_file):
    reader = PyPDF2.PdfReader(uploaded_file)
    text = ""
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text + "\n"
    return text

@st.cache_resource
def chunk_text(text, max_chars=1000):
    paragraphs = text.split("\n")
    chunks = []
    current_chunk = ""
    for para in paragraphs:
        if len(current_chunk) + len(para) < max_chars:
            current_chunk += " " + para
        else:
            chunks.append(current_chunk.strip())
            current_chunk = para
    if current_chunk:
        chunks.append(current_chunk.strip())
    return chunks

@st.cache_resource
def embed_chunks(chunks):
    vectorizer = TfidfVectorizer().fit(chunks)
    embeddings = vectorizer.transform(chunks)
    return vectorizer, embeddings

def find_relevant_chunks(question, vectorizer, embeddings, chunks, top_k=3):
    question_vec = vectorizer.transform([question])
    similarities = cosine_similarity(question_vec, embeddings).flatten()
    top_indices = similarities.argsort()[-top_k:][::-1]
    return [chunks[i] for i in top_indices]

def ask_llm_with_context(context, question):
   API_URL = "https://api-inference.huggingface.co/models/HuggingFaceH4/zephyr-7b-alpha"
    headers = {
        "Authorization": f"Bearer {st.secrets['hf_token']}"
    }
    prompt = f"Based on the following context from an insurance policy, answer the question.\n\nContext:\n{context}\n\nQuestion: {question}"
    payload = {"inputs": prompt}

    response = requests.post(API_URL, headers=headers, json=payload)
    if response.status_code != 200:
        return f"שגיאה מה-LLM: {response.status_code} - {response.text}"
    try:
        return response.json()[0]['generated_text']
    except Exception as e:
        return f"שגיאה בפענוח JSON: {e}"

uploaded_file = st.file_uploader("העלה את מסמך הפוליסה שלך (PDF)", type=["pdf"])

if uploaded_file:
    with st.spinner("מטעין את הפוליסה..."):
        full_text = extract_text_from_pdf(uploaded_file)
        chunks = chunk_text(full_text)
        vectorizer, embeddings = embed_chunks(chunks)
    st.success(f"המסמך חולק ל-{len(chunks)} קטעים.")

    question = st.text_input("מה ברצונך לשאול את הבוט?")
    if question:
        relevant_chunks = find_relevant_chunks(question, vectorizer, embeddings, chunks)
        context = "\n".join(relevant_chunks)
        with st.spinner("חושב על התשובה..."):
            answer = ask_llm_with_context(context, question)
        st.subheader("תשובת הבוט:")
        st.write(answer)