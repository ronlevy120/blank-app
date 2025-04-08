import streamlit as st
import PyPDF2
import requests
from sklearn.feature_extraction.text
import TfidfVectorizer
from sklearn.metrics.pairwise
import cosine_similarity

st.set_page_config(page_title="שאל את הבוט על הפוליסה שלך - RAG", layout="wide") st.title("שאל את הבוט על הפוליסה שלך (RAG עם Zephyr + תרגום + פולו-אפ)")

@st.cache_data def extract_text_from_pdf(uploaded_file): reader = PyPDF2.PdfReader(uploaded_file) text = "" for page in reader.pages: page_text = page.extract_text() if page_text: text += page_text + "\n" return text

@st.cache_resource def chunk_text(text, max_chars=1000): paragraphs = text.split("\n") chunks = [] current_chunk = "" for para in paragraphs: if len(current_chunk) + len(para) < max_chars: current_chunk += " " + para else: chunks.append(current_chunk.strip()) current_chunk = para if current_chunk: chunks.append(current_chunk.strip()) return chunks

@st.cache_resource def embed_chunks(chunks): vectorizer = TfidfVectorizer().fit(chunks) embeddings = vectorizer.transform(chunks) return vectorizer, embeddings

def find_relevant_chunks(question, vectorizer, embeddings, chunks, top_k=3): question_vec = vectorizer.transform([question]) similarities = cosine_similarity(question_vec, embeddings).flatten() top_indices = similarities.argsort()[-top_k:][::-1] return [chunks[i] for i in top_indices]

def translate_hebrew_to_english(text): API_URL = "https://api-inference.huggingface.co/models/Helsinki-NLP/opus-mt-he-en" headers = { "Authorization": f"Bearer {st.secrets['hf_token']}" } payload = {"inputs": text} response = requests.post(API_URL, headers=headers, json=payload) if response.status_code == 200: return response.json()[0]['translation_text'] return text

def translate_english_to_hebrew(text): API_URL = "https://api-inference.huggingface.co/models/Helsinki-NLP/opus-mt-en-he" headers = { "Authorization": f"Bearer {st.secrets['hf_token']}" } payload = {"inputs": text} response = requests.post(API_URL, headers=headers, json=payload) if response.status_code == 200: return response.json()[0]['translation_text'] return text

def ask_llm_with_context(context, question, chat_history): API_URL = "https://api-inference.huggingface.co/models/HuggingFaceH4/zephyr-7b-alpha" headers = { "Authorization": f"Bearer {st.secrets['hf_token']}" } prompt = f"You are a helpful insurance assistant.\nYour task is to answer user questions based on the provided context.\nBe concise, yet make sure to include all relevant details from the context.\nIf the question is about coverages, list them clearly and number them (1, 2, 3...) so the user can follow up.\nYou must not return the full context in your answer, only the distilled answer.\n\nPrevious questions and answers:\n{chat_history}\n\nContext:\n{context}\n\nQuestion: {question}" payload = {"inputs": prompt}

response = requests.post(API_URL, headers=headers, json=payload)
if response.status_code != 200:
    return f"שגיאה מה-LLM: {response.status_code} - {response.text}"
try:
    return response.json()[0]['generated_text']
except Exception as e:
    return f"שגיאה בפענוח JSON: {e}"

uploaded_file = st.file_uploader("העלה את מסמך הפוליסה שלך (PDF)", type=["pdf"])

if uploaded_file: with st.spinner("מטעין את הפוליסה..."): full_text = extract_text_from_pdf(uploaded_file) chunks = chunk_text(full_text) vectorizer, embeddings = embed_chunks(chunks) st.success(f"המסמך חולק ל-{len(chunks)} קטעים.")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = ""

question = st.text_input("מה ברצונך לשאול את הבוט?")
if question:
    translated_question = translate_hebrew_to_english(question)
    relevant_chunks = find_relevant_chunks(translated_question, vectorizer, embeddings, chunks)
    context = "\n".join(relevant_chunks)
    with st.spinner("חושב על התשובה..."):
        english_answer = ask_llm_with_context(context, translated_question, st.session_state.chat_history)
        hebrew_answer = translate_english_to_hebrew(english_answer)
    st.session_state.chat_history += f"Question: {translated_question}\nAnswer: {english_answer}\n"
    st.subheader("תשובת הבוט:")
    st.write(hebrew_answer)

