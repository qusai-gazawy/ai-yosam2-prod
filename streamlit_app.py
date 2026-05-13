import streamlit as st
import os
import PIL.Image
import google.generativeai as genai
from openai import OpenAI
import base64
import json
import fitz  # PyMuPDF
from google.oauth2 import service_account

# --- PAGE CONFIG ---
st.set_page_config(page_title="Ortho-AI Research Lab", layout="wide")

# --- 1. SECRETS & CLIENTS ---
# In the Cloud, use st.secrets instead of keys.json
try:
    OPENAI_API_KEY = st.secrets["openai_api_key"]
    DEEPSEEK_API_KEY = st.secrets["deepseek_api_key"]
    # For Gemini Service Account (Paste the JSON content into Streamlit Secrets)
    sa_info = json.loads(st.secrets["google_service_account"])
    credentials = service_account.Credentials.from_service_account_info(sa_info)
    genai.configure(credentials=credentials)
except Exception as e:
    st.error("Missing Secrets! Ensure openai_api_key, deepseek_api_key, and google_service_account are set.")

client_openai = OpenAI(api_key=OPENAI_API_KEY)
client_deepseek = OpenAI(api_key=DEEPSEEK_API_KEY, base_url="https://api.deepseek.com")

# --- 2. HELPER FUNCTIONS ---
def extract_pdf_context(pdf_path):
    doc = fitz.open(pdf_path)
    return "".join([page.get_text() for page in doc])

def get_chunks(text, chunk_size=2000):
    return [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]

def retrieve_relevant_context(query, chunks, top_k=5):
    keywords = query.lower().split()
    scored_chunks = []
    for chunk in chunks:
        score = sum(1 for word in keywords if word in chunk.lower())
        scored_chunks.append((score, chunk))
    scored_chunks.sort(key=lambda x: x[0], reverse=True)
    return "\n---\n".join([c[1] for c in scored_chunks[:top_k]])

def encode_image_base64(image_bytes):
    return base64.b64encode(image_bytes).decode('utf-8')

# --- 3. PROMPTS ---
system_prompt = """You are an expert AI Radiologist... [Insert your full system_prompt here]"""
groundedness_rater_prompt = """You are an evaluation assistant... [Insert your full groundedness_rater_prompt here]"""
relevance_rater_prompt = """You are an expert orthopedic radiology evaluator... [Insert your full relevance_rater_prompt here]"""

# --- 4. APP INTERFACE ---
st.title("🏥 YO-SAM: Medical Image Interpretation")
st.sidebar.header("Upload Data")

uploaded_pdf = st.sidebar.file_uploader("Clinical Manual (PDF)", type="pdf")
uploaded_img = st.sidebar.file_uploader("Knee X-Ray (JPG/PNG)", type=["jpg", "png", "jpeg"])

if uploaded_pdf and uploaded_img:
    # Process PDF once
    with open("temp_manual.pdf", "wb") as f:
        f.write(uploaded_pdf.getbuffer())
    
    full_text = extract_pdf_context("temp_manual.pdf")
    text_chunks = get_chunks(full_text)
    
    st.image(uploaded_img, caption="Target X-Ray", width=400)
    
    if st.button("Generate & Audit Analysis"):
        col1, col2 = st.columns(2)
        
        # RAG Context
        relevant_context = retrieve_relevant_context("orthopedic implant knee", text_chunks)
        img_pil = PIL.Image.open(uploaded_img)
        img_base64 = encode_image_base64(uploaded_img.getvalue())

        with st.spinner("Analyzing with GPT and Gemini..."):
            # GPT Analysis (Note: using gpt-4o as gpt-5.2 doesn't exist yet)
            res_oa = client_openai.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": [
                        {"type": "text", "text": "Analyze this image."},
                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_base64}"}}
                    ]}
                ]
            )
            oa_ans = res_oa.choices[0].message.content

            # Gemini Analysis (Note: using gemini-1.5-flash)
            model_gemini = genai.GenerativeModel(model_name='gemini-1.5-flash', system_instruction=system_prompt)
            res_gem = model_gemini.generate_content(["Analyze this image.", img_pil])
            gem_ans = res_gem.text

        # Display Results
        with col1:
            st.subheader("GPT-4o Report")
            st.info(oa_ans)
        with col2:
            st.subheader("Gemini 1.5 Report")
            st.success(gem_ans)

        # --- DEEPSEEK AUDIT ---
        st.divider()
        st.header("🧠 DeepSeek Reasoner Audit")
        
        for name, ans in [("GPT-4o", oa_ans), ("Gemini 1.5", gem_ans)]:
            with st.expander(f"Audit for {name}"):
                audit_query = f"CONTEXT: {relevant_context}\n\nANSWER: {ans}\n\n{groundedness_rater_prompt}"
                audit_res = client_deepseek.chat.completions.create(
                    model="deepseek-reasoner",
                    messages=[{"role": "user", "content": audit_query}]
                )
                st.write(audit_res.choices[0].message.content)
else:
    st.warning("Please upload both the Clinical Manual and an X-ray image in the sidebar.")
