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
st.set_page_config(page_title="AI-YOSAM2 Research Lab", layout="wide")

# --- 1. SECRETS & CLIENTS ---
OPENAI_API_KEY = None
DEEPSEEK_API_KEY = None

try:
    OPENAI_API_KEY = st.secrets["openai_api_key"]
    DEEPSEEK_API_KEY = st.secrets["deepseek_api_key"]
    sa_info = json.loads(st.secrets["google_service_account"])
    credentials = service_account.Credentials.from_service_account_info(sa_info)
    genai.configure(credentials=credentials)
    
    client_openai = OpenAI(api_key=OPENAI_API_KEY)
    client_deepseek = OpenAI(api_key=DEEPSEEK_API_KEY, base_url="https://api.deepseek.com")
except Exception as e:
    st.error(f"Configuration Error: {e}")
    st.stop()

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
user_prompt_text = "Describe what you observe in the provided X-Ray image according to the provided context."

system_prompt = """You are an expert AI Radiologist specializing in Musculoskeletal (MSK) Imaging... [Your Full System Prompt]"""

groundedness_rater_prompt = """You are an evaluation assistant. Rate GROUNDEDNESS... [Your Full Groundedness Prompt]"""

relevance_rater_prompt = """You are an expert orthopedic radiology evaluator. Rate QUALITY... [Your Full Relevance Prompt]"""

# --- 4. APP INTERFACE ---
st.title("🏥 AI-YOSAM2: Multi-Method Evaluation")
st.sidebar.header("Upload Research Data")

uploaded_pdf = st.sidebar.file_uploader("Clinical Manual (PDF)", type="pdf")
uploaded_img = st.sidebar.file_uploader("Knee X-Ray", type=["jpg", "png", "jpeg"])

if uploaded_pdf and uploaded_img:
    with open("temp_manual.pdf", "wb") as f:
        f.write(uploaded_pdf.getbuffer())
    
    full_text = extract_pdf_context("temp_manual.pdf")
    text_chunks = get_chunks(full_text)
    relevant_context = retrieve_relevant_context("orthopedic implant knee", text_chunks)
    
    img_pil = PIL.Image.open(uploaded_img)
    img_b64 = encode_image_base64(uploaded_img.getvalue())

    if st.button("🚀 Run Triple-Method Comparison"):
        results = [] # To store (MethodName, ModelName, OutputText)

        # DEFINING THE 3 METHODS
        methods = [
            {"name": "1. Basic User Prompt", "sys": "", "user_ext": ""},
            {"name": "2. System Prompt Only", "sys": system_prompt, "user_ext": ""},
            {"name": "3. RAG (Context + System)", "sys": system_prompt, "user_ext": f"\nCONTEXT:\n{relevant_context}"}
        ]

        progress_bar = st.progress(0)
        
        for idx, m in enumerate(methods):
            st.subheader(f"Method: {m['name']}")
            col1, col2 = st.columns(2)
            
            # --- GPT Execution ---
            with col1:
                st.write("**GPT-5.2**")
                res_oa = client_openai.chat.completions.create(
                    model="gpt-5.2",
                    messages=[
                        {"role": "system", "content": m["sys"]},
                        {"role": "user", "content": [
                            {"type": "text", "text": user_prompt_text + m["user_ext"]},
                            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_b64}"}}
                        ]}
                    ]
                )
                gpt_out = res_oa.choices[0].message.content
                st.info(gpt_out)
                results.append((m["name"], "GPT-5.2", gpt_out))

            # --- Gemini Execution ---
            with col2:
                st.write("**Gemini 1.5 Pro**")
                # Using 1.5 Pro/Flash for 2026 stability
                model_gem = genai.GenerativeModel(model_name='gemini-2.5-pro', system_instruction=m["sys"])
                res_gem = model_gem.generate_content([user_prompt_text + m["user_ext"], img_pil])
                gem_out = res_gem.text
                st.success(gem_out)
                results.append((m["name"], "Gemini 1.5 Pro", gem_out))
            
            progress_bar.progress((idx + 1) / 3)

        # --- 5. DEEPSEEK AUDIT (6 Reports total) ---
        st.divider()
        st.header("🧠 DeepSeek Reasoner: Comparative Audit")
        
        for method_name, model_name, report_text in results:
            with st.expander(f"Audit: {method_name} | {model_name}"):
                audit_col1, audit_col2 = st.columns(2)
                
                # Groundedness Audit
                with audit_col1:
                    st.write("📊 **Groundedness Score**")
                    g_query = f"CONTEXT: {relevant_context}\n\nANSWER: {report_text}\n\n{groundedness_rater_prompt}"
                    g_res = client_deepseek.chat.completions.create(
                        model="deepseek-reasoner",
                        messages=[{"role": "user", "content": g_query}]
                    )
                    st.write(g_res.choices[0].message.content)
                
                # Relevance Audit
                with audit_col2:
                    st.write("🎯 **Relevance Score**")
                    r_query = f"ANSWER: {report_text}\n\n{relevance_rater_prompt}"
                    r_res = client_deepseek.chat.completions.create(
                        model="deepseek-reasoner",
                        messages=[{"role": "user", "content": r_query}]
                    )
                    st.write(r_res.choices[0].message.content)

else:
    st.warning("Upload Clinical Manual and X-Ray to begin.")
