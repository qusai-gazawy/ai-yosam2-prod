import streamlit as st
import os
import PIL.Image
import google.generativeai as genai
from openai import OpenAI
import base64
import json
import fitz  # PyMuPDF
from google.oauth2 import service_account

# --- 5. APP INTERFACE ---
st.title("AI-YOSAM2: Knee Implant Interpretation in X-rays")

# Create a clean layout column structure
text_col, logo_col1, logo_col2, logo_col3 = st.columns([3, 1, 1, 1])

with text_col:
    # Small top padding to shift the text down to align with the middle of the logos
    st.write("")
    st.markdown("""
    <div style="font-size: 0.85rem; color: #666; line-height: 1.4;">
        <strong>Supported by NSF DART</strong> (Award No. OIA-1946391) & <strong>SAU</strong>.<br>
        <em>Any opinions, findings, or conclusions expressed do not necessarily reflect the views of the National Science Foundation.</em>
    </div>
    """, unsafe_allow_html=True)

# Helper paths for your local image folder setup
logo_sau_path = "demo_images/logo_sau.png"
logo_nsf_path = "demo_images/logo_nsf.png"
logo_dart_path = "demo_images/logo_dart.png"

# Render the images using native Streamlit functions
with logo_col1:
    if os.path.exists(logo_sau_path):
        # Shifting the short SAU banner down slightly to line up with the middle of the NSF circle
        st.write("")
        st.image(logo_sau_path, use_container_width=True)

with logo_col2:
    if os.path.exists(logo_nsf_path):
        # The tall circular NSF logo stays at the top of its column
        st.image(logo_nsf_path, use_container_width=True)

with logo_col3:
    if os.path.exists(logo_dart_path):
        # Shifting the short DART text logo down to match SAU
        st.write("")
        st.image(logo_dart_path, use_container_width=True)

st.write("")

# --- 1. LOCAL DEMO CONFIG (FULLY OFFLINE - 2 BOOKS & 5 IMAGES) ---
DEMO_BOOKS = {
    "📕 Textbook 1: Osteoarthritis of the Knee": "demo_books/Osteoarthritis-of-the-knee.pdf",
    "📗 Textbook 2: Principle of Orthopedic Implants": "demo_books/Principle_of_Orthopedic_Implants.pdf"
}

DEMO_IMAGES = {
    "📸 Knee Implant Image 1": "demo_images/Knee_Implant_Image_1.png",
    "📸 Knee Implant Image 2": "demo_images/Knee_Implant_Image_2.png",
    "📸 Knee Implant Image 3": "demo_images/Knee_Implant_Image_3.png",
    "📸 Knee Implant Image 4": "demo_images/Knee_Implant_Image_4.png",
    "📸 Knee Implant Image 5": "demo_images/Knee_Implant_Image_5.png"
}

# --- 2. SECRETS & CLIENTS ---
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

# --- 3. HELPER FUNCTIONS ---
def extract_pdf_context(pdf_path):
    """Safely opens and extracts text from a PDF file."""
    if not os.path.exists(pdf_path):
        st.error(f"File not found: {pdf_path}. Please check your local directory structure.")
        return ""
    with fitz.open(pdf_path) as doc:
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

# --- 4. PROMPTS ---
user_prompt_text = "\nDescribe what you observe in the provided X-Ray image according to the provided context.\n"

system_prompt = """You are an expert AI Radiologist specializing in Musculoskeletal (MSK) Imaging and orthopedic implants.
Your role is to provide a formal, technical interpretation of knee X-ray images, with a focus on identifying and evaluating orthopedic implants.

OUTPUT STRUCTURE:
- Findings
- Impression"""

groundedness_rater_prompt = """
You are an evaluation assistant. Your job is to rate the GROUNDEDNESS of an answer using the provided medical context.

IMPORTANT:
- The context is general medical knowledge (e.g., textbook).
- The answer may include image-based observations.
- HOWEVER, higher scores REQUIRE meaningful use of the provided context.

Definition:
- Grounded = the answer correctly uses, reflects, or aligns with concepts from the provided context.
- Not grounded = the answer ignores the context, introduces unsupported reasoning, or contradicts it.

Scoring Guide:

5 = Fully grounded
- Clearly uses and reflects concepts from the provided context
- Medical reasoning aligns strongly with the context
- Terminology and explanations show direct connection to the material

4 = Mostly grounded
- Generally consistent with the context
- Uses some relevant concepts or terminology from the context
- Minor gaps in explicit connection

3 = Partially grounded
- Medically reasonable but only loosely connected to the context
- Limited or implicit use of context concepts

2 = Weakly grounded
- Mostly generic medical reasoning
- Minimal or no clear connection to the provided context

1 = Not grounded
- Ignores the context or contradicts it
- Uses irrelevant or incorrect medical reasoning

Instructions:
1) Compare the answer to the provided context.
2) Reward answers that explicitly use or reflect context concepts.
3) Do NOT give a high score if the answer is only generally correct but does not use the context.
4) Provide:
   - A groundedness score from 1 to 5
   - A brief justification (2–5 bullets)
   - If score < 5, list up to 5 unsupported or weak claims

Return your output in this exact format:

Score: <1-5>
Justification:
- ...
Unsupported claims:
- ...
"""

relevance_rater_prompt = """
You are an expert orthopedic radiology evaluator.

Your task is to evaluate the QUALITY of the answer as a clinical radiology report.

IMPORTANT:
- You do NOT have access to the original image.
- Evaluate ONLY structure, completeness, and clinical reasoning.

A high-quality report must:
- Follow radiology structure (Findings, Impression, etc.)
- Address arthroplasty-specific elements:
  - components (femoral, tibial, patellar)
  - alignment
  - fixation interfaces
  - periprosthetic findings
- Use correct orthopedic terminology

Scoring:
5 = full professional report (complete, structured, domain-specific)
4 = strong but minor gaps
3 = acceptable but incomplete
2 = weak or generic
1 = not a medical report

DO NOT penalize for missing image access.
"""

st.sidebar.header("Data Selection Mode")
data_mode = st.sidebar.radio("Choose Input Method:", ["Run Quick Demo Setup", "Upload Custom Files"])

relevant_context = ""
img_pil = None
img_b64 = None
ready_to_run = False

if data_mode == "Run Quick Demo Setup":
    st.sidebar.subheader("Select Local Demo Assets")
    selected_demo_book = st.sidebar.selectbox("Choose a Textbook:", list(DEMO_BOOKS.keys()))
    selected_demo_img = st.sidebar.selectbox("Choose a Test Image:", list(DEMO_IMAGES.keys()))
    
    # Process Local PDF Book
    local_pdf_path = DEMO_BOOKS[selected_demo_book]
    with st.sidebar.spinner("Parsing local textbook..."):
        full_text = extract_pdf_context(local_pdf_path)
        
    if full_text:
        text_chunks = get_chunks(full_text)
        relevant_context = retrieve_relevant_context("orthopedic knee arthroplasty implant hardware", text_chunks)
        
        # Process Local Image
        local_image_path = DEMO_IMAGES[selected_demo_img]
        if os.path.exists(local_image_path):
            try:
                with open(local_image_path, "rb") as img_file:
                    img_bytes = img_file.read()
                img_pil = PIL.Image.open(local_image_path)
                img_b64 = encode_image_base64(img_bytes)
                ready_to_run = True
                st.sidebar.success("✅ Local assets parsed and verified.")
            except Exception as e:
                st.sidebar.error(f"Error reading image file: {e}")
        else:
            st.sidebar.error(f"Image file not found at: {local_image_path}")
else:
    # Custom Manual Uploader
    uploaded_pdf = st.sidebar.file_uploader("Clinical Manual (PDF)", type="pdf")
    uploaded_img = st.sidebar.file_uploader("Knee X-Ray", type=["jpg", "png", "jpeg"])
    
    if uploaded_pdf and uploaded_img:
        with open("temp_manual.pdf", "wb") as f:
            f.write(uploaded_pdf.getbuffer())
        
        full_text = extract_pdf_context("temp_manual.pdf")
        text_chunks = get_chunks(full_text)
        relevant_context = retrieve_relevant_context("orthopedic implant knee", text_chunks)
        
        img_bytes = uploaded_img.getvalue()
        img_pil = PIL.Image.open(uploaded_img)
        img_b64 = encode_image_base64(img_bytes)
        ready_to_run = True
# --- 6. MAIN WORKBENCH EXECUTION ---
if ready_to_run:
    # Display the target X-ray image without the text context preview expander
    st.image(img_pil, caption="Target Evaluation Image", width=400)

    if st.button("🚀 Run Multimodal Comparison"):
        results = []

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
                with st.spinner("GPT generating report..."):
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
                st.write("**Gemini 2.5 Flash**")
                with st.spinner("Gemini generating report..."):
                    model_gem = genai.GenerativeModel(model_name='gemini-2.5-flash', system_instruction=m["sys"] if m["sys"] else None)
                    res_gem = model_gem.generate_content([user_prompt_text + m["user_ext"], img_pil])
                    gem_out = res_gem.text
                st.success(gem_out)
                results.append((m["name"], "Gemini 2.5 Flash", gem_out))
            
            progress_bar.progress((idx + 1) / 3)

        # --- 7. DEEPSEEK AUDIT ---
        st.divider()
        st.header("🧠 DeepSeek Reasoner: Comparative Audit")
        
        for method_name, model_name, report_text in results:
            with st.expander(f"Audit: {method_name} | {model_name}"):
                audit_col1, audit_col2 = st.columns(2)
                
                with audit_col1:
                    st.write("📊 **Groundedness Score**")
                    with st.spinner("DeepSeek checking grounding..."):
                        g_query = f"CONTEXT: {relevant_context}\n\nANSWER: {report_text}\n\n{groundedness_rater_prompt}"
                        g_res = client_deepseek.chat.completions.create(
                            model="deepseek-reasoner",
                            messages=[{"role": "user", "content": g_query}]
                        )
                    st.write(g_res.choices[0].message.content)
                
                with audit_col2:
                    st.write("🎯 **Relevance Score**")
                    with st.spinner("DeepSeek evaluating clinical metrics..."):
                        r_query = f"ANSWER: {report_text}\n\n{relevance_rater_prompt}"
                        r_res = client_deepseek.chat.completions.create(
                            model="deepseek-reasoner",
                            messages=[{"role": "user", "content": r_query}]
                        )
                    st.write(r_res.choices[0].message.content)
else:
    st.info("💡 Select 'Run Quick Demo Setup' or upload custom files in the sidebar to populate the research workbench.")
