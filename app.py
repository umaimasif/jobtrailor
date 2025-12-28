import streamlit as st
import os
import json
import pdfplumber
from typing import TypedDict, Optional
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

# 1. Configuration & State
load_dotenv()

class JobApplicationState(TypedDict):
    job_posting_url: str
    github_url: str
    resume_text: str
    personal_writeup: str
    job_requirements: Optional[dict]
    candidate_profile: Optional[dict]
    tailored_resume: Optional[str]
    interview_materials: Optional[str]
    approved: Optional[bool]
    retries: Optional[int]

# 2. LLM Setup
llm = ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0.2,
    api_key=os.getenv("GROQ_API_KEY")
)

# 3. Utility Functions
def read_resume_file(file) -> str:
    text = ""
    with pdfplumber.open(file) as pdf:
        for page in pdf.pages:
            text += page.extract_text() + "\n"
    return text

def parse_json_safe(text: str, default: dict):
    text = text.strip().replace("```json", "").replace("```", "")
    try:
        return json.loads(text)
    except:
        return default

# 4. Node Definitions
def research_job(state: JobApplicationState):
    prompt = f"Analyze job URL: {state['job_posting_url']}. Return ONLY JSON with skills, experience, qualifications."
    response = llm.invoke(prompt)
    return {"job_requirements": parse_json_safe(response.content, {}), "retries": 0}

def build_profile(state: JobApplicationState):
    prompt = f"Resume: {state['resume_text']}\nGithub: {state['github_url']}\nSummary: {state['personal_writeup']}\nReturn ONLY JSON profile."
    response = llm.invoke(prompt)
    return {"candidate_profile": parse_json_safe(response.content, {})}

def tailor_resume(state: JobApplicationState):
    prompt = f"Tailor this resume based on: {state['job_requirements']} and {state['candidate_profile']}. Do not invent info."
    response = llm.invoke(prompt)
    return {"tailored_resume": response.content}

def validate_resume(state: JobApplicationState):
    prompt = f"Is this resume relevant and honest? {state['tailored_resume']}\nAnswer ONLY YES or NO."
    response = llm.invoke(prompt).content.strip()
    is_approved = "YES" in response.upper()
    return {"approved": is_approved, "retries": state.get("retries", 0) + 1}

def prepare_interview(state: JobApplicationState):
    prompt = f"Create interview questions for this job and resume: {state['tailored_resume']}"
    response = llm.invoke(prompt)
    return {"interview_materials": response.content}

# 5. Graph Construction
def get_app():
    builder = StateGraph(JobApplicationState)
    builder.add_node("research_job", research_job)
    builder.add_node("build_profile", build_profile)
    builder.add_node("tailor_resume", tailor_resume)
    builder.add_node("validate_resume", validate_resume)
    builder.add_node("prepare_interview", prepare_interview)

    builder.set_entry_point("research_job")
    builder.add_edge("research_job", "build_profile")
    builder.add_edge("build_profile", "tailor_resume")
    builder.add_edge("tailor_resume", "validate_resume")
    
    builder.add_conditional_edges(
        "validate_resume",
        lambda x: "approved" if x["approved"] or x["retries"] > 2 else "retry",
        {"approved": "prepare_interview", "retry": "tailor_resume"}
    )
    builder.add_edge("prepare_interview", END)

    # Use MemorySaver to allow the human review pause
    memory = MemorySaver()
    # We interrupt BEFORE prepare_interview so the user can see the resume first
    return builder.compile(checkpointer=memory, interrupt_before=["prepare_interview"])
# Check if app is already in session, if not, create it
if "app" not in st.session_state:
    st.session_state.app = get_app()

# Use the app from session state instead of recreating it
app = st.session_state.app
thread = {"configurable": {"thread_id": "streamlit_user"}}
# 6. Streamlit Interface
# --- 6. STREAMLIT INTERFACE (Improved with Session State) ---
st.set_page_config(page_title="Job Trailer Agent", layout="wide")
st.title("💼 Job Trailer Agent: Resume Tailoring & Interview Prep")

# Initialize Session State variables if they don't exist
if "step" not in st.session_state:
    st.session_state.step = "input"
if "tailored_resume" not in st.session_state:
    st.session_state.tailored_resume = None
if "interview_materials" not in st.session_state:
    st.session_state.interview_materials = None

# Sidebar Inputs
with st.sidebar:
    st.header("Inputs")
    job_url = st.text_input("Job URL")
    github_url = st.text_input("GitHub URL")
    uploaded_file = st.file_uploader("Upload Original Resume", type="pdf")
    personal_writeup = st.text_area("Personal Summary")
    
    if st.button("Generate Tailored Resume", type="primary"):
        st.session_state.step = "processing"

# Initialize graph and thread
app = get_app()
thread = {"configurable": {"thread_id": "streamlit_user"}}

# STEP 1: Processing
if st.session_state.step == "processing" and uploaded_file:
    with st.spinner("AI is researching and tailoring..."):
        resume_text = read_resume_file(uploaded_file)
        initial_input = {
            "job_posting_url": job_url,
            "github_url": github_url,
            "resume_text": resume_text,
            "personal_writeup": personal_writeup
        }
        
        # Use the session_state app
        st.session_state.app.invoke(initial_input, thread)
        
        snapshot = st.session_state.app.get_state(thread)
        st.session_state.tailored_resume = snapshot.values.get("tailored_resume")
        st.session_state.step = "review"
        st.rerun()

# STEP 2: Review
if st.session_state.step == "review":
    st.subheader("📝 Review Your Tailored Resume")
    st.markdown(st.session_state.tailored_resume)
    
    st.divider()
    col1, col2 = st.columns(2)
    with col1:
        if st.button("✅ Approve & Get Interview Prep"):
            with st.spinner("Generating materials..."):
                # Resume using the persistent app instance
                # The thread_id tells the app which checkpoint to resume from
                final_state_output = st.session_state.app.invoke(None, thread)
                
                st.session_state.interview_materials = final_state_output.get("interview_materials")
                st.session_state.step = "final"
                st.rerun()

# STEP 3: Final Output
if st.session_state.step == "final":
    st.success("Analysis Complete!")
    tab1, tab2 = st.tabs(["Final Resume", "Interview Materials"])
    with tab1:
        st.markdown(st.session_state.tailored_resume)
        st.download_button(
            "Download Resume",
            st.session_state.tailored_resume,
            file_name="tailored_resume.md"
        )
    with tab2:
        st.markdown(st.session_state.interview_materials)
        st.download_button(
            "Download Interview Materials",
            st.session_state.interview_materials,
            file_name="interview_materials.md"
        )
        