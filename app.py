# ================================================================
#  app.py  —  Kathford BSc CSIT AI Study Buddy
#  Run:  uv run streamlit run app.py
# ================================================================

import subprocess, sys
import streamlit as st
from pathlib import Path
from dotenv import load_dotenv
from src.chroma_client import get_chroma_client
from src.rag_engine import get_answer

BASE_DIR = Path(__file__).resolve().parent
load_dotenv(dotenv_path=BASE_DIR / ".env", override=False)

# ── Complete TU BSc CSIT Syllabus (all subjects, skip only pure math) ──
SYLLABUS = {
    "Semester 1": {
        "num": "1",
        "subjects": {
            "introduction_to_it": "Introduction to IT (CSC109)",
            "c_programming":      "C Programming (CSC110)",
            "digital_logic":      "Digital Logic (CSC111)",
            "physics":            "Physics (PHY113)",
        }
    },
    "Semester 2": {
        "num": "2",
        "subjects": {
            "discrete_structures":             "Discrete Structures (CSC160)",
            "object_oriented_programming_cpp": "OOP in C++ (CSC161)",
            "microprocessor":                  "Microprocessor (CSC162)",
        }
    },
    "Semester 3": {
        "num": "3",
        "subjects": {
            "computer_architecture": "Computer Architecture (CSC208)",
            "computer_graphics":     "Computer Graphics (CSC209)",
        }
    },
    "Semester 4": {
        "num": "4",
        "subjects": {
            "theory_of_computation":      "Theory of Computation (CSC257)",
            "computer_networks":          "Computer Networks (CSC258)",
            "operating_systems":          "Operating Systems (CSC259)",
            "database_management_system": "DBMS (CSC260)",
            "artificial_intelligence":    "Artificial Intelligence (CSC261)",
        }
    },
    "Semester 5": {
        "num": "5",
        "subjects": {
            "system_analysis_and_design": "System Analysis & Design (CSC315)",
            "cryptography":               "Cryptography (CSC316)",
            "web_technology":             "Web Technology (CSC318)",
            "multimedia_computing":       "Multimedia Computing",
        }
    },
    "Semester 6": {
        "num": "6",
        "subjects": {
            "software_engineering":     "Software Engineering (CSC364)",
            "egovernance":              "E-Governance (CSC366)",
            "dotnet_centric_computing": ".NET Centric Computing (CSC367)",
            "technical_writing":        "Technical Writing (CSC368)",
            "ecommerce":                "E-Commerce",
        }
    },
    "Semester 7": {
        "num": "7",
        "subjects": {
            "advanced_java_programming":   "Advanced Java (CSC409)",
            "data_warehousing_and_mining": "Data Warehousing & Mining (CSC410)",
            "principles_of_management":    "Principles of Management (MGT411)",
            "software_project_management": "Software Project Management (CSC415)",
        }
    },
    "Semester 8": {
        "num": "8",
        "subjects": {
            "advanced_database":                        "Advanced Database (CSC475)",
            "advanced_networking_ipv6":                 "Advanced Networking with IPv6 (CSC477)",
            "distributed_networking":                   "Distributed Networking (CSC478)",
            "game_technology":                          "Game Technology (CSC479)",
            "distributed_and_object_oriented_database": "Distributed & OO Database (CSC480)",
            "cloud_computing":                          "Cloud Computing (CSC481)",
            "geographical_information_system":          "Geographical Information System (CSC482)",
            "mobile_application_development":           "Mobile Application Development (CSC484)",
            "real_time_systems":                        "Real Time Systems (CSC485)",
            "network_and_system_administration":        "Network & System Administration (CSC486)",
            "embedded_systems_programming":             "Embedded Systems Programming (CSC487)",
        }
    },
}

@st.cache_data(ttl=30)
def indexed_cols() -> set:
    try:
        c = get_chroma_client()
        return {col.name for col in c.list_collections()}
    except Exception:
        return set()

def pretty(key: str) -> str:
    return key.replace("_", " ").title()

# ================================================================
#  PAGE CONFIG + CUSTOM CSS  (Kathford brand: #2196f3 blue)
# ================================================================
st.set_page_config(page_title="Kathford Study Buddy", page_icon="📚", layout="wide")

st.markdown("""
<style>
/* ── Import font ── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

/* ── Hide default streamlit chrome ── */
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding-top: 0 !important; }

/* ── Top navbar ── */
.kb-navbar {
    background: #ffffff;
    border-bottom: 1px solid #e8f4fd;
    padding: 12px 32px;
    display: flex;
    align-items: center;
    justify-content: space-between;
    position: sticky;
    top: 0;
    z-index: 999;
    box-shadow: 0 1px 8px rgba(33,150,243,0.08);
}
.kb-navbar-left { display: flex; align-items: center; gap: 12px; }
.kb-logo-box {
    background: #2196f3;
    color: white;
    font-weight: 700;
    font-size: 15px;
    padding: 6px 14px;
    border-radius: 6px;
    letter-spacing: 0.5px;
}
.kb-college-name { font-size: 13px; color: #444; font-weight: 500; }
.kb-college-sub  { font-size: 11px; color: #2196f3; font-weight: 500; }
.kb-badge {
    background: #e3f2fd;
    color: #1565c0;
    font-size: 11px;
    font-weight: 600;
    padding: 4px 12px;
    border-radius: 20px;
    border: 1px solid #bbdefb;
}

/* ── Hero banner ── */
.kb-hero {
    background: linear-gradient(135deg, #1565c0 0%, #2196f3 50%, #42a5f5 100%);
    padding: 40px 32px 36px;
    color: white;
    margin-bottom: 0;
}
.kb-hero-tag {
    font-size: 11px;
    font-weight: 600;
    letter-spacing: 1.5px;
    text-transform: uppercase;
    opacity: 0.85;
    margin-bottom: 8px;
}
.kb-hero-title {
    font-size: 28px;
    font-weight: 700;
    margin: 0 0 8px;
    line-height: 1.25;
}
.kb-hero-sub {
    font-size: 14px;
    opacity: 0.88;
    margin: 0;
    max-width: 520px;
    line-height: 1.6;
}
.kb-hero-stats {
    display: flex;
    gap: 28px;
    margin-top: 24px;
}
.kb-stat { text-align: center; }
.kb-stat-num { font-size: 22px; font-weight: 700; }
.kb-stat-lbl { font-size: 11px; opacity: 0.8; margin-top: 2px; }

/* ── Sidebar styling ── */
[data-testid="stSidebar"] {
    background: #f8fbff !important;
    border-right: 1px solid #e3f2fd;
}
[data-testid="stSidebar"] .stSelectbox label { color: #1565c0 !important; font-weight: 600 !important; font-size: 12px !important; text-transform: uppercase; letter-spacing: 0.5px; }

/* ── Subject status pills ── */
.subj-pill {
    display: inline-flex;
    align-items: center;
    gap: 5px;
    padding: 4px 10px;
    border-radius: 20px;
    font-size: 11px;
    font-weight: 500;
    margin: 2px 0;
    width: 100%;
}
.subj-pill.ready   { background: #e8f5e9; color: #2e7d32; border: 1px solid #c8e6c9; }
.subj-pill.missing { background: #fff3e0; color: #e65100; border: 1px solid #ffe0b2; }

/* ── Chat bubbles ── */
.kb-user-msg {
    background: #2196f3;
    color: white;
    padding: 12px 16px;
    border-radius: 16px 16px 4px 16px;
    margin: 8px 0;
    max-width: 80%;
    margin-left: auto;
    font-size: 14px;
    line-height: 1.5;
}
.kb-bot-msg {
    background: #ffffff;
    color: #1a1a2e;
    padding: 14px 18px;
    border-radius: 16px 16px 16px 4px;
    margin: 8px 0;
    max-width: 85%;
    font-size: 14px;
    line-height: 1.6;
    border: 1px solid #e3f2fd;
    box-shadow: 0 2px 8px rgba(33,150,243,0.06);
}

/* ── Source citation pills ── */
.src-pill {
    display: inline-block;
    background: #e3f2fd;
    color: #1565c0;
    font-size: 11px;
    padding: 3px 10px;
    border-radius: 12px;
    margin: 2px 3px;
    border: 1px solid #bbdefb;
}

/* ── Welcome cards ── */
.kb-welcome-card {
    background: white;
    border: 1px solid #e3f2fd;
    border-top: 3px solid #2196f3;
    border-radius: 10px;
    padding: 16px;
    text-align: center;
    box-shadow: 0 2px 8px rgba(33,150,243,0.06);
}
.kb-welcome-card .icon { font-size: 24px; margin-bottom: 8px; }
.kb-welcome-card .title { font-size: 13px; font-weight: 600; color: #1565c0; margin-bottom: 4px; }
.kb-welcome-card .desc { font-size: 12px; color: #666; }

/* ── Warning / info boxes ── */
.kb-info {
    background: #e3f2fd;
    border-left: 4px solid #2196f3;
    padding: 12px 16px;
    border-radius: 0 8px 8px 0;
    font-size: 13px;
    color: #1565c0;
    margin: 16px 0;
}
.kb-warn {
    background: #fff8e1;
    border-left: 4px solid #ffc107;
    padding: 12px 16px;
    border-radius: 0 8px 8px 0;
    font-size: 13px;
    color: #795548;
    margin: 16px 0;
}

/* ── Upload button ── */
.stButton > button {
    background: #2196f3 !important;
    color: white !important;
    border: none !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
    font-size: 13px !important;
    padding: 8px 20px !important;
    transition: background 0.2s !important;
}
.stButton > button:hover { background: #1565c0 !important; }

/* ── Chat input ── */
[data-testid="stChatInput"] textarea {
    border: 2px solid #e3f2fd !important;
    border-radius: 12px !important;
    font-size: 14px !important;
}
[data-testid="stChatInput"] textarea:focus {
    border-color: #2196f3 !important;
    box-shadow: 0 0 0 3px rgba(33,150,243,0.1) !important;
}

/* ── Footer ── */
.kb-footer {
    text-align: center;
    padding: 20px;
    font-size: 11px;
    color: #aaa;
    border-top: 1px solid #f0f0f0;
    margin-top: 32px;
}
.kb-footer a { color: #2196f3; text-decoration: none; }
</style>
""", unsafe_allow_html=True)


# ── TOP NAVBAR ───────────────────────────────────────────────────
st.markdown("""
<div class="kb-navbar">
  <div class="kb-navbar-left">
    <div class="kb-logo-box">K</div>
    <div>
      <div class="kb-college-name">Kathford International College</div>
      <div class="kb-college-sub">BSc CSIT · Tribhuvan University · Balkumari, Lalitpur</div>
    </div>
  </div>
  <div class="kb-badge">🤖 AI Study Buddy</div>
</div>
""", unsafe_allow_html=True)


# ── HERO BANNER ──────────────────────────────────────────────────
total_indexed = len(indexed_cols())
st.markdown(f"""
<div class="kb-hero">
  <div class="kb-hero-tag">Kathford AI Lab · Powered by RAG Technology</div>
  <div class="kb-hero-title">Discover, Build &amp; Advance Intelligence</div>
  <div class="kb-hero-sub">
    Ask questions about your actual BSc CSIT textbooks and get precise answers
    grounded only in your course material — not generic internet answers.
  </div>
  <div class="kb-hero-stats">
    <div class="kb-stat"><div class="kb-stat-num">8</div><div class="kb-stat-lbl">Semesters</div></div>
    <div class="kb-stat"><div class="kb-stat-num">30+</div><div class="kb-stat-lbl">Subjects</div></div>
    <div class="kb-stat"><div class="kb-stat-num">{total_indexed}</div><div class="kb-stat-lbl">Indexed Now</div></div>
    <div class="kb-stat"><div class="kb-stat-num">24/7</div><div class="kb-stat-lbl">Available</div></div>
  </div>
</div>
""", unsafe_allow_html=True)


# ── SESSION STATE ────────────────────────────────────────────────
if "history"  not in st.session_state: st.session_state.history  = []
if "sem_num"  not in st.session_state: st.session_state.sem_num  = None
if "subj_key" not in st.session_state: st.session_state.subj_key = None


# ================================================================
#  SIDEBAR
# ================================================================
with st.sidebar:
    st.markdown("### 🎓 Select Semester")
    sem_label = st.selectbox("Semester", ["All Semesters"] + list(SYLLABUS.keys()), label_visibility="collapsed")

    sem_num  = None
    subj_key = None
    subjects = {}

    if sem_label != "All Semesters":
        sem_num  = SYLLABUS[sem_label]["num"]
        subjects = SYLLABUS[sem_label]["subjects"]
        indexed  = indexed_cols()

        st.markdown("### 📖 Select Subject")
        subj_options = ["All subjects"] + list(subjects.values())
        subj_display = st.selectbox("Subject", subj_options, label_visibility="collapsed")
        if subj_display != "All subjects":
            subj_key = next(k for k, v in subjects.items() if v == subj_display)

        # Subject index status
        st.markdown("#### 📊 Index Status")
        for key, name in subjects.items():
            col_name = f"sem{sem_num}_{key}"
            if col_name in indexed:
                st.markdown(f'<div class="subj-pill ready">✅ {name}</div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="subj-pill missing">📂 {name}</div>', unsafe_allow_html=True)

    st.divider()

    # Upload section
    st.markdown("### 📥 Upload Study Material")
    if sem_num and subjects:
        upload_to = st.selectbox("Upload to subject:", list(subjects.values()), key="upload_to")
        upload_key = next(k for k, v in subjects.items() if v == upload_to)

        files = st.file_uploader("Choose .epub or .pdf", type=["epub","pdf"], accept_multiple_files=True)
        if files and st.button("📥 Index Now", use_container_width=True):
            folder = BASE_DIR / "Data" / f"Sem{sem_num}" / upload_key
            folder.mkdir(parents=True, exist_ok=True)
            for f in files:
                (folder / f.name).write_bytes(f.read())
            with st.spinner(f"Indexing {len(files)} file(s)..."):
                r = subprocess.run(
                    [sys.executable, str(BASE_DIR / "src" / "ingest.py"),
                     "--sem", sem_num, "--subject", upload_key],
                    capture_output=True, text=True, cwd=BASE_DIR
                )
            if r.returncode == 0:
                st.success(f"✅ {len(files)} file(s) indexed!")
                indexed_cols.clear()
                st.rerun()
            else:
                st.error("Indexing failed — check terminal")
                st.code(r.stderr[-800:])
    else:
        st.markdown('<div class="kb-info">Select a semester above to upload study material.</div>', unsafe_allow_html=True)

    st.divider()
    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.history = []
        st.rerun()

    st.markdown("""
    <div style="text-align:center; padding: 12px 0; font-size: 11px; color: #aaa; line-height: 1.8;">
        <b style="color:#2196f3">Kathford AI Study Buddy</b><br>
        Powered by RAG + Ollama<br>
        Answers grounded in your textbooks only
    </div>
    """, unsafe_allow_html=True)


# ================================================================
#  MAIN CHAT AREA
# ================================================================
filter_str = ""
if sem_num and subj_key:
    filter_str = f" — {subjects[subj_key]}"
elif sem_num:
    filter_str = f" — {sem_label} (all subjects)"

st.markdown(f"<h3 style='margin:20px 32px 4px; color:#1565c0; font-size:18px;'>💬 Ask Your Question{filter_str}</h3>", unsafe_allow_html=True)

# Welcome cards
if not st.session_state.history:
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown('<div class="kb-welcome-card"><div class="icon">🧠</div><div class="title">OS & Networks</div><div class="desc">What is a deadlock? Explain OSI model.</div></div>', unsafe_allow_html=True)
    with c2:
        st.markdown('<div class="kb-welcome-card"><div class="icon">🗄️</div><div class="title">DBMS</div><div class="desc">What is normalisation? Explain SQL joins.</div></div>', unsafe_allow_html=True)
    with c3:
        st.markdown('<div class="kb-welcome-card"><div class="icon">🤖</div><div class="title">AI & ML</div><div class="desc">Explain A* search. What is a neural network?</div></div>', unsafe_allow_html=True)
    with c4:
        st.markdown('<div class="kb-welcome-card"><div class="icon">💻</div><div class="title">Programming</div><div class="desc">What is polymorphism in OOP?</div></div>', unsafe_allow_html=True)

    if total_indexed == 0:
        st.markdown('<div class="kb-warn">⚠️ No study material indexed yet. Select a semester → upload your epub → click "Index Now" to get started.</div>', unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="kb-info">✅ {total_indexed} subject(s) ready. Select a semester & subject from the sidebar to filter answers, or ask a question below to search across all material.</div>', unsafe_allow_html=True)

# Render history
for turn in st.session_state.history:
    with st.chat_message("user"):
        st.markdown(f'<div class="kb-user-msg">{turn["question"]}</div>', unsafe_allow_html=True)
    with st.chat_message("assistant", avatar="📚"):
        st.markdown(f'<div class="kb-bot-msg">{turn["answer"]}</div>', unsafe_allow_html=True)
        if turn.get("sources"):
            src_html = "".join([
                f'<span class="src-pill">📄 {s["file"]} · Ch.{s["page"]}</span>'
                for s in turn["sources"]
            ])
            st.markdown(f"<div style='margin-top:6px'>{src_html}</div>", unsafe_allow_html=True)

# Chat input
query = st.chat_input("Ask anything about your BSc CSIT subjects...")

if query:
    with st.chat_message("user"):
        st.markdown(f'<div class="kb-user-msg">{query}</div>', unsafe_allow_html=True)

    with st.chat_message("assistant", avatar="📚"):
        with st.spinner("Searching your study material..."):
            result = get_answer(
                query   = query,
                sem     = sem_num,
                subject = subj_key,
                history = [{"question": t["question"], "answer": t["answer"]} for t in st.session_state.history],
            )
        st.markdown(f'<div class="kb-bot-msg">{result["answer"]}</div>', unsafe_allow_html=True)
        if result.get("sources"):
            src_html = "".join([
                f'<span class="src-pill">📄 {s["file"]} · Ch.{s["page"]}</span>'
                for s in result["sources"]
            ])
            st.markdown(f"<div style='margin-top:6px'>{src_html}</div>", unsafe_allow_html=True)

    st.session_state.history.append({
        "question": query,
        "answer":   result["answer"],
        "sources":  result.get("sources", []),
    })

# Footer
st.markdown("""
<div class="kb-footer">
    <b>Kathford International College of Engineering & Management</b><br>
    Balkumari, Lalitpur Metropolitan City · 
    <a href="https://kathford.edu.np">kathford.edu.np</a> · 
    info@kathford.edu.np · 01-5186046<br>
    <span style="color:#2196f3">AI Study Buddy</span> — Built by BSc CSIT Students · Tribhuvan University Affiliated
</div>
""", unsafe_allow_html=True)
