import streamlit as st
import os
import time
import datetime
import tempfile

from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langchain_community.embeddings import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFDirectoryLoader

# ─────────────────────────────────────────────────────────────────────────────
#  PAGE CONFIG
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Lexis AI",
    page_icon="✦",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# ─────────────────────────────────────────────────────────────────────────────
#  CSS
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Geist:wght@300;400;500;600;700&family=Geist+Mono:wght@400;500&display=swap');

:root {
  --bg:         #0a0a0a;
  --bg-2:       #111111;
  --bg-3:       #1a1a1a;
  --surface:    #1c1c1c;
  --surface-2:  #242424;
  --border:     rgba(255,255,255,.08);
  --border-2:   rgba(255,255,255,.12);
  --blue:       #5b8cf5;
  --blue-dim:   rgba(91,140,245,.12);
  --blue-glow:  rgba(91,140,245,.2);
  --tx-1: #f5f5f5;
  --tx-2: #a0a0a0;
  --tx-3: #555;
  --r-sm: 10px;
  --r-md: 14px;
  --r-lg: 20px;
  --r-xl: 28px;
  --font: 'Geist', system-ui, sans-serif;
  --mono: 'Geist Mono', monospace;
}

*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
html, body, [class*="css"] { font-family: var(--font); color: var(--tx-1); }
#MainMenu, footer, header { visibility: hidden; }

.stApp {
  background: var(--bg);
  min-height: 100vh;
}

/* hide default sidebar toggle, collapse sidebar fully */
[data-testid="stSidebar"] { display: none !important; }
section[data-testid="stSidebarContent"] { display: none !important; }

.block-container {
  max-width: 780px !important;
  padding: 0 16px !important;
  margin: 0 auto !important;
}

/* ── hide streamlit labels ─── */
.stTextInput > label,
.stTextArea  > label,
.stFileUploader > label { display: none !important; }

/* ── TOP HEADER ─── */
.app-header {
  display: flex;
  flex-direction: column;
  align-items: center;
  padding: 48px 0 32px;
  gap: 8px;
}
.app-logo {
  width: 44px; height: 44px;
  background: linear-gradient(135deg, var(--blue), #8b5cf6);
  border-radius: 13px;
  display: flex; align-items: center; justify-content: center;
  font-size: 20px;
  box-shadow: 0 0 0 1px rgba(91,140,245,.3), 0 8px 24px rgba(91,140,245,.15);
  margin-bottom: 4px;
}
.app-title {
  font-size: 1.55rem;
  font-weight: 700;
  letter-spacing: -.04em;
  color: var(--tx-1);
}
.app-title span { color: var(--blue); }
.app-subtitle {
  font-size: .82rem;
  color: var(--tx-3);
  letter-spacing: .01em;
}

/* ── UPLOAD ZONE (compact, chip-style) ─── */
.upload-area {
  display: flex;
  align-items: center;
  flex-wrap: wrap;
  gap: 8px;
  margin-bottom: 16px;
}

/* file chip */
.file-chip {
  display: inline-flex;
  align-items: center;
  gap: 7px;
  background: var(--surface);
  border: 1px solid var(--border-2);
  border-radius: 8px;
  padding: 7px 12px 7px 10px;
  font-size: .78rem;
  color: var(--tx-2);
  font-family: var(--mono);
  max-width: 220px;
}
.file-chip-icon {
  width: 20px; height: 20px;
  background: #ef4444;
  border-radius: 5px;
  display: flex; align-items: center; justify-content: center;
  font-size: 10px; color: white; flex-shrink: 0;
}
.file-chip-name {
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
  flex: 1;
  min-width: 0;
}
.file-chip-size {
  font-size: .68rem;
  color: var(--tx-3);
  flex-shrink: 0;
}

/* index status pill */
.index-pill {
  display: inline-flex; align-items: center; gap: 5px;
  padding: 4px 10px; border-radius: 100px;
  font-size: .7rem; font-weight: 600; font-family: var(--mono);
}
.index-pill.ready {
  background: rgba(52,211,153,.08);
  border: 1px solid rgba(52,211,153,.2);
  color: #34d399;
}
.index-pill.pending {
  background: rgba(251,191,36,.08);
  border: 1px solid rgba(251,191,36,.2);
  color: #fbbf24;
}
.index-dot {
  width: 6px; height: 6px; border-radius: 50%;
  background: currentColor;
  animation: pulse 2s ease-in-out infinite;
}
@keyframes pulse {
  0%,100% { opacity:1; }
  50% { opacity:.4; }
}

/* ── CHAT MESSAGES ─── */
.chat-wrapper {
  display: flex;
  flex-direction: column;
  gap: 24px;
  padding: 8px 0 24px;
}

.msg-user {
  display: flex;
  justify-content: flex-end;
}
.msg-user .bubble-user {
  background: var(--blue);
  color: #fff;
  border-radius: 18px 18px 4px 18px;
  padding: 12px 16px;
  font-size: .9rem;
  line-height: 1.65;
  max-width: 75%;
  box-shadow: 0 2px 12px rgba(91,140,245,.25);
}

.msg-bot {
  display: flex;
  gap: 12px;
  align-items: flex-start;
}
.bot-avatar {
  width: 30px; height: 30px;
  background: linear-gradient(135deg, var(--surface-2), var(--surface));
  border: 1px solid var(--border-2);
  border-radius: 50%; flex-shrink: 0;
  display: flex; align-items: center; justify-content: center;
  font-size: 13px;
  margin-top: 2px;
}
.bubble-bot {
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: 4px 18px 18px 18px;
  padding: 14px 18px;
  font-size: .9rem;
  line-height: 1.72;
  color: var(--tx-1);
  flex: 1;
  box-shadow: 0 1px 8px rgba(0,0,0,.2);
}
.bubble-bot p { margin: 0 0 10px; }
.bubble-bot p:last-child { margin: 0; }
.bubble-bot ul, .bubble-bot ol { padding-left: 1.2em; margin: 6px 0; }
.bubble-bot li { margin-bottom: 4px; }

/* ── source chips ─── */
.sources-row {
  margin-top: 12px;
  padding-top: 10px;
  border-top: 1px solid var(--border);
  display: flex;
  flex-wrap: wrap;
  gap: 5px;
  align-items: center;
}
.sources-label {
  font-size: .65rem;
  text-transform: uppercase;
  letter-spacing: .1em;
  color: var(--tx-3);
  font-weight: 600;
  margin-right: 2px;
}
.src-chip {
  display: inline-flex; align-items: center; gap: 4px;
  background: var(--bg-3);
  border: 1px solid var(--border);
  border-radius: 100px;
  padding: 3px 9px;
  font-size: .68rem;
  font-family: var(--mono);
  color: var(--blue);
}

/* ── msg timestamp ─── */
.msg-time {
  font-size: .63rem;
  color: var(--tx-3);
  font-family: var(--mono);
  margin-top: 4px;
  text-align: right;
}
.msg-bot .msg-time { text-align: left; padding-left: 42px; }

/* ── EMPTY STATE ─── */
.empty-state {
  text-align: center;
  padding: 60px 24px 40px;
  color: var(--tx-3);
}
.empty-state .big-icon { font-size: 2.5rem; margin-bottom: 12px; opacity: .4; }
.empty-state .headline { font-size: .95rem; font-weight: 600; color: var(--tx-2); margin-bottom: 6px; }
.empty-state .sub { font-size: .8rem; line-height: 1.7; }

/* ── INPUT BAR ─── */
.input-bar-wrap {
  position: sticky;
  bottom: 0;
  background: linear-gradient(0deg, var(--bg) 70%, transparent);
  padding: 16px 0 20px;
  margin-top: 8px;
}

/* streamlit input override */
.stTextInput > div > div > input {
  background: var(--surface) !important;
  border: 1px solid var(--border-2) !important;
  border-radius: var(--r-xl) !important;
  color: var(--tx-1) !important;
  font-family: var(--font) !important;
  font-size: .9rem !important;
  padding: 14px 60px 14px 20px !important;
  transition: border-color .18s, box-shadow .18s !important;
  caret-color: var(--blue) !important;
  height: 52px !important;
}
.stTextInput > div > div > input:focus {
  border-color: rgba(91,140,245,.5) !important;
  box-shadow: 0 0 0 3px rgba(91,140,245,.1) !important;
  outline: none !important;
}
.stTextInput > div > div > input::placeholder { color: var(--tx-3) !important; }
.stTextInput > div > div > input:disabled {
  opacity: .5 !important;
  cursor: not-allowed !important;
}

/* send button  */
.stButton > button {
  background: var(--blue) !important;
  color: #fff !important;
  border: none !important;
  border-radius: 50% !important;
  width: 40px !important; height: 40px !important;
  padding: 0 !important;
  font-size: 1rem !important;
  box-shadow: 0 2px 10px rgba(91,140,245,.3) !important;
  transition: filter .14s, transform .14s !important;
  display: flex; align-items: center; justify-content: center !important;
  min-width: unset !important;
}
.stButton > button:hover {
  filter: brightness(1.15) !important;
  transform: scale(1.05) !important;
}
.stButton > button:disabled {
  background: var(--surface-2) !important;
  box-shadow: none !important;
  cursor: not-allowed !important;
}

/* file uploader */
[data-testid="stFileUploader"] { background: transparent !important; }
[data-testid="stFileDropzone"] {
  background: var(--surface) !important;
  border: 1.5px dashed var(--border-2) !important;
  border-radius: var(--r-md) !important;
}
[data-testid="stFileDropzone"] p, [data-testid="stFileDropzone"] span {
  color: var(--tx-3) !important;
  font-family: var(--font) !important;
  font-size: .8rem !important;
}

/* clear btn */
.clear-btn > button {
  background: transparent !important;
  border: none !important;
  color: var(--tx-3) !important;
  font-size: .75rem !important;
  padding: 4px 8px !important;
  height: auto !important; width: auto !important;
  box-shadow: none !important;
  border-radius: 6px !important;
  min-width: unset !important;
}
.clear-btn > button:hover {
  color: var(--tx-2) !important;
  background: var(--surface) !important;
  filter: none !important; transform: none !important;
}

/* delete btn */
.del-btn > button {
  background: transparent !important;
  border: none !important;
  color: var(--tx-3) !important;
  font-size: .75rem !important;
  padding: 2px 6px !important;
  height: auto !important; width: auto !important;
  box-shadow: none !important;
  border-radius: 4px !important;
  min-width: unset !important;
}
.del-btn > button:hover {
  color: #fb7185 !important;
  background: rgba(251,113,133,.08) !important;
  filter: none !important; transform: none !important;
}

/* spinner */
.stSpinner > div { border-top-color: var(--blue) !important; }

/* expander */
[data-testid="stExpander"] {
  background: var(--surface) !important;
  border: 1px solid var(--border) !important;
  border-radius: var(--r-md) !important;
}
[data-testid="stExpander"] summary {
  font-size: .78rem !important; font-weight: 600 !important;
  color: var(--tx-2) !important; padding: 10px 14px !important;
}

/* chunk card */
.chunk-card {
  background: var(--bg-3);
  border-left: 2px solid rgba(103,232,249,.4);
  border-radius: 6px;
  padding: 10px 14px;
  margin-bottom: 8px;
  font-size: .78rem;
  line-height: 1.7;
  color: var(--tx-2);
}
.chunk-tag {
  font-family: var(--mono);
  font-size: .6rem;
  color: #67e8f9;
  letter-spacing: .1em;
  text-transform: uppercase;
  margin-bottom: 6px;
  font-weight: 600;
}

/* footer disclaimer */
.footer-note {
  text-align: center;
  font-size: .7rem;
  color: var(--tx-3);
  padding: 8px 0 16px;
}

hr { border: none !important; border-top: 1px solid var(--border) !important; margin: 16px 0 !important; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
#  LOAD ENV & INIT
# ─────────────────────────────────────────────────────────────────────────────
load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")

llm = ChatGroq(groq_api_key=groq_api_key, model_name="llama-3.1-8b-instant")

prompt = ChatPromptTemplate.from_template("""
Answer the question based only on the provided context.
Be thorough, accurate, and well-structured in your response.

<context>
{context}
</context>

Question: {input}
""")

# ─────────────────────────────────────────────────────────────────────────────
#  SESSION STATE
# ─────────────────────────────────────────────────────────────────────────────
for k, v in {
    "chat_history":   [],
    "uploaded_files": [],
    "upload_dir":     None,
}.items():
    if k not in st.session_state:
        st.session_state[k] = v

if st.session_state.upload_dir is None or not os.path.isdir(st.session_state.upload_dir or ""):
    st.session_state.upload_dir = tempfile.mkdtemp(prefix="lexis_uploads_")

# ─────────────────────────────────────────────────────────────────────────────
#  HELPERS
# ─────────────────────────────────────────────────────────────────────────────
def format_bytes(n):
    for unit in ("B", "KB", "MB"):
        if n < 1024:
            return f"{n:.0f} {unit}"
        n /= 1024
    return f"{n:.1f} GB"

def save_uploaded_file(uf):
    dest = os.path.join(st.session_state.upload_dir, uf.name)
    with open(dest, "wb") as f:
        f.write(uf.getbuffer())
    return dest

def remove_uploaded_file(filename):
    path = os.path.join(st.session_state.upload_dir, filename)
    if os.path.exists(path):
        os.remove(path)
    st.session_state.uploaded_files = [
        f for f in st.session_state.uploaded_files if f["name"] != filename
    ]
    for key in ("vectors","embeddings","docs","final_documents","text_splitter","loader"):
        st.session_state.pop(key, None)

def create_vector_embedding(pdf_dir):
    st.session_state.embeddings = OllamaEmbeddings(model="nomic-embed-text")
    st.session_state.loader     = PyPDFDirectoryLoader(pdf_dir)
    st.session_state.docs       = st.session_state.loader.load()
    st.session_state.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    st.session_state.final_documents = st.session_state.text_splitter.split_documents(st.session_state.docs)
    st.session_state.vectors = FAISS.from_documents(st.session_state.final_documents, st.session_state.embeddings)

def run_query(user_input):
    doc_chain       = create_stuff_documents_chain(llm, prompt)
    retriever       = st.session_state.vectors.as_retriever()
    retrieval_chain = create_retrieval_chain(retriever, doc_chain)
    t0   = time.process_time()
    resp = retrieval_chain.invoke({"input": user_input})
    return resp, time.process_time() - t0

# ─────────────────────────────────────────────────────────────────────────────
#  HEADER
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="app-header">
  <div class="app-logo">✦</div>
  <div class="app-title">Lexis <span>AI</span></div>
  <div class="app-subtitle">Chat with your documents</div>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
#  DOCUMENT ATTACHMENT AREA
# ─────────────────────────────────────────────────────────────────────────────
kb_ready = "vectors" in st.session_state
has_files = len(st.session_state.uploaded_files) > 0

# ── Upload widget (compact expander) ─────────────────────────────────────────
with st.expander("📎 Upload PDF documents", expanded=not has_files):
    uploaded = st.file_uploader(
        "Upload PDFs",
        type=["pdf"],
        accept_multiple_files=True,
        label_visibility="collapsed",
    )
    if uploaded:
        existing = {f["name"] for f in st.session_state.uploaded_files}
        added = False
        for uf in uploaded:
            if uf.name not in existing:
                path = save_uploaded_file(uf)
                st.session_state.uploaded_files.append({"name": uf.name, "size": uf.size, "path": path})
                for key in ("vectors","embeddings","docs","final_documents","text_splitter","loader"):
                    st.session_state.pop(key, None)
                added = True
        if added:
            # Auto-build index immediately after new files are saved
            with st.spinner("Indexing documents…"):
                create_vector_embedding(st.session_state.upload_dir)
            st.rerun()

    # File list inside expander with remove buttons
    if st.session_state.uploaded_files:
        st.markdown('<div style="margin-top:10px;"></div>', unsafe_allow_html=True)
        for fi in list(st.session_state.uploaded_files):
            row = st.columns([7, 1])
            with row[0]:
                st.markdown(
                    f'<div style="display:flex;align-items:center;gap:8px;padding:4px 0;">'
                    f'<span style="background:#ef4444;color:#fff;font-size:.6rem;font-weight:700;'
                    f'padding:2px 5px;border-radius:4px;">PDF</span>'
                    f'<span style="font-size:.78rem;font-family:var(--mono);color:var(--tx-2);">'
                    f'{fi["name"]}</span>'
                    f'<span style="font-size:.68rem;color:var(--tx-3);">({format_bytes(fi["size"])})</span>'
                    f'</div>',
                    unsafe_allow_html=True,
                )
            with row[1]:
                st.markdown('<div class="del-btn">', unsafe_allow_html=True)
                if st.button("✕", key=f"del_{fi['name']}"):
                    remove_uploaded_file(fi["name"])
                    # Re-build index if files remain
                    if st.session_state.uploaded_files:
                        with st.spinner("Re-indexing…"):
                            create_vector_embedding(st.session_state.upload_dir)
                    st.rerun()
                st.markdown('</div>', unsafe_allow_html=True)

# ── File chips + status shown above chat (when files exist) ──────────────────
kb_ready = "vectors" in st.session_state   # recompute after possible auto-build
has_files = len(st.session_state.uploaded_files) > 0

if has_files:
    chips = ""
    for fi in st.session_state.uploaded_files:
        chips += (
            f'<div class="file-chip">'
            f'<div class="file-chip-icon">PDF</div>'
            f'<span class="file-chip-name">{fi["name"]}</span>'
            f'<span class="file-chip-size">{format_bytes(fi["size"])}</span>'
            f'</div>'
        )
    status = (
        '<span class="index-pill ready"><span class="index-dot"></span>Indexed</span>'
        if kb_ready else
        '<span class="index-pill pending"><span class="index-dot"></span>Indexing…</span>'
    )
    st.markdown(
        f'<div style="display:flex;flex-wrap:wrap;align-items:center;gap:8px;margin:12px 0 4px;">'
        f'{chips}{status}</div>',
        unsafe_allow_html=True,
    )

# ─────────────────────────────────────────────────────────────────────────────
#  CHAT HISTORY
# ─────────────────────────────────────────────────────────────────────────────
if not st.session_state.chat_history:
    st.markdown("""
    <div class="empty-state">
      <div class="big-icon">🔍</div>
      <div class="headline">Ask anything about your documents</div>
      <div class="sub">Upload a PDF above and start chatting — indexing happens automatically.<br>Sources will appear below each response.</div>
    </div>
    """, unsafe_allow_html=True)
else:
    st.markdown('<div class="chat-wrapper">', unsafe_allow_html=True)
    for msg in st.session_state.chat_history:
        ts = msg.get("time", "")
        if msg["role"] == "user":
            st.markdown(f"""
            <div class="msg-user">
              <div>
                <div class="bubble-user">{msg['content']}</div>
                <div class="msg-time">{ts} ✓✓</div>
              </div>
            </div>
            """, unsafe_allow_html=True)
        else:
            sources_html = ""
            if msg.get("sources"):
                pills = "".join(
                    f'<span class="src-chip">📄 {s}</span>'
                    for s in msg["sources"]
                )
                sources_html = f'<div class="sources-row"><span class="sources-label">Sources</span>{pills}</div>'
            st.markdown(f"""
            <div class="msg-bot">
              <div class="bot-avatar">✦</div>
              <div style="flex:1;min-width:0;">
                <div class="bubble-bot">{msg['content']}{sources_html}</div>
                <div class="msg-time">{ts}</div>
              </div>
            </div>
            """, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
#  SOURCE CHUNKS (collapsible, only last response)
# ─────────────────────────────────────────────────────────────────────────────
last_bot = next(
    (m for m in reversed(st.session_state.chat_history) if m["role"] == "assistant"),
    None,
)
if last_bot and last_bot.get("chunks"):
    with st.expander("📄 View source passages", expanded=False):
        for i, chunk in enumerate(last_bot["chunks"]):
            st.markdown(
                f'<div class="chunk-card"><div class="chunk-tag">Chunk {i+1} · {chunk["label"]}</div>{chunk["text"]}</div>',
                unsafe_allow_html=True,
            )

# ─────────────────────────────────────────────────────────────────────────────
#  INPUT BAR
# ─────────────────────────────────────────────────────────────────────────────
st.markdown('<div class="input-bar-wrap">', unsafe_allow_html=True)
inp_col, btn_col = st.columns([10, 1])
with inp_col:
    user_prompt = st.text_input(
        "Ask",
        placeholder="Ask anything about your documents…",
        label_visibility="collapsed",
        key="user_input",
        disabled=not kb_ready,
    )
with btn_col:
    st.markdown('<div style="display:flex;align-items:center;height:52px;">', unsafe_allow_html=True)
    ask_clicked = st.button("↑", disabled=not kb_ready)
    st.markdown('</div>', unsafe_allow_html=True)

if not kb_ready:
    hint = "Upload a PDF above to start chatting — indexing is automatic." if not has_files else "Indexing in progress, please wait…"
    st.markdown(f'<div style="text-align:center;font-size:.73rem;color:var(--tx-3);margin-top:6px;">{hint}</div>', unsafe_allow_html=True)

# clear conversation link
if st.session_state.chat_history:
    st.markdown('<div style="text-align:center;margin-top:4px;">', unsafe_allow_html=True)
    st.markdown('<div class="clear-btn" style="display:inline-block;">', unsafe_allow_html=True)
    if st.button("Clear conversation"):
        st.session_state.chat_history = []
        st.rerun()
    st.markdown('</div></div>', unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)  # input-bar-wrap

# ─────────────────────────────────────────────────────────────────────────────
#  PROCESS QUERY
# ─────────────────────────────────────────────────────────────────────────────
if ask_clicked and user_prompt and kb_ready:
    with st.spinner("Searching…"):
        response, latency = run_query(user_prompt)

    answer  = response["answer"]
    context = response.get("context", [])

    source_labels = []
    for i, doc in enumerate(context):
        src   = doc.metadata.get("source", "")
        pg    = doc.metadata.get("page", "")
        label = os.path.basename(src) if src else f"Chunk {i+1}"
        if pg != "":
            label += f" · p{pg+1}"
        source_labels.append(label)

    ts     = datetime.datetime.now().strftime("%I:%M %p")
    chunks = [{"label": source_labels[i], "text": doc.page_content} for i, doc in enumerate(context)]

    st.session_state.chat_history.append({"role": "user",      "content": user_prompt, "time": ts})
    st.session_state.chat_history.append({"role": "assistant", "content": answer, "sources": source_labels, "chunks": chunks, "latency": latency, "time": ts})
    st.rerun()

# ─────────────────────────────────────────────────────────────────────────────
#  FOOTER
# ─────────────────────────────────────────────────────────────────────────────
st.markdown('<div class="footer-note">Lexis AI may make mistakes. Please verify important information.</div>', unsafe_allow_html=True)