import streamlit as st
import arxiv
from sentence_transformers import SentenceTransformer
import chromadb
import hashlib
import time
from openai import OpenAI
import os
groq_key = os.environ.get("GROQ_API_KEY", "")
# ── Page Config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="ArXiv Research Agent",
    page_icon="⚗️",
    layout="wide",
    initial_sidebar_state="expanded"
)

LLM_MODEL = "meta-llama/llama-4-scout-17b-16e-instruct"

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Outfit:wght@300;400;500;600&display=swap');

html, body, [class*="css"] { font-family: 'Outfit', sans-serif; }
#MainMenu, footer { visibility: hidden; }

.user-msg {
    background: #16162a;
    border: 1px solid #2a2a4a;
    border-radius: 14px 14px 2px 14px;
    padding: 14px 18px;
    margin: 10px 0 10px 15%;
    color: #e2e2f0;
    font-size: 15px;
    line-height: 1.65;
}
.assistant-msg {
    background: #11111e;
    border: 1px solid #1c1c32;
    border-radius: 2px 14px 14px 14px;
    padding: 14px 18px;
    margin: 10px 15% 4px 0;
    color: #d8d8f0;
    font-size: 15px;
    line-height: 1.65;
}
.source-row { margin: 0 15% 14px 0; }
.source-chip {
    display: inline-block;
    background: #16102e;
    border: 1px solid #3d2f8f;
    border-radius: 20px;
    padding: 3px 11px;
    font-size: 11px;
    color: #9b8de8;
    margin: 3px 4px 3px 0;
    font-family: 'Space Mono', monospace;
    cursor: default;
}
.paper-card {
    background: #111120;
    border: 1px solid #1c1c32;
    border-radius: 10px;
    padding: 11px 13px;
    margin-bottom: 8px;
    transition: border-color 0.2s;
}
.paper-card:hover { border-color: #3d2f8f; }
.paper-title {
    color: #c0b8f5;
    font-size: 12.5px;
    font-weight: 500;
    line-height: 1.45;
    margin-bottom: 5px;
}
.paper-authors { color: #44445a; font-size: 11px; margin-bottom: 3px; }
.paper-cat {
    display: inline-block;
    background: #1a1030;
    border: 1px solid #2a1f50;
    border-radius: 8px;
    padding: 1px 8px;
    font-size: 10px;
    color: #6655aa;
    font-family: 'Space Mono', monospace;
}
.stat-row { display: flex; gap: 8px; margin-bottom: 14px; }
.stat-box {
    flex: 1;
    background: #111120;
    border: 1px solid #1c1c32;
    border-radius: 10px;
    padding: 12px 8px;
    text-align: center;
}
.stat-num { font-size: 22px; font-weight: 600; color: #7c6cf8; font-family: 'Space Mono', monospace; }
.stat-label { font-size: 10px; color: #44445a; margin-top: 2px; }
.hero-title { font-size: 28px; font-weight: 600; color: #e2e2f0; margin-bottom: 4px; }
.hero-sub { color: #44445a; font-size: 14px; margin-bottom: 20px; font-family: 'Space Mono', monospace; }
.empty-state { text-align: center; padding: 40px 20px; color: #333350; font-size: 14px; }
.step-badge {
    display: inline-block;
    background: #1a1030;
    border: 1px solid #3d2f8f;
    border-radius: 50%;
    width: 22px; height: 22px;
    line-height: 22px;
    text-align: center;
    font-size: 11px;
    color: #7c6cf8;
    font-family: 'Space Mono', monospace;
    margin-right: 8px;
}
</style>
""", unsafe_allow_html=True)

# ── Cached Resources ──────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading embedding model...")
def load_embed_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

@st.cache_resource(show_spinner="Setting up vector store...")
def load_vector_store():
    chroma = chromadb.EphemeralClient()
    return chroma.get_or_create_collection("papers")

@st.cache_resource(show_spinner="Setting up Arxiv Client...")
def load_arxivclient():
    return arxiv.Client(
    delay_seconds=10,
    num_retries=3
    )

embed_model = load_embed_model()
store = load_vector_store()
arxClient = load_arxivclient()
# ── LLM Helper ────────────────────────────────────────────────────────────────
def get_llm():
    return OpenAI(api_key=groq_key, base_url="https://api.groq.com/openai/v1")

def ask(prompt):
    llm = get_llm()
    resp = llm.chat.completions.create(
        model=LLM_MODEL,
        messages=[{"role": "user", "content": prompt}]
    )
    return resp.choices[0].message.content

# ── RAG Pipeline ──────────────────────────────────────────────────────────────
def get_papers(query, max_results=5):
    url = 'https://export.arxiv.org/api/query'
    search = arxiv.Search(
        query=query,
        max_results=3,
        sort_by=arxiv.SortCriterion.Relevance
    )
    time.sleep(4)
    papers = []
    for paper in arxClient.results(search):
        title = paper.title
        sum = paper.summary
        authors = [i.name for i in paper.authors]
        category = paper.categories
        link = paper.links[1]
        papers.append({"title": title, "summary": sum, "authors": authors,"category":category,"link":str(link)})
    return papers

# Paste this at the top of your script / notebook, then call generation() normally

def get_papers_demo(query, max_results=5):
    return [
        {
            "title": "Wave-Particle Duality and the Double-Slit Experiment",
            "summary": "We examine the double-slit experiment as the canonical demonstration of quantum superposition. When both slits are open, the wavefunction passes through both simultaneously and interferes with itself, producing an interference pattern on the screen. This pattern is not the sum of two single-slit diffraction patterns because the cross-terms in |ψ1 + ψ2|² — the interference terms — are non-zero. Closing one slit destroys the superposition and eliminates these cross-terms, recovering classical addition of intensities.",
            "authors": ["Richard Feynman", "Albert Hibbs"],
            "category": ["quant-ph", "physics.optics"],
            "link": "https://arxiv.org/abs/quant-ph/0001001",
            "published": "2001-03-15",
        },
        {
            "title": "Quantum Interference and the Failure of Classical Probability",
            "summary": "Classical probability predicts that the probability of an event through two paths is the sum of individual probabilities: P = P1 + P2. Quantum mechanics replaces probabilities with probability amplitudes (complex numbers), so P = |A1 + A2|² = |A1|² + |A2|² + 2Re(A1*A2). The third term, 2Re(A1*A2), is the interference term and has no classical analogue. In the double-slit experiment this term produces bright and dark fringes. When only one slit is open, A2=0 and the interference term vanishes, giving simple single-slit diffraction with no fringes.",
            "authors": ["Paul Dirac", "John von Neumann"],
            "category": ["quant-ph"],
            "link": "https://arxiv.org/abs/quant-ph/0002002",
            "published": "2003-07-22",
        },
        {
            "title": "Path Integrals and Multi-Slit Interference",
            "summary": "Using Feynman's path integral formulation, every path a particle can take contributes a complex amplitude e^(iS/ℏ) where S is the classical action. In the double-slit setup, paths through slit 1 and paths through slit 2 have different path lengths and therefore different phases. Summing amplitudes before squaring — not squaring before summing — is what produces interference. When which-path information is available (even in principle), the cross-terms average to zero and the pattern collapses to a classical sum, demonstrating the role of quantum coherence.",
            "authors": ["Murray Gell-Mann", "James Hartle"],
            "category": ["quant-ph", "physics.gen-ph"],
            "link": "https://arxiv.org/abs/quant-ph/0003003",
            "published": "2005-11-10",
        },
        {
            "title": "Decoherence and the Quantum-to-Classical Transition in Slit Experiments",
            "summary": "Decoherence theory explains why macroscopic objects do not show double-slit interference even though they obey quantum mechanics. When the environment (air molecules, photons) entangles with the particle and records which-path information, the off-diagonal elements of the density matrix — precisely the interference terms — decay exponentially fast. The resulting mixed state produces an intensity pattern indistinguishable from classical P = P1 + P2. This framework unifies the quantum measurement problem with the observed classical behaviour of large objects.",
            "authors": ["Wojciech Zurek", "Erich Joos"],
            "category": ["quant-ph", "cond-mat.mes-hall"],
            "link": "https://arxiv.org/abs/quant-ph/0004004",
            "published": "2008-02-28",
        },
    ]

def indexing(topic, max_results=3):
    query = query_construct(topic)
    papers = get_papers(query, max_results)
    saved = 0
    for p in papers:
        uid = hashlib.md5(p['title'].encode()).hexdigest()
        try:
            if store.get(ids=[uid])['ids']:
                continue
        except Exception:
            pass
        small = distill(p['summary'])
        full_text = f"{p['title']}. {p['summary']}"
        embedding = embed_model.encode(small).tolist()
        store.add(
            documents=[full_text],
            embeddings=[embedding],
            metadatas=[{
                "title": p["title"],
                "authors": ", ".join(p["authors"][:3]),
                "category": ", ".join(p["category"][:2]),
                "link" : p['link']
            }],
            ids=[uid]
        )
        saved += 1
    st.write(f"Stored {saved} new papers in store.")

def retrival(query, n_res=5):
    count = store.count()
    if count == 0:
        return []
    n_res = min(n_res, count)
    embedded = embed_model.encode(query).tolist()
    res = store.query(query_embeddings=[embedded], n_results=n_res)
    docs = res['documents'][0]
    ids = res['ids'][0]
    titles = [m["title"] for m in res['metadatas'][0]]
    return list(zip(titles, docs,ids))

def query_construct(question):
    prompt = f"""
    You are an AI research assistant. Given a question, generate the best possible query to search for the arxiv api for the question : {question}
    ti:	Title
    au:	Author
    abs:	Abstract
    co	:Comment
    jr	:Journal Reference
    cat	:Subject Category;
    Use Boolean logic, synonyms, field prefixes mentioned above, and relevant categories with `cat:`. Prefer precision over noise. Only output the search query, an example of the query for question anti-gravity is
    (ti:antigravity OR ti:"anti-gravity" OR abs:antigravity OR abs:"anti-gravity" OR all:antigravity OR all:"anti-gravity" OR all:"repulsive gravity" OR all:"negative mass") AND (cat:physics.gen-ph OR cat:gr-qc OR cat:physics.class-ph)
    and only output the query nothing else
    """
    resp = ask(prompt)
    return resp

def multi_query_generation(question, n=3):
    prompt = f"""You are an AI research assistant. Given a question, generate {n} different
    search queries to find relevant papers on arXiv.

    Each query should be 3-6 plain keywords, no quotes, no site: operators, no numbering.
    Return ONLY the queries, one per line.

    Question: {question}"""
    resp = ask(prompt)
    print("Multi-query generation completed")
    queries = resp.split('\n')
    queries = [q.strip() for q in queries if q.strip()]  # remove empty lines
    return queries

def grade(question, doc):
    prompt = f"""Grade relevance of this research paper to the question.
    Question: {question}
    Document: {doc[:400]}
    Return ONLY a single digit — 1 (relevant), 2 (partial), or 3 (irrelevant)."""
    try:
        result = ask(prompt).strip().strip('.')[0]
        return int(result)
    except Exception:
        return 1

def distill(text):
    return ask(f"""Distill the following research paper summary into 3-4 sentences capturing only the core idea, key method, and main finding. Return only the distilled summary, no preamble.
    Summary: {text}""")

def reciprocal_rank_fusion(results_list, k=60):
    scores = {}
    for results in results_list:
        for rank, doc in enumerate(results):
            key = doc if not isinstance(doc, list) else str(doc)
            scores[key] = scores.get(key, 0) + 1 / (rank + k)
    return sorted(scores, key=scores.get, reverse=True)

def fusion_retrival(question, n=3):
    st.write("⚖️ Grading relevance & fusing results...")
    queries = multi_query_generation(question,n)
    all_context = []
    for query in queries:
        indexing(query)
        retrieved = retrival(query)
        temp = []
        for title, doc,id in retrieved:
            relevance = grade(question, doc)
            if relevance == 1:
                temp.append((title, doc,id))
            elif relevance == 2:
                temp.append((title, distill(doc),id))
        all_context.append(temp)
    ranked = reciprocal_rank_fusion(all_context)
    return ranked[:n]

def generation(question):
    results = fusion_retrival(question)
    sources = []
    context_parts = []
    for item in results:
        if isinstance(item, tuple) and len(item) == 3:
            title, doc,id = item
            sources.append((title,id))
            context_parts.append(f"[{title}]:\n{doc}")
        else:
            context_parts.append(str(item))

    if not context_parts:
        return ask(question), []

    context = "\n\n".join(context_parts)
    prompt = f"""You are a research assistant answering questions based on provided papers.
        {context}
        Question: {question}
        Write a clear, concise answer in 5-6 sentences. Cite papers by their actual title in brackets like [Title].
        Do not repeat the same point twice. Write in plain English, not bullet points."""
    return ask(prompt), sources

# ── Session State ─────────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []
if "total_queries" not in st.session_state:
    st.session_state.total_queries = 0

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚗️ ArXiv Agent")
    st.markdown("<p style='color:#44445a;font-size:12px;margin-top:-10px;font-family:Space Mono'>RAG + CRAG + Fusion</p>", unsafe_allow_html=True)
    st.divider()


    st.divider()
    st.markdown("### 📚 Paper Library")

    col1, col2 = st.columns(2)
    with col1:
        if st.button("🔄 Refresh", use_container_width=True):
            st.rerun()
    with col2:
        if st.button("🗑️ Clear chat", use_container_width=True):
            st.session_state.messages = []
            st.rerun()

    all_papers = store.get(include=["metadatas", "documents"])
    paper_count = len(all_papers["ids"]) if all_papers["ids"] else 0

    st.markdown(f"""
    <div class="stat-row">
        <div class="stat-box">
            <div class="stat-num">{paper_count}</div>
            <div class="stat-label">papers</div>
        </div>
        <div class="stat-box">
            <div class="stat-num">{st.session_state.total_queries}</div>
            <div class="stat-label">queries</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    if paper_count > 0:
        search_filter = st.text_input("🔍 Filter", placeholder="Search paper titles...", label_visibility="collapsed")
        st.markdown(f"<p style='color:#44445a;font-size:11px;margin-bottom:8px'>{paper_count} paper{'s' if paper_count != 1 else ''} indexed</p>", unsafe_allow_html=True)

        shown = 0
        for meta in all_papers["metadatas"]:
            title = meta.get("title", "Unknown")
            authors = meta.get("authors", "")
            category = meta.get("category", "")
            link = meta.get("link","")
            if search_filter and search_filter.lower() not in title.lower():
                continue

            short_authors = authors[:45] + "..." if len(authors) > 45 else authors
            cats = [c.strip() for c in category.split(",") if c.strip()]
            cat_chips = "".join([f'<span class="paper-cat">{c}</span> ' for c in cats[:2]])

            st.markdown(f"""
                <a href="{link}" target="_blank" rel="noopener noreferrer" style="text-decoration:none; color:inherit;">
                <div class="paper-card" >
                    <div class="paper-title">{title}</div>
                    <div class="paper-authors">{short_authors}</div>
                    <div style="margin-top:5px">{cat_chips}</div>
                </div>
                </a>
            """, unsafe_allow_html=True)
            shown += 1

        if search_filter and shown == 0:
            st.markdown("<p style='color:#44445a;font-size:12px'>No matches found.</p>", unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="empty-state">
            <div style="font-size:28px;margin-bottom:8px">📭</div>
            No papers yet.<br>Ask a question to start indexing.
        </div>
        """, unsafe_allow_html=True)

# ── Main Area ─────────────────────────────────────────────────────────────────
st.markdown('<div class="hero-title">ArXiv Research Agent</div>', unsafe_allow_html=True)
st.markdown('<div class="hero-sub">multi-query · rag fusion · crag · multi-representation</div>', unsafe_allow_html=True)

# How it works expander
with st.expander("⚙️ How this works"):
    st.markdown("""
    <div style="font-size:14px;color:#888899;line-height:1.8">
        <span class="step-badge">1</span> Your question is <b style="color:#c0b8f5">rewritten into 5 search queries</b> from different angles<br>
        <span class="step-badge">2</span> Each query <b style="color:#c0b8f5">fetches papers from arXiv</b> and indexes them with distilled embeddings<br>
        <span class="step-badge">3</span> Retrieved papers are <b style="color:#c0b8f5">graded for relevance</b> — irrelevant ones are filtered, partial ones are distilled<br>
        <span class="step-badge">4</span> <b style="color:#c0b8f5">Reciprocal Rank Fusion</b> merges all result sets into one ranked list<br>
        <span class="step-badge">5</span> Top papers become context for <b style="color:#c0b8f5">cited answer generation</b>
    </div>
    """, unsafe_allow_html=True)

st.divider()

# Chat history
for msg in st.session_state.messages:
    if msg["role"] == "user":
        st.markdown(f'<div class="user-msg">&nbsp;&nbsp;{msg["content"]}</div>', unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="assistant-msg">🔬&nbsp;&nbsp;{msg["content"]}</div>', unsafe_allow_html=True)
        sources = msg.get("sources", [])
        if sources:
            seen = set()
            chips = ""
            for s in sources:
                if s not in seen:
                    seen.add(s)
                    label = s[:55] + "..." if len(s) > 55 else s
                    chips += f'<span class="source-chip">📄 {label[0]}</span>'
            st.markdown(f'<div class="source-row">{chips}</div>', unsafe_allow_html=True)

# Empty state
if not st.session_state.messages:
    st.markdown("""
    <div class="empty-state" style="padding:60px 40px">
        <div style="font-size:40px;margin-bottom:12px">🔬</div>
        <div style="font-size:16px;color:#555570;margin-bottom:8px">Ask any research question</div>
        <div style="font-size:13px;color:#333350">
            Try: "How is machine learning used in drug discovery?" <br>
            or "What are recent advances in quantum error correction?"
        </div>
    </div>
    """, unsafe_allow_html=True)

# Input
question = st.chat_input("Ask a research question...")

if question:
    if False:
        st.error("⚠️ Please enter your Groq API key in the sidebar first.")
    else:
        st.session_state.messages.append({"role": "user", "content": question})
        st.session_state.total_queries += 1
        st.markdown(f'<div class="user-msg">💻&nbsp;&nbsp;{question}</div>', unsafe_allow_html=True)

        with st.status("Researching arXiv...", expanded=True) as status:
            # st.write("🔍 Generating search queries...")
            # queries = multi_query_generation(question)
            #
            # st.write(f"📥 Fetching & indexing papers for {len(queries)} queries...")
            # for q in queries:
            #     indexing(q)
            #
            # st.write("⚖️ Grading relevance & fusing results...")
            # results = fusion_retrival(question)

            st.write("✍️ Generating cited answer...")
            answer, sources = generation(question)

            status.update(label="Done!", state="complete", expanded=False)

        st.session_state.messages.append({
            "role": "assistant",
            "content": answer,
            "sources": sources
        })

        st.markdown(f'<div class="assistant-msg">🔬&nbsp;&nbsp;{answer}</div>', unsafe_allow_html=True)

        if sources:
            seen = set()
            chips = ""
            for s in sources:
                if s not in seen:
                    seen.add(s)
                    label = s[:55] + "..." if len(s) > 55 else s
                    chips += f'<span class="source-chip">📄 {label}</span>'
            st.markdown(f'<div class="source-row">{chips}</div>', unsafe_allow_html=True)

        st.rerun()