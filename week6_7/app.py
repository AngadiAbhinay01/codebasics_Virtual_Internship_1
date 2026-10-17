"""MediAssist AI - Streamlit user interface.

Run with:  streamlit run app.py
"""

import os
from pathlib import Path

import pandas as pd
import streamlit as st
from dotenv import load_dotenv

# Make .env and chroma_db resolve relative to the project folder, regardless
# of the directory Streamlit was launched from.
BASE_DIR = Path(__file__).resolve().parent
os.chdir(BASE_DIR)
load_dotenv(BASE_DIR / ".env", override=True)

from chroma_store import ChromaStore
from data_ingestion import (
    MAX_ARTICLES,
    fetch_articles,
    prepare_documents,
    retrieve_pmids,
)
from rag_pipeline import MediAssistRAG

DEFAULT_SEARCH_TERM = (
    '("intermittent fasting"[Title/Abstract]) AND '
    '("obesity" OR "type 2 diabetes" OR "metabolic health")'
)

st.set_page_config(
    page_title="MediAssist AI",
    page_icon="🩺",
    layout="wide",
)


# --------------------------------------------------------------------------
# Cached resources (loaded once per server process, not on every rerun)
# --------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading embedding model and vector store...")
def get_store():
    return ChromaStore()


@st.cache_resource(show_spinner="Connecting to the language model...")
def get_rag():
    """Raises on failure, so a failed attempt is never cached."""
    if not os.getenv("GROQ_API_KEY"):
        try:  # deployed apps (e.g. Streamlit Cloud) keep the key in st.secrets
            os.environ["GROQ_API_KEY"] = st.secrets["GROQ_API_KEY"]
        except Exception:
            pass
    rag = MediAssistRAG()
    rag.vector_store = get_store()  # share one store instance
    return rag


store = get_store()
try:
    rag, rag_error = get_rag(), None
except Exception as error:
    rag, rag_error = None, str(error)


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------
def abstract_to_text(abstract):
    if isinstance(abstract, dict):
        return "\n\n".join(
            f"**{label}:** {text}" for label, text in abstract.items() if text
        )
    return str(abstract or "")


def has_abstract(article):
    abstract = article.get("abstract")
    if isinstance(abstract, dict):
        return bool(abstract) and abstract != {"SUMMARY": "No Abstract"}
    return bool(abstract)


def build_results_table(articles):
    rows = []
    for article in articles:
        usable = has_abstract(article)
        rows.append(
            {
                "Select": usable,
                "PMID": str(article["pmid"]),
                "Title": article.get("title") or "No Title",
                "Journal": article.get("journal") or "Unknown",
                "Year": article.get("publication_date") or "Unknown",
                "Abstract": usable,
            }
        )
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# Session state
# --------------------------------------------------------------------------
st.session_state.setdefault("articles", [])
st.session_state.setdefault("search_id", 0)
st.session_state.setdefault("last_search_term", "")
st.session_state.setdefault("qa_result", None)


# --------------------------------------------------------------------------
# Sidebar, part 1: PubMed search
# --------------------------------------------------------------------------
with st.sidebar:
    st.title("🩺 MediAssist AI")
    st.subheader("1. Search PubMed")

    search_term = st.text_area(
        "PubMed query",
        value=DEFAULT_SEARCH_TERM,
        height=120,
        help="Standard PubMed syntax works, e.g. "
        '("keto diet"[Title/Abstract]) AND "epilepsy".',
    )
    max_results = st.slider(
        "Maximum articles",
        min_value=10,
        max_value=MAX_ARTICLES,
        value=50,
        step=10,
        help="Larger searches take longer because PubMed requests are throttled.",
    )

    if st.button("🔍 Search PubMed", use_container_width=True, type="primary"):
        if not search_term.strip():
            st.warning("Enter a search query first.")
        else:
            try:
                with st.spinner("Searching PubMed and fetching abstracts..."):
                    pmids = retrieve_pmids(search_term.strip(), max_results)
                    articles = fetch_articles(pmids) if pmids else []
                if not articles:
                    st.session_state.articles = []
                    st.warning("No articles found. Try broadening the query.")
                else:
                    st.session_state.articles = articles
                    st.session_state.search_id += 1  # resets the table selection
                    st.session_state.last_search_term = search_term.strip()
            except Exception as error:
                st.error(f"PubMed search failed: {error}")


# --------------------------------------------------------------------------
# Main page
# --------------------------------------------------------------------------
st.title("MediAssist AI")
st.caption(
    "Search PubMed, build a research knowledge base, and ask questions "
    "grounded in the ingested abstracts."
)
st.info(
    "This tool summarises published research abstracts for information only. "
    "It is not medical advice; consult a qualified clinician for health decisions.",
    icon="⚠️",
)

tab_ask, tab_results = st.tabs(["💬 Ask a question", "📄 PubMed search results"])

# ---- Tab: search results (must render before the sidebar ingest button) ----
selected_pmids = []
with tab_results:
    articles = st.session_state.articles
    if not articles:
        st.write("Use the sidebar to search PubMed. Results will appear here.")
    else:
        df = build_results_table(articles)
        st.write(
            f"**{len(df)}** articles found for: `{st.session_state.last_search_term}`. "
            f"**{int(df['Abstract'].sum())}** have an abstract and can be ingested. "
            "Untick any article you want to leave out."
        )
        edited = st.data_editor(
            df,
            key=f"results_editor_{st.session_state.search_id}",
            hide_index=True,
            use_container_width=True,
            height=420,
            disabled=["PMID", "Title", "Journal", "Year", "Abstract"],
            column_config={
                "Select": st.column_config.CheckboxColumn("Select", width="small"),
                "PMID": st.column_config.TextColumn("PMID", width="small"),
                "Title": st.column_config.TextColumn("Title", width="large"),
                "Year": st.column_config.TextColumn("Year", width="small"),
                "Abstract": st.column_config.CheckboxColumn(
                    "Has abstract", width="small"
                ),
            },
        )
        selected_pmids = edited.loc[
            edited["Select"] & edited["Abstract"], "PMID"
        ].tolist()

        with st.expander("Preview an abstract"):
            preview_pmid = st.selectbox(
                "Article", options=df["PMID"].tolist(),
                format_func=lambda p: f"{p} - "
                f"{df.loc[df['PMID'] == p, 'Title'].iloc[0][:90]}",
            )
            preview = next(a for a in articles if str(a["pmid"]) == preview_pmid)
            st.markdown(f"**{preview.get('title', 'No Title')}**")
            st.caption(
                f"{preview.get('journal', '')} · {preview.get('publication_date', '')} · "
                f"{preview.get('authors', '')}"
            )
            st.markdown(abstract_to_text(preview.get("abstract")))

# ---- Sidebar, part 2: ingest + store status ----
with st.sidebar:
    st.divider()
    st.subheader("2. Ingest into vector store")
    st.write(f"Selected articles: **{len(selected_pmids)}**")

    if st.button(
        "📥 Ingest selected documents",
        use_container_width=True,
        disabled=not selected_pmids,
    ):
        try:
            chosen = [
                a for a in st.session_state.articles
                if str(a["pmid"]) in set(selected_pmids)
            ]
            with st.spinner("Embedding and storing documents..."):
                documents = prepare_documents(chosen)
                store.ingest_documents(documents)
            st.success(f"Ingested {len(documents)} articles.")
        except Exception as error:
            st.error(f"Ingestion failed: {error}")

    st.divider()
    st.metric("Articles in vector store", store.get_collection_count())

    with st.expander("Danger zone"):
        confirm = st.checkbox("I want to delete all stored articles")
        if st.button("🗑️ Clear vector store", disabled=not confirm):
            ids = store.collection.get(include=[])["ids"]
            if ids:
                store.collection.delete(ids=ids)
            st.session_state.qa_result = None
            st.rerun()

# ---- Tab: question answering ----
with tab_ask:
    if store.get_collection_count() == 0:
        st.warning(
            "The vector store is empty. Search PubMed and ingest articles "
            "from the sidebar before asking questions."
        )
    if rag is None:
        st.warning(
            f"AI answer generation is unavailable ({rag_error}). "
            "Retrieved articles will still be shown."
        )

    with st.form("query_form", clear_on_submit=False):
        question = st.text_input(
            "Ask a question about the ingested articles",
            placeholder="What are the effects of intermittent fasting on "
            "body weight and type 2 diabetes?",
        )
        n_results = st.slider("Articles to retrieve", 1, 10, 5)
        submitted = st.form_submit_button("Ask", type="primary")

    if submitted:
        if not question.strip():
            st.warning("Please enter a question.")
        elif store.get_collection_count() == 0:
            st.warning("Ingest some articles first.")
        else:
            try:
                with st.spinner("Retrieving research and generating an answer..."):
                    sources = store.search_documents(question.strip(), n_results)
                    answer = None
                    if rag is not None:
                        answer = rag.answer_question(
                            query=question.strip(), n_results=n_results
                        )
                st.session_state.qa_result = {
                    "question": question.strip(),
                    "answer": answer,
                    "sources": sources,
                }
            except Exception as error:
                st.session_state.qa_result = None
                st.error(f"Unable to generate an answer: {error}")

    result = st.session_state.qa_result
    if result:
        st.divider()
        st.subheader("Answer")
        if result["answer"]:
            st.markdown(result["answer"])
        else:
            st.write("No generated answer (language model not configured).")

        st.subheader(f"Retrieved sources ({len(result['sources'])})")
        for i, source in enumerate(result["sources"], start=1):
            meta = source.get("metadata", {})
            pmid = meta.get("pmid", "Unknown")
            with st.expander(
                f"{i}. {meta.get('title', 'Untitled')} (PMID {pmid})"
            ):
                st.caption(
                    f"{meta.get('journal', 'Unknown journal')} · "
                    f"{meta.get('publication_date', 'Unknown year')} · "
                    f"distance {source['distance']:.3f} (lower = more relevant)"
                )
                st.write(source["text"])
                st.markdown(
                    f"[Open on PubMed](https://pubmed.ncbi.nlm.nih.gov/{pmid}/)"
                )
