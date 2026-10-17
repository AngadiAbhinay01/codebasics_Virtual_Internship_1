
from pubmed import PubMedRetriever


MAX_ARTICLES = 300


def retrieve_pmids(search_term, max_results=MAX_ARTICLES):
    """Retrieve unique PubMed IDs, capped at 300."""
    max_results = max(0, min(max_results, MAX_ARTICLES))

    if max_results == 0:
        return []

    pmids = PubMedRetriever.search_pubmed_articles(
        search_term=search_term,
        max_results=max_results
    )

    # Preserve order while removing duplicates
    unique_pmids = list(dict.fromkeys(pmids))

    print(f"Retrieved {len(unique_pmids)} unique PMIDs.")
    return unique_pmids[:MAX_ARTICLES]


def fetch_articles(pmids):
    """Fetch article details in batches of at most 100."""
    pmids = list(dict.fromkeys(pmids))[:MAX_ARTICLES]

    if not pmids:
        return []

    articles = PubMedRetriever.fetch_pubmed_abstracts(pmids)

    # Exclude malformed records without an identifier
    valid_articles = [
        article for article in articles
        if article.get("pmid")
    ]

    print(f"Fetched {len(valid_articles)} article records.")
    return valid_articles


def prepare_documents(articles):
    """Convert PubMed records into ChromaDB documents."""
    documents = []

    for article in articles:
        abstract_sections = article.get("abstract", {})

        if isinstance(abstract_sections, dict):
            abstract_text = " ".join(
                f"{section}: {text}"
                for section, text in abstract_sections.items()
                if text
            )
        else:
            abstract_text = str(abstract_sections or "")

        title = article.get("title") or "No Title"

        # Avoid storing empty or placeholder abstracts
        if not abstract_text.strip() or abstract_text.strip() == "SUMMARY: No Abstract":
            continue

        document_text = (
            f"Title: {title}\n"
            f"Abstract: {abstract_text}"
        )

        metadata = {
            "pmid": str(article["pmid"]),
            "title": title[:1000],
            "journal": str(
                article.get("journal") or "Unknown Journal"
            )[:500],
            "authors": str(
                article.get("authors") or "No Authors"
            )[:2000],
            "publication_date": str(
                article.get("publication_date") or "Unknown Year"
            )[:100]
        }

        documents.append({
            "pmid": str(article["pmid"]),
            "text": document_text,
            "metadata": metadata
        })

    print(f"Prepared {len(documents)} documents for ChromaDB.")
    return documents