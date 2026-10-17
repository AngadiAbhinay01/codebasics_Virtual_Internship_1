
from data_ingestion import (
    retrieve_pmids,
    fetch_articles,
    prepare_documents
)
from chroma_store import ChromaStore
from rag_pipeline import MediAssistRAG


MAX_ARTICLES = 300

SEARCH_TERM = (
    '("intermittent fasting"[Title/Abstract]) AND '
    '("obesity" OR "type 2 diabetes" OR "metabolic health")'
)


def ingest_articles():
    """Retrieve PubMed articles and store them in ChromaDB."""

    print("=" * 60)
    print("MediAssist AI - Data Ingestion Pipeline")
    print("=" * 60)

    # Step 1: Retrieve PMIDs
    pmids = retrieve_pmids(
        search_term=SEARCH_TERM,
        max_results=MAX_ARTICLES
    )

    if not pmids:
        print("No PMIDs found. Check the search term or API response.")
        return False

    print(f"\nRetrieved {len(pmids)} PMIDs.")

    # Step 2: Fetch article details
    articles = fetch_articles(pmids)

    if not articles:
        print("No article records were retrieved.")
        return False

    print(f"Fetched {len(articles)} article records.")

    # Step 3: Prepare documents
    documents = prepare_documents(articles)

    if not documents:
        print("No documents with usable abstracts were found.")
        return False

    # Step 4: Store documents in ChromaDB
    vector_store = ChromaStore()
    vector_store.ingest_documents(documents)

    # Step 5: Validate ingestion
    print("\nIngestion completed successfully.")
    print("Total documents in collection:",
          vector_store.get_collection_count())

    return True


def test_retrieval():
    """Test natural-language retrieval from ChromaDB."""

    print("\n" + "=" * 60)
    print("Testing ChromaDB Retrieval")
    print("=" * 60)

    vector_store = ChromaStore()

    if vector_store.get_collection_count() == 0:
        print("The vector store is empty. Ingest articles first.")
        return

    query = (
        "What are the effects of intermittent fasting "
        "on body weight and type 2 diabetes?"
    )

    print("\nQuery:", query)

    results = vector_store.search_documents(
        query=query,
        n_results=3
    )

    for index, result in enumerate(results, start=1):
        metadata = result.get("metadata", {})

        print(f"\nResult {index}")
        print("PMID:", metadata.get("pmid", "Unknown"))
        print("Title:", metadata.get("title", "Unknown"))
        print("Journal:", metadata.get("journal", "Unknown"))
        print("Distance:", round(result["distance"], 4))
        print("Content:", result["text"][:500], "...")


def healthcare_qa():
    """Answer healthcare questions using RAG and Groq Llama."""

    print("\n" + "=" * 60)
    print("MediAssist AI - Healthcare Q&A")
    print("=" * 60)

    rag = MediAssistRAG()

    if rag.vector_store.get_collection_count() == 0:
        print("The vector store is empty. Ingest articles first.")
        return

    while True:
        question = input(
            "\nEnter your healthcare question "
            "(or type 'exit' to quit): "
        ).strip()

        if question.lower() in {"exit", "quit"}:
            print("Exiting MediAssist AI.")
            break

        if not question:
            print("Please enter a valid question.")
            continue

        try:
            print("\nRetrieving research and generating an answer...\n")

            answer = rag.answer_question(
                query=question,
                n_results=5
            )

            print("-" * 60)
            print("MediAssist AI Answer")
            print("-" * 60)
            print(answer)

        except Exception as error:
            print(f"Unable to generate an answer: {error}")


def main():
    print("\nWelcome to MediAssist AI")

    print("\n1. Ingest PubMed articles")
    print("2. Test vector-store retrieval")
    print("3. Ask a healthcare question (RAG)")
    print("4. Exit")

    while True:
        choice = input("\nSelect an option (1-4): ").strip()

        if choice == "1":
            ingest_articles()

        elif choice == "2":
            test_retrieval()

        elif choice == "3":
            healthcare_qa()

        elif choice == "4":
            print("Thank you for using MediAssist AI.")
            break

        else:
            print("Invalid choice. Please enter 1, 2, 3, or 4.")


if __name__ == "__main__":
    main()