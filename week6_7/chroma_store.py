
import chromadb
from chromadb.utils.embedding_functions import (
    SentenceTransformerEmbeddingFunction
)


class ChromaStore:
    def __init__(
        self,
        persist_directory="chroma_db",
        collection_name="mediassist_articles"
    ):
        # Persistent local vector database
        self.client = chromadb.PersistentClient(
            path=persist_directory
        )

        # Embedding model converts text into vectors
        self.embedding_function = SentenceTransformerEmbeddingFunction(
            model_name="all-MiniLM-L6-v2"
        )

        # Create the collection if it does not already exist
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            embedding_function=self.embedding_function,
            metadata={"description": "PubMed intermittent fasting articles"}
        )

    def ingest_documents(self, documents):
        """Insert or update article documents in ChromaDB."""
        if not documents:
            print("No documents available for ingestion.")
            return

        self.collection.upsert(
            ids=[str(doc["pmid"]) for doc in documents],
            documents=[doc["text"] for doc in documents],
            metadatas=[doc["metadata"] for doc in documents]
        )

        print(f"Processed {len(documents)} articles.")

    def search_documents(self, query, n_results=5):
        """Retrieve the most relevant articles for a query."""
        total_documents = self.collection.count()

        if total_documents == 0:
            return []

        results = self.collection.query(
            query_texts=[query],
            n_results=min(n_results, total_documents),
            include=["documents", "metadatas", "distances"]
        )

        retrieved_articles = []

        for i, document in enumerate(results["documents"][0]):
            retrieved_articles.append({
                "text": document,
                "metadata": results["metadatas"][0][i],
                "distance": results["distances"][0][i]
            })

        return retrieved_articles

    def get_collection_count(self):
        """Return the number of stored articles."""
        return self.collection.count()