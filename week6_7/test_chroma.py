
from chroma_store import ChromaStore

store = ChromaStore()

print("Total stored articles:", store.get_collection_count())

results = store.search_documents(
    "Intermittent fasting and insulin resistance",
    n_results=5
)

for result in results:
    print("\nPMID:", result["metadata"]["pmid"])
    print("Title:", result["metadata"]["title"])
    print("Distance:", result["distance"])