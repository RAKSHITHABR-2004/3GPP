import json
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from pathlib import Path
from chromadb import PersistentClient
import os
import subprocess
import sys
from langchain_huggingface import HuggingFaceEmbeddings


CHROMA_PATH = os.getenv("CHROMA_PATH", os.path.join(os.path.dirname(__file__), "../chroma_storage"))
print("Using ChromaDB path:", CHROMA_PATH)
client = PersistentClient(path=CHROMA_PATH)

version1_name = os.getenv("VERSION1", "re10")
version2_name = os.getenv("VERSION2", "re17")

existing_collections = [col.name for col in client.list_collections()]

missing = []
if version1_name not in existing_collections:
    print(f'"{version1_name}" embedding not present. Provide this "{version1_name}" file.')
    missing.append(version1_name)

if version2_name not in existing_collections:
    print(f'"{version2_name}" embedding not present. Provide this "{version2_name}" file.')
    missing.append(version2_name)

if missing:
    for i in range(len(missing)):
        print(f"Running embed2.py ({i+1}/{len(missing)}) to generate missing embeddings...")
        subprocess.run([sys.executable, str(Path(__file__).parent / "embed2.py")], check=True)

BASE_DIR = os.path.dirname(__file__)
CHROMA_PATH = os.path.join(BASE_DIR, "../chroma_storage")

client = PersistentClient(path=CHROMA_PATH)
collection_v1 = client.get_collection(name=version1_name)
collection_v2 = client.get_collection(name=version2_name)


version_1_raw = collection_v1.get(include=["embeddings", "documents"])
version_2_raw = collection_v2.get(include=["embeddings", "documents"])

version_1_embeddings = [
    {"id": id_, "text": doc, "embedding": embedding}
    for id_, doc, embedding in zip(version_1_raw["ids"], version_1_raw["documents"], version_1_raw["embeddings"])
]
version_2_embeddings = [
    {"id": id_, "text": doc, "embedding": embedding}
    for id_, doc, embedding in zip(version_2_raw["ids"], version_2_raw["documents"], version_2_raw["embeddings"])
]


v1_embeds = np.array([item["embedding"] for item in version_1_embeddings])
v2_embeds = np.array([item["embedding"] for item in version_2_embeddings])
similarity_matrix = cosine_similarity(v1_embeds, v2_embeds)


threshold = 0.85
matches = []
deleted = set()
added = set(range(len(version_2_embeddings)))

for i, row in enumerate(similarity_matrix):
    max_sim = max(row)
    j = np.argmax(row)
    if max_sim >= threshold:
        matches.append((i, j, max_sim))
        added.discard(j)
    else:
        deleted.add(i)


def classify_semantic_change_heuristic(text_v1, text_v2, jaccard_threshold=0.6):
    tokens_v1 = set(text_v1.lower().split())
    tokens_v2 = set(text_v2.lower().split())
    intersection = tokens_v1 & tokens_v2
    union = tokens_v1 | tokens_v2
    jaccard_sim = len(intersection) / len(union) if union else 0.0
    added_words = list(tokens_v2 - tokens_v1)
    removed_words = list(tokens_v1 - tokens_v2)

    if jaccard_sim >= jaccard_threshold:
        return {
            "status": "Unchanged",
            "jaccard_similarity": round(jaccard_sim, 2),
            "summary": "Minor or no change detected.",
            "added_words": added_words,
            "removed_words": removed_words
        }
    else:
        summary = "Semantic-level change detected. "
        if added_words:
            summary += f"Added: {', '.join(added_words)}. "
        if removed_words:
            summary += f"Removed: {', '.join(removed_words)}. "
        if not added_words and not removed_words:
            summary += "Structure or order changed."
        return {
            "status": "Modified",
            "jaccard_similarity": round(jaccard_sim, 2),
            "summary": summary,
            "added_words": added_words,
            "removed_words": removed_words
        }


results = {"unchanged": [], "deleted": [], "added": [], "modified": []}
version_mapping = []

for i, j, sim in matches:
    v1 = version_1_embeddings[i]
    v2 = version_2_embeddings[j]
    version_mapping.append({
        "id_v1": v1["id"],
        "id_v2": v2["id"],
        "similarity_score": round(sim, 3)
    })

    change_info = classify_semantic_change_heuristic(v1["text"], v2["text"])
    if change_info["status"] == "Unchanged":
        results["unchanged"].append({
            "id": v1["id"],
            "text": v1["text"],
            "jaccard_similarity": change_info["jaccard_similarity"]
        })
    else:
        results["modified"].append({
            "id_v1": v1["id"],
            "id_v2": v2["id"],
            "original": v1["text"],
            "modified": v2["text"],
            "jaccard_similarity": change_info["jaccard_similarity"],
            "summary_change": change_info["summary"],
            "added_words": change_info["added_words"],
            "removed_words": change_info["removed_words"]
        })

for i in deleted:
    results["deleted"].append({
        "id": version_1_embeddings[i]["id"],
        "text": version_1_embeddings[i]["text"]
    })

for j in added:
    results["added"].append({
        "id": version_2_embeddings[j]["id"],
        "text": version_2_embeddings[j]["text"]
    })


final_output = {
    "version_1": version1_name,
    "version_2": version2_name,
    "results": results,
    "version_mapping": version_mapping
}

output_file = f"change_detection_{version1_name}_vs_{version2_name}.json"
with open(output_file, "w", encoding="utf-8") as f:
    json.dump(final_output, f, ensure_ascii=False, indent=2)

print(f"Change detection completed. Results saved to '{output_file}'.")


with open(output_file, "r", encoding="utf-8") as f:
    summary_text = f.read()

embed_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
embedding = embed_model.embed_query(summary_text)

summary_collection_name = f"{version1_name}_vs_{version2_name}"
summary_collection = client.get_or_create_collection(name=summary_collection_name)

summary_collection.add(
    documents=[summary_text],
    embeddings=[embedding],
    ids=[f"{version1_name}_vs_{version2_name}_embedded"]
)

print(f"Embedded summary saved to ChromaDB collection '{summary_collection_name}'.")