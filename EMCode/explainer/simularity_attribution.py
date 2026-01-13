# python
import os

import torch
from transformers import AutoTokenizer
import torch.nn.functional as f

from EMCode.model.em_small_test_model import EMModel
from EMCode.scripts.data_loader import infer_left_right_columns_from_csv, load_data_from_file, project_root

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Pfade relativ zur Root
ROOT = project_root()
MODEL_PATH = os.path.abspath(os.path.join(ROOT, "models/small_test_model.pt"))
DATASET_PATH = os.path.abspath(os.path.join(ROOT, "datasets/"))


# -------------------------
# Model + tokenizer loading
# -------------------------
def load_model(model_path):
    """
    Loads tokenizer and trained EM EMCode.
    """
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    model = EMModel().to(DEVICE)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.eval()
    return tokenizer, model


# -------------------------
# Embedding extraction
# -------------------------
def encode_text(tokenizer, model, text):
    """
    Encodes a single text into a CLS embedding.
    """
    enc = tokenizer(text, return_tensors="pt", truncation=True, padding=True).to(DEVICE)

    with torch.no_grad():
        embedding = model.encode(
            enc["input_ids"],
            enc["attention_mask"],
            enc.get("token_type_ids")
        )

    return embedding.squeeze(0)


# -------------------------
# Similarity Attribution
# -------------------------
def similarity_attribution(
        query_text,
        reference_data,
        tokenizer,
        model,
        top_k=5
):
    """
    Explains a query by retrieving top-k most similar reference examples.
    """
    query_emb = encode_text(tokenizer, model, query_text)

    similarities = []

    for idx, ref in enumerate(reference_data):
        ref_emb = encode_text(tokenizer, model, ref["text"])
        sim = f.cosine_similarity(query_emb, ref_emb, dim=0).item()

        similarities.append({
            "index": idx,
            "similarity": sim,
            "label": ref["label"],
            "text": ref["text"]
        })

    similarities.sort(key=lambda x: x["similarity"], reverse=True)
    return similarities[:top_k]


# -------------------------
# Display
# -------------------------
def print_similarity_results(results):
    """
    Prints similarity attribution results.
    """
    print("\nTop similar examples:\n")
    for r in results:
        print(f"Index: {r['index']}")
        print(f"Similarity: {r['similarity']:.4f}")
        print(f"Label: {r['label']}")
        print(f"Text: {r['text'][:200]}...")
        print("-" * 50)


# -------------------------
# Main
# -------------------------
def main():
    tokenizer, model = load_model(MODEL_PATH)
    left_cols, right_cols = infer_left_right_columns_from_csv(os.path.join(DATASET_PATH, "test_short.csv"))
    test_data = load_data_from_file(
        os.path.join(DATASET_PATH, "test_short.csv"),
        text_cols_left=left_cols,
        text_cols_right=right_cols)

    query_sample = test_data[1]

    results = similarity_attribution(
        query_sample["text"],
        test_data,
        tokenizer,
        model,
        top_k=5
    )

    print("Query example:")
    print(query_sample["text"])
    print_similarity_results(results)


if __name__ == "__main__":
    main()
