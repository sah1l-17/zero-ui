"""
Script to build all embeddings (product, category, intent patterns).
Run this script once whenever the source data changes.
"""

import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
import json
import os
import hashlib
import pickle
import sys

# Add parent directory to path to import shared utilities
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ===============================
# 🔧 Cache Configuration
# ===============================

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(SCRIPT_DIR)
CACHE_DIR = os.path.join(ROOT_DIR, "embeddings_cache")

PRODUCT_EMBEDDINGS_FILE = os.path.join(CACHE_DIR, "product_embeddings.npy")
CATEGORY_EMBEDDINGS_FILE = os.path.join(CACHE_DIR, "category_embeddings.npy")
PATTERN_EMBEDDINGS_FILE = os.path.join(CACHE_DIR, "pattern_embeddings.npy")
FAISS_INDEX_FILE = os.path.join(CACHE_DIR, "faiss_index.bin")
DF_CACHE_FILE = os.path.join(CACHE_DIR, "df_cache.pkl")
METADATA_FILE = os.path.join(CACHE_DIR, "cache_metadata.json")


def compute_file_hash(filepath):
    """Compute MD5 hash of a file to detect changes."""
    hasher = hashlib.md5()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def save_cache(excel_path, intents_path, df, embeddings, category_embeddings,
               categories, index, pattern_embeddings, pattern_texts, labels):
    """Save all embeddings and metadata to disk."""
    os.makedirs(CACHE_DIR, exist_ok=True)

    np.save(PRODUCT_EMBEDDINGS_FILE, embeddings)
    np.save(CATEGORY_EMBEDDINGS_FILE, category_embeddings)
    np.save(PATTERN_EMBEDDINGS_FILE, pattern_embeddings)
    faiss.write_index(index, FAISS_INDEX_FILE)

    with open(DF_CACHE_FILE, "wb") as f:
        pickle.dump({
            "df": df,
            "categories": categories,
            "pattern_texts": pattern_texts,
            "labels": labels,
        }, f)

    metadata = {
        "excel_hash": compute_file_hash(excel_path),
        "intents_hash": compute_file_hash(intents_path),
    }
    with open(METADATA_FILE, "w") as f:
        json.dump(metadata, f)

    print("✅ Embeddings cached to:", CACHE_DIR)


def build_embeddings():
    """Build all embeddings from scratch."""
    
    # ===============================
    # 1️⃣ File Paths
    # ===============================
    
    file_path = os.path.join(ROOT_DIR, "Products Data.xlsx")
    intents_path = os.path.join(ROOT_DIR, "intents.json")
    
    print("🔄 Building embeddings from source data...")
    print(f"📊 Excel file: {file_path}")
    print(f"📝 Intents file: {intents_path}")
    
    # ===============================
    # 2️⃣ Load Excel (All Sheets)
    # ===============================
    
    all_sheets = pd.read_excel(file_path, sheet_name=None)

    # Normalize column names PER SHEET before concat to avoid
    # duplicate columns (e.g. " Price" vs "Price" vs "Original Price")
    cleaned_sheets = []
    for sheet_name, sheet_df in all_sheets.items():
        sheet_df.columns = sheet_df.columns.astype(str).str.strip().str.lower()
        # Unify price column names → "price"
        price_renames = {}
        for col in sheet_df.columns:
            if col != "price" and "price" in col:
                price_renames[col] = "price"
        if price_renames:
            sheet_df = sheet_df.rename(columns=price_renames)
        cleaned_sheets.append(sheet_df)

    df = pd.concat(cleaned_sheets, ignore_index=True)
    
    print("Total Products (Raw):", len(df))
    
    # ===============================
    # 3️⃣ Clean & Normalize
    # ===============================
    
    df = df.fillna("")
    
    # Columns already normalized per-sheet; drop any remaining dupes
    df = df.loc[:, ~df.columns.duplicated()]
    df = df.map(lambda x: str(x).strip().lower())
    
    if "product type" not in df.columns:
        raise ValueError("Column 'product type' not found in Excel.")
    
    df["product type"] = df["product type"].replace("", np.nan)
    df = df.dropna(subset=["product type"])
    df = df.reset_index(drop=True)
    
    print("Total Products (After Cleaning):", len(df))
    
    # ===============================
    # 4️⃣ Build Embedding Text
    # ===============================
    
    def build_embedding_text(row):
        text = "This is a product listing.\n\n"
        for col in df.columns:
            if row[col] != "":
                text += f"{col}: {row[col]}\n"
        return text.strip()
    
    corpus = df.apply(build_embedding_text, axis=1).tolist()
    
    # ===============================
    # 5️⃣ Create Product Embeddings
    # ===============================
    
    print("📦 Loading BGE model for product embeddings...")
    bge_model = SentenceTransformer("BAAI/bge-base-en-v1.5")
    
    print("🔢 Creating product embeddings...")
    embeddings = bge_model.encode(
        corpus,
        batch_size=32,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True
    )
    
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)
    index.add(embeddings)
    
    print("FAISS index created with", index.ntotal, "products")
    
    # ===============================
    # 6️⃣ Category Embeddings
    # ===============================
    
    print("🏷️  Creating category embeddings...")
    categories = df["product type"].unique().tolist()
    
    category_texts = []
    for cat in categories:
        sample_products = df[df["product type"] == cat]["product name"].head(10).tolist()
        combined_text = f"Category: {cat}. Example products: " + ", ".join(sample_products)
        category_texts.append(combined_text)
    
    category_embeddings = bge_model.encode(
        category_texts,
        convert_to_numpy=True,
        normalize_embeddings=True
    )
    
    # ===============================
    # 7️⃣ Intent Pattern Embeddings
    # ===============================
    
    print("🎯 Creating intent pattern embeddings...")
    with open(intents_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    pattern_texts = []
    labels = []
    
    for intent in data["intents"]:
        tag = intent["tag"]
        for pattern in intent["patterns"]:
            pattern_texts.append(pattern)
            labels.append(tag)
    
    print("📦 Loading MiniLM model for intent embeddings...")
    minilm_model = SentenceTransformer("all-MiniLM-L6-v2")
    pattern_embeddings = minilm_model.encode(pattern_texts)
    
    # ===============================
    # 💾 Save Cache
    # ===============================
    
    print("💾 Saving all embeddings to cache...")
    save_cache(file_path, intents_path, df, embeddings, category_embeddings,
               categories, index, pattern_embeddings, pattern_texts, labels)
    
    print("\n✨ Embedding creation complete!")
    print(f"📊 Products indexed: {len(df)}")
    print(f"🏷️  Categories: {len(categories)}")
    print(f"🎯 Intent patterns: {len(pattern_texts)}")


if __name__ == "__main__":
    build_embeddings()
