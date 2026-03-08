# Scripts Directory

This folder contains scripts that need to be run **once** to create embeddings.

## 📦 build_embeddings.py

Creates all embeddings required for the application:

- Product embeddings (using BAAI/bge-base-en-v1.5)
- Category embeddings
- Intent pattern embeddings (using all-MiniLM-L6-v2)
- FAISS index for fast similarity search

### When to Run

Run this script:

- **First time setup**: Before running main.py for the first time
- **Data updates**: Whenever `Products Data.xlsx` or `intents.json` changes
- **Fresh rebuild**: If you want to rebuild all embeddings from scratch

### How to Run

```bash
# From the root directory
python scripts/build_embeddings.py
```

### What it Does

1. Loads and cleans product data from `Products Data.xlsx`
2. Creates product embeddings and FAISS index
3. Generates category embeddings for semantic category detection
4. Builds intent pattern embeddings for intent classification
5. Saves all embeddings to `embeddings_cache/` folder

### Output Files

All embeddings are saved in the `embeddings_cache/` folder:

- `product_embeddings.npy` - Product embeddings
- `category_embeddings.npy` - Category embeddings
- `pattern_embeddings.npy` - Intent pattern embeddings
- `faiss_index.bin` - FAISS index for fast search
- `df_cache.pkl` - Cached dataframe and metadata
- `cache_metadata.json` - File hashes for validation

## ⚠️ Important Notes

- This script may take several minutes to run depending on your dataset size
- Requires sufficient disk space for storing embeddings
- Make sure the required models can be downloaded (internet connection needed on first run)
- The script will use the models from HuggingFace's model hub
