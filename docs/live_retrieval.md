# m(m)ore Live Retrieval API Documentation

## Overview

The **Live Retrieval API** enables users to **upload, update, download, delete files**, and perform **semantic search** across indexed documents. This API is designed for retrieval-augmented generation (RAG) and search applications.

# Backend Server setup

Setup Instructions

1. **(Optional)** **Set Environment Variables**

If you would like to use a specific database and collection name please set the environment variables using the code below. Otherwise, the following default values will be used:

- Milvus uri = demo.db
- Milvus database name = my_db
- Collection name = my_documents

```bash
export MILVUS_URI="your_milvus_uri"
export MILVUS_DB="your_database_name"
export DEFAULT_COLLECTION="your_collection_name"
```

2. **Run the Server**

To start the server, run this command:

```bash
python3 -m mmore live-retrieval --host the_host --port the_port
```

This command:

- Starts the Uvicorn ASGI server on the specified host and port
- Loads the FastAPI application from the `src/mmore/run_live_retrieval.py` file

> **Important**: Keep this terminal window open. The backend runs in the foreground and closing the terminal will shut down the server.

---

# API Usage

## 📂 Upload, 🔁 Update, 🗑️ Delete, 📥 Download Endpoints

_Check the [Index API Documentation](./index_api.md)_

---

## 🔍 Context Retrieval

### 🔎 `POST /v1/retrieve`

**Search for files based on content similarity**

| Parameter | Type | Description |
| --- | --- | --- |
| `fileIds` | `List[str]` (body) | List of file IDs to search within |
| `maxMatches` | `int` | Maximum number of matches to return |
| `minSimilarity` | `float` | Minimum similarity score for results (ranges from -1.0 to 1.0) |
| `query` | `str` (body) | Search query |

- Searches for content within the specified files.
- Returns results sorted by similarity score.

**Response**:

```json
[
  {
    "fileId": "example123",
    "chunkId": "32",
    "content": "Content...",
    "similarity": 0.85
  },
  {
    "fileId": "example456",
    "chunkId": "2",
    "content": "Content...",
    "similarity": 0.78
  }
]
```

---

## How it works

- **Uploading**, **Processing**, **Indexing** ➜ _Check the [Index API Documentation](./index_api.md)_

- **Retrieval**:
  The query is converted into sparse and dense embeddings and compared by the Milvus database with documents from the authorized file ids. The list of results is narrowed down by the parameters `maxMatches` and `minSimilarity` and sorted with the most similar being given earlier.

## 🧰 Developer Notes

_Check the [Index API Documentation](./index_api.md)_

### 💡 Tips

- Avoid duplicate `fileId` unless using `PUT` to update.
- You can test endpoints via Swagger UI at `/docs`.