from unittest.mock import patch, mock_open
import unittest
import pytest

import os
import requests
import time

os.chdir(os.path.dirname(os.path.dirname(__file__)))

from mmore.run_rag import rag, RAGInferenceConfig, read_queries, save_results
from mmore.utils import load_config

def test_load_config():
    path = os.path.relpath("examples/rag/config.yaml", os.path.dirname(os.path.dirname(__file__)))
    config = load_config(path, RAGInferenceConfig)

    assert config.rag

    assert config.rag.llm
    llm = config.rag.llm
    assert llm.llm_name == "HuggingFaceTB/SmolLM2-1.7B-Instruct"
    assert llm.max_new_tokens == 250

    assert config.rag.retriever
    retriever = config.rag.retriever
    assert retriever.db
    assert retriever.db.uri == "./demo.db"
    assert retriever.hybrid_search_weight == 0.5
    assert retriever.k == 5

    assert config.rag.system_prompt == "Use the following context to answer the questions.\n\nContext:\n{context}"

    assert config.mode == "local"
    assert config.mode_args
    assert config.mode_args.input_file == "./examples/rag/queries.jsonl"
    assert config.mode_args.output_file == "./examples/rag/output.json"

def test_load_config2():
    path = os.path.relpath("examples/rag/config_api.yaml", os.path.dirname(os.path.dirname(__file__)))
    config = load_config(path, RAGInferenceConfig)

    assert config.mode == "api"
    assert config.mode_args
    mode_args = config.mode_args

    assert mode_args.endpoint == "/rag"
    assert mode_args.port == 8000
    assert mode_args.host == "localhost"

def test_read_queries():
    path = os.path.relpath("examples/rag/config.yaml", os.path.dirname(os.path.dirname(__file__)))
    config = load_config(path, RAGInferenceConfig)

    queries = read_queries(config.mode_args.input_file)
    assert len(queries) == 4, f"Expected 4 queries, got {len(queries)}"
    assert queries[0]["input"] == "When was Barack Obama born?"

@patch("mmore.run_rag.save_results", side_effect=save_results)
@patch("mmore.run_rag.read_queries", side_effect=read_queries)
def test_local_query(read_queries, save_results):
    path = os.path.relpath("examples/rag/config.yaml", os.path.dirname(os.path.dirname(__file__)))

    start_time = time.time()
    rag(path)
    end_time = time.time()

    read_queries.assert_called_once()
    save_results.assert_called_once()

    path_input = "./examples/rag/queries.jsonl"
    path_output = "./examples/rag/output.json"

    assert os.path.exists(path_output)
    assert start_time <= os.path.getmtime(path_output)
    assert os.path.getmtime(path_output) <= end_time

if __name__ == '__main__':
    unittest.main()