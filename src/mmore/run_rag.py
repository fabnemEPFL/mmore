import argparse

from typing import Literal, List, Dict, Union
from dataclasses import dataclass

from pathlib import Path
import json

import uvicorn

from fastapi import FastAPI, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

from langserve import add_routes

from pydantic import BaseModel

from .rag.llm import _OPENAI_MODELS as AVAILABLE_MODELS
from .rag.pipeline import RAGPipeline, RAGConfig
from .rag.types import MMOREOutput, MMOREInput, Msg
from .utils import load_config

import logging
RAG_EMOJI = "🧠"
logger = logging.getLogger(__name__)
logging.basicConfig(format=f'[RAG {RAG_EMOJI} -- %(asctime)s] %(message)s', level=logging.INFO, datefmt='%Y-%m-%d %H:%M:%S')

from dotenv import load_dotenv
load_dotenv() 

@dataclass
class LocalConfig:
    input_file: str
    output_file: str

@dataclass
class APIConfig:
    endpoint: str = '/rag'
    port: int = 8000
    host: str = '0.0.0.0'
    
@dataclass
class RAGInferenceConfig:
    rag: RAGConfig
    mode: str
    mode_args: Union[LocalConfig, APIConfig] = None

    def __post_init__(self):
        if self.mode_args is None and self.mode == 'api':
            self.mode_args = APIConfig()

class CompletionPayload(BaseModel):
    messages: list[Msg]
    model: str
    collection_name: str

def read_queries(input_file: Path) -> List[str]:
    with open(input_file, 'r') as f:
        return [json.loads(line) for line in f]

def save_results(results: List[Dict], output_file: Path):
    results = [
        {key: d[key] for key in {'input', 'context', 'answer'} if key in d} 
        for d in results
    ]   
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)

def create_api(rags: dict[str, RAGPipeline], config: RAGInferenceConfig, endpoint: str):
    app = FastAPI(
        title="RAG Pipeline API",
        description="API for question answering using RAG",
        version="1.0",
    )

    llm_default = config.rag.llm.llm_name

    rag = rags[llm_default]
    RUNNABLES = {llm_name: rag.rag_chain.with_types(input_type=MMOREInput, output_type=MMOREOutput) for llm_name, rag in rags.items()}
    runnable = RUNNABLES[llm_default]

    # Add routes for the RAG chain
    add_routes(
        app,
        runnable,
        path=endpoint,
        playground_type="chat"
    )

    @app.get("/health")
    def health_check():
        return {"status": "healthy"}
    
    # @app.get("/available_models")
    # def available_models():
    #     """Get the list of available models"""
    #     return {"message": [name for name, val in RUNNABLES.items() if val]}

    @app.post("/v1/chat/completions")
    async def openai_completions(payload: CompletionPayload):
        """ Mimics OpenAI API structure """

        # Stream response from LLM
        async def generate():
            for chunk in RUNNABLES[payload.model].stream(payload.messages):
                yield f"data: {json.dumps({'content': chunk})}\n\n"

        return StreamingResponse(generate(), media_type="text/event-stream")

    return app

def load_all(config):
    initial_llm = config.rag.llm.llm_name

    rags = dict()
    for llm_name in AVAILABLE_MODELS:
        config.rag.llm.llm_name = llm_name
        rags[llm_name] = RAGPipeline.from_config(config.rag)
    
    config.rag.llm.llm_name = initial_llm

    return rags

def rag(config_file):
    """Run RAG."""
    config = load_config(config_file, RAGInferenceConfig)
    
    logger.info('Creating the RAG Pipeline...')
    logger.info('RAG pipeline initialized!')

    if config.mode == 'local':
        queries = read_queries(config.mode_args.input_file)
        rag = RAGPipeline.from_config(config.rag)
        results = rag(queries, return_dict=True)
        save_results(results, config.mode_args.output_file)
    elif config.mode == 'api':
        llm_name = config.rag.llm.llm_name
        if llm_name == "all":
            rags = load_all(config)
        else:
            rags = {llm_name: RAGPipeline.from_config(config.rag)}

        app = create_api(rags, config, config.mode_args.endpoint)
        uvicorn.run(app, host=config.mode_args.host, port=config.mode_args.port)
    else:
        raise ValueError(f"Unknown inference mode: {config.mode}. Should be in [api, local]")
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-file", required=True, help="Path to the rag configuration file.")
    args = parser.parse_args()

    rag(args.config_file)