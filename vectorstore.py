from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterable, List, Tuple

from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.docstore.document import Document

from config import settings


logger = logging.getLogger(__name__)


def get_embeddings_model() -> HuggingFaceEmbeddings:
    """
    Multilingual MiniLM embeddings for Hindi/English/Hinglish queries and content.
    """
    model_name = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    return HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs={"device": "cpu"},  # change to "cuda" if you have GPU
        encode_kwargs={"normalize_embeddings": True},
    )


def build_documents(chunks: Iterable[Tuple[str, dict]]) -> List[Document]:
    docs: List[Document] = []
    for text, metadata in chunks:
        if not text or not text.strip():
            continue
        docs.append(Document(page_content=text.strip(), metadata=metadata))
    return docs


def create_or_load_vectorstore(
    collection_name: str,
    docs: List[Document] | None = None,
) -> Chroma:
    """
    Create (and optionally populate) or load an existing Chroma collection.
    """
    persist_dir = Path(settings.chroma_db_dir)
    persist_dir.mkdir(parents=True, exist_ok=True)

    embeddings = get_embeddings_model()

    if docs:
        logger.info("Creating new Chroma collection '%s' with %d documents", collection_name, len(docs))
        vs = Chroma.from_documents(
            documents=docs,
            embedding=embeddings,
            persist_directory=str(persist_dir),
            collection_name=collection_name,
        )
    else:
        logger.info("Loading existing Chroma collection '%s'", collection_name)
        vs = Chroma(
            embedding_function=embeddings,
            persist_directory=str(persist_dir),
            collection_name=collection_name,
        )

    return vs

