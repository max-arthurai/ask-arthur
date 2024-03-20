import json
from langchain.retrievers import EnsembleRetriever
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_community.embeddings import CohereEmbeddings, HuggingFaceEmbeddings
from langchain_community.retrievers import BM25Retriever
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import SentenceTransformersTokenTextSplitter
from typing import List, Union


def docs_from_csv(
    filename: str = "retrievers/arthur_index_315.csv",
    tokens_per_chunk=128, 
    chunk_overlap=32
):
    arthur_index_csv_loader = CSVLoader(filename, source_column='source', metadata_columns=['content_type'])
    data = arthur_index_csv_loader.load()
    splitter = SentenceTransformersTokenTextSplitter(tokens_per_chunk=tokens_per_chunk, chunk_overlap=chunk_overlap)
    docs = [doc.model_dump_json() for doc in splitter.split_documents(data)]
    return docs


# print("loading arthur data")
# arthur_index_docs = docs_from_csv()


# def save_docs_to_jsonl(docs, file_path)->None:
#     with open(file_path, 'w') as jsonl_file:
#         for doc in docs:
#             jsonl_file.write(doc.json() + '\n')
#
#
# save_docs_to_jsonl(arthur_index_docs, 'retrievers/arthur_index.jsonl')


def load_docs_from_jsonl(file_path) -> List[Document]:
    array = []
    with open(file_path, 'r') as jsonl_file:
        for line in jsonl_file:
            data = json.loads(line)
            obj = Document(**data)
            array.append(obj)
    return array


arthur_index_docs = load_docs_from_jsonl('retrievers/arthur_index.jsonl')


def get_langchain_ensemble_retriever(faiss_name: str, bm25k=2, faissk=2) -> BaseRetriever:
    """
    https://python.langchain.com/docs/modules/data_connection/retrievers/ensemble

    TODO: rerank
    """
    bm25_retriever = BM25Retriever.from_documents(arthur_index_docs)
    bm25_retriever.k = bm25k
    name = faiss_name.replace("retrievers/arthur_faiss_", "")
    if 'text-embedding' in faiss_name:
        embedding = OpenAIEmbeddings(model=name)
    elif 'embed-english' in name:
        embedding = CohereEmbeddings(model=name)
    else:
        embedding = HuggingFaceEmbeddings(model_name=name, model_kwargs={'trust_remote_code': True})
    faiss_vectorstore = FAISS.load_local(faiss_name, embedding, allow_dangerous_deserialization=True)
    faiss_retriever = faiss_vectorstore.as_retriever(search_kwargs={"k": faissk})

    return EnsembleRetriever(
        retrievers=[bm25_retriever, faiss_retriever], weights=[0.5, 0.5]
    )



