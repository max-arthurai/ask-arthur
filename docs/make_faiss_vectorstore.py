from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_community.embeddings import CohereEmbeddings, HuggingFaceEmbeddings
from langchain_community.vectorstores.faiss import FAISS
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import SentenceTransformersTokenTextSplitter
from typing import List


def docs_from_csv(
    filename: str = "arthur_index_315.csv",
    tokens_per_chunk=128, 
    chunk_overlap=32
) -> List[Document]:
    arthur_index_csv_loader = CSVLoader(filename, source_column='source', metadata_columns=['content_type'])
    data = arthur_index_csv_loader.load()
    print("splitting into chunks")
    splitter = SentenceTransformersTokenTextSplitter(tokens_per_chunk=tokens_per_chunk, chunk_overlap=chunk_overlap)
    docs = splitter.split_documents(data)
    return docs
print("loading arthur index docs...")
arthur_index_docs = docs_from_csv()

def make_faiss_vectorstores(names: List[str]) -> None:
    '''
    https://python.langchain.com/docs/integrations/vectorstores/faiss
    '''

    for name in names:
        print(name)

        if 'text-embedding' in name:
            embedding = OpenAIEmbeddings(model=name)
        elif 'embed-english' in name:
            embedding = CohereEmbeddings(model=name)
        else:
            embedding = HuggingFaceEmbeddings(model_name=name, model_kwargs={'trust_remote_code':True})
        faiss_vectorstore = FAISS.from_documents(arthur_index_docs, embedding)
        faiss_vectorstore.save_local(f"arthur_faiss_{name}")


embedding_models = [
    "sentence-transformers/all-MiniLM-L12-v2", 
    "nomic-ai/nomic-embed-text-v1.5",
    "text-embedding-3-small",
    "text-embedding-3-large",
    "text-embedding-ada-002",
    "embed-english-v3.0",
    "embed-english-light-v3.0",
    "embed-english-v2.0",
    "embed-english-light-v2.0",
]
make_faiss_vectorstores(embedding_models)