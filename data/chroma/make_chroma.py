import chromadb
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_text_splitters import SentenceTransformersTokenTextSplitter
from sentence_transformers import SentenceTransformer

nomic_v15 = SentenceTransformer(
    model_name_or_path="nomic-ai/nomic-embed-text-v1.5",
    trust_remote_code=True
)


def nomic_embed(texts: list[str]) -> list[list[float]]:
    return [nomic_v15.encode(x).tolist() for x in texts]


def docs_from_csv(
    filename: str = "../docs/arthur_index_315.csv",
    tokens_per_chunk=250,
    chunk_overlap=50
):
    arthur_index_csv_loader = CSVLoader(filename, source_column='source', metadata_columns=['content_type'])
    data = arthur_index_csv_loader.load()
    splitter = SentenceTransformersTokenTextSplitter(
        tokens_per_chunk=tokens_per_chunk,
        chunk_overlap=chunk_overlap,
    )
    docs = [doc.to_json() for doc in splitter.split_documents(data)]
    return docs


print("making docs from csv")
arthur_index = docs_from_csv()
arthur_text = [
    x['kwargs']['page_content']
    for x in arthur_index
]
print("making embeddings of docs")
arthur_embeddings = nomic_embed(arthur_text)
arthur_text_metadata = [
    x['kwargs']['metadata']
    for x in arthur_index
]

print("making chroma db")
chroma_client = chromadb.PersistentClient()
collection = chroma_client.create_collection(name="arthur_index")
collection.add(
    documents=arthur_text,
    embeddings=arthur_embeddings,
    metadatas=arthur_text_metadata,
    ids=[f"id{i}" for i in range(len(arthur_text))]
)
