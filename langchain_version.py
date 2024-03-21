import chromadb
from langchain.retrievers import EnsembleRetriever
from langchain_anthropic import ChatAnthropic
from langchain_community.chat_models.cohere import ChatCohere
from langchain_community.embeddings import CohereEmbeddings, HuggingFaceEmbeddings
from langchain_community.llms.huggingface_pipeline import HuggingFacePipeline
from langchain_community.retrievers import BM25Retriever
from langchain_community.vectorstores.chroma import Chroma
from langchain_core.language_models.chat_models import BaseLanguageModel
from langchain_core.output_parsers import StrOutputParser as GetOutput
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.retrievers import BaseRetriever
from langchain_core.runnables import RunnablePassthrough as GetInput
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
import pandas as pd


def get_langchain_ensemble_retriever(name: str, k=10) -> BaseRetriever:
    """
    https://python.langchain.com/docs/modules/data_connection/retrievers/ensemble

    TODO: rerank
    """
    arthur_index = pd.read_csv("docs/arthur_index_315.csv").dropna()
    bm25_retriever = BM25Retriever.from_texts(arthur_index['text'])
    bm25_retriever.k = k
    name = name.replace("retrievers/arthur_faiss_", "")
    if 'text-embedding' in name:
        embedding = OpenAIEmbeddings(model=name)
    elif 'embed-english' in name:
        embedding = CohereEmbeddings(model=name)
    else:
        embedding = HuggingFaceEmbeddings(model_name=name, model_kwargs={'trust_remote_code': True})
    persistent_client = chromadb.PersistentClient()
    langchain_chroma = Chroma(
        client=persistent_client,
        collection_name="arthur_index",
        embedding_function=embedding,
    ).as_retriever(search_kwargs={"k": k})
    return EnsembleRetriever(retrievers=[bm25_retriever, langchain_chroma], weights=[0.5, 0.5])


def get_langchain_llm(name: str) -> BaseLanguageModel:
    """
    https://python.langchain.com/docs/modules/data_connection/retrievers/ensemble
    """
    if 'gpt' in name:
        return ChatOpenAI(model_name=name, max_tokens=2000, temperature=0)
    elif 'claude' in name:
        return ChatAnthropic(model_name=name, max_tokens=2000, temperature=0)
    elif 'command' in name:
        return ChatCohere(model_name=name, max_tokens=2000, temperature=0)
    else:
        return HuggingFacePipeline.from_model_id(
            model_id=name,
            task="text-generation",
            pipeline_kwargs={"max_new_tokens": 2000, "temperature": 0},
        )


rag_template = ChatPromptTemplate.from_template("""
You are Arthur, the AI Performance & Security Engine
You are the chat interface for Arthur, the enterprise AI solution for monitoring, compliance, analysis, 
and development of AI applications ranging from simple tabular classifiers all the way to advanced LLM RAG applications.
You answer questions for users, always provide in depth explanations, and give example python code where appropriate.
IMPORTANT: if you are responding with code, that code MUST already exist in the retrieved context.

Answer the question based only on the following context:
===
{context}
===
Question: {question}
===
Answer: """)


def get_chain(retriever, llm):
    return (
        {"context": retriever, "question": GetInput()} | rag_template | llm | GetOutput()
    )


def run(args):
    retriever = get_langchain_ensemble_retriever(
        name=f"retrievers/arthur_faiss_{args.embedding}",
        k=args.k
    )
    llm = get_langchain_llm(args.llm)
    chain = get_chain(retriever, llm)
    for text in chain.stream(args.prompt):
        print(text, end='', flush=True)
