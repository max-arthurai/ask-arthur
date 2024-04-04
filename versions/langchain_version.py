import chromadb
from langchain.retrievers import EnsembleRetriever
from langchain_anthropic import ChatAnthropic
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms.huggingface_pipeline import HuggingFacePipeline
from langchain_community.retrievers import BM25Retriever
from langchain_community.vectorstores.chroma import Chroma
from langchain_core.language_models.chat_models import BaseLanguageModel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.retrievers import BaseRetriever
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI
import pandas as pd


def get_langchain_ensemble_retriever(embedding, k=3, retrieval='chroma') -> BaseRetriever:
    """Ensemble of classic retrieval and modern vector retrieval

    Classic retrieval: BM25 full-document retrieval
    Modern vector retrieval: dense-vector-embedded chunk retrieval
    https://python.langchain.com/docs/modules/data_connection/retrievers/ensemble

    TODO: rerank
    """
    arthur_index = pd.read_csv("../data/docs/arthur_index_315.csv").dropna()
    bm25_retriever = BM25Retriever.from_texts(arthur_index['text'])
    bm25_retriever.k = 1
    embedding = HuggingFaceEmbeddings(model_name=embedding, model_kwargs={'trust_remote_code': True})
    assert retrieval == 'chroma'  # todo allow other options
    persistent_client = chromadb.PersistentClient("data")
    langchain_chroma = Chroma(
        client=persistent_client,
        collection_name="arthur_index",
        embedding_function=embedding,
    ).as_retriever(search_kwargs={"k": k})
    return EnsembleRetriever(retrievers=[bm25_retriever, langchain_chroma], weights=[0.5, 0.5])


def get_langchain_llm(llm_name, temperature=0.0, max_tokens=250) -> BaseLanguageModel:
    if 'gpt' in llm_name:
        return ChatOpenAI(model_name=llm_name, max_tokens=max_tokens, temperature=temperature)
    elif 'claude' in llm_name:
        return ChatAnthropic(model_name=llm_name, max_tokens=max_tokens, temperature=temperature)
    else:
        return HuggingFacePipeline.from_model_id(
            model_id=llm_name,
            task="text-generation",
            pipeline_kwargs={"max_new_tokens": max_tokens, "temperature": temperature},
        )


def get_chain(retriever, llm, prompt_template):
    return (
        {"context": retriever, "question": RunnablePassthrough()} | prompt_template | llm | StrOutputParser()
    )


default_rag_template = ChatPromptTemplate.from_template("""
You are Arthur, the AI Performance & Security Engine
You are the chat interface for Arthur, the enterprise AI solution for monitoring, compliance, analysis, 
and development of AI applications ranging from simple tabular classifiers all the way to advanced LLM RAG applications.
You answer questions for users, always provide in depth explanations, and give example python code where appropriate.
IMPORTANT: if you are responding with code, that code MUST already exist in the retrieved context.

Answer the question BRIEFLY (1-2 sentences and/or a short list) based only on the following context:
===
{context}
===
Question: {question}
===
Answer: """)


def run(
    prompt: str,
    llm_name: str = "gpt-4-0125-preview",
    embedding_name: str = "nomic-ai/nomic-embed-text-v1.5",
    rag_template: ChatPromptTemplate = default_rag_template,
    temperature: float = 0.0,
    max_tokens: int = 250,
    k: int = 5
):
    assert rag_template.input_variables == ['context', 'question']
    retriever = get_langchain_ensemble_retriever(embedding_name, k=k)
    llm = get_langchain_llm(llm_name, temperature=temperature, max_tokens=max_tokens)
    chain = get_chain(retriever, llm, rag_template)
    return str(chain.invoke(prompt))
