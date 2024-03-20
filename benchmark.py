from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from typing import List

from langchain.generators.generators import get_langchain_llm
from langchain.retrievers.retrievers import get_langchain_ensemble_retriever
#
# class Retrieval(BaseModel):
#     retriever: BaseRetriever
#     name: Optional[str] = None
#     input_size: Optional[int] = None
#     cost: Optional[float] = None
#
#     @classmethod
#     def from_name(cls, name: str):
#         retriever = get_langchain_ensemble_retriever("retrievers/arthur_faiss_" + embname)
#         sizemap = {
#
#         }
#
# class Generation(BaseModel):
#     lm: BaseLanguageModel
#     name: str
#     input_size: int
#     output_size: int
#     cost: float
#
# class RAG(BaseModel):
#     retrieval: Retrieval
#     generation: Generation
#     name: Optional[str] = None
#
#     @classmethod
#     def from_names(cls, embname: str, llmname: str):
#
#         lm = get_langchain_llm(llmname)
#         return cls(
#             embedder=EmbeddingModel(
#                 retriever=retriever,
#             ),
#             language_model=LanguageModel(
#                 lm=lm
#             )
#         )

test_prompt = ChatPromptTemplate.from_template("""Answer the question based only on the following context:
{context}

Question: {question}
""")


def test(
    queries: List[str], 
    embedding_names: List[str],
    generator_names: List[str]
) -> None:
    for embname in embedding_names:
        print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n', embname)
        retriever = get_langchain_ensemble_retriever("retrievers/arthur_faiss_"+embname)
        for llmname in generator_names:
            print('+++++++++++++++\n', llmname)
            llm = get_langchain_llm(llmname)
            for q in queries:
                print('??????\n', q)
                chain = (
                    {"context": retriever, "question": RunnablePassthrough()}
                    | test_prompt
                    | llm
                    | StrOutputParser()
                )
                output = chain.invoke(q)
                print('\n', output, '\n')

# info needed:
# input size
# cost

# https://huggingface.co/sentence-transformers/all-MiniLM-L12-v2
# https://huggingface.co/nomic-ai/nomic-embed-text-v1.5
# https://openai.com/pricing
# https://platform.openai.com/docs/guides/embeddings/embedding-models
# https://cohere.com/pricing
# https://docs.cohere.com/reference/embed

embedding_models = [
    # "sentence-transformers/all-MiniLM-L12-v2", 
    "nomic-ai/nomic-embed-text-v1.5",
    # "text-embedding-3-small",
    # "text-embedding-3-large",
    "text-embedding-ada-002",
    # "embed-english-v3.0",
    # "embed-english-light-v3.0",
    # "embed-english-v2.0",
    # "embed-english-light-v2.0",
]


# info needed:
# input size
# output size
# cost

# https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.2
# https://docs.anthropic.com/claude/docs/models-overview
# https://openai.com/pricing
# https://platform.openai.com/docs/guides/text-generation
# https://cohere.com/pricing
# https://docs.cohere.com/reference/chat

llms = [
    # 'mistralai/Mistral-7B-Instruct-v0.2', # context 8k, cost 0
    # 'mistralai/Mistral-7B-v0.1', # context 8k, cost 0
    # 'gpt-3.5-turbo-0613', # context 8k, cost
    # 'gpt-4-turbo-0125', # context 8k, output 4k, cost,
    'gpt-3.5-turbo-0125', # context 128k, output 4k, cost
    # 'gpt-4-turbo-0125', # context 8k, output 4k, cost,
    # 'claude-3-opus-20240229', # 200k, output 4k, cost,
    # 'claude-3-sonnet-20240229', # 200k, output 4k, cost,
    # 'claude-3-haiku-20240307', # 200k, output 4k, cost,
    # 'command', #
    # 'command-r'
]

queries = [
    'what products does arthur offer', # site, answer is scope bench shield chat
    'what can Arthur do to support my tabular classifier', # scope, tabular, classification
    'image explainability bug', # scope, CV, explainability
    'send ground truth after inferences', # scope
    'llm testing', # bench
    'how does arthur reckon with social inequality', # scope, bias
    'what metrics are used for model performance tracking', # scope, arthur algorithms, performance, anomaly, drift, explainability
    'how to respond to change in distribution of data in production models', # scope, drift, anomaly
    'methods for reducing bias in machine learning models', # scope, bias
    'methods for improving application using an open source LLM', # bench
    'how to implement continuous monitoring of AI systems', # scope, shield, chat, blog
    'techniques for anomaly detection in model predictions', # scope, arthur algorithms
    'ways to ensure data privacy in AI applications', # shield, chat, blog
    'challenges in scaling AI monitoring systems', # scope, blog,
    'evaluating the robustness of NLP models and LLMs', # scope, bench
    'integrating external data sources for dynamic model updating', # scope, AWS, S3, SageMaker
    'how can i track a model that is classifying customer written feedback in Arthur', # scope, NLP, answer is nlp classification monitoring
    'how can i track a model that is summarizing customer written feedback in Arthur', # shield, LLM, answer is hallucination check
    'what do i need to prepare to enable explainability', # scope, answer is To enable explainability you will need:  1. A python script that wraps your models predict function 2. your serialized model file and 3. A requirements.txt with the dependencies to support the above
    'What are the additional requirements for on-prem deployment specific to the airgapped mode?', # scope, answer is "For airgap onprem deployment, you must have the following in addition to the online onprem deployment requirements: * An existing private container registry * Existing private Python registries (PyPI, Anaconda) when the model explanation feature is enabled * Access to your private container and Python registries
    'Is KMS required for Arthur\'s backup and restore capability?' # scope, answer is "No, but Arthur highly recommends that your EBS volumes are encrypted with KMS.",
    'how many parameters can a model have in arthur', # scope, answer is as many as you want bc arthur only monitoring inputs/outputs not the model object itself
    'How do I create the prediction to ground truth mapping?', # scope, answer is "The prediction to ground truth mapping needs to identify which columns are prediction and ground truth (GT) but also map the labels within the columns to one another. There are several ways to create this mapping, depending on your reference dataset: Method 1 - If your dataset contains a single prediction column and single GT column, map the name of the prediction column to its corresponding ground truth value ({"Name of predicted column": correspoding value in GT column}). Method 2 - If your dataset contains multiple prediction columns and a single ground truth column, map the name of each predicted column to its corresponding ground truth value ({"name of prediction column 1": corresponding value in GT column, "name of prediction column 2": corresponding value in GT column}). Method 3 - If your dataset contains multiple prediction and GT columns, map the name of each prediction column to the name of its corresponding GT column ({"name of prediction column1" : "name of GT 1 column", "name of prediction column2" : "name of GT column2"})."
    'What performance metrics does Arthur support for classification models?', # scope, answer is "Arthur's out of the box performance metrics for classification models include accuracy rate, balanced accuracy rate, AUC, recall, precision, specificity (TNR), F1, false positive rate, false negative rate. Arthur also supports user defined metrics for classification models. "
    'Can I restore a deleted Arthur model?', # scope, answer is ""No, you cannot restore your deleted model to what it once was. The most notable thing that occurs is that the inference data collected by your model are deleted immediately. This means that all of the production data you have been tracking will be deleted. However, there is enough information stored immediately after deletion to re-onboard your model object with its original reference dataset."
]

test(queries, embedding_models, llms)
